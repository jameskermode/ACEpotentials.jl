# `lammps-export` Correctness and CPU Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `lammps-export` library correct for multi-species models with the pair term, re-entrant, and ≤ 1.2x `pair_style pace recursive` per core in the exact `:polynomial` mode, on two reference models.

**Architecture:** All changes are in `export/` of the ACEsuit `lammps-export` branch: the export generator (`export/src/*.jl`) is rewritten stage by stage so the generated evaluator becomes isomorphic to ML-PACE's `ace_recursive.cpp` (per-neighbour stack kernel, species-block A accumulation, DAG AA with a C-tilde-seeded backward, no per-neighbour buffers), the C API gains an opaque workspace handle, and the LAMMPS plugin holds one workspace per OpenMP thread. Every step is gated on exactness against the previous generator (1e-13) and against the Julia `ETACEPotential` (1e-12), then timed by difference against `pace recursive`.

**Tech Stack:** Julia 1.12.2, JuliaC.jl 0.3.10 (`juliac --trim=safe`), EquivariantTensors 0.4.3, Polynomials4ML 0.5.8, StaticArrays; LAMMPS 22Jul2025 with the `pair_style ace` plugin (C++17, CMake, optional OpenMP); ML-PACE build for the comparator; Python 3 + ctypes/ASE for the C-API check.

**Spec:** `docs/plans/lammps_export_parity_design.md` (read it first; this plan argues from it).

## Global Constraints

- Every step is verified before it is timed: generated code vs `ETACEPotential`/`StackedCalculator` in Julia at **1e-12** (energies, forces, virial); library via the Python C API at **1e-12**; `pair_style ace` in LAMMPS vs Julia at **1e-10**, 1 and 2 MPI ranks; `OMP_NUM_THREADS=4` vs serial at **1e-12** once the workspace API exists.
- Each performance step (Tasks 5–7) keeps the previous generator's exported model as its reference: any force differing by more than **1e-13 relative** is a bug, not a speed-up.
- Hermite mode is compared to the **splinified** model at 1e-12 and its error vs the **fitted** model is reported in the test log; no tolerance is loosened anywhere; every test states its reference.
- Timing: single MPI rank pinned to one idle core (`taskset -c N`, `OMP_NUM_THREADS=1`), `timestep 0.0`, 100 steps, two runs within 3 % (else a third and the median), `pace recursive` re-run the same day on the same core, `/proc/loadavg` recorded before each block, no block on a contended core.
- `EquivariantTensors` and `ACEpotentials.ETModels` are **not** modified. `src/` is not modified.
- Branch `export/perf-parity` off `origin/lammps-export` (075d3859); nothing is pushed unless the maintainer says so. Commit after every task with the trailer lines given in the executor's session (if none are given, plain commits).
- `:polynomial` is the default mode; `:hermite_spline` stays, fixed, non-default.

## Cold-start facts (moriarty)

- Checkout: `~/ace-potentials-julia-1.2/ACEpotentials.jl`. It is on a host-local branch `lammps-export-verify` (one commit 39d853dc = the two-line Hermite dispatch fix; `stash@{0}` = a pre-existing Manifest edit — leave it). Julia: `julia` = 1.12.2. Never touch `~/ACEpotentials-jax` (another checkout; only its data file is read).
- Export project: `cd export && julia --project=. -e 'using Pkg; Pkg.instantiate()'` (already instantiated once; `log.instantiate` in `verify_cantor/`). Tests: `cd export/test && julia --project=.. runtests.jl` (43 pass at the tip; `lammps` and `mpi` arguments select the LAMMPS/MPI groups; `check_lammps_available()` looks for `lmp` on PATH).
- juliac recipe: `verify_cantor/compile_lib.jl <model.jl> <out.so> [cpu_target]` (JuliaC `ImageRecipe(trim_mode="safe", add_ccallables=true)` + `LinkRecipe`), 30–40 s per library.
- LAMMPS: `~/lammps/lammps-22Jul2025/build/lmp`; plugin build dir `verify_cantor/plugin_build/` (`aceplugin.so`, built from `export/lammps/plugin/cmake` with `BUILD_OMP=OFF`); runner `verify_cantor/run_lammps.sh <tag> <lib.so>` and input `verify_cantor/in.cantor_ace` (10 fresh LAMMPS starts on the held-out geometries; `LD_LIBRARY_PATH` lines inside are required). ML-PACE comparator: `/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-mlpace/lmp` with its build dir FIRST on `LD_LIBRARY_PATH`, then `/storage/eng/essswb/venvs/lammps-jax/lib`, `/software/easybuild/software/CUDA/12.9.1/lib64`, `/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib` (copy from `~/si-ace/ACEpotentials/acejax/bench/run_bench.sh`).
- Reference model 1 (Cantor): `verify_cantor/chain_cantor.jl` lines 20–40 build it (`ExtXYZ.load("/home/eng/essswb/ACEpotentials-jax/cantor1k_b_mh1.xyz")`, `train = data[1:250]`, `held = data[991:1000]`, `M.ace_model(; elements=(:Cr,:Mn,:Fe,:Co,:Ni), order=3, Ytype=:solid, level=M.TotalDegree(), max_level=6, pair_maxn=30, rin0cuts (rin 0, r0 per pair, rcut 6.25), init_WB=:zeros, init_Wpair=:onehot, ...)` — copy those lines verbatim), fitted `ps, st, E0s` in `verify_cantor/cantor_v010_params.jld2` (`JLD2.load(f, "ps", "st", "E0s")`). **Do not refit; never load `lsq_cantor.jld2` (1.7 GB).** Held-out LAMMPS geometries: `~/si-ace/spike_yace/cantor/cantor_{1..10}.data` (rotated copies of `held[k]`); 17-digit Julia references for every chain stage in `verify_cantor/ref_{1..10}.txt`; matched `.yace` for pace: same directory (see `in.bench_pace`).
- Existing comparison scripts to reuse: `verify_cantor/compare_common.py`, `compare_pylib.py` (ctypes C-API check), `compare_lammps.py` (dump vs reference), `in.bench_ace`, `in.bench_pace`, `profile/profile_export.jl` (stage timer).
- Shared host: check `cat /proc/loadavg` and `who` before any timing; yield to others; `nvidia-smi` is irrelevant (CPU work).

---

### Task 0: Branch, fixture harness, baseline

Creates the working branch and a reusable Julia harness that every later task calls: build the Cantor `StackedCalculator` from the saved parameters, evaluate the 10 held-out configurations through an exported model file, and compare to the Julia calculator. Establishes the baseline numbers.

**Files:**
- Create: `export/test/fixtures/cantor_fixture.jl`
- Create: `export/test/check_export.jl`
- Create: `export/bench/README.md` (protocol; filled in Task 4)

**Interfaces:**
- Produces: `load_cantor_fixture() -> (; model, ps, st, E0s, stacked, held, rcut)` where `stacked::StackedCalculator` is `(ETOneBody, ETPairModel, ETACE)` from `ETM.convert2et_full`; `site_sets(held, rcut) -> Vector{Vector{(Rs, Zs, Z0, ids)}}` neighbour sets per config; `check_export(model_file, stacked, held; tol, reference=:fitted) -> (maxdE, maxdF, maxdV)` which `Base.include`s the exported file into a fresh `Module`, evaluates `site_energy_forces_virial` on every site of every held-out config, assembles total energy/forces/virial, and compares to `stacked` (or to a passed calculator) — asserting `≤ tol` and printing the maxima.

- [ ] **Step 1: Create the branch**

```bash
cd ~/ace-potentials-julia-1.2/ACEpotentials.jl
git status --short            # must be clean apart from untracked verify_cantor/
git fetch origin
git checkout -b export/perf-parity origin/lammps-export   # 075d3859
git cherry-pick 39d853dc                                   # Hermite dispatch fix (replaced properly in Task 2)
git log --oneline -3
```

Expected: HEAD = cherry-picked fix on top of 075d3859.

- [ ] **Step 2: Write the fixture loader**

`export/test/fixtures/cantor_fixture.jl`:

```julia
# Cantor (CrMnFeCoNi) reference model: the fit from verify_cantor/chain_cantor.jl,
# reloaded from saved parameters.  Do not refit here (43 min).
using ACEpotentials, ExtXYZ, JLD2, StaticArrays, Lux, LuxCore, NeighbourLists
using ACEpotentials.Models, ACEpotentials.ETModels
const M = ACEpotentials.Models
const ETM = ACEpotentials.ETModels
using AtomsBase: atomic_number, position
using Unitful: ustrip, @u_str

const CANTOR_XYZ = "/home/eng/essswb/ACEpotentials-jax/cantor1k_b_mh1.xyz"
const CANTOR_PARAMS = joinpath(homedir(), "ace-potentials-julia-1.2/ACEpotentials.jl/verify_cantor/cantor_v010_params.jld2")

function load_cantor_fixture()
    data = ExtXYZ.load(CANTOR_XYZ)
    held = data[991:1000]
    # --- copy of chain_cantor.jl lines 24-40: MUST stay identical to the fit ---
    elements = (:Cr, :Mn, :Fe, :Co, :Ni)
    NZ = length(elements); rcut = 6.25
    r0 = M._default_rin0cuts(elements)          # per-pair r0 as the fit used
    rin0cuts = SMatrix{NZ, NZ}([(rin = 0.0, r0 = r0[i, j].r0, rcut = rcut) for i in 1:NZ, j in 1:NZ])
    model = M.ace_model(; elements = elements, order = 3, Ytype = :solid,
                          level = M.TotalDegree(), max_level = 6,
                          pair_maxn = 30, rin0cuts = rin0cuts,
                          init_WB = :zeros, init_Wpair = :onehot)
    # --- end copy ---
    ps, st, E0s = JLD2.load(CANTOR_PARAMS, "ps", "st", "E0s")
    stacked = ETM.convert2et_full(model, ps, st)        # (E0, pair, many-body)
    return (; model, ps, st, E0s, stacked, held, rcut)
end
```

Before committing, diff the model-construction lines against `verify_cantor/chain_cantor.jl:24-40` and make them identical (the saved `ps` only fits that exact spec; a mismatch shows up as a `Lux` parameter-shape error or a wrong energy).

- [ ] **Step 3: Write the export checker**

`export/test/check_export.jl`:

```julia
# Evaluate an exported model file on the held-out configs and compare to a Julia calculator.
include(joinpath(@__DIR__, "fixtures", "cantor_fixture.jl"))
using AtomsCalculators: potential_energy, forces, virial
using LinearAlgebra: norm

"Neighbour sets (Rs, Zs, Z0, js) for every site of `sys`, full list within rcut."
function site_sets(sys, rcut)
    nl = PairList(sys, rcut * u"Å")
    out = Vector{Tuple{Vector{SVector{3,Float64}}, Vector{Int}, Int, Vector{Int}}}()
    Zs_all = atomic_number(sys, :)
    for i in 1:length(sys)
        js, Rs = NeighbourLists.neigs(nl, i)
        push!(out, ([SVector{3,Float64}(ustrip.(u"Å", R)) for R in Rs],
                    [Int(Zs_all[j]) for j in js], Int(Zs_all[i]), collect(js)))
    end
    return out
end

"Total E, F, V of `sys` through the exported module `ex`."
function exported_efv(ex, sys, rcut)
    N = length(sys); E = 0.0
    F = zeros(SVector{3,Float64}, N); V = zeros(SMatrix{3,3,Float64,9})
    for (i, (Rs, Zs, Z0, js)) in enumerate(site_sets(sys, rcut))
        Ei, Fi, Vi = ex.site_energy_forces_virial(Rs, Zs, Z0)
        E += Ei; V += Vi
        for (k, j) in enumerate(js); F[j] += Fi[k]; F[i] -= Fi[k]; end
    end
    return E, F, V
end

function check_export(model_file, calc, held, rcut; tol, label = model_file)
    ex = Module(Symbol("Exported_", hash(model_file)))
    Base.include(ex, model_file)
    maxdE = maxdF = maxdV = 0.0
    for sys in held
        E, F, V = Base.invokelatest(exported_efv, ex, sys, rcut)
        Eref = ustrip(u"eV", potential_energy(sys, calc))
        Fref = [SVector{3,Float64}(ustrip.(u"eV/Å", f)) for f in forces(sys, calc)]
        Vref = SMatrix{3,3,Float64,9}(ustrip.(u"eV", virial(sys, calc)))
        maxdE = max(maxdE, abs(E - Eref) / length(sys))
        maxdF = max(maxdF, maximum(norm.(F .- Fref)))
        maxdV = max(maxdV, maximum(abs.(V .- Vref)))
    end
    println("$label: max|dE|/atom = $maxdE  max|dF| = $maxdF  max|dV| = $maxdV  (tol $tol)")
    @assert maxdE <= tol && maxdF <= tol && maxdV <= tol "$label exceeds tol=$tol"
    return (maxdE, maxdF, maxdV)
end
```

If `NeighbourLists.neigs` returns vectors in different units/ordering than assumed, adapt to what `chain_cantor.jl:98-123` does (it builds the same sets) — the reference is that script, not this sketch.

- [ ] **Step 4: Baseline run**

```bash
cd ~/ace-potentials-julia-1.2/ACEpotentials.jl/export
julia --project=. -e '
  include("test/check_export.jl"); fx = load_cantor_fixture()
  vc = joinpath(homedir(), "ace-potentials-julia-1.2/ACEpotentials.jl/verify_cantor")
  check_export(joinpath(vc, "cantor_poly_model.jl"), fx.stacked, fx.held, fx.rcut; tol = 1e-8, label = "tip :polynomial (no pair)")'
```

Expected: the assertion FAILS on forces by up to ~7 eV/Å (the tip drops the pair term) — record the number. Then re-run against the many-body-only stack to confirm the harness itself is right:

```bash
julia --project=. -e '
  include("test/check_export.jl"); fx = load_cantor_fixture()
  mb = ETM.StackedCalculator((fx.stacked.calcs[1], fx.stacked.calcs[3]))   # E0 + many-body (check index order with typeof.(getfield.(fx.stacked.calcs, :model)))
  vc = joinpath(homedir(), "ace-potentials-julia-1.2/ACEpotentials.jl/verify_cantor")
  check_export(joinpath(vc, "cantor_poly_model.jl"), mb, fx.held, fx.rcut; tol = 1e-12, label = "tip :polynomial vs E0+MB")'
```

Expected: PASS at 1e-12 (the finding measured 6.7e-14). If this fails, the harness (units, neighbour ordering, virial sign) is wrong — fix the harness until it reproduces `ref_k.txt` values, never the tolerance.

- [ ] **Step 5: Run the branch's own tests once**

```bash
cd ~/ace-potentials-julia-1.2/ACEpotentials.jl/export/test && julia --project=.. runtests.jl 2>&1 | tail -5
```

Expected: 43 passed (as on the tip).

- [ ] **Step 6: Commit**

```bash
git add export/test/fixtures/cantor_fixture.jl export/test/check_export.jl
git commit -m "export/test: Cantor reference fixture and exported-model checker (baseline: tip drops the pair term)"
```

---

### Task 1: Export the pair potential (A1)

**Files:**
- Modify: `export/src/export_ace_model.jl:60-90` (StackedCalculator handling), `:131-236` (writer calls)
- Modify: `export/src/write_radial.jl` (new `_write_pair_basis`)
- Modify: `export/src/write_evaluation.jl` (pair term in `site_energy`, `site_energy_forces`, `site_energy_forces_virial`)
- Test: `export/test/test_pair_export.jl` (new), registered in `export/test/runtests.jl`

**Interfaces:**
- Consumes: `ETPairModel` from `src/et_models/et_pair.jl` (`rembed = EdgeEmbed(EnvRBranchL(envelope, EmbedDP(trans, polys, SelectLinL W[N_PAIR, N_PAIRPOLYS, NZ^2])))`, `readout = SelectLinL(N_PAIR → 1, NZ)`); `ETOneBody`; `convert2et_full` builds `StackedCalculator((E0, pair, ace))` (order as in `src/et_models/et_calculators.jl:699-730` — check).
- Produces: generated constants `N_PAIRPOLYS`, `PAIRPOLY_A/B/C`, `PAIR_TRANSFORM_PARAMS` (per ordered pair), `PAIR_ENV_P`, `PAIR_ENV_RCUT`, `PAIR_C = (SVector{N_PAIRPOLYS}(...), …)` (one per ordered pair `(iz0, jz)`: the readout-folded coefficients `c_pair[iz0,jz][q] = Σ_n Wread[iz0][n] · W[(iz0,jz)][n,q]`), and generated functions `pair_energy(r, iz0, jz)::Float64`, `pair_energy_d(r, iz0, jz)::Tuple{Float64,Float64}`. `export_ace_model(::StackedCalculator, …)` raises on any calc that is not `ETOneBody`/`ETPairModel`/`ETACE`.

- [ ] **Step 1: Write the failing test**

`export/test/test_pair_export.jl`:

```julia
using Test
include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

@testset "Pair potential is exported" begin
    fx = load_cantor_fixture()
    build = mkpath(joinpath(@__DIR__, "build"))
    f = joinpath(build, "cantor_pair_poly.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f; for_library = false, radial_basis = :polynomial)
    src = read(f, String)
    @test occursin("const N_PAIRPOLYS", src)
    @test occursin("function pair_energy_d", src)
    # full stack (E0 + pair + many-body) reproduced to 1e-12
    dE, dF, dV = check_export(f, fx.stacked, fx.held, fx.rcut; tol = 1e-12, label = "poly+pair vs stacked")
    @test dF <= 1e-12
end

@testset "Unknown calculator in the stack is refused" begin
    fx = load_cantor_fixture()
    bogus = ETM.StackedCalculator((fx.stacked.calcs..., fx.stacked.calcs[1]))  # duplicate E0 is fine; use a non-ET calc if one is handy
    # The refusal test: a stack whose model is not one of the three types
    struct NotAModel end
    fake = (; model = NotAModel(), ps = nothing, st = nothing)
    @test_throws ErrorException export_ace_model(ETM.StackedCalculator((fake,)), tempname())
end
```

(If `StackedCalculator` cannot hold a plain NamedTuple, build the refusal test around whatever it does accept, e.g. a second `ETACE`; the behaviour under test is "anything other than E0/pair/ACE raises".)

Register in `runtests.jl` next to `test_multispecies.jl`.

- [ ] **Step 2: Run to verify it fails**

```bash
cd export/test && julia --project=.. -e 'include("test_pair_export.jl")' 2>&1 | tail -20
```

Expected: `occursin("const N_PAIRPOLYS")` fails.

- [ ] **Step 3: Extract the pair model in `export_ace_model(::StackedCalculator)`**

Replace the loop at `export_ace_model.jl:64-71`:

```julia
    e0_calc = nothing; pair_calc = nothing; etace_calc = nothing
    for subcalc in calc.calcs
        m = subcalc.model
        if m isa ETOneBody;        e0_calc = subcalc
        elseif m isa ETPairModel;  pair_calc = subcalc
        elseif m isa ETACE;        etace_calc = subcalc
        else error("export_ace_model: cannot export a $(typeof(m)) calculator (only ETOneBody, ETPairModel, ETACE)")
        end
    end
    etace_calc === nothing && error("StackedCalculator must contain an ETACE model")
```

and pass `pair_calc` through: `export_ace_model(etace_calc, filename; E0_dict, pair_calc, kwargs...)`. Add `pair_calc=nothing` to the `ETACEPotential` method's keywords, and `using ACEpotentials.ETModels: ETPairModel` at the top.

- [ ] **Step 4: Write the pair basis generator**

In `write_radial.jl` add:

```julia
# Pair potential: E_pair(site) = Σ_j env(r_ij) · dot(PAIR_C[(iz0,jz)], P(y_ij)),
# with the readout folded into per-pair polynomial coefficients at export time.
function _write_pair_basis(io, pair_calc, NZ)
    pm, ps, st = pair_calc.model, pair_calc.ps, pair_calc.st
    branch = pm.rembed.layer            # EnvRBranchL(envelope, rbasis)
    rb = branch.rbasis                  # EmbedDP(trans, polys, SelectLinL)
    polys = rb.basis                    # Chebyshev basis (check: rb.basis or rb.basis.l)
    pA, pB, pC = polys.refstate.A, polys.refstate.B, polys.refstate.C
    W = ps.rembed.rbasis.post.W         # [N_PAIR, N_PAIRPOLYS, NZ^2]  (verify path with keys(ps.rembed))
    Wr = ps.readout.W                   # [1, N_PAIR, NZ]
    env = st.rembed.envelope            # refstate (rcut, p) of PolyEnvelope1sR (verify with keys(st.rembed))
    trans = rb.trans.refstate.params    # Agnesi params per pair, same layout as the many-body radials
    nq = length(pA)
    println(io, "const N_PAIRPOLYS = $nq")
    println(io, "const PAIRPOLY_A = SVector{$nq,Float64}($(repr(collect(pA))))")
    println(io, "const PAIRPOLY_B = SVector{$nq,Float64}($(repr(collect(pB))))")
    println(io, "const PAIRPOLY_C = SVector{$nq,Float64}($(repr(collect(pC))))")
    println(io, "const PAIR_ENV_RCUT = $(Float64(env.rcut))")
    println(io, "const PAIR_ENV_P = $(Int(env.p))")
    println(io, "const PAIR_C = (")
    for iz0 in 1:NZ, jz in 1:NZ
        k = (iz0 - 1) * NZ + jz           # ordered pair index, same convention as RBASIS_W
        c = vec(Wr[1, :, iz0]' * W[:, :, k])   # length N_PAIRPOLYS
        println(io, "    SVector{$nq,Float64}($(repr(collect(c)))),  # ($iz0,$jz)")
    end
    println(io, ")")
    println(io, "const PAIR_TRANSFORM_PARAMS = (")
    for k in 1:NZ^2      # same loop body as write_radial.jl:120-135 (TRANSFORM_PARAMS), with `trans` and the pair rcut
        iz = (k - 1) ÷ NZ + 1; jz = (k - 1) % NZ + 1
        pr = trans[k]    # if `trans` is stored per symmetric pair, map (iz,jz) -> symmetric index exactly as write_radial.jl:122-126 does
        println(io, "    (rin=$(Float64(pr.rin)), req=$(Float64(pr.req)), rcut=$(Float64(env.rcut)), pin=$(Int(pr.pin)), pcut=$(Int(pr.pcut)), a=$(Float64(pr.a)), b0=$(Float64(pr.b0)), b1=$(Float64(pr.b1))),  # ($iz,$jz)")
    end
    println(io, ")")
    println(io, raw"""
@inline function _pair_env_d(r::Float64)
    s = r / PAIR_ENV_RCUT
    s >= 1.0 && return 0.0, 0.0
    sp = s^(-PAIR_ENV_P)
    e = (sp - 1.0) * (1.0 - s)
    de = (-PAIR_ENV_P * sp / s) * (1.0 - s) - (sp - 1.0)
    return e, de / PAIR_ENV_RCUT
end

@inline function pair_energy_d(r::Float64, iz0::Int, jz::Int)
    k = (iz0 - 1) * NZ + jz
    @inbounds p = PAIR_TRANSFORM_PARAMS[k]
    y, dy = agnesi_transform_d(r, p)
    e, de = _pair_env_d(r)
    e <= 0.0 && return 0.0, 0.0
    @inbounds c = PAIR_C[k]
    # Clenshaw-free three-term recurrence, values and derivatives
    P1 = PAIRPOLY_A[1]; dP1 = 0.0
    P2 = PAIRPOLY_A[2] * y + PAIRPOLY_B[2]; dP2 = PAIRPOLY_A[2]
    v = c[1] * P1 + c[2] * P2; dv = c[2] * dP2
    @inbounds for n in 3:N_PAIRPOLYS
        Pn = (PAIRPOLY_A[n] * y + PAIRPOLY_B[n]) * P2 + PAIRPOLY_C[n] * P1
        dPn = PAIRPOLY_A[n] * P2 + (PAIRPOLY_A[n] * y + PAIRPOLY_B[n]) * dP2 + PAIRPOLY_C[n] * dP1
        v += c[n] * Pn; dv += c[n] * dPn
        P1, P2, dP1, dP2 = P2, Pn, dP2, dPn
    end
    return e * v, de * v + e * dv * dy
end
@inline pair_energy(r::Float64, iz0::Int, jz::Int) = pair_energy_d(r, iz0, jz)[1]
""")
end
```

The exact field paths (`rb.basis` vs `rb.basis.l`, `ps.rembed.rbasis.post.W`, `st.rembed.envelope`) must be confirmed interactively (`keys(ps.rembed)`, `typeof(branch.rbasis)`) against a fixture-built `pair_calc`; the envelope formula is `_eval_env_1sr` in `src/et_models/convert.jl:233-237` (`(s^-p - 1)(1 - s)` for `s < 1`).

- [ ] **Step 5: Emit the pair term in the evaluation functions**

In `_write_evaluation_functions` (now called with `has_pair = pair_calc !== nothing`): in `site_energy`, after `val = dot(B, WB_iz)`, and in the two force functions, add a neighbour loop

```julia
    # pair potential
    @inbounds for j in 1:nneigh
        r = norm(Rs[j]); r > 1e-10 || continue
        jz = z2i(Zs[j])
        val += pair_energy(r, iz0, jz)
    end
```

and in the force assembly loop, before `forces[j] = -f`:

```julia
            ep, dep = pair_energy_d(r, iz0, z2i(Zs[j]))   # ordered pair: centre species first
            Ei += ep
            f = f + dep * rhat
            virial = virial - Rj * (dep * rhat)'   # virial function only
```

Call `_write_pair_basis(io, pair_calc, NZ)` from `export_ace_model` after the many-body radial section when `pair_calc !== nothing`; when it is `nothing`, emit `@inline pair_energy_d(r, iz0, jz) = (0.0, 0.0)` and `pair_energy(r, iz0, jz) = 0.0` so the evaluation code is mode-independent.

- [ ] **Step 6: Run the test**

```bash
cd export/test && julia --project=.. -e 'include("test_pair_export.jl")' 2>&1 | tail -20
```

Expected: PASS at 1e-12 on the full stack. If forces disagree only by the pair contribution's sign or by a factor, the envelope derivative or the `(iz0, jz)` order is wrong — compare per-site against `fx.stacked.calcs[2]` alone (a pair-only `StackedCalculator`) to isolate.

- [ ] **Step 7: Full export test suite, then commit**

```bash
cd export/test && julia --project=.. runtests.jl 2>&1 | tail -5
git add export/src export/test/test_pair_export.jl export/test/runtests.jl
git commit -m "export: emit the ETPairModel term (readout folded per ordered pair); refuse unknown calculators"
```

---

### Task 2: One pair-index convention, Hermite fix, `:polynomial` default, multi-species tests (A2, A3, A5)

**Files:**
- Modify: `export/src/write_radial.jl:4-18` and `:33-40` (`zz2pair_sym` removed), `export/src/codegen.jl:380-414` (dispatch), `export/src/splinify.jl` (`extract_hermite_spline_data` keyed by ordered pair)
- Modify: `export/src/export_ace_model.jl` (`radial_basis=:polynomial` default and docstring), `export/scripts/build_deployment.jl` (default), `export/README.md` (mode table)
- Modify: `export/src/write_evaluation.jl:45,84` (assert message)
- Test: `export/test/test_multispecies.jl` (rewrite)

**Interfaces:**
- Produces: a single generated helper `@inline pair_idx(iz::Int, jz::Int) = (iz - 1) * NZ + jz` emitted once (in `_write_species`), used by every per-pair table in both modes; Hermite tables `PAIR_k_*` exist for every ordered pair `k = pair_idx(iz, jz)` the model has.

- [ ] **Step 1: Rewrite the multi-species test (fails first)**

Replace `export/test/test_multispecies.jl` with a test that builds an NZ=3 model with asymmetric cutoffs, random (seeded) weights, converts, exports in **both** modes and compares to the ET calculators at 1e-12:

```julia
using Test, Random, StaticArrays, Lux, LuxCore
using ACEpotentials, ACEpotentials.Models, ACEpotentials.ETModels
const M = ACEpotentials.Models; const ETM = ACEpotentials.ETModels
include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))
using AtomsBuilder, AtomsBase

function three_species_model()
    elements = (:Ti, :Al, :V)
    rin0cuts = M._default_rin0cuts(elements)
    # asymmetric cutoffs: every ordered pair different
    rin0cuts = SMatrix{3,3}([(rin = 0.0, r0 = rin0cuts[i,j].r0, rcut = 4.6 + 0.2i + 0.3j) for i in 1:3, j in 1:3])
    m = M.ace_model(; elements, order = 3, Ytype = :solid, level = M.TotalDegree(), max_level = 6,
                      pair_maxn = 8, rin0cuts, init_WB = :glorot_normal, init_Wpair = :glorot_normal)
    ps, st = Lux.setup(MersenneTwister(7), m)
    return m, ps, st
end

function rattled_configs(n)
    at = bulk(:Ti, cubic = true) * (2, 2, 2)              # 32 atoms
    Z = [22, 13, 23]
    out = []
    for k in 1:n
        rng = MersenneTwister(100 + k)
        sys = FlexibleSystem([Atom(Z[mod1(i + k, 3)], position(at, i) + 0.15u"Å" * randn(rng, SVector{3})) for i in 1:length(at)], cell(at))
        push!(out, sys)
    end
    out
end

@testset "Multi-species export, both modes, 1e-12" begin
    m, ps, st = three_species_model()
    stacked = ETM.convert2et_full(m, ps, st)
    held = rattled_configs(5)
    rcut = maximum(c.rcut for c in M._default_rin0cuts((:Ti,:Al,:V))) + 1.0  # covers 4.6+0.2i+0.3j ≤ 6.1
    build = mkpath(joinpath(@__DIR__, "build"))
    # polynomial: exact vs fitted model
    fpoly = joinpath(build, "ms3_poly.jl")
    Base.invokelatest(export_ace_model, stacked, fpoly; radial_basis = :polynomial)
    @test check_export(fpoly, stacked, held, 6.1; tol = 1e-12, label = "NZ=3 :polynomial")[2] <= 1e-12
    # hermite: exact vs the SPLINIFIED model; error vs fitted reported, not asserted
    ace = stacked.calcs[end]                       # the ETACE calculator (check type)
    spl = ETM.splinify(ace.model, ace.ps, ace.st; Nspl = 50)
    calc50 = ETM.ETACEPotential(spl, ace.ps, ace.st, ace.rcut)   # follow chain_cantor.jl:130-136 for the exact call
    stacked50 = ETM.StackedCalculator((stacked.calcs[1:end-1]..., calc50))
    fherm = joinpath(build, "ms3_hermite50.jl")
    Base.invokelatest(export_ace_model, stacked50, fherm; radial_basis = :hermite_spline)
    @test check_export(fherm, stacked50, held, 6.1; tol = 1e-12, label = "NZ=3 :hermite_spline vs splinified")[2] <= 1e-12
    dF_fit = check_export_report(fherm, stacked, held, 6.1; label = "NZ=3 :hermite_spline vs FITTED (Nspl=50, informational)")
    @info "Hermite Nspl=50 error vs fitted model" dF_fit
end

@testset "> 256 neighbours is not a silent cap" begin
    m, ps, st = three_species_model()
    stacked = ETM.convert2et_full(m, ps, st)
    f = joinpath(mkpath(joinpath(@__DIR__, "build")), "ms3_poly_dense.jl")
    Base.invokelatest(export_ace_model, stacked, f; radial_basis = :polynomial)
    ex = Module(:Dense); Base.include(ex, f)
    Rs = [SVector(2.0 + 0.01k, 0.1k, -0.05k) for k in 1:300]; Zs = fill(22, 300)
    # until Task 6 removes the cap this must be a clear error, afterwards it must evaluate
    r = try; ex.site_energy(Rs, Zs, 22); catch e; e; end
    @test r isa Number || occursin("neighbours", sprint(showerror, r))
end
```

Add `check_export_report` to `check_export.jl`: same as `check_export` but returns the maxima without asserting.

- [ ] **Step 2: Run to verify it fails**

```bash
cd export/test && julia --project=.. -e 'include("test_multispecies.jl")' 2>&1 | tail -20
```

Expected: the Hermite export either errors on `zz2pair_sym` indexing or fails the 1e-12 comparison for the asymmetric pairs; the polynomial branch should already pass (post Task 1).

- [ ] **Step 3: Single pair-index convention**

In `_write_species` (export_ace_model.jl:~300) add after `z2i`:

```julia
# Ordered species-pair index (centre first) used by EVERY per-pair table
@inline pair_idx(iz::Int, jz::Int) = (iz - 1) * NZ + jz
```

Delete both `zz2pair_sym` definitions (`write_radial.jl:13-16`, `:35-38`) and `_write_spline_radial_basis_header`'s copy. In `write_radial.jl:225-249` replace `pair_idx = (iz - 1) * NZ + jz` by `pair_idx(iz, jz)`. In `codegen.jl:387,403` replace `zz2pair_sym(iz, jz)` by `pair_idx(iz, jz)`, and make `generate_hermite_spline_code` emit tables for **every ordered pair** `k = pair_idx(iz, jz)`, `k = 1:NZ^2`, from `hermite_data[k]`; in `splinify.jl`'s `extract_hermite_spline_data` key the dictionary by that same ordered index (read how it currently enumerates pairs and what `TransSelSplines`' per-pair table index is; `chain_cantor.jl` and the host commit 39d853dc show the two-line version of this fix — this task replaces it with the one convention). Remove the cherry-picked commit's now-redundant lines if they conflict.

- [ ] **Step 4: Defaults, cap message, docs**

- `export_ace_model.jl` docstring and signature: `radial_basis::Symbol=:polynomial` (already the default in code; the docstring at `:98-125` says Hermite is "exact, ~3-4x faster" — rewrite: polynomial = exact and default; hermite = approximate, error scales like `(h·N_POLYS)^3·N_POLYS` on derivatives, for small-N learned radials).
- `build_deployment.jl`: default `radial_basis=:polynomial`; grep for `:hermite_spline` defaults in `scripts/` and `examples/etace_lammps_tutorial.jl` and switch them.
- `write_evaluation.jl:45,84`: `@assert nneigh <= MAX_NEIGHBORS "site has $nneigh neighbours; this export supports at most $MAX_NEIGHBORS (re-export after Task 6 removes the cap)"`.
- `README.md` mode table: replace with `| :polynomial | exact (1e-12 vs fitted) | default | ~150 µs/site after Task 5 (fill in from Task 8) | any model |` and `| :hermite_spline | approximate: 2.7e-4 eV/Å at Nspl=50, 3e-6 at Nspl=200 on the Cantor model | opt-in | … | learned radials, small N_POLYS |`; delete "machine precision" and "3–4x faster".

- [ ] **Step 5: Run the tests**

```bash
cd export/test && julia --project=.. -e 'include("test_multispecies.jl")' 2>&1 | tail -20
cd export/test && julia --project=.. runtests.jl 2>&1 | tail -5
```

Expected: both modes pass at 1e-12 against their references; the informational Hermite-vs-fitted number is printed (~1e-4–1e-3 for this random model); whole suite green.

- [ ] **Step 6: Commit**

```bash
git add export/src export/scripts export/README.md export/examples export/test/test_multispecies.jl export/test/check_export.jl
git commit -m "export: one ordered pair index for all per-pair tables (fixes Hermite for NZ>=2); :polynomial default; NZ=3 asymmetric tests at 1e-12 in both modes"
```

---

### Task 3: LAMMPS and Hermite tests at the real tolerances; CI (A4)

**Files:**
- Modify: `export/test/test_lammps.jl` (tolerance `:224`, pair term, 2 ranks), `export/test/test_hermite_accuracy.jl:233,282` (references stated), `.github/workflows/export-ci.yml`
- Create: `export/lammps/test/run_two_ranks.sh`

**Interfaces:**
- Consumes: Task 1's pair export (LAMMPS energies now include it), Task 0's `check_export.jl` for the Julia reference values.
- Produces: `export/lammps/test/compare_dump.py <dump> <ref.txt> <tol>` (copy of `verify_cantor/compare_lammps.py` generalised to any model) used by CI and by Task 8.

- [ ] **Step 1: Tighten `test_lammps.jl`**

At `test_lammps.jl:224` the comparison to Python is at 1e-6. Change the reference to Julia (`check_export`'s `exported_efv` on the same geometry the LAMMPS input builds — write the geometry out with `write_data` and read it back so both sides evaluate identical coordinates) and the tolerance to **1e-10** on forces and energy/atom; keep the Python comparison as a second check at 1e-12 (library vs Julia) so a failure is attributable. Add a two-rank run (`mpirun -np 2 lmp …`) compared to the one-rank dump at 1e-12 (copy the pattern of `acejax/lammps/test_si_bundle.sh` in the ACEpotentials `pr/lammps-throughput` branch: dump both, `compare_dump.py`).

- [ ] **Step 2: Make the Hermite tests say what they test**

`test_hermite_accuracy.jl:233,282`: keep `atol=1e-8` **against the splinified reference** with a comment saying so; add, for each Nspl the test exports, an `@info` line reporting max |ΔF| against the *fitted* model (from `check_export_report`), and for the `:polynomial` export assert 1e-12 against the fitted model.

- [ ] **Step 3: CI**

In `.github/workflows/export-ci.yml`: the Julia test job runs `test_multispecies.jl` (both modes) and `test_pair_export.jl`; the LAMMPS job builds the plugin with `-DBUILD_OMP=OFF` **until Task 6**, runs `test_lammps.jl` at 1e-10, then the two-rank script. Keep the JuliaC step mandatory (it is: `b5a77f4e`).

- [ ] **Step 4: Run what can run on the host**

```bash
cd export/test && julia --project=.. runtests.jl lammps 2>&1 | tail -15
```

Expected: LAMMPS group passes at 1e-10 (with the pair term now present on both sides); 2-rank dump agrees at 1e-12.

- [ ] **Step 5: Commit**

```bash
git add export/test .github/workflows/export-ci.yml export/lammps/test
git commit -m "export/test: LAMMPS vs Julia at 1e-10 with pair term, two ranks; Hermite tests state their reference; CI runs both modes"
```

---

### Task 4: Large reference model, benchmark harness, baseline rows (measurement infrastructure)

**Files:**
- Create: `export/bench/fit_tial_order4.jl`, `export/bench/bench_parity.sh`, `export/bench/make_pace_basis.md` (pointer), `export/bench/README.md`
- Create (host, untracked artefacts): `verify_cantor/../bench_parity/` outputs

**Interfaces:**
- Produces: `export/bench/bench_parity.sh <model-tag> <lib.so> <yace> <nsteps>` printing one row `tag  ace_ms_per_step  pace_ms_per_step  ratio  loadavg` from pinned single-core runs of `pair_style ace` and `pair_style pace recursive` on the same 1728–2000-atom box; a saved TiAl order-4 model `bench_parity/tial_o4_params.jld2` + fixture loader `load_tial_fixture()` in `export/test/fixtures/tial_fixture.jl` (same shape as the Cantor one).

- [ ] **Step 1: Fit the large model once**

`export/bench/fit_tial_order4.jl` (run with `julia --project=export -t 8`; ~30–60 min; save `ps, st, E0s`):

```julia
using ACEpotentials, ACEfit, JLD2, StaticArrays, Lux
const M = ACEpotentials.Models
data, _, meta = ACEpotentials.example_dataset("TiAl_tutorial")
train = data[1:5:end]; held = data[2:50:end][1:10]
elements = (:Ti, :Al); rcut = 5.5; NZ = 2
r0 = M._default_rin0cuts(elements)
rin0cuts = SMatrix{NZ,NZ}([(rin = 0.0, r0 = r0[i,j].r0, rcut = rcut) for i in 1:NZ, j in 1:NZ])
# choose max_level so that length(model.basis) ≈ 1500-2500 at order 4 (try 10, 11, 12 and print the size)
model = M.ace_model(; elements, order = 4, Ytype = :solid, level = M.TotalDegree(), max_level = 11,
                      pair_maxn = 20, rin0cuts, init_WB = :zeros, init_Wpair = :onehot)
@info "many-body basis size" length(model.basis)   # adjust max_level until 1500 ≤ n ≤ 2500
pot = ACEpotentials.ACEPotential(model)
acefit!(train, pot; energy_key = "energy", force_key = "force", virial_key = "virial",
        solver = ACEfit.BLR(), repulsion_restraint = true, verbose = false)   # keys as in the tutorial dataset (check meta)
ACEpotentials.compute_errors(held, pot; energy_key = "energy", force_key = "force", virial_key = "virial")
JLD2.jldsave(joinpath(@__DIR__, "..", "..", "bench_parity", "tial_o4_params.jld2"); ps = pot.ps, st = pot.st, E0s = pot.model.Vref)   # match how chain_cantor.jl saved E0s
```

Follow `chain_cantor.jl:41-72` for the exact `acefit!`/`compute_errors` keyword names on this branch. Write `export/test/fixtures/tial_fixture.jl` mirroring the Cantor fixture (same construction lines, `convert2et_full`).

- [ ] **Step 2: Matched pace basis for TiAl**

Build a `.yace` of the same many-body size with `pyace` exactly as `~/si-ace/ACEpotentials/acejax/bench/make_pace_basis.py <n_B> <order> <lmax> <rcut> <out.ace>` does (cp39 venv notes in that script's docstring; random coefficients are fine for timing), two elements. The Cantor comparator already exists (`~/si-ace/spike_yace/cantor/*.yace`, see `verify_cantor/in.bench_pace`).

- [ ] **Step 3: The benchmark script**

`export/bench/bench_parity.sh`:

```bash
#!/usr/bin/env bash
# bench_parity.sh <tag> <libace.so> <pace.ace|.yace> <steps>   -- one pinned core, timestep 0.0
set -uo pipefail
TAG=$1; LIB=$(readlink -f $2); YACE=$(readlink -f $3); STEPS=${4:-100}; CORE=${CORE:-31}
HERE=$(cd "$(dirname "$0")" && pwd)
LMP_ACE=~/lammps/lammps-22Jul2025/build/lmp
PLUGIN=${PLUGIN:-~/ace-potentials-julia-1.2/ACEpotentials.jl/verify_cantor/plugin_build/aceplugin.so}
LMP_PACE=/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-mlpace/lmp
V=/storage/eng/essswb/venvs/lammps-jax
export OMP_NUM_THREADS=1
echo "# loadavg $(cat /proc/loadavg) $(date)"
run() { # $1 = lmp, $2 = input, $3.. = -var args
  local lmp=$1 in=$2; shift 2
  taskset -c $CORE $lmp -in $in -log none -screen /tmp/bench_$TAG.$$ "$@" >/dev/null 2>&1
  local t=$(grep -m1 "Loop time of" /tmp/bench_$TAG.$$ | awk '{print $4}')
  python3 -c "print(1000*$t/$STEPS)"
}
LD_LIBRARY_PATH=$(dirname $LIB):/software/easybuild/software/GCCcore/14.3.0/lib64:${LD_LIBRARY_PATH:-}
A1=$(run $LMP_ACE $HERE/in.bench_ace -var lib $LIB -var plugin $PLUGIN -var steps $STEPS)
A2=$(run $LMP_ACE $HERE/in.bench_ace -var lib $LIB -var plugin $PLUGIN -var steps $STEPS)
export LD_LIBRARY_PATH=$(dirname $LMP_PACE):$V/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:${LD_LIBRARY_PATH:-}
P1=$(run $LMP_PACE $HERE/in.bench_pace -var yace $YACE -var steps $STEPS)
P2=$(run $LMP_PACE $HERE/in.bench_pace -var yace $YACE -var steps $STEPS)
python3 - <<EOF
a=[$A1,$A2]; p=[$P1,$P2]
def agree(x): return abs(x[0]-x[1])/min(x) <= 0.03
print(f"$TAG  ace_ms/step={sum(a)/2:.2f} (agree={agree(a)})  pace_recursive_ms/step={sum(p)/2:.2f} (agree={agree(p)})  ratio={(sum(a)/2)/(sum(p)/2):.2f}")
EOF
```

`in.bench_ace` / `in.bench_pace` are copied from `verify_cantor/` (they build the 2000-atom-class box; ensure `pair_style pace recursive`, not `product`, and `timestep 0.0`). If either pair disagrees by > 3 %, re-run a third time and take the median by hand; record the loadavg line with every row.

- [ ] **Step 4: Baseline rows**

With the Task 3 generator (pair term included, tip kernel), export both reference models in `:polynomial` and `:hermite_spline` (Nspl=50), compile, and run:

```bash
cd ~/ace-potentials-julia-1.2/ACEpotentials.jl
for tag in cantor_poly cantor_h50 tial_poly tial_h50; do CORE=31 export/bench/bench_parity.sh $tag lib/libace_$tag.so $CANTOR_YACE or $TIAL_YACE 100; done | tee bench_parity/rows_task4.txt
```

Expected: Cantor `:polynomial` ≈ 7x, Hermite ≈ 2.6x (from the finding); TiAl rows are new. Write them into `export/bench/README.md` with the date and loadavg.

- [ ] **Step 5: Commit**

```bash
git add export/bench export/test/fixtures/tial_fixture.jl
git commit -m "export/bench: parity benchmark harness (pinned core, pace recursive comparator) and TiAl order-4 reference model; baseline rows"
```

---

### Task 5: Sparse radial mixing, pruning, integer powers (B1)

**Files:**
- Modify: `export/src/write_radial.jl:100-114` (RBASIS_W), `:118-136` (params), `:138-205` (transform), `:210-250` (`_evaluate_Rnl_pair`, `_d_pair`)
- Modify: `export/src/codegen.jl:180-199` (Hermite tables pruned to used rows)
- Test: `export/test/test_generator_parity.jl` (new; reused by Tasks 6–7)

**Interfaces:**
- Produces: generated constants `RNL_USED = (…)` (the (n,l) indices any A function uses), per ordered pair `RBASIS_ROWS_k::SVector{m_k,Int}` (which `Rnl` rows are nonzero for pair k) and either `RBASIS_SEL_k::SVector{m_k,Int}` (one-hot: the polynomial index feeding each row) or `RBASIS_Wk::SMatrix{m_k, N_POLYS}` (dense); `evaluate_Rnl_d(r, iz, jz)` keeps its signature and return type (`SVector{N_RNL}` pair) in this task — the width change comes with the kernel in Task 6. `TRANSFORM_PARAMS` carries `pin::Int`, `pcut::Int`.
- Test helper: `generator_parity(export_fn_old, export_fn_new, fixture; tol=1e-13)` — exports the same calculator with the previous commit's generator (checked out into a temp dir via `git show <sha>:export/src/... > tmp/`) and the current one, evaluates both on the held-out configs, asserts relative agreement.

- [ ] **Step 1: Write the parity test**

`export/test/test_generator_parity.jl`:

```julia
using Test
include(joinpath(@__DIR__, "check_export.jl"))
const REF_SHA = get(ENV, "EXPORT_REF_SHA", "HEAD~1")   # generator to compare against

function export_with(sha, calc, file; mode)
    tmp = mktempdir()
    for f in ("export_ace_model.jl", "write_radial.jl", "write_evaluation.jl", "write_c_interface.jl", "codegen.jl", "splinify.jl")
        write(joinpath(tmp, f), read(`git -C $(dirname(dirname(@__DIR__))) show $sha:export/src/$f`, String))
    end
    m = Module(Symbol("Gen_", replace(sha, r"[^A-Za-z0-9]" => "_")))
    Base.include(m, joinpath(tmp, "export_ace_model.jl"))
    Base.invokelatest(m.export_ace_model, calc, file; for_library = false, radial_basis = mode)
end

@testset "generator parity vs $REF_SHA" begin
    for (name, fx) in (("cantor", load_cantor_fixture()), ("tial", load_tial_fixture()))
        build = mkpath(joinpath(@__DIR__, "build"))
        fold = joinpath(build, "$(name)_ref.jl"); fnew = joinpath(build, "$(name)_new.jl")
        export_with(REF_SHA, fx.stacked, fold; mode = :polynomial)
        Base.invokelatest(export_ace_model, fx.stacked, fnew; for_library = false, radial_basis = :polynomial)
        exo = Module(:Old); Base.include(exo, fold); exn = Module(:New); Base.include(exn, fnew)
        worst = 0.0
        for sys in fx.held
            Eo, Fo, Vo = Base.invokelatest(exported_efv, exo, sys, fx.rcut)
            En, Fn, Vn = Base.invokelatest(exported_efv, exn, sys, fx.rcut)
            scale = max(maximum(norm.(Fo)), 1.0)
            worst = max(worst, abs(Eo - En) / abs(Eo), maximum(norm.(Fo .- Fn)) / scale, maximum(abs.(Vo .- Vn)) / max(maximum(abs.(Vo)), 1.0))
        end
        @info "generator parity $name" worst
        @test worst <= 1e-13
        @test check_export(fnew, fx.stacked, fx.held, fx.rcut; tol = 1e-12, label = "$name new vs fitted")[2] <= 1e-12
    end
end
```

- [ ] **Step 2: Run it before changing anything**

```bash
cd export/test && EXPORT_REF_SHA=HEAD julia --project=.. -e 'include("test_generator_parity.jl")' 2>&1 | tail -8
```

Expected: PASS trivially (same generator) — this proves the harness; from now on `EXPORT_REF_SHA` defaults to `HEAD~1`.

- [ ] **Step 3: Emit mixing from `W`'s structure**

In `_write_etace_radial_basis`, after computing `W_radial`, detect and emit:

```julia
    abasis_rnl = unique(first.(collect(tensor.abasis.spec)))          # Rnl indices any A uses
    println(io, "const RNL_USED = $(repr(Tuple(sort(abasis_rnl))))")
    onehot = all(all(x -> x == 0.0 || x == 1.0, W_radial[:, :, k]) && all(sum(W_radial[:, :, k] .!= 0, dims = 2) .<= 1) for k in 1:n_pairs)
    println(io, "const RBASIS_ONEHOT = $onehot")
    for k in 1:n_pairs
        Wk = W_radial[:, :, k]
        rows = [t for t in sort(abasis_rnl) if any(Wk[t, :] .!= 0)]
        println(io, "const RBASIS_ROWS_$k = SVector{$(length(rows)),Int}($(repr(rows)))")
        if onehot
            sel = [findfirst(!=(0.0), Wk[t, :]) for t in rows]
            println(io, "const RBASIS_SEL_$k = SVector{$(length(rows)),Int}($(repr(sel)))")
        else
            println(io, "const RBASIS_W_$k = SMatrix{$(length(rows)),$(n_polys),Float64,$(length(rows)*n_polys)}($(repr(vec(Wk[rows, :]))))")
        end
    end
```

(`tensor` must be passed into the radial writer — add the argument.) Then replace the dense products at `write_radial.jl:225-226,247-249` by a per-pair generated function

```julia
@inline function _mix_$k(P_env::SVector{N_POLYS,Float64}, dP_env::SVector{N_POLYS,Float64})
    Rnl = zero(MVector{N_RNL,Float64}); dRnl = zero(MVector{N_RNL,Float64})
    @inbounds for (i, t) in enumerate(RBASIS_ROWS_$k)
        q = RBASIS_SEL_$k[i]; Rnl[t] = P_env[q]; dRnl[t] = dP_env[q]     # one-hot
    end
    return SVector(Rnl), SVector(dRnl)
end
```

(dense variant: `v = RBASIS_W_$k * P_env; Rnl[RBASIS_ROWS_$k[i]] = v[i]`), dispatched by `pair_idx` through an `if`-chain as `codegen.jl` does for Hermite. Emit `pin`, `pcut` as `Int` in `TRANSFORM_PARAMS` (`write_radial.jl:134-136` currently `Float64(p.pin)`) so `s^p.pin` is an integer power. Prune: in `codegen.jl:180-199` write Hermite `F`/`G` rows only for `RBASIS_ROWS_k` (and the lookup fills only those rows).

- [ ] **Step 4: Parity, exactness, timing**

```bash
cd export/test && julia --project=.. -e 'include("test_generator_parity.jl")' 2>&1 | tail -8   # 1e-13 vs HEAD~1... use the SHA of Task 4's commit
cd export/test && julia --project=.. runtests.jl 2>&1 | tail -5
# compile + bench (Cantor + TiAl, :polynomial and :hermite_spline)
julia --project=export verify_cantor/compile_lib.jl build/cantor_new.jl lib/libace_cantor_b1.so
CORE=31 export/bench/bench_parity.sh cantor_b1 lib/libace_cantor_b1.so $CANTOR_YACE 100 | tee -a bench_parity/rows.txt   # CANTOR_YACE = the file verify_cantor/in.bench_pace points at under ~/si-ace/spike_yace/cantor/
```

Expected: parity ≤ 1e-13; `:polynomial` Cantor from ~522 to ~150 µs/site (ratio ~7x → ~2x); Hermite tables 4–5x smaller. Record rows.

- [ ] **Step 5: Commit**

```bash
git add export/src export/test/test_generator_parity.jl
git commit -m "export: radial mixing emitted from W's sparsity (one-hot gather / dense GEMV), unused (n,l) pruned, integer transform powers -- B1"
```

---

### Task 6: Per-neighbour kernel, workspace API, plugin and Python calculator (B2, §3)

**Files:**
- Rewrite: `export/src/write_evaluation.jl` (entire evaluation section: no `WORK_*`, no `MAX_NEIGHBORS`)
- Modify: `export/src/write_c_interface.jl` (workspace handle on every `ace_site_*` and the batch entry; `ace_workspace_new/free`)
- Modify: `export/src/write_radial.jl`, `export/src/codegen.jl` (per-neighbour evaluators return only the pair's used rows: `evaluate_rnl_k(r) -> (SVector{m_k}, SVector{m_k})`)
- Modify: `export/lammps/plugin/src/pair_ace.cpp:120-175` (symbols), `:392-636` (compute), `pair_ace.h`; `export/lammps/plugin/cmake/CMakeLists.txt:49` (OpenMP default stays ON now that it is safe)
- Modify: `export/ase-ace/src/ase_ace/library_calculator.py` (workspace per instance), `export/docs/C_INTERFACE_API.md`
- Test: `export/test/test_generator_parity.jl` (runs unchanged), `export/test/test_workspace.jl` (new), `export/lammps/test/test_omp_vs_serial.sh` (new)

**Interfaces:**
- Produces (C ABI, documented in `C_INTERFACE_API.md`):
  - `void* ace_workspace_new(void)`; `void ace_workspace_free(void*)`.
  - `double ace_site_energy(void* ws, int z0, int nneigh, const int* neighbor_z, const double* neighbor_Rij)`
  - `double ace_site_energy_forces(void* ws, int z0, int nneigh, const int* neighbor_z, const double* neighbor_Rij, double* forces)`
  - `double ace_site_energy_forces_virial(void* ws, …, double* forces, double* virial)`
  - `ace_batch_energy_forces_virial(void* ws, …)`; `ace_site_basis(void* ws, …)`.
  - Generated Julia: `struct Workspace; A::Vector{Float64}; AA::Vector{Float64}; ∂A::Vector{Float64}; ∂AA::Vector{Float64}; nb::Vector{NeighCache}; end` where `NeighCache = NamedTuple{(:iz,:r,:rhat,:R,:dR,:Y,:dY), …}` holds the per-neighbour derivative terms (sizes: `m_max` radials, `N_YLM` harmonics); `site_energy_forces_virial!(ws::Workspace, Rs, Zs, Z0, forces::AbstractVector{SVector{3,Float64}}) -> (E, V)`; the old positional functions remain as thin wrappers allocating a workspace (used by tests and `_write_main`).

- [ ] **Step 1: Write the workspace/re-entrancy test (fails first)**

`export/test/test_workspace.jl`:

```julia
using Test, Base.Threads
include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

@testset "workspace API: re-entrant and cap-free" begin
    fx = load_cantor_fixture()
    f = joinpath(mkpath(joinpath(@__DIR__, "build")), "cantor_ws.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f; radial_basis = :polynomial)
    ex = Module(:WS); Base.include(ex, f)
    sets = site_sets(fx.held[1], fx.rcut)
    # serial reference
    ref = [Base.invokelatest(ex.site_energy_forces_virial, s[1], s[2], s[3]) for s in sets]
    # concurrent evaluation with one workspace per thread must reproduce it exactly
    out = Vector{Any}(undef, length(sets))
    wss = [Base.invokelatest(ex.new_workspace) for _ in 1:nthreads()]
    @threads for i in eachindex(sets)
        Rs, Zs, Z0, _ = sets[i]
        F = Vector{SVector{3,Float64}}(undef, length(Rs))
        E, V = Base.invokelatest(ex.site_energy_forces_virial!, wss[threadid()], Rs, Zs, Z0, F)
        out[i] = (E, F, V)
    end
    for i in eachindex(sets)
        @test out[i][1] == ref[i][1] && out[i][2] == ref[i][2] && out[i][3] == ref[i][3]   # bitwise
    end
    # 300 neighbours evaluate (no cap)
    Rs = [SVector(2.0 + 0.01k, 0.1k, -0.05k) for k in 1:300]
    @test isfinite(Base.invokelatest(ex.site_energy, Rs, fill(24, 300), 24))
    @test !occursin("MAX_NEIGHBORS", read(f, String))
end
```

Run with `julia -t 4 --project=.. -e 'include("test_workspace.jl")'`. Expected: fails (`new_workspace` undefined).

- [ ] **Step 2: Generate the kernel**

Replace `_write_evaluation_functions` so it emits, for a model with `m_k` used radials per pair and `nA_k` A-functions per pair block (the A functions whose `Rnl` row belongs to species `jz`'s rows — derive `ABLOCK_k = SVector{nA_k,Int}` (indices into A) and `ABLOCK_R_k`, `ABLOCK_Y_k` (local radial slot and Ylm index) at export from `ABASIS_SPEC` and `RBASIS_ROWS_k`):

```julia
struct NeighCache
    iz::Int; r::Float64; rhat::SVector{3,Float64}
    R::SVector{M_MAX,Float64}; dR::SVector{M_MAX,Float64}          # padded to M_MAX = max_k m_k
    Y::SVector{N_YLM,Float64}; dY::SVector{N_YLM,SVector{3,Float64}}
end
mutable struct Workspace
    A::Vector{Float64}; AA::Vector{Float64}; ∂A::Vector{Float64}; ∂AA::Vector{Float64}
    nb::Vector{NeighCache}
end
new_workspace() = Workspace(zeros(N_A), zeros(N_AA), zeros(N_A), zeros(N_AA), NeighCache[])

# pass 1: embeddings on the stack, A accumulated into the neighbour species' block only
@inline function _accumulate_A!(A, ws::Workspace, Rs, Zs, iz0)
    nneigh = length(Rs); resize!(ws.nb, nneigh); fill!(A, 0.0)
    @inbounds for j in 1:nneigh
        r = norm(Rs[j]); jz = z2i(Zs[j])
        if r <= 1e-10; ws.nb[j] = NeighCache(jz, r, zero(SVector{3,Float64}), zero(SVector{M_MAX,Float64}), zero(SVector{M_MAX,Float64}), zero(SVector{N_YLM,Float64}), zero(SVector{N_YLM,SVector{3,Float64}})); continue; end
        rhat = Rs[j] / r
        R, dR = evaluate_rnl_pair_d(r, iz0, jz)         # SVector{M_MAX} (padded), dispatch on pair_idx
        Y, dY = eval_ylm_ed(Rs[j])
        _accumulate_A_block!(A, R, Y, iz0, jz)           # generated: for each (a, s, y) in ABLOCK_k: A[a] += R[s] * Y[y]
        ws.nb[j] = NeighCache(jz, r, rhat, R, dR, Y, dY)
    end
end

# pass 2: forces from ∂A
@inline function _forces_from_∂A!(forces, virial_acc, ∂A, ws::Workspace, Rs, iz0, with_virial::Bool)
    @inbounds for j in 1:length(Rs)
        nb = ws.nb[j]; nb.r <= 1e-10 && (forces[j] = zero(SVector{3,Float64}); continue)
        f = _force_block(∂A, nb, iz0)                   # generated: Σ_{(a,s,y) in ABLOCK_k} ∂A[a] * (dR[s]*Y[y]*rhat + R[s]*dY[y])
        ep, dep = pair_energy_d(nb.r, iz0, nb.iz)
        f = f + dep * nb.rhat
        forces[j] = -f
        with_virial && (virial_acc[] -= Rs[j] * f')
    end
end

function site_energy_forces_virial!(ws::Workspace, Rs, Zs, Z0, forces)
    iz0 = z2i(Z0); nneigh = length(Rs)
    nneigh == 0 && return E0_of(iz0), zero(SMatrix{3,3,Float64,9})
    _accumulate_A!(ws.A, ws, Rs, Zs, iz0)
    E = tensor_energy_and_∂A!(ws.∂A, ws.AA, ws.∂AA, ws.A, iz0)   # Task 6: flat AA + WB readout as today, but writing ∂A; Task 7: DAG + ctilde
    E += _pair_energy_sum(ws, iz0)
    vir = Ref(zero(SMatrix{3,3,Float64,9}))
    _forces_from_∂A!(forces, vir, ws.∂A, ws, Rs, iz0, true)
    return E + E0_of(iz0), vir[]
end
```

`_accumulate_A_block!` and `_force_block` are emitted as an `if iz0 == …; if jz == …` dispatch over generated straight-line loops over `ABLOCK_k` (static tuples), which is what makes the species-block accumulation free of zero multiplies. `E0_of(iz0)` replaces the `_emit_species_dispatch` E0 chains. Keep `site_energy(Rs, Zs, Z0)`, `site_energy_forces`, `site_energy_forces_virial` as wrappers (`ws = new_workspace()`), so `check_export.jl` and the tests keep working; `tensor_energy_and_∂A!` in this task is the existing flat `evaluate_aabasis!`/A2B/`pullback_aabasis!` sequence re-targeted onto workspace vectors (the DAG replaces it in Task 7).

- [ ] **Step 3: C interface and plugin**

`write_c_interface.jl`: emit

```julia
Base.@ccallable function ace_workspace_new()::Ptr{Cvoid}
    ws = new_workspace(); push!(WORKSPACES, ws)          # keep alive: const WORKSPACES = Workspace[]
    return pointer_from_objref(ws)
end
Base.@ccallable function ace_workspace_free(p::Ptr{Cvoid})::Cvoid
    ws = unsafe_pointer_to_objref(p)::Workspace
    filter!(w -> w !== ws, WORKSPACES); return nothing
end
```

(`WORKSPACES` must be guarded by a `ReentrantLock` for `new/free`; the hot entries take the pointer and do `unsafe_pointer_to_objref(p)::Workspace`.) Every `ace_site_*` and the batch entry gains `ws::Ptr{Cvoid}` first; forces are written directly through `unsafe_wrap(Array, forces_ptr, nneigh*3)` reinterpreted as `SVector{3}` (no `Vector{SVector}` allocation). `pair_ace.cpp`: add the two function-pointer typedefs and `dlsym` lookups at `:134-170`; in `init_style` allocate `workspaces[t] = ace_workspace_new()` for `t < omp_get_max_threads()` (1 without OpenMP); free in the destructor; pass `workspaces[tid]` at `:532`; call the virial entry only when `vflag_global || vflag_atom`, otherwise `ace_site_energy_forces`. Remove the header's "thread-safe" claim and replace it with "re-entrant given one workspace per thread". `library_calculator.py`: `self._ws = lib.ace_workspace_new()` in `__init__`, freed in `__del__`, passed to every call; drop the "single-threaded" note. Update `C_INTERFACE_API.md`.

- [ ] **Step 4: OpenMP-vs-serial LAMMPS test**

`export/lammps/test/test_omp_vs_serial.sh`: build the plugin with `-DBUILD_OMP=ON`, run `in.cantor_ace` (config 3) with `OMP_NUM_THREADS=1` and `=4` (`-sf omp` is not needed; the plugin uses OpenMP internally), dump forces, `compare_dump.py dump.omp1 dump.omp4 1e-12`. Add to CI's LAMMPS job (Task 3 wired the job with `BUILD_OMP=OFF` — switch it to ON here).

- [ ] **Step 5: Parity, exactness, threads, timing**

```bash
cd export/test && julia -t 4 --project=.. -e 'include("test_workspace.jl")' 2>&1 | tail -6
cd export/test && julia --project=.. -e 'include("test_generator_parity.jl")' 2>&1 | tail -8   # EXPORT_REF_SHA = Task 5 commit
cd export/test && julia --project=.. runtests.jl lammps 2>&1 | tail -8
export/lammps/test/test_omp_vs_serial.sh
# compile + bench both models
CORE=31 export/bench/bench_parity.sh cantor_b2 lib/libace_cantor_b2.so $CANTOR_YACE 100 | tee -a bench_parity/rows.txt
CORE=31 export/bench/bench_parity.sh tial_b2 lib/libace_tial_b2.so $TIAL_YACE 100 | tee -a bench_parity/rows.txt
```

Expected: bitwise thread reproducibility; parity ≤ 1e-13; LAMMPS 1e-10; OMP 1e-12; Cantor `:polynomial` from ~150 to ~60–80 µs/site. If juliac rejects the kernel (`--trim=safe` error naming a dynamic call), the usual culprits are `resize!` on `Vector{NeighCache}` with a non-isbits `NeighCache` (make it `isbits`: it is, if all fields are `SVector`/scalars) and closures — fix in the generator, never by relaxing `trim_mode`.

- [ ] **Step 6: Commit**

```bash
git add export/src export/lammps export/ase-ace export/docs export/test .github/workflows/export-ci.yml
git commit -m "export: per-neighbour stack kernel, species-block A accumulation, forces from dA; opaque workspace API (re-entrant, no neighbour cap); plugin holds one workspace per OpenMP thread -- B2"
```

---

### Task 7: DAG AA products with a C-tilde-seeded backward (B3)

**Files:**
- Modify: `export/src/export_ace_model.jl` (`_write_tensor`: emit DAG + ctilde instead of `AABASIS_SPECS_*`, `A2BMAP_*`, `WB_*`), `export/src/write_evaluation.jl` (`tensor_energy_and_∂A!`)
- Test: `export/test/test_dag.jl` (new) + `test_generator_parity.jl`

**Interfaces:**
- Consumes: `EquivariantTensors.SparseSymmProdDAG(spec)` (`src/ace/symmprod_dag.jl:31`; fields `nodes::Vector{Tuple{Int,Int}}`, `num1` leaves, `projection`, `has0` — read the struct), `tensor.aabasis.specs`, `tensor.A2Bmaps[1]`, `W_readout[1, :, iz]`.
- Produces: generated `const DAG_NODES = ((n1,n2),…)` for the non-leaf nodes, `const N_DAG`, `const CTILDE_iz::Vector{Float64}` (length `N_DAG`, one per species, `= projectionᵀ · (A2Bmapᵀ · WB_iz)` so that `E = Σ_n CTILDE[n] · AA_dag[n]`), and
  `tensor_energy_and_∂A!(∂A, AAd, ∂AAd, A, iz0)`: forward `AAd[1:N_A] = A; AAd[n] = AAd[n1]*AAd[n2]`; `E = dot(CTILDE_iz, AAd)`; backward `∂AAd .= CTILDE_iz; for n in N_DAG:-1:N_A+1: (n1,n2) = DAG_NODES[n]; w = ∂AAd[n]; ∂AAd[n1] += w*AAd[n2]; ∂AAd[n2] += w*AAd[n1]`; `∂A .= ∂AAd[1:N_A]`.

- [ ] **Step 1: Structural test (fails first)**

`export/test/test_dag.jl`:

```julia
using Test, EquivariantTensors, SparseArrays, LinearAlgebra, Random
include(joinpath(@__DIR__, "check_export.jl"))
include(joinpath(dirname(@__DIR__), "src", "export_ace_model.jl"))

@testset "DAG reproduces the flat AA basis and the readout" begin
    fx = load_cantor_fixture()
    ace = fx.stacked.calcs[end]; tensor = ace.model.basis
    dag = EquivariantTensors.SparseSymmProdDAG(vcat(tensor.aabasis.specs...))   # check the constructor signature in symmprod_dag.jl
    A = randn(MersenneTwister(3), length(tensor.abasis))
    AA_flat = EquivariantTensors.evaluate(tensor.aabasis, A)
    AA_dag = EquivariantTensors.evaluate(dag, A)                                # DAG node values
    P = dag.projection                                                          # flat index -> dag node
    @test maximum(abs.(AA_flat .- AA_dag[P])) <= 1e-13 * maximum(abs.(AA_flat))
    # readout fold
    A2B = tensor.A2Bmaps[1]; WB = ace.ps.readout.W[1, :, 1]
    ct_flat = A2B' * WB
    E_flat = dot(ct_flat, AA_flat)
    ct_dag = zeros(length(AA_dag)); ct_dag[P] .+= ct_flat
    @test abs(dot(ct_dag, AA_dag) - E_flat) <= 1e-12 * abs(E_flat)
end

@testset "exported DAG evaluator is exact" begin
    fx = load_cantor_fixture()
    f = joinpath(mkpath(joinpath(@__DIR__, "build")), "cantor_dag.jl")
    Base.invokelatest(export_ace_model, fx.stacked, f; radial_basis = :polynomial)
    @test occursin("const DAG_NODES", read(f, String)) && !occursin("A2BMAP_1_I", read(f, String))
    @test check_export(f, fx.stacked, fx.held, fx.rcut; tol = 1e-12, label = "DAG export")[2] <= 1e-12
end
```

- [ ] **Step 2: Run to verify it fails**

Expected: first testset passes (it tests ET itself — if the projection semantics differ, fix the test to ET's actual API, cf. `acejax/spike_recursive/dag.py` for the structural reconstruction check), second fails on `DAG_NODES`.

- [ ] **Step 3: Emit the DAG**

In `_write_tensor`: build `dag` as above; emit `N_A`, `N_DAG = length(dag.nodes)`, `DAG_NODES` for nodes `N_A+1:N_DAG` as a tuple of `(n1, n2)` (`Int32` pairs to keep the constant small), and per species `CTILDE_iz` computed as in the test; stop emitting `AABASIS_SPECS_*`, `A2BMAP_*`, `WB_*` for the evaluation path (keep `site_basis` working by emitting `A2BMAP_*` only when `for_library` requests `ace_site_basis` — or drop `ace_site_basis` from the C API with a note; decide and document). Handle `hasconst`/`has0` (constant term) the way `symmprod_dag.jl` does. If the order-1 AA entries are identity copies of A, they are leaves — do not duplicate them.

- [ ] **Step 4: Emit `tensor_energy_and_∂A!`** as in the interface block (two plain loops over static tuples; `@inbounds`, no allocation), replacing Task 6's flat version.

- [ ] **Step 5: Parity, exactness, timing**

```bash
cd export/test && julia --project=.. -e 'include("test_dag.jl")' 2>&1 | tail -8
cd export/test && julia --project=.. -e 'include("test_generator_parity.jl")' 2>&1 | tail -8    # vs Task 6 commit: ≤1e-13 (re-association is roundoff-level; if it exceeds 1e-13 relative on forces, report the value — the acejax spike saw 8e-16)
cd export/test && julia --project=.. runtests.jl lammps 2>&1 | tail -8
CORE=31 export/bench/bench_parity.sh cantor_b3 lib/libace_cantor_b3.so $CANTOR_YACE 100 | tee -a bench_parity/rows.txt
CORE=31 export/bench/bench_parity.sh tial_b3 lib/libace_tial_b3.so $TIAL_YACE 100 | tee -a bench_parity/rows.txt
```

Expected: Cantor ~1.1–1.4x on top of B2; TiAl (order 4) larger. **This row is the gate: both ratios ≤ 1.2x.** If the gate is missed, profile by difference (delete stages in a copy of the generated file, as `verify_cantor/profile/profile_export.jl` does) and report the breakdown; do not tune Hermite instead.

- [ ] **Step 6: Commit**

```bash
git add export/src export/test/test_dag.jl
git commit -m "export: DAG AA products (EquivariantTensors SparseSymmProdDAG) with C-tilde-seeded backward; B, A2B and WB removed from the evaluation path -- B3"
```

---

### Task 8: Final measurement table, docs, finding update (B4 close-out)

**Files:**
- Modify: `export/bench/README.md` (full table), `export/README.md` (performance section, mode table filled), `export/docs/C_INTERFACE_API.md` (final)
- Create: `docs/findings/FINDINGS_lammps_export_parity.md` in the ACEpotentials `pr/lammps-throughput` checkout **if that checkout is available on the host** (`~/si-ace/ACEpotentials` is an rsync'd copy without git; otherwise write the finding into `export/bench/FINDINGS_parity.md` on this branch and say so)

- [ ] **Step 1: Same-day full table**

On a quiet core (`loadavg` < 2 preferred; otherwise record it), run `bench_parity.sh` for both models in `:polynomial` for the Task 4 (baseline), 5, 6, 7 libraries (re-compile from the tagged commits if the `.so` files were not kept) and `pace recursive`, two runs each, in one session; also one Hermite (Nspl=50) row per model at Task 7. Table columns: step, Cantor µs/site, ratio, TiAl µs/site, ratio, loadavg.

- [ ] **Step 2: Multi-rank sanity**

`mpirun -np 4` on the 2000-atom box, 100 steps, `%varavg` on `Pair` from the log: should be a few percent like pace's (the finding's 22–34 % was attributed to host load); record it.

- [ ] **Step 3: Docs**

`export/README.md`: the mode table with measured numbers, the workspace API in the C-interface section, "known limitations" (LAMMPS-side neighbour copy, Julia runtime size), and the verification protocol pointer. `export/bench/README.md`: the protocol and the table.

- [ ] **Step 4: Finding**

Write the finding: what changed per step with the measured factor, the gate verdict on both models, exactness numbers at every level, must-fix items closed (pair term, Hermite dispatch, cap, re-entrancy, CI blindness, tolerances), what remains (Julia runtime beside LAMMPS; Hermite is approximate by construction; `ace_site_basis` status).

- [ ] **Step 5: Commit and hand over**

```bash
git add export/bench export/README.md export/docs
git commit -m "export/bench: parity table (both reference models), docs; B4 close-out"
git log --oneline origin/lammps-export..HEAD
```

Report the branch name and the table to the maintainer; do not push.
