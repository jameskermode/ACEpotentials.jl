# Finding: where the `lammps-export` evaluator spends its time, and what it would take to reach ML-PACE's CPU throughput (2026-09-15)

**Question.** The ACEsuit `lammps-export` branch (tip 075d3859) generates a
plain-Julia site evaluator from an `ETACE` model, compiles it with
`juliac --trim=safe`, and calls it per atom from a `pair_style ace` plugin.
Its own record puts it 2-4.6x behind ML-PACE per basis function. Where does
the time go, and how hard is it to close the gap to `pair_style pace`
(`recursive`, ML-PACE's CPU evaluator)?

**Answer in one paragraph.** The generated evaluator is structurally
reasonable -- it is a true reverse pass (dE/dB -> dE/dAA -> dE/dA -> per-neighbour
forces), not forward-mode -- but it is *flat* (independent products per AA
function, no DAG), *unfolded* (goes through B and a sparse A2B matrix in both
directions instead of a C-tilde vector), keeps its per-neighbour work in
`[MAX_NEIGHBORS, feature]` arrays that every per-neighbour store, zero and
force-assembly loop walks with a 2 KB stride, and in `:polynomial` mode does
the radial species mixing as a fully unrolled, scalar, *dense* 74x45 static
matvec whose entries are 99.5 % zeros. Measured on the five-element Cantor
model of `FINDINGS_lammps_export.md` (moriarty, one pinned core, loadavg 5-6):
**`pair_style ace` = 522 us/atom (`:polynomial`), 195 us/atom
(`:hermite_spline`, dispatch fixed); `pair_style pace` (fork, same 1348-function
basis, 1e4-node yace) = 75 us/atom.** So the exact mode is 7.0x and the
recommended mode 2.6x behind ML-PACE on this host. In `:polynomial` mode 84 %
of the site time is the dense radial mixing; in `:hermite_spline` mode the
tensor stages (A, AA, A2B, their pullbacks) are only 15 %, and the rest is
radial table lookup + strided stores (38 %), force/virial assembly (26 %) and
zeroing the strided gradient buffers (16 %). None of this is architectural:
every item is a code-generator change (days each, ~2 weeks in total), none
needs EquivariantTensors or the C++ plugin to change, and the DAG that
`SparseSymmProdDAG` already provides is the *smallest* of them at this basis
size. After all of it the per-site cost should land at roughly 30-40 us on
this host against ML-PACE's 75 -- i.e. parity is plausible, with no structural
residue expected on the evaluator side; what remains is the Julia runtime next
to LAMMPS and a currently unsafe OpenMP path.

## 0. What was measured and how

Host `moriarty` (Xeon Silver 4216, 32 cores), Julia 1.12.2, checkout
`~/ace-potentials-julia-1.2/ACEpotentials.jl` on branch `lammps-export-verify`
(= `acesuit/lammps-export` 075d3859 + the two-line Hermite dispatch fix
39d853dc). Other users' jobs kept `/proc/loadavg` at **4.6-6.1** throughout;
all absolute times below are under that load, and every comparison is
same-host, same-load, single thread (`JULIA_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, LAMMPS runs pinned with `taskset -c 30`).

Model: the Cantor `:polynomial` and (fixed) `:hermite_spline` exports of
`FINDINGS_lammps_export.md` (`verify_cantor/cantor_{poly,hermite50}_model.jl`).
Sizes, read from the generated files: NZ = 5, N_RNL = 74, N_POLYS = 45,
MAXL = 2 (N_YLM = 9), nA = 72, nAA = 1964 (orders 1/2/3: 30/393/1541),
N_BASIS = 1348, A2B nnz = 1964 (one AA per B), rcut 6.25 A. Of the 74 radial
functions, 44 are referenced by `ABASIS_SPEC`; the other 30 (the l = 3
radials, `MAXL` is 2) are computed per neighbour and never used.

Geometry: 2048-atom random fcc CrMnFeCoNi, a = 3.59 A, +-0.05 A displacements
(the `in.bench_ace` box); **84.2 neighbours/atom** within 6.25 A (min 81,
max 86). Three instruments, all on that box:

1. **Julia in-process** (`verify_cantor/profile/profile_export.jl`): the
   generated `.jl` `include`d into a fresh module and `site_energy_forces_virial`
   called site by site (2048 sites x 5 repeats), `@allocated` per call, a
   straight-line stage timer that re-executes the function's own stages with
   `time_ns()` between them (stage sum is checked against the whole-function
   time), and `Profile` (`profile_{flat,tree}_*.txt`).
2. **The compiled library through its C entry point** (`time_lib.py`,
   ctypes, one fixed 85-neighbour random neighbourhood, 1000 calls).
3. **LAMMPS** (`run_bench.sh`): `in.bench_ace` (100 steps, `timestep 0.0`,
   `pair_style ace`, plugin built with `BUILD_OMP=OFF`) and `in.bench_pace`
   (the yace spike's fork binary, `cantor_n10000_exact.yace`, the v0.6 twin
   with the same 1348 many-body functions per species). Both compute E0 +
   many-body only (neither carries the pair term), so the comparison is
   like-for-like.

| instrument | `:polynomial` | `:hermite_spline` Nspl 50 | `:hermite_spline` Nspl 200 | `pair_style pace` |
|---|---|---|---|---|
| Julia in-process, us/site (E+F+V) | 502-522 | 187-189 | -- | -- |
| Julia in-process, us/site (E only) | 104 | 54 | -- | -- |
| compiled `.so` via ctypes, us/site (E+F+V) | 584-596 | 202-208 | 143-154 | -- |
| compiled `.so` via ctypes, us/site (E only) | 100-105 | 58 | 54-68 | -- |
| **LAMMPS Pair time, 100 steps, 2048 atoms** | **106.8 s = 522 us/atom** | **39.9 s = 195 us/atom** | -- | **15.36 s = 75 us/atom** |
| LAMMPS wall / RSS | 109 s / 302 MB | 42 s / 303 MB | -- | (48 s yace load) |

Three things follow immediately. The compiled library runs at the same
speed as the JIT-compiled source (within 15 %), so **a Julia-side profile is
representative of the `.so`**. The plugin adds nothing measurable: LAMMPS
per-atom time equals the ctypes per-site time, although the LAMMPS neighbour
list holds 201 neighbours/atom (2 A skin) that the plugin filters down to 84.
And the exact mode is 7.0x, the recommended mode 2.6x behind `pace
recursive` on the same core (the branch's own TiAl figure of 4.6x per basis
function is consistent: `benchmark/fair_comparison/results/fair_etace_spline_np1.log`
gives 94 us/atom for 308 functions at 112 neighbours vs ML-PACE 80 us/atom for
1166).

## 1. Structure of the generated evaluator

All line numbers are in the generator (`export/src/*.jl` at 075d3859) and,
where useful, in the generated Cantor `:polynomial` file
(`verify_cantor/cantor_poly_model.jl`, "gen:") which is what juliac compiled.

### 1.1 Per-neighbour embeddings

`compute_embeddings_ed` (`write_evaluation.jl:82-132`, gen:465) loops over
neighbours; for each it calls `evaluate_Rnl_d(r, iz0, jz)` and `eval_ylm_ed(R)`
and copies the returned `SVector`s element by element into global work
arrays `WORK_Rnl[j, t]`, `WORK_dRnl[j, t]`, `WORK_Ylm[j, t]`, `WORK_dYlm[j, t]`
(`write_evaluation.jl:104-116`). The arrays are `zeros(Float64, MAX_NEIGHBORS, N)`
with `MAX_NEIGHBORS = 256` (`write_evaluation.jl:14-21`): column-major with
the *neighbour* index fast. The layout is chosen so that the A-basis
reduction over neighbours is contiguous (`write_evaluation.jl:17`), which it
is -- but every per-neighbour write (and, later, every per-neighbour read in
the force assembly and every per-neighbour zeroing of the gradient buffers)
then walks memory with a stride of 256 x 8 B = 2 KB, one cache line per
element, and with a 2 KB stride the 74 stores of one neighbour land in only
two L1 sets (64 B lines, 4 KB per way), so they also evict each other. This is the single largest inefficiency in the
Hermite mode (section 3).

**Radials, `:polynomial` mode** (`write_radial.jl:23-265`): per neighbour, an
Agnesi transform with three `Float64^Float64` powers (`write_radial.jl:172-205`;
`pin`/`pcut` are written as floats at `:134-136`), a Chebyshev three-term
recurrence for N_POLYS = 45 values and derivatives (`eval_polys_ed`,
`write_radial.jl:79-95`), the quartic envelope, and then the species mixing
**as a dense static matvec**: `W = RBASIS_W[pair_idx]; Rnl = W * SVector(P_env);
dRnl = W * SVector(dP_env_dr)` with `W::SMatrix{74,45}` (`write_radial.jl:225-226,
247-249`, gen:262-264). `RBASIS_W` is a tuple of 25 such matrices (one per
ordered species pair, `write_radial.jl:106-114`), each 26.6 KB. For this model
-- and for every model built with `init_Wradial = :onehot` and then fitted
linearly, i.e. every ACE1-style fit -- **W is a 0/1 selection with 14-16
nonzeros out of 3330** (checked on all 25 pairs: the only nonzero value is 1.0;
the radial index encodes (polynomial degree, neighbour species), so only the
~15 rows belonging to species jz are nonzero). StaticArrays compiles the
74x45 product to an unrolled chain of 3330 scalar `muladd`s
(`StaticArrays/src/matrix_multiply.jl:108`, the unrolled path; 7277 of 10702
profile samples land on that `muladd`). Two of them per neighbour, 85
neighbours: **6660 scalar FMAs per neighbour, 99.5 % of them multiplying
zero**.

**Radials, `:hermite_spline` mode** (`codegen.jl:128-414`): per ordered pair a
cubic Hermite table `PAIR_k_F`, `PAIR_k_G` of `N_KNOTS` x `SVector{74}` in the
transformed coordinate y in [-1, 1] (`codegen.jl:180-199`), evaluated as four
`SVector{74}` gathers plus Horner (`codegen.jl:297-376`), envelope applied
after the spline (`:330`, exact), transform per pair (`:207-272`, integer
powers). Dispatch is a 25-way `if pair_idx == k` chain (`codegen.jl:387-411`),
which at the tip indexes with `zz2pair_sym` (the multi-species bug of
`FINDINGS_lammps_export.md`; fixed on the host branch). All 74 functions are
tabulated for every pair, including the ~58 that are identically zero for
that pair's neighbour species and the 30 l = 3 functions no A uses.

**Solid harmonics**: SpheriCart's code generator emits straight-line code for
maxl at export time (`codegen.jl:28-115`); values and Cartesian gradients
returned as `SVector`s. Cheap (1-2 % of site time) and correct; nothing to do
here.

### 1.2 A basis

`evaluate_abasis!` (`write_evaluation.jl:293-305`, gen:659): for each of the
nA = 72 `(Rnl index, Ylm index)` pairs in `ABASIS_SPEC`, a `@simd ivdep`
reduction over the neighbour column. That is nA x nneigh = 72 x 85 FMAs on
contiguous data -- 2.3-2.7 us/site, the cheapest stage. Its pullback
(`pullback_abasis!`, `write_evaluation.jl:212-224`) scatters `∂A[iA]` into
`∂Rnl[j, ϕ1]` and `∂Ylm[j, ϕ2]` for every neighbour, i.e. it materialises a
dense `[nneigh, N_RNL]` and `[nneigh, N_YLM]` gradient that the force
assembly then reads back -- ML-PACE never forms these; it contracts dE/dA
with the per-neighbour derivative on the fly.

Note the species structure the evaluator does not exploit: with one-hot
radials each neighbour contributes to only the ~14 A functions of its own
species block (nA = 72 over five blocks), so 4/5 of the 72 x 85 FMAs, of the
scatter, and of the force assembly's 74 radial terms per neighbour are
multiplications by zero.

### 1.3 AA products: flat, not a DAG

`evaluate_aabasis!` (`write_evaluation.jl:316-354`, gen:676-700) is emitted
per correlation order as `AA[i] = A[ϕ[1]] * A[ϕ[2]] * A[ϕ[3]]` over
`AABASIS_SPECS_3` etc. -- **independent products per basis function, exactly
`jnp.prod` over gathered rows; no subproduct sharing**. The pullback
(`pullback_aabasis!`, `write_evaluation.jl:234-283`, gen:622) does the
order-3 pattern `∂A[ϕ[1]] += ∂AA_i * a2 * a3` (three products of two per
term). Orders > 4 fall back to `_static_prod_ed` (`:191-208`). EquivariantTensors
0.4.3 ships `SparseSymmProdDAG` (`src/ace/symmprod_dag.jl:31`,
kernels `symmprod_dag_kernels.jl:15-47` forward `AA[i] = AA[n1]*AA[n2]`,
`:51-83` backward `Δ̃[n1] = muladd(wi, AA[n2], Δ̃[n1])`), which is exactly
ML-PACE's recursive scheme; the generator does not use it.

### 1.4 Readout: through B, not folded

`tensor_evaluate` (`write_evaluation.jl:361-381`, gen:702) computes
`B = A2Bmap * AA` as a COO loop over `A2BMAP_1_{I,J,V}` (nnz = 1964) and the
site energy is `dot(B, WB_iz)` (`:159`, per centre species). The backward
pass seeds `WORK_∂B[k] = WB_iz[k]` for all 1348 k (`:442`, gen:872-884), then
`∂AA = A2Bmap' * ∂B` as a second COO loop (`:396-400`). **The readout is not
folded**: the C-tilde vector `c̃_iz = A2Bmapᵀ WB_iz` (length nAA, one per
centre species) would replace B, WB, both COO loops and the 1348-element copy
with one dot product forward and a constant seed backward. Cheap in absolute
terms here (A2B + readout + `∂AA` ≈ 9 us of 187, plus the `∂B` copy), but it
is also what makes the DAG usable (section 4a).

### 1.5 Forces: a real reverse pass, then per-neighbour assembly

`site_energy_forces_virial` (`write_evaluation.jl:489-571`, gen:840):
embeddings -> `tensor_evaluate` -> seed `∂B` -> zero the `[nneigh, N_RNL]`
and `[nneigh, N_YLM]` gradient buffers row by row (`:531-534`, strided) ->
`tensor_pullback!` (`:390-409`: `∂AA`, `∂A`, `∂Rnl`/`∂Ylm`) -> per neighbour
`f = Σ_t ∂Rnl[j,t]·dRnl[j,t]·r̂ + Σ_t ∂Ylm[j,t]·dYlm[j,t]` (`:546-557`). So:
**backward accumulation of dE/dA through AA and A, no ForwardDiff/Zygote,
no per-neighbour forward mode.** Two details cost real time: every one of
the 83 terms per neighbour does a 3x3 outer product `virial -= Rj * df'`
(`:550, :556`) instead of one `Rj ⊗ f_j` per neighbour (identical by
linearity), and all reads are the 2 KB-strided ones. The virial is also
computed unconditionally -- the plugin always calls the `_virial` entry
(`pair_ace.cpp:532`) -- although LAMMPS uses F·r for the global virial by
default (`vflag_fdotr`, `pair_ace.cpp:635`), in which case `vflag_global` is
0 and the site virial is discarded (`:578`).

### 1.6 Allocations and state

Per site call: `forces = Vector{SVector{3}}(undef, nneigh)`
(`write_evaluation.jl:538`) -- 2120 B measured with `@allocated` at 85
neighbours, 0 B for energy-only; the C entry adds `Zs` and `Rs` vectors
(`write_c_interface.jl:101-118`, ~2.8 KB). ~5 KB/site, ~10 MB per MD step at
2048 atoms; the GC cost of that is small and `benchmark/PERFORMANCE_ANALYSIS.md`
reached the same conclusion for the older export. All other scratch is the
set of **global `const WORK_*` arrays** (`write_evaluation.jl:14-35`): the
evaluator has shared mutable state and is therefore not re-entrant (section 2).

## 2. The C boundary and the plugin

`pair_ace.cpp` (636 lines) dlopens the library, resolves four symbols
(`:134-170`), requests a full neighbour list and requires `newton pair on`
(`:363-367`). `compute()` (`:392-636`) loops over local atoms; per atom it
counts and copies the neighbours within `cutoff` into flat `int`/`double`
buffers (`:463-523`, two passes over the 201-entry LAMMPS list), calls
`ace_site_energy_forces_virial(z0, nneigh, Z, Rij, forces, virial)`
(`:532`), adds the returned forces-on-neighbours to `f[j]` and their negative
sum to `f[i]` (`:545-575`; Newton on, ghosts summed by LAMMPS's reverse
comm), and accumulates energy and the Voigt-mapped virial. On the Julia side
the entry (`write_c_interface.jl:197-228`) copies the buffers into Julia
`Vector`s, calls `site_energy_forces_virial`, and `unsafe_store!`s the
results back. Per-call overhead is a few hundred ns against 200-500 us of
work -- **not a factor**, as the LAMMPS = ctypes = in-process timings show.

**OpenMP.** The plugin is built with `BUILD_OMP=ON` by default
(`cmake/CMakeLists.txt:49`) and then runs the atom loop as
`#pragma omp parallel` / `omp for schedule(dynamic)` with per-thread neighbour
buffers and per-thread force arrays (`pair_ace.cpp:413-435, 552-570`,
reduced after the region, `:609-620`). The header claims "the Julia library
is thread-safe for concurrent calls" (`:11-13`). It is not: every
`ace_site_*` call writes the same global `WORK_Rnl`, `WORK_A`, `WORK_∂B`, ...
(`write_evaluation.jl:14-35`). Calling a `@ccallable` from a foreign thread
is itself supported (Julia >= 1.9 adopts foreign threads on entry), so the
code does not crash -- which is why `benchmark/PERFORMANCE_ANALYSIS.md` could
report a 1x8 OpenMP run at 678 % CPU -- but with `OMP_NUM_THREADS > 1` the
site evaluations race on shared buffers and the forces are wrong. Nothing in
the branch checks OpenMP results against serial ones. Until the work
buffers are per-call (or handed in by the caller), the plugin must be built
with `BUILD_OMP=OFF` or run with one thread, as the verification did.

**MPI load imbalance.** In `benchmark/results/` the ETACE runs show
`%varavg` on `Pair` of 34 % (Hermite, 2 ranks), 22 % (4), 8 % (8), 24.5 %
(poly, 4 ranks), 2.4 % (poly, 8), while on the same 2000-atom B2 box the
`pace` runs show 1.1-1.4 % at 2, 4 and 8 ranks and the older `oldace` Julia
library 1.5-2.7 %. The decomposition is therefore balanced; the ETACE numbers
are erratic in rank count and consistent with a shared, loaded host
(`full_benchmark.log` shows the runs were made on moriarty). There is nothing
rank-dependent in the plugin or the library. One thing worth ruling out when
it is re-measured: the embedded Julia runtime in each rank starts its own GC
threads, so eight ranks on eight cores may oversubscribe (`ps -T` on a running
rank would show it).

## 3. Profile

### 3.1 Stage timings (Julia in-process, us/site, 2048 sites x 5, loadavg 5-6)

Stages are the straight-line pieces of `site_energy_forces_virial`, timed in
sequence with the same work arrays; the stage sum is within 5-15 % of the
whole-function time (the remainder is `fill!`s, the `∂B` copy, `view`
construction and the `forces` allocation). The `Profile` shares (flat and tree
files in `verify_cantor/profile/`) agree with the stage timer to a few percent.

| stage | `:polynomial` | share | `:hermite_spline` 50 | share |
|---|---|---|---|---|
| radials `evaluate_Rnl_d` + strided store | **419.4** | **83.9 %** | **61.8** | **38.5 %** |
| solid harmonics + store | 2.5 | 0.5 % | 2.7 | 1.7 % |
| A accumulation `evaluate_abasis!` | 2.3 | 0.5 % | 2.7 | 1.6 % |
| AA products `evaluate_aabasis!` | 2.9 | 0.6 % | 3.4 | 2.1 % |
| A2B + `dot(B,WB)` + `∂B = WB` | 4.9 | 1.0 % | 5.5 | 3.4 % |
| zero `∂Rnl`/`∂Ylm` rows (strided) | 21.3 | 4.3 % | **25.8** | **16.1 %** |
| `∂AA = A2Bᵀ∂B` | 2.9 | 0.6 % | 3.6 | 2.2 % |
| `∂A` `pullback_aabasis!` | 7.3 | 1.5 % | 9.2 | 5.7 % |
| `∂Rnl, ∂Ylm` `pullback_abasis!` | 3.1 | 0.6 % | 3.7 | 2.3 % |
| force + virial assembly | 33.5 | 6.7 % | **42.3** | **26.4 %** |
| stage sum / whole function | 500 / 522 | | 161 / 189 | |

Within the Hermite "radials" stage the tree profile splits it as 555 samples
in `evaluate_Rnl_d` (the table gathers, Horner, transform) against 707 in
the `WORK_*[j, t] = ...` stores (`compute_embeddings_ed` lines 7675-7686 of the
generated file): **the strided stores cost more than the spline evaluation
itself**. Adding the zeroing (500 samples) and the strided reads in the force
assembly, roughly 45 % of the Hermite site time is memory access pattern,
not arithmetic. The whole tensor block (A, AA, A2B and the three pullbacks)
is 28 us = 15 %.

### 3.2 Attribution by difference

Two one-change variants of the generated Cantor files, re-timed the same way
(`verify_cantor/profile/cantor_poly_sparseW_model.jl`,
`cantor_hermite50_virial1_model.jl`; the generators are not touched):

| variant | change | whole function | radials stage |
|---|---|---|---|
| `:polynomial` as generated | -- | 522 us | 419 us |
| `:polynomial`, W applied as a one-nonzero-per-row gather (a 74-iteration branchy loop; still not what a generator should emit) | `write_radial.jl:247-249` | **254 us** | 126 us |
| `:hermite_spline` as generated | -- | 189 us | -- |
| `:hermite_spline`, one `Rj ⊗ f_j` per neighbour instead of 83 outer products | `write_evaluation.jl:550, 556` | **167 us** | -- |

In the gather variant the tree profile puts 57 % of the remaining radial
time in the gather loop itself (the `c == 0` branch and the dynamic
`SVector` index), 26 % in the Chebyshev recurrence and 9 % in the transform;
a generator that emits the 14-16 `(row, column)` pairs as straight-line
assignments removes the gather, leaving ~45 us + the stores. The virial
variant saves 22 us/site, i.e. the per-term outer products were half of the
force assembly.

Energy-only (`site_energy`) is 104 us (poly) / 54 us (Hermite): the
derivative path is 5x / 3.5x the value path, whereas in ML-PACE the two are
within ~2x. The extra is the second matvec (poly), the `dRnl`/`dYlm` stores,
the gradient-buffer zeroing, the scatter pullback and the assembly -- all
items in section 4.

### 3.3 The branch's TiAl benchmark, re-read with these numbers

`benchmark/fair_comparison` (TiAl, 2000 atoms, 112 neighbours, order 3,
totaldegree 8, maxl 4): nA = 58, nAA = 550, N_BASIS = 308, N_RNL = 50 (28
used), N_POLYS = 24. Hermite 94 us/atom, poly 149 us/atom, ML-PACE 80 us/atom
at 1166 functions. The poly-minus-Hermite difference (55 us/atom = 0.5
us/neighbour) is the 50x24 dense one-hot matvec x2; the Hermite number itself
is 0.84 us/neighbour for a model whose arithmetic is a few hundred flops per
neighbour -- the same strided-buffer and assembly overheads as above, at a
basis size where the tensor is negligible.

## 4. Gap analysis to `pace recursive`

ML-PACE's per-site cost on this host for the same basis is 75 us/atom at 84
neighbours. Its structure (ACERecursiveEvaluator): radial functions from a
per-pair Hermite table in *r* (`deltaSplineBins` 0.001 A, both the base
Chebyshev and the mixed `fr(n,l)`), A accumulated per neighbour into the
block of the neighbour's species only, AA through a DAG with one
multiplication per node, backward pass seeded by C-tilde, forces per
neighbour as `Σ_{nlm} (dE/dA_{μ_j n l m}) ∂(R_{nl} Y_{lm})/∂r_j` with no
materialised per-neighbour gradient buffers, no allocation, no virial work
unless requested. Each item below is the distance to one of those.

| # | item | layer | expected gain on this model | effort |
|---|---|---|---|---|
| a | DAG AA (`SparseSymmProdDAG`) with C-tilde-seeded backward, B/A2B/`∂B` removed | export generator only (ET already has the DAG constructor) | tensor block 28 us -> ~10 us: **1.1x now, ~1.4x after c-e** | 2-3 days |
| b | radial mixing emitted from W's sparsity (one-hot -> static gather; general W -> vectorised GEMV), unused (n,l) pruned, `pin`/`pcut` integer, transform per pair | generator | **poly mode 522 -> ~150 us (3.5x)**; Hermite tables shrink 4-5x | 1-2 days |
| c | per-neighbour kernel: keep `Rnl`,`dRnl`,`Ylm`,`dYlm` in stack `SVector`s, accumulate A per neighbour (species block only), compute `f_j` directly from `∂A` in a second neighbour pass, no `∂Rnl`/`∂Ylm` buffers, one virial outer product per neighbour (or none when not requested), `forces` written to the caller's buffer | generator | Hermite stores+zeroing+assembly 130 us -> ~25 us: **~2.5x**; also fixes the `MAX_NEIGHBORS = 256` hard limit | 3-4 days |
| d | thread safety: per-call work buffers (or a caller-supplied workspace pointer in the C API), OMP-vs-serial test | generator + 20 lines of plugin | correctness; enables the plugin's existing OpenMP path (ML-PACE has none, so MPI-only parity is the target anyway) | 1-2 days |
| e | polynomial-mode cost | covered by b; after b the recurrence is ~0.1 us/neighbour | exact mode becomes the *fast* mode | -- |

Notes on each.

**(a)** The forward DAG replaces two multiplications per order-3 term with one
per node and the backward three-products-of-two with two FMAs per node
(`symmprod_dag_kernels.jl:36-38, 68-72`); folding `c̃_iz = A2Bmapᵀ WB_iz`
(projected onto the DAG's node indexing via `dag.projection`,
`symmprod_dag.jl:11-30`) removes both COO loops, the `dot`, and the
1348-element `∂B` copy. The generator would emit `const DAG_NODES = ((n1,n2),...)`
and `const CTILDE_iz` and two loops -- the same shape of code it already
emits. On this model the tensor is 15 % of the Hermite time, so the DAG is
worth ~1.1x today and ~1.4x once (b)-(c) have removed the rest; it matters
more at order 4 / larger nAA, where ML-PACE's own `product` vs `recursive`
gap on CPU is 1.7x-3.7x (`acejax/bench/results.md`, last section) and the
acejax DAG spike's reverse-pass buffer objection was a GPU one. Nothing in
EquivariantTensors or ACEpotentials needs to change.

**(b)** Why ET's `splinify` is so much less accurate than ML-PACE's table:
`trans_splines` (`EquivariantTensors/src/embed/transsplines.jl:26-62`) calls
`P4ML.splinify(y -> W*P(y), -1, 1, nspl)` -- cubic Hermite on `nspl` uniform
knots in the *transformed* coordinate of the *mixed* functions, whose highest
component is a Chebyshev polynomial of degree N_POLYS-1 = 44. With
Nspl = 50, h = 0.041 and the top polynomial has ~1.1 knots per half-period;
the value error scales like (h n)^4/384 and the derivative error like
(h n)^3 n, which is why forces are at 2.7e-4 (50) and 3.0e-6 (200) despite the
smoothness prior damping the high-n weights. ML-PACE tabulates low-degree
radials (nradbase ~ 10-20) at 6000 knots in r; to get this model's
derivative error to <= 1e-9 the same way needs h n <~ 0.02, i.e. **Nspl ≈
2000-3000 in y, a 60-90 MB table for 25 pairs x 74 functions** -- the same
wall the yace route hit (185 MB at 1e4 nodes). Tabulation is the wrong tool
for degree-45 radials. The right one is the recurrence the `:polynomial`
mode already has (45 + 45 FMAs, latency-bound, ~0.1-0.2 us/neighbour after
the `pow` calls are made integer) plus a mixing that costs what W actually
contains: for one-hot W a static gather of ~15 values (free); for a learned
W a 74x45 GEMV that vectorises to ~0.4 us with AVX2 -- comparable to the
Hermite lookup and exact. So the recommendation is to make `:polynomial`
fast rather than to make `:hermite_spline` accurate; the Hermite mode then
only earns its place for genuinely learned radials with small N_POLYS, where
it is also accurate. Pruning the 30 unused l = 3 radials and the 58-of-74
per-pair zero rows shrinks Hermite tables and per-neighbour work by 4-5x as a
side effect.

**(c)** This is where ML-PACE's structure differs most from the generated
code and where most of the Hermite time is. Per neighbour: evaluate the
~15 nonzero radials and 9 harmonics into stack `SVector`s, accumulate
`A[k] += R·Y` for the ~14 A functions of species jz (a static list per pair,
emitted by the generator), and keep nothing per neighbour. After the tensor
pass, loop over neighbours again, re-evaluate (or keep, they are 24
doubles) the per-neighbour derivatives and form
`f_j = Σ_k ∂A[k] (dR_k Y_k r̂ + R_k dY_k)`; `Rj ⊗ f_j` once. No
`[MAX_NEIGHBORS, feature]` arrays, no zeroing, no strided access, no
neighbour cap. `forces` should be written straight into the caller's
buffer through the C interface rather than allocated. The measured 130 us of
stores + zeroing + assembly becomes ~25 us of contiguous arithmetic.

**(d)** Two viable designs: pass a `void* workspace` from the plugin's
per-thread buffers into the C API (ML-PACE's plugin also keeps per-instance
scratch), or allocate per call and rely on Julia's allocator (adopted
threads allocate fine; ~5 KB/site is already allocated today). Either way
`WORK_*` globals go. Add a LAMMPS test that compares `OMP_NUM_THREADS=4`
forces to serial at 1e-12.

**(e)** The polynomial mode's 3.5x deficit is entirely item (b); after it the
polynomial recurrence is cheaper than the Hermite gather for this model.

**What remains after all of it.** Per-neighbour arithmetic for this model is
~15 radials (recurrence + gather), 9 harmonics with gradients, ~14 A
accumulations and ~14 force terms -- of order 300-500 flops, ~0.2-0.3 us
scalar, x 84 neighbours ≈ 20-25 us/site; the DAG over 1964 nodes forward and
backward is ~6k flops ≈ 3-5 us; plus a few us of fixed cost. **~30-40 us/site
against ML-PACE's 75 us/site on this core** -- parity is plausible, and on
this model there is no structural residue on the evaluator side (the
generated code can be made isomorphic to `ace_recursive.cpp`). Two
non-evaluator residues stay: the ~38 MiB Julia runtime shipped next to
LAMMPS (start-up is already ~0 s over LAMMPS itself), and the fact that
every one of these changes is a code-generator change that must be
re-verified at 1e-13 on multi-species models, which the branch's test suite
cannot currently do (section 5). Total effort for (a)-(d): about two weeks
of generator work, no EquivariantTensors or ETModels changes required,
plugin changes limited to (d).

## 5. Must-fix items independent of performance

1. **Hermite multi-species dispatch** (`codegen.jl:388, 403`: `zz2pair_sym`
   on asymmetric per-pair tables): 13.5 eV/A errors for any NZ >= 2 at the
   tip; two-line fix on the host branch (39d853dc), not pushed. Better: emit
   only the tables the model has and keep one indexing convention.
2. **The pair potential is silently dropped**
   (`export_ace_model.jl:60-90`: the `StackedCalculator` loop keeps `ETOneBody`
   and `ETACE` and ignores `ETPairModel`; `_write_evaluation_functions(io,
   tensor, NZ, false)` at `:236`, `has_pair` unused). A deployed library is
   one-body + many-body only; on the Cantor model that removes up to 6.9
   eV/A of repulsive force. Either export the pair basis (the same
   recurrence, N_PAIR = 30 per pair) or refuse to export a stack that
   contains one.
3. **CI blindness**: `test_multispecies.jl` exports `:polynomial` only and
   asserts `isfinite(E)`, `E2 ≈ E` and `length(F) == 3` (`:141-148`) -- it
   never compares to the ETACE calculator, in either mode; the Hermite tests
   are single-species Si. A multi-species (NZ >= 3, asymmetric `rin0cuts`)
   test of both modes against `ETACEPotential` at 1e-12 is the minimum, and
   the LAMMPS test compares to Python at 1e-6 only (`test_lammps.jl:224`).
4. **Force tolerance 1e-8** (`test_hermite_accuracy.jl:233, 282`:
   `atol=1e-8 rtol=1e-6`) hides a 2.7e-4 splinification error only because
   the reference there is the splined model; against the fitted model both
   Hermite node counts fail. The tests should state which reference they
   use and, for `:polynomial`, assert 1e-12.
5. **OpenMP path** (section 2): racy on the global `WORK_*` arrays; build
   with `BUILD_OMP=OFF` or fix (d) before anyone runs it with threads.
6. Minor: `MAX_NEIGHBORS = 256` is a silent hard limit (`@assert` at
   `write_evaluation.jl:45, 84`) that a dense or small-cutoff system will
   trip; the README's "~3-4x faster" for Hermite is 1.6x-2.7x measured, and
   its "machine precision" claim holds only against the splined model.

## Artefacts

Host `moriarty:~/ace-potentials-julia-1.2/ACEpotentials.jl/verify_cantor/profile/`
(untracked; nothing in the branch checkout modified): `profile_export.jl`
(stage timer + Profile driver), `make_variant_sparseW.jl`, `time_lib.py`,
`run_bench.sh`; logs `log.profile_{poly,hermite50,poly_sparseW,hermite50_virial1}`,
`profile_{flat,tree}_*.txt`, `log.run_bench`, `log.bench_ace_{hermite50,poly}`,
`log.bench_pace`; variant sources `cantor_poly_sparseW_model.jl`,
`cantor_hermite50_virial1_model.jl`. Branch read through a throwaway
worktree of `origin/lammps-export` at 075d3859 (removed). Local: this file
only.
