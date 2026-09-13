# PR draft: Fix the ~200x design-matrix assembly regression (`evaluate_basis_ed` / `energy_forces_virial_basis`)

Branch: `fix/basis-ed-performance` (off `main` at 266f84eb).

## Summary

Assembling the linear least-squares system (`ACEfit.feature_matrix` ->
`energy_forces_virial_basis` -> `evaluate_basis_ed`) cost 350-1550x one
`energy_forces_virial` per structure and ~10-26 GB of allocation per 32-atom
structure.  Two causes, both in the classic `ACEModel` path:

1. `evaluate_basis_ed` (`src/models/ace.jl:663-693` on `main`) took a
   `ForwardDiff.jacobian` of `evaluate_basis` over the **full** basis vector
   (NZ x n_B entries, of which only the centre species' block is nonzero)
   and then went through `collect(dB_vec')[:]`, `reinterpret`, `reshape`,
   `permutedims`, `collect`.  The result inferred as
   `Tuple{Vector{Float64}, Any}` and cost 16-136 ms and 60-270 MB per site.
2. The consumer loop in `energy_forces_virial_basis`
   (`src/models/calculators.jl:311-318` on `main`) ran
   `F[Js[a], k] -= dv[k, a] * force_unit(calc)` on that `Any`, with a
   boxed `SVector{3, Quantity}` and dynamic dispatch per element, over all
   `length_basis x nneigh` entries (4/5 of them structurally zero for a
   5-element model): ~2.4e8 allocations per structure; 90 % of the wall
   time for categorical models.

Diagnosis and profiles: `docs/findings/FINDINGS_assembly_profile.md`
(jax-eval branch) and `acejax/bench/profile_assembly/`.

This PR replaces both with a hand-written forward-mode pushforward and a
type-stable accumulation.  Design matrices are reproduced to <= 6e-15
relative; `energy_forces_virial_basis` is **174-218x faster** single-threaded
on a 32-atom, 5-element structure and allocates 20-60 MB instead of
9.5-26 GB.

## What changed

### `src/models/basis_ed.jl` (new)

* `BasisEDWorkspace{T}`: all intermediates (`rs`, `∇rs`, `Rnl`, `dRnl`,
  `∂Rnl`, `Ylm`, `∂Ylm`, `A`, `∂A`, `AA`, `∂AA`, `Bi`, `∂Bi`, `Rpair`,
  `dRpair`) sized for a maximum neighbour count; grows by doubling if a site
  has more neighbours.
* `evaluate_basis_ed!(ws, model, Rs, Zs, Z0, ps, st)`: the pushforward with
  `SVector{3}` tangents,
  `rs -> Rnl (evaluate_ed_batched!), Ylm (P4ML.evaluate_ed!) -> A -> AA -> Bi = A2B * AA`,
  plus the pair basis, computing only the block of the centre species.
  Kernels: `_pf_A!` (pooled sparse product), `_pf_AA!`/`_pf_AA_N!`
  (symmetric product, `@generated` over the correlation order, uses a local
  `_prod_ed`), `_pf_A2B!` (CSC loop; generic fallback for a dense map).
  Works for both radial bases (`SplineRnlrzzBasis` via `ace1_model`,
  `LearnableRnlrzzBasis` via `ace_model`) since both provide
  `evaluate_ed_batched!`.
* `evaluate_basis_ed(model, Rs, Zs, Z0, ps, st; ws = nothing)`: same
  signature and return convention as before - `B::Vector{T}` of length
  `length_basis(model)` and `dB::Matrix{SVector{3,T}}` of size
  `(length_basis, nneigh)` with `dB[k, j] = ∂B[k]/∂Rs[j]` - but now
  concretely typed (`@inferred` passes) and ~12 ms -> ~1 ms per site.  A
  workspace can be passed to avoid re-allocating the intermediates.

### `src/models/calculators.jl`

* `energy_forces_virial_basis(at, calc, ps, st; domain, executor, ntasks, nlist, ws)`:
  - `_efv_basis_site!` is a function barrier (`get_neighbours` returns
    `Zs`, `z0` as `Any`) that runs the pushforward into the workspace and
    accumulates the species block and pair block directly from `ws.∂Bi` /
    `ws.dRpair` into unit-less `E::Vector{T}`, `F::Matrix{SVector{3,T}}`,
    `V::Vector{SMatrix{3,3,T,9}}`; no per-element units, no `dv[k, :]`
    copies, no loop over the structurally-zero rows.
  - Units are attached once at the end; the returned types are unchanged
    (`Vector{Quantity{eV}}`, `Matrix{SVector{3, Quantity{eV/Å}}}`,
    `Vector{SMatrix{3,3,Quantity{eV}}}`), so `ACEfit.feature_matrix`,
    `test_silicon.jl`, `test_calculator.jl` etc. are unaffected.
  - Sites are split into `ntasks` chunks (default `Threads.nthreads()`, as
    for `energy_forces_virial`), each evaluated on its own task with its own
    workspace and accumulators, summed in chunk order.  `ntasks = 1` or
    `executor = SequentialEx()` gives the serial path.  The chunked result
    differs from the serial one only by floating-point summation order
    (tested to 1e-12 relative); on ACEfit workers (1 thread) it is serial.
  - The old loop called `get_neighbours(at, V, nlist, i)` with the virial
    array `V` in place of the potential (harmless, only used for `get_id`
    dispatch, but a latent bug and one of the sources of `z0::Any`); the
    new code passes `calc`.

### `src/models/ace.jl`

* Removed the ForwardDiff `evaluate_basis_ed` and the dead
  `evaluate_basis_ed_old`.  `__vec`/`__svecs` stay (used by
  `jacobian_grad_params`).

### `src/models/models.jl`

* `include("basis_ed.jl")` after `ace.jl`.

### Tests: `test/models/test_basis_ed.jl` (registered in `test/models/test_models.jl`)

For `ace1_model(elements = [:Si, :O, :C], order = 3, totaldegree = 8)`
(splines) and `ace_model(...; max_level = 8, maxl = 4)` (learnable radial
basis), and the splinified learnable model:

* `evaluate_basis_ed` against the previous ForwardDiff implementation (kept
  verbatim in the test file) to 1e-12 relative, on 5 random environments
  each;
* directional derivatives against `ForwardDiff.derivative` of
  `evaluate_basis` (1e-12) and central finite differences (1e-6);
* `@inferred` return type `Tuple{Vector{Float64}, Matrix{SVector{3,Float64}}}`;
  workspace growth from an undersized workspace gives identical results;
  the empty neighbourhood returns zeros and a `(length_basis, 0)` matrix;
* an allocation bound per call with a workspace (4x the size of the returned
  `(B, dB)` plus 2 MB; the old code allocated ~100 MB here) - not a wall
  time, so not flaky;
* `energy_forces_virial_basis` against the previous unit-ful accumulation
  loop (identical types; values to 1e-12 relative) on random 16-atom
  3-element structures; serial versus `ntasks = 3` to 1e-12; contraction
  with random linear parameters reproduces `energy_forces_virial` (E, F, V);
  `domain` as a range and as a vector, serial and chunked, against
  `potential_energy_basis`.

## Verification

Commands (Julia 1.12.7, macOS arm64; the test environment is
`test/Project.toml` minus CUDA/LuxCUDA/MLDataDevices/TestEnv, with
`ACEpotentials` pointing at the branch):

```
julia +1.12 -t 3 --project=<testenv> -e 'using Test; @testset "Basis ED" begin include("test/models/test_basis_ed.jl") end'
   Basis ED      |  158    158  47.7s
julia +1.12 -t 2 --project=<testenv> -e 'using Test; @testset "ACE Model" begin include("test/models/test_ace.jl") end; @testset "ACE Calculator" begin include("test/models/test_calculator.jl") end'
   ACE Model     |  324    324  47.6s
   ACE Calculator |  140    140  11.9s
julia +1.12 -t 2 --project=<testenv> -e 'using Test, LazyArtifacts; @testset "Test silicon" begin include("test/test_silicon.jl") end'
   Test silicon  |   50     50  1m45.5s
```

`test_ace.jl` includes the existing `evaluate_basis_ed` checks against
`evaluate_ed` and against `jacobian_grad_params` (before and after
splinification); `test_calculator.jl` includes the existing
`energy_forces_virial_basis` versus `energy_forces_virial` check.

## Before / after

Machine: Mac (Apple Silicon, 12 cores), Julia 1.12.7, `-t 1`, other Julia
jobs idle.  Structure: 32-atom CrMnFeCoNi (`cantor1k_b_mh1.xyz`, keys
`mace_energy/force/virial`), 80 neighbours per site at rcut = 6.25 Å;
`ace1_model(elements = [:Cr,:Mn,:Fe,:Co,:Ni], order = 3, totaldegree = D, r0 = 2.5, rcut = 6.25)`
with random linear parameters.  "Old" is the `main` implementation copied
verbatim into the benchmark script and run in the same process
(`bench_basis_ed.jl` in the session scratchpad, log `bench_t1.log`); min of 3-5 runs.

| | D = 6 (6 890 columns) | D = 8 (19 320 columns) |
|---|---|---|
| `evaluate_basis_ed`, one site, old | 15.9 ms, 102 MB | 37.6 ms, 267 MB |
| `evaluate_basis_ed`, one site, new | 1.29 ms, 19.8 MB (1.09 ms, 12.7 MB with `ws`) | 3.77 ms, 62.8 MB (3.09 ms, 35.6 MB with `ws`) |
| `energy_forces_virial_basis`, old | **5.82 s**, 9 548 MB, GC 33 % | **14.5 s**, 26 354 MB, GC 31 % |
| `energy_forces_virial_basis`, new (serial) | **26.7 ms**, 20 MB, GC 0 % | **83.5 ms**, 61 MB, GC 0 % |
| speed-up | **218x** | **174x** |
| new / one `energy_forces_virial` (4.4 / 6.8 ms) | 6x | 12x |
| max rel. difference old vs new (E / F / V) | 2.7e-16 / 5.7e-15 / 1.3e-15 | 1.8e-16 / 3.8e-15 / 1.1e-15 |
| `ACEfit.feature_matrix`, new | 28.4 ms, 28 MB | 87 ms, 87 MB |
| `ACEpotentials.assemble`, 32 structures, 1 process, new | 11.0 s = 0.345 s/structure, GC 88 % | 12.1 s = 0.378 s/structure, GC 67 % |
| old `energy_forces_virial_basis` over the same 32 structures (lower bound for the old assemble) | 200.7 s = 6.27 s/structure | 581.6 s = 18.2 s/structure |
| assemble ratio | 18x | 48x |
| `energy_forces_virial_basis`, new, `-t 4` (4 chunks) | 13.3 ms, 111 MB (serial in the same run: 25.2 ms) | 47.0 ms, 322 MB (serial: 77.3 ms) |
| chunked vs serial max rel. difference (E / F / V) | 1.3e-16 / 2.5e-16 / 3.2e-16 | 1.8e-16 / 3.3e-16 / 1.9e-16 |

(The single-site "new" numbers include building the full
`(length_basis x nneigh)` `dB` matrix, 13-40 MB, which the
`energy_forces_virial_basis` path does not do.)

The `assemble` ratios (18x / 48x) are far below the
`energy_forces_virial_basis` ratios (218x / 174x) because, once the feature
matrix costs 28-87 ms, the remaining ~0.3 s per structure is ACEfit's own
overhead: 67-88 % of the assembly wall time is now garbage collection, i.e.
the explicit `GC.gc()` after every task in `ACEfit.assemble`.  See the
follow-ups below.  (The "old" assemble was not run in full; the old
`energy_forces_virial_basis` alone over the same structures is a lower
bound, consistent with the 16.4 s/structure measured for the full old
assemble in the findings document.)

With 4 threads the chunked site loop gives 1.6-1.9x over serial at the
price of one `(natoms x length_basis)` force accumulator per chunk
(memory-bound; the same behaviour as the scratch `efv_basis_v3t`).

## ACEfit follow-ups (not in this PR; separate package)

Measured in `docs/findings/FINDINGS_assembly_profile.md` §2d and visible
in the `assemble` numbers above (`ACEfit/src/assemble.jl:36-47`, ACEfit 0.3.x):

1. `GC.gc()` after every structure: a full collection costs 0.15-0.4 s per
   call (growing with the worker's heap) and is now the single largest item
   per structure - ~90 % of the serial assembly time with this PR.  Drop it,
   or make it `GC.gc(false)`, or run it every N tasks.
2. The `pmap` closure captures `basis`, so the model (2-9 MB serialised) is
   shipped with every task: 8-50 ms per task versus 0.3-1.7 ms when read
   from a worker global.  `sendto(workers(), basis = basis)` on line 36
   already defines that global but the closure never uses it.
3. `Array(A)` on line 47 copies the whole design matrix out of the
   `SharedArray`: transient 2x peak memory (2 x 156 GB for the
   3200-structure degree-10 case).  Return the `SharedArray` / `sdata(A)`
   or write `feature_matrix!` straight into the view.

## Deliberately left out

* Batched / GPU-ready evaluation via `EquivariantTensors._jacobian_X`: it
  uses `promote_type(Float64, SVector{3}) = Any` for the tangent element
  type and cannot take vector tangents as is; a 3-line upstream change
  would let ACEpotentials call the batched kernel for all sites of a
  structure at once.  The hand-written kernels here follow its structure.
* A specialised `ACEfit.feature_matrix(::AtomsData, ::ACEPotential{<:ACEModel})`
  writing straight into the row layout: the unit-strip/copy-out now costs
  ~2 ms of the 28 ms per structure, not worth a second code path.
* Reducing the remaining allocations in the radial bases
  (`evaluate_ed(::SplineRnlrzzBasis, r, ...)` allocates two vectors per
  neighbour); they are ~10 % of the new per-structure cost.
* Any change to the feature-matrix row layout or to the returned types of
  `energy_forces_virial_basis`.
* Bit-identical serial/threaded results: the chunked sum differs from the
  serial one by summation order (<= 1e-15 relative here).
