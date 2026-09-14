# PR draft: Fast design-matrix assembly (~200x) and fast forward evaluation (2.5-4x / core, 12x on 4 threads) for the classic `ACEModel`

Branch: `fix/basis-ed-performance` (off `main` at 266f84eb).

**Depends on EquivariantTensors >= 0.5.2** (branch
`feat/pushforward-vector-tangents`, PR to ACEsuit/EquivariantTensors.jl):
the tensor part of the Part A pushforward (`A -> AA -> B` with one
`SVector{3}` tangent per neighbour) is `EquivariantTensors.pushforward_rows!`
from that release; the compat entry goes from `"0.4.3"` to `"0.4.3, 0.5"`.
See "EquivariantTensors dependency" below.

## Summary

Two independent performance fixes for the classic (non-ET) `ACEModel`
path, both exact with respect to the previous implementation and covered by
new tests that keep the previous code as the reference:

* **Part A - design-matrix assembly** (`evaluate_basis_ed`,
  `energy_forces_virial_basis`, hence `ACEfit.feature_matrix` /
  `acefit!`): 174-218x faster per structure, allocation 9.5-26 GB -> 20-60 MB.
* **Part B - forward evaluation** (`energy_forces_virial` of a fitted
  model, the MD path): 2.3-3.0x faster per core, 11-12x on 4 threads
  versus the old single-threaded path, allocation 53-97 MB -> 0.2-0.7 MB
  per 256-atom call; the site loop is allocation-free.

## Part A: design-matrix assembly

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

### What changed (Part A)

#### `src/models/basis_ed.jl` (new)

* `BasisEDWorkspace{T}`: the embedding intermediates (`rs`, `∇rs`, `Rnl`,
  `dRnl`, `∂Rnl`, `Ylm`, `∂Ylm`) and the outputs (`Bi`, `∂Bi`, `Rpair`,
  `dRpair`) sized for a maximum neighbour count; grows by doubling if a site
  has more neighbours.
* `evaluate_basis_ed!(ws, model, Rs, Zs, Z0, ps, st)`: the pushforward with
  `SVector{3}` tangents,
  `rs -> Rnl (evaluate_ed_batched!), Ylm (P4ML.evaluate_ed!) -> A -> AA -> Bi = A2B * AA`,
  plus the pair basis, computing only the block of the centre species.
  The tensor part `A -> AA -> Bi` is one call,
  `EquivariantTensors.pushforward_rows!(Bi, ∂Bi, model.tensor, Rnl, Ylm, ∂Rnl, ∂Ylm)`,
  the row-wise pushforward of `SparseACEbasis` added to ET in 0.5.2 (see
  below); its `A`, `∂A`, `AA`, `∂AA` intermediates live on ET's Bumper
  stack.  (Earlier revisions of this branch carried these three kernels
  locally as `_pf_A!`, `_pf_AA!`, `_pf_A2B!`; they are now upstream, where
  they sit next to the `evaluate!` / `pullback!` kernels of the same
  layers.)
  Works for both radial bases (`SplineRnlrzzBasis` via `ace1_model`,
  `LearnableRnlrzzBasis` via `ace_model`) since both provide
  `evaluate_ed_batched!`.
* `evaluate_basis_ed(model, Rs, Zs, Z0, ps, st; ws = nothing)`: same
  signature and return convention as before - `B::Vector{T}` of length
  `length_basis(model)` and `dB::Matrix{SVector{3,T}}` of size
  `(length_basis, nneigh)` with `dB[k, j] = ∂B[k]/∂Rs[j]` - but now
  concretely typed (`@inferred` passes) and ~12 ms -> ~1 ms per site.  A
  workspace can be passed to avoid re-allocating the intermediates.

#### EquivariantTensors dependency (`Project.toml`)

`EquivariantTensors = "0.4.3, 0.5"` (was `"0.4.3"`).  The three tensor
kernels of Part A are now ET's `pushforward_rows!` (ET branch
`feat/pushforward-vector-tangents`, version 0.5.2, one small PR separate
from the spec-ordering fix):

```julia
# one tangent per input row j (neighbour); nothing is summed over j;
# the tangent type only needs T * T∂ -> T∂, so SVector{3} tangents work
pushforward_rows!(A, ∂A, abasis::PooledSparseProduct{NB}, (Rnl, Ylm), (∂Rnl, ∂Ylm))
pushforward_rows!(AA, ∂AA, aabasis::SparseSymmProd, A, ∂A)
pushforward_rows!(B, ∂B, tensor::SparseACEbasis, Rnl, Ylm, ∂Rnl, ∂Ylm)   # composition + A2Bmaps[1]
pushforward_rows(tensor::SparseACEbasis, Rnl, Ylm, ∂Rnl, ∂Ylm)           # allocating
```

with `whatalloc` methods, `@inferred`/allocation-free layer kernels, and
tests against ForwardDiff, finite differences and the adjoint identity with
the existing `pullback` (`test/ace/test_pushforward_rows.jl` in ET).
ACEpotentials `main` is already on ET's 0.4 API; the 0.4.3 -> 0.5.x jump
needs no code change here (`test/models/test_ace.jl`,
`test_calculator.jl`, `test_forward_fast.jl`, `test_basis_ed.jl` pass
unchanged on 0.5.2).  One pre-existing threshold in `test/test_silicon.jl`
is marginally exceeded on ET 0.5.x independently of this branch: the BLR
fit's `liq` energy RMSE is 0.4203 meV against the 0.4 meV bound
(`rmse_blr`, line 84), identically with the old local kernels and with the
ET call (they agree to 1e-12); ET 0.5.0 orthonormalised the coupling
coefficients, which changes the basis the BLR prior acts on.  Not
addressed here.

#### `src/models/calculators.jl`

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

#### `src/models/ace.jl`

* Removed the ForwardDiff `evaluate_basis_ed` and the dead
  `evaluate_basis_ed_old`.  `__vec`/`__svecs` stay (used by
  `jacobian_grad_params`).

#### `src/models/models.jl`

* `include("basis_ed.jl")` after `ace.jl`.

#### Tests: `test/models/test_basis_ed.jl` (registered in `test/models/test_models.jl`)

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

## Part B: forward evaluation

`AtomsCalculators.energy_forces_virial(sys, calc)` for a fitted
`ACEPotential{ACEModel}` with spline radial bases cost 4-10 ms per 32-atom
structure (8e3 atom-steps/s per core on the 5-element models) and ~1150
allocations per site, which also capped thread scaling at ~3x.  Diagnosis
and profiles: `docs/findings/FINDINGS_forward_profile.md` and
`acejax/bench/profile_forward/` (jax-eval branch).  Causes:

1. **Radial splines, 45-71 % of a site.**  `SplineRnlrzzBasis` stores one
   cubic B-spline per species pair whose value is an `SVector{LEN}`
   (LEN = 74-126 for the categorical models) and evaluated it per edge
   through `Interpolations` with `Dual` numbers (`Rnl_splines.jl:88`,
   called from `ace.jl:341` and, allocating, `:384`).  But for the
   one-hot weights of `ace1_model` (`set_onehot_weights!`) every column
   `Rnl[:, n]` is a scalar multiple of one of NU ~ 7-10 polynomials,
   independent of `l`, and 4/5 of the columns are zero on any edge of a
   5-element model.  The transform / envelope evaluation with `Dual`s and
   run-time integer `^` (`radial_transforms.jl:31`, `radial_envelopes.jl:84`)
   cost as much again as the B-spline itself.
2. **~60 allocations per site + ~6 per edge** in `evaluate_ed`
   (`ace.jl:322-400`) and the generic `SitePotential` driver: boxed
   spline/envelope objects from `SMatrix` indexing, `atomic_number(sys, j)`
   per edge in `get_neighbours`, `ka_evaluate` launches, the A basis
   computed twice, two sparse `A2B` matvecs per site, `Quantity`-valued
   force accumulation on an un-inferred `eval_grad_site`.
3. The neighbour list rebuilt on every call (5-8 % of the old call, ~20 %
   of the new one).

### What changed (Part B)

#### `src/models/Rnl_basis.jl`, `src/models/Rnl_splines.jl`

* `RnlSplineTables{T, TT, TENV}`: the B-spline coefficients of a
  `SplineRnlrzzBasis` as plain arrays, with the column factorisation
  `Rnl[:, n] = c[n, pair] * P[:, u[n, pair], pair]` (`P` is
  `(nnodes+2, NU, NZ^2)`).  The factorisation is found numerically at
  construction (`_factorise_columns`, columns proportional to 1e-12
  relative, then verified against the raw tables); a basis whose columns
  are not proportional gets the trivial factorisation (NU = LEN, `c = 1`,
  `u = n`) and goes through exactly the same kernel.  The tables also
  hold plain `Matrix` copies of the transforms and envelopes (indexing the
  `SMatrix` of non-isbits objects boxed per edge).  Found:
  `ace1_model` D6/D8 (Si,O,C): LEN 47/75 -> NU 7/9; the 5-element D6/D8:
  74/126 -> 7/9; Si D10: 37 -> 10; pair bases 24-40 -> 7-10; a splinified
  random-weight `ace_model`: NU = LEN.
* `SplineRnlrzzBasis` gets a `tables` field (concretely typed from the
  existing type parameters); the previous 7-argument positional
  constructor builds the tables, so `splinify` and `ace1_model` are
  unchanged and `basis.splines` is kept for anything that reads it
  (`fasteval.jl`, `_spline_zz`).  `meta["radial_factorisation"] =
  (LEN, NU, factorised)`.
* `evaluate_batched!` / `evaluate_ed_batched!` (signatures unchanged, so
  Part A's `evaluate_basis_ed!` and `evaluate` / `grad_params` use them
  as before) call `_spline_tables_batched!`: per edge, transform and
  envelope once (scalar `Dual` for the derivative), the cubic B-spline
  position and the 4 value / 4 gradient weights (same arithmetic as
  `Interpolations`' `Cubic(Line(OnGrid()))`: `_cubic_pos`,
  `_cubic_value_weights`, `_cubic_gradient_weights`), the NU distinct
  columns' value and r-derivative, then the expansion to LEN columns with
  SIMD loops.  Scratch space comes from Bumper (`@no_escape`/`@alloc`).
  Generic in the element type of `rs`, so `Dual` positions
  (`jacobian_grad_params`, `ad_hessian_site`) still work.  The scalar
  `evaluate` / `evaluate_ed` (one edge, `SVector{LEN}`) are unchanged.
* `radial_transforms.jl`, `radial_envelopes.jl`: `_intpow(x, n)` expands
  the run-time integer powers of `GeneralizedAgnesiTransform` and
  `PolyEnvelope2sX` (exponents 0-4) into multiplications (5-7 % of the
  optimised call).

#### `src/models/site_ed.jl` (new)

* `SiteEDWorkspace{T}` (rs, ∇rs, Rnl, dRnl, Ylm, dYlm, A, AA, ∂A, ∂Rnl,
  ∂Ylm, ∇Ei, sbuf, Rpair, dRpair), grown by doubling when a site has more
  neighbours.
* `fold_readout_weights(model, ps[, iz])`: `wAA[iz] = A2B' * WB[:, iz]`.
* `evaluate_ed!(ws, model, Rs, Zs, Z0, ps, st, wAA) -> Ei` with `∇Ei` in
  `ws.∇Ei`: radial and Ylm embeddings into the workspace, `ET.evaluate!`
  for A and AA once, `Ei = wAA . AA`, `ET.pullback!` for ∂A and (∂Rnl,
  ∂Ylm) into the workspace, `_assemble_grad_fast!` (radial part reduced to
  a scalar per edge before the vector update), pair basis contracted
  column by column.  No B, no A2B matvecs, no KernelAbstractions launches,
  zero allocations per site (tested).
* `evaluate_ed(model, Rs, Zs, Z0, ps, st)` (`ace.jl`) is now a wrapper
  that allocates a workspace and folds the weights per call; return type
  unchanged (`(Float64, Vector{SVector{3,Float64}})`, `@inferred`).
  `_assemble_grad_ed!` is removed with the old body.

#### `src/models/calculators.jl`

* `energy_forces_virial(at, V::ACEPotential{<:ACEModel}, ps, st; domain,
  executor, ntasks, nlist, ws)`: species gathered once per call
  (`atomic_number(at, :)`), weights folded once, the sites split into
  `ntasks` chunks on `Threads.@spawn` tasks (default
  `Threads.nthreads()`; `ntasks = 1` or `SequentialEx()` is serial), each
  chunk (`_efv_chunk`, a function barrier on the concrete types) reading
  its neighbourhoods with `NeighbourLists.neigs!` into reused buffers and
  accumulating unit-free `E::T`, `F::Vector{SVector{3,T}}`,
  `V::SMatrix{3,3,T}`; chunks summed in order, units attached once.  The
  returned types are exactly those of `zero_energy` / `zero_forces` /
  `zero_virial` (tested).  `nlist` (existing keyword, now used) and `ws`
  (a vector of `SiteEDWorkspace`s) can be passed to reuse the neighbour
  list and the workspaces across calls, e.g. from an MD driver.
* New method `energy_forces_virial(at, V::ACEPotential{<:ACEModel}; kwargs...)`
  so that `AtomsCalculators.energy_forces_virial(sys, calc)` (and
  `energy_forces`, `forces`) take this path instead of the generic
  `SitePotential` driver.  `potential_energy` and `virial` still go
  through the generic driver (`eval_site` / `eval_grad_site`), which now
  uses the new kernels but still allocates per site.
* The chunked result differs from the serial one only by summation order
  (tested to 1e-12; observed <= 4e-16 relative).

#### Tests: `test/models/test_forward_fast.jl` (registered in `test/models/test_models.jl`)

The previous `evaluate_ed` (per-edge `Interpolations` splines via the
scalar `evaluate_ed(basis, r, ...)`, `ET.evaluate` + `ET.pullback`,
`_assemble_grad_ed!`) and the generic driver loop are kept verbatim as the
reference.  Models: `ace1_model` with (Si, O, C) at degrees 6 and 8,
`ace1_model([:Si], totaldegree = 10)`, `ace_model` with the learnable
radial basis un-splinified and splinified (random weights: NU = LEN, the
non-factorisable case).  Checks: spline tables against the per-edge
splines (1e-13 / 1e-12) including the explicitly unfactorised tables
(`factorise = false`) and `Dual` numbers through the value kernel;
`evaluate_ed` against the reference (energies 1e-12, gradients 1e-10),
`@inferred`, workspace growth, zero allocations of `evaluate_ed!` for
spline bases, empty neighbourhood, symmetric `ForwardDiff` Hessian through
`evaluate_ed`; `energy_forces_virial` against the reference driver on
random 16-atom structures (types identical, energies 1e-12, forces and
virials 1e-10), the `AtomsCalculators` entry point, serial versus
`ntasks = 3` (1e-12), `potential_energy`, `domain` as a range and a
vector, `nlist` / `ws` reuse and its allocation bound (bytes and count),
which the old path exceeds by ~100x.

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

# after Part B (both parts on the branch):
julia +1.12 -t 3 --project=<testenv> -e 'using Test; @testset "Forward fast" begin include("test/models/test_forward_fast.jl") end'
   Forward fast  |  313    313  41.9s
julia +1.12 -t 2 --project=<testenv> -e 'using Test, LazyArtifacts; @testset "all" begin @testset "Models" begin include("test/models/test_models.jl") end; @testset "Test silicon" begin include("test/test_silicon.jl") end end'
   all           | 1300   1300  3m03.9s      (test_models.jl in full + test_silicon.jl)
julia +1.12 -t 2 --project=<testenv> -e '... test_recompw.jl, test_json.jl, test_io.jl, test_bugs.jl ...'
   13 pass, 1 "Unexpected Pass": the pre-existing `@test_broken` in test/test_bugs.jl:47
   (Julia >= 1.12 basis-ordering issue) passes on this Mac, as its own comment says it does;
   unrelated to this PR.
```
```
# with the tensor kernels from EquivariantTensors 0.5.2 (pushforward_rows!),
# ET dev'd from the feat/pushforward-vector-tangents branch:
julia +1.12 -t 3 --project=<testenv> -e '... test/models/test_basis_ed.jl ...'
   Basis ED      |  158    158  56.5s       (unchanged test, same 1e-12 tolerances)
julia +1.12 -t 2 --project=<testenv> -e '... test_forward_fast.jl, test_ace.jl, test_calculator.jl, test_silicon.jl ...'
   Forward fast  |  313    313  50.6s
   ACE Model     |  324    324  24.5s
   ACE Calculator |  140    140   9.0s
   Test silicon  |   49      1  1m53.9s     (the BLR liq-E threshold, see above; same with the old kernels on ET 0.5.2)
# EquivariantTensors itself, full suite on the branch:
julia +1.12 --project=<ET> -e 'using Pkg; Pkg.test()'
   EquivariantTensors.jl | 5916   5916  2m54.1s
```
The ET-backend tests (`test/et_models`, `test/etmodels`) do not touch the
changed code and were not run.

`test_ace.jl` includes the existing `evaluate_basis_ed` checks against
`evaluate_ed` and against `jacobian_grad_params` (before and after
splinification); `test_calculator.jl` includes the existing
`energy_forces_virial_basis` versus `energy_forces_virial` check.

## Before / after

### Part A (design-matrix assembly)

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
`energy_forces_virial_basis` path does not do.  These Part A numbers were
taken before Part B; with Part B's radial kernel the serial
`energy_forces_virial_basis` is 23.0 ms / 18.5 MB (D = 6) and
75.8 ms / 58.9 MB (D = 8) on the same structure, i.e. 250x / 190x.)

With the tensor kernels from ET 0.5.2 instead of the local ones (same
script, same machine, same session-to-session comparison with ET 0.5.2 in
both runs, `-t 1`, `NASSEMBLE=4`):

| serial, 32-atom structure | D = 6 | D = 8 |
|---|---|---|
| `evaluate_basis_ed` with `ws`, local kernels | 0.964 ms, 12.7 MB | 3.05 ms, 35.6 MB |
| `evaluate_basis_ed` with `ws`, ET `pushforward_rows!` | 0.833 ms, 12.7 MB | 2.74 ms, 35.6 MB |
| `energy_forces_virial_basis`, local kernels | 23.5 ms, 18.5 MB (197x old) | 80.0 ms, 59.4 MB (196x old) |
| `energy_forces_virial_basis`, ET `pushforward_rows!` | 20.5 ms, 15.9 MB (215x old) | 66.2 ms, 39.6 MB (229x old) |

i.e. unchanged within noise or slightly faster (the `A`/`AA` intermediates
are no longer heap arrays in the workspace but Bumper-stack allocations
inside ET); results agree with the old ForwardDiff path to 3-4e-15 as
before.

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

### Before / after, Part B (forward evaluation)

Same machine and Julia.  "before" = the original checkout's environment
(`acejax/julia`, whose `ACEpotentials` is the jax-eval branch containing
`main`'s 266f84eb kernel; the numbers reproduce
`FINDINGS_forward_profile.md` §1), "after" = this branch, both timing
`AtomsCalculators.energy_forces_virial(sys, m)` (min of 7 after warm-up;
`bench_forward.jl`, logs `bench_forward_{before,after}_t{1,4}.log`).
`nlist` = the neighbour list passed in; `+ws` = neighbour list and
workspaces reused.  Structures: the 32-atom Cantor cell and its 2x2x2
replica (80 neighbours/site), a rattled 64-atom Si cell and its 2x2x2
replica (46 neighbours/site).  Energies agree to all printed digits
(1e-6 eV) across the two processes; in-process against the verbatim old
kernel: |dE| <= 4.5e-13 eV, max |dF| <= 4.7e-14 eV/Å, max |dV| <= 8.4e-11 eV.

`-t 1`:

| model | atoms | before ms / atom-steps/s / MB / allocs | after ms / steps/s / MB / allocs | after +nlist | after +nlist+ws (KB, allocs) | speed-up (plain / +nlist+ws) |
|---|---|---|---|---|---|---|
| cat_D6 | 32 | 3.87 / 8.3e3 / 6.6 / 37 119 | 1.43 / 2.2e4 / 0.9 / 1 322 | 1.22 / 2.6e4 | 1.20 / 2.7e4 / 172 KB / 64 | **2.7x / 3.2x** |
| cat_D6 | 256 | 31.3 / 8.2e3 / 52.9 / 294 992 (GC 4 %) | 11.3 / 2.3e4 / 4.1 / 8 499 | 9.42 / 2.7e4 | 9.49 / 2.7e4 / 205 KB / 69 | **2.8x / 3.3x** |
| cat_D8 | 32 | 6.31 / 5.1e3 / 12.1 / 37 183 | 2.74 / 1.2e4 / 1.6 / 1 325 | 2.47 / 1.3e4 | 2.49 / 1.3e4 / 652 KB / 64 | **2.3x / 2.5x** |
| cat_D8 | 256 | 53.1 / 4.8e3 / 96.8 / 295 504 (GC 6 %) | 21.1 / 1.2e4 / 4.9 / 8 502 | 19.2 / 1.3e4 | 19.3 / 1.3e4 / 685 KB / 69 | **2.5x / 2.8x** |
| Si_D10 | 64 | 2.98 / 2.2e4 / 5.5 / 45 959 | 1.64 / 3.9e4 / 1.2 / 2 317 | 0.94 / 6.8e4 | 0.94 / 6.8e4 / 18 KB / 38 | **1.8x / 3.2x** |
| Si_D10 | 512 | 19.9 / 2.6e4 / 40.9 / 365 860 | 10.0 / 5.1e4 / 5.1 / 16 667 | 7.50 / 6.8e4 | 7.57 / 6.8e4 / 93 KB / 45 | **2.0x / 2.6x** |

`-t 4` (before: the generic driver's `Folds` threading over
`Threads.nthreads()` chunks; after: `ntasks = 4` `@spawn` chunks):

| model | atoms | before ms / steps/s / allocs | after ms / steps/s / allocs | after +nlist+ws | speed-up vs before-4 | vs before-1 |
|---|---|---|---|---|---|---|
| cat_D6 | 32 | 1.44 / 2.2e4 / 38 096 | 0.72 / 4.5e4 / 1 971 | 0.37 / 8.8e4 / 176 | 2.0x / 3.9x | 5.4x / 10.6x |
| cat_D6 | 256 | 12.4 / 2.1e4 / 300 234 (GC 10 %) | 4.68 / 5.5e4 / 8 951 | 2.65 / 9.7e4 / 167 | 2.6x / 4.7x | 6.7x / **11.8x** |
| cat_D8 | 32 | 2.03 / 1.6e4 / 38 512 | 1.16 / 2.8e4 / 1 983 | 0.78 / 4.1e4 / 176 | 1.7x / 2.6x | 5.4x / 8.1x |
| cat_D8 | 256 | 16.7 / 1.5e4 / 303 562 | 7.39 / 3.5e4 / 8 963 | 5.35 / 4.8e4 / 167 | 2.3x / 3.1x | 7.2x / **9.9x** |
| Si_D10 | 64 | 1.54 / 4.2e4 / 46 328 | 1.03 / 6.2e4 / 2 820 | 0.29 / 2.2e5 / 137 | 1.5x / 5.3x | 2.9x / 10.2x |
| Si_D10 | 512 | 7.75 / 6.6e4 / 366 241 | 5.10 / 1.0e5 / 17 110 | 2.16 / 2.4e5 / 143 | 1.5x / 3.6x | 3.9x / **9.2x** |

(The 4-thread "after" without `nlist` is bounded by the serial
`PairList` rebuild - 2.1 ms of the 4.7 ms for cat_D6/256 - which is why
an MD driver should pass `nlist`; with it the 4-task path scales 3.6x
over 1 task and the site loop allocates ~170 objects per call, none per
site.)

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

## Other follow-ups (not in this PR)

* **EquivariantTensors `_jacobian_X`** (`sparse_ace_basis.jl`,
  `sparseprodpool.jl`, `sparsesymmprod.jl`): the batched
  `(maxneigs, nnodes, nfeat)` KernelAbstractions version of the same
  operation still takes `promote_type(Float64, SVector{3}) = Any` for the
  tangent element type; the single-node `pushforward_rows!` now in ET
  (which Part A uses) could be the CPU reference for fixing it, after
  which Part A could evaluate all sites of a structure in one batched call.
* **EquivariantTensors `_pb_evaluate_pbAA!`** (`sparsesymmprod.jl:174`,
  vector version): no `@inbounds`, generic over the correlation order; it
  is now the largest single item of the optimised site for the degree-8
  categorical model (~23 us of ~67 us per site).  `@inbounds` plus
  order-1/2/3 specialisations would give ~1.5-2x on that stage.
* **3-factor A basis** `A = Σ_j P_{n'}(r_j) Y_lm(r_j) e_k(z_j)`
  (`PooledSparseProduct{3}` over `(P, Ylm, E)` with `P` only NU wide,
  `E` the one-hot or embedding row): removes the LEN-wide `Rnl`, `dRnl`,
  `∂Rnl` arrays and the expansion / pullback / assemble passes over them,
  which are now ~half of the remaining site time for the categorical
  models and would make an embedded model's cost independent of the
  embedding dimension.  Projected ~2x on top of Part B; a spec-generation
  change in `_generate_ace_model` / `ace1_compat.jl`.
* Type the `ps` / `st` / `co_ps` fields of `ACEPotential` (currently
  `Any`): `eval_grad_site` and the generic `potential_energy` / `virial`
  drivers would then infer; the fast `energy_forces_virial` is already a
  function barrier so this only matters for the generic paths.
* Contract the pair basis with `Wpair` at `splinify` time (one scalar
  spline per pair) - the pair basis is ~15 % of the optimised site.
* The learnable (`LearnableRnlrzzBasis`) `evaluate_ed_batched!` still
  allocates per edge; only the spline basis got the new kernel (production
  models are splinified).

## Deliberately left out

* Batched / GPU-ready evaluation via `EquivariantTensors._jacobian_X`: it
  uses `promote_type(Float64, SVector{3}) = Any` for the tangent element
  type and cannot take vector tangents as is.  The per-site kernels went
  upstream as `pushforward_rows!` instead (ET 0.5.2); batching over the
  sites of a structure is a follow-up.
* A specialised `ACEfit.feature_matrix(::AtomsData, ::ACEPotential{<:ACEModel})`
  writing straight into the row layout: the unit-strip/copy-out now costs
  ~2 ms of the 28 ms per structure, not worth a second code path.
* Reducing the remaining allocations in the radial bases
  (`evaluate_ed(::SplineRnlrzzBasis, r, ...)` allocates two vectors per
  neighbour); they are ~10 % of the new per-structure cost.
* Any change to the feature-matrix row layout or to the returned types of
  `energy_forces_virial_basis`.
* Bit-identical serial/threaded results: the chunked sum differs from the
  serial one by summation order (<= 1e-15 relative here), in both parts.
* Routing `potential_energy` and `virial` through the fast driver (they
  still use the generic `SitePotential` `Folds` loops with the new
  kernels); `energy_forces_virial` / `energy_forces` / `forces` are the
  MD-relevant entry points and do take the fast path.
* An `ACEPotential` field for cached folded weights: `wAA = A2B' * WB` is
  recomputed per call (NZ sparse matvecs, ~10 us), which keeps
  `set_linear_parameters!` / committee parameter swaps safe.
