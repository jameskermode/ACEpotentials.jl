# Profiling the forward evaluator (energy / forces / virial)

Scope: `AtomsCalculators.energy_forces_virial(sys, model)` for a fitted
`ACEPotential{ACEModel}` with spline radial bases -- the path an MD driver
calls.  The design-matrix / basis path (`evaluate_basis_ed`,
`energy_forces_virial_basis`) is covered separately in
`FINDINGS_assembly_profile.md`.

Everything here was measured on the local Mac (Apple M3 Pro, 6P+6E cores,
18 GB), Julia 1.13.0, `--project=acejax/julia`, branch `jax-eval`, which
contains main's `266f84eb` "Fix ~13x force-evaluation regression" (the
`_assemble_grad_ed!` function barrier) -- all numbers are on top of that
fix.  Scripts and logs: `acejax/bench/profile_forward/`.  Nothing in `src/`
was changed; the optimised evaluator is a scratch override
(`fast_efv.jl`) built from an `ACEPotential`.

Models (linear parameters randomised so evaluation does real work):

| name | construction | n_B/elem | nR (radial width) | nY | nA | nAA | npair |
|---|---|---|---|---|---|---|---|
| cat_D6 | `ace1_model(elements=[Cr,Mn,Fe,Co,Ni], order=3, totaldegree=6)` | 1348 | 74 | 9 | 72 | 1964 | 30 |
| emb16_D6 | `ace_embedding_model(..., totaldegree=6, embedding=MH-1, d_max=16)` | 323 | 224 | 9 | 177 | 415 | 6 |
| cat_D8 | as cat_D6, degree 8 | 3824 | 126 | 25 | 157 | 7600 | 40 |
| emb16_D8 | as emb16_D6, degree 8 | 753 | 384 | 25 | 380 | 1217 | 8 |
| Si_D10 | `ace1_model(elements=[:Si], order=3, totaldegree=10)` | 110 | 37 | 25 | 43 | 230 | 10 |

Structures: `cantor1k_b_mh1.xyz` frame 1 (32 atoms, rcut 6.25 A, 79
neighbours/atom) and its 2x2x2 / 3x3x3 supercells (256 / 864 atoms); for Si
the largest `Si_tiny` cell (64 atoms, 43 neighbours) and its 2x2x2 (512).

## 1. Measurement table (production path, `01_measure.jl`)

Min of 5 timed calls after warm-up.  "allocs" is the number of heap
allocations per call; GC never triggered inside the timed calls (0 %), but the
allocation volume (~220 KB and ~1150 allocations **per site**) is what stops
the threaded path scaling, see section 6.

| model | natoms | threads | ms/call | atom-steps/s | MB alloc | allocs | GC % |
|---|---|---|---|---|---|---|---|
| cat_D6 | 32 | 1 | 3.99 | 8.0e3 | 6.95 | 36 969 | 0 |
| cat_D6 | 256 | 1 | 30.4 | 8.4e3 | 55.5 | 293 946 | 0 |
| emb16_D6 | 32 | 1 | 6.20 | 5.2e3 | 7.97 | 36 937 | 0 |
| emb16_D6 | 256 | 1 | 46.9 | 5.5e3 | 63.7 | 293 690 | 0 |
| cat_D8 | 32 | 1 | 6.61 | 4.8e3 | 12.7 | 37 033 | 0 |
| cat_D8 | 256 | 1 | 51.0 | 5.0e3 | 101.5 | 294 458 | 0 |
| emb16_D8 | 32 | 1 | 10.1 | 3.2e3 | 12.1 | 37 097 | 0 |
| emb16_D8 | 256 | 1 | 77.7 | 3.3e3 | 97.0 | 294 970 | 0 |
| Si_D10 | 64 | 1 | 3.03 | 2.1e4 | 5.6 | 43 887 | 0 |
| Si_D10 | 512 | 1 | 20.2 | 2.5e4 | 41.4 | 349 423 | 0 |
| cat_D6 | 32 | 4 | 1.47 | 2.2e4 | 7.0 | 37 946 | 0 |
| cat_D6 | 256 | 4 | 10.5 | 2.4e4 | 55.9 | 299 188 | 0 |
| emb16_D6 | 32 | 4 | 1.95 | 1.6e4 | 8.0 | 37 306 | 0 |
| emb16_D6 | 256 | 4 | 13.8 | 1.9e4 | 63.8 | 294 068 | 0 |
| cat_D8 | 256 | 4 | 15.7 | 1.6e4 | 102.2 | 302 516 | 0 |
| emb16_D8 | 256 | 4 | 22.1 | 1.2e4 | 97.1 | 295 348 | 0 |
| Si_D10 | 512 | 4 | 7.65 | 6.7e4 | 41.5 | 349 804 | 0 |

These reproduce the lestrade numbers in `acejax/bench/distil/RESULTS.md`
(8.3e3 categorical, 4.7e3 embedded, 2.6e3 embedded deg 8, one thread): an
M3 Pro core is ~1.1x a lestrade core on this code.

Size scaling: atom-steps/s is flat from 32 to 256 (and 864) atoms -- the
evaluator is linear in system size.  The neighbour list **is rebuilt on every
call** (`PairList(sys, cutoff)` is the default of the `nlist` kwarg in
`AtomsCalculatorsUtilities.SitePotentials.energy_forces_virial`); it costs
0.30 ms at 32 atoms and 2.1 ms at 256 atoms, i.e. 5-8 % of a production call
(measured by passing `nlist=` explicitly) but ~20 % of the optimised call.

Threading: the site loop is threaded, but not in ACEpotentials -- the call
dispatches to the generic `SitePotential` driver in
`AtomsCalculatorsUtilities/src/sitepotentials/assembly.jl:56`, which does
`Folds.sum(collect(index_chunks(domain; n=ntasks)), ThreadedEx())` with
`ntasks = Threads.nthreads()`, each chunk accumulating into its own
`zero_forces` copy.  (ACEpotentials' own `energy_forces_virial(at, V, ps, st)`
in `calculators.jl:139` is the same pattern but is only reached through the
4-argument / rrule path.)  No `Threads.@threads` / `@floop` anywhere in
`src/models/`.  4 threads give 2.6-3.4x on 256 atoms.

## 2. Where the time goes (production path)

### 2a. Per-site stage timings (`02_profile.jl`, one site with 80 neighbours, min over repeats)

| stage | cat_D6 | emb16_D6 | cat_D8 | emb16_D8 | Si_D10 (45 nb) |
|---|---|---|---|---|---|
| `get_neighbours` (Js, Rs, Zs gather) | 5.0 us | 5.0 | 5.0 | 5.0 | 3.0 |
| `radii_ed!` | 0.1 | 0.1 | 0.1 | 0.1 | 0.0 |
| **radial `evaluate_ed_batched!` (splines)** | **39.6** | **99.3** | **55.7** | **196.4** | **9.7** |
| `P4ML.evaluate_ed!` (Ylm) | 1.5 | 0.7 | 4.5 | 2.3 | 2.4 |
| A basis `evaluate!` | 0.5 | 1.4 | 1.2 | 3.1 | 0.2 |
| A basis `ka_evaluate` (as used inside `ET.evaluate`) | 3.3 | 8.1 | 7.3 | 17.7 | 1.1 |
| AA `evaluate!` / `ka_evaluate` | 1.2 / 3.1 | 0.2 / 0.8 | 5.3 / 10.2 | 0.6 / 1.5 | 0.1 / 0.6 |
| B = A2B*AA (sparse matvec) | 2.5 | 0.4 | 13.3 | 1.5 | 0.3 |
| `ET.evaluate(tensor)` total (as used) | 9.7 | 10.0 | 32.3 | 21.4 | 2.3 |
| dAA = A2B' * dB (sparse matvec) | 2.2 | 0.5 | 8.6 | 1.4 | 0.2 |
| dA `pullback!`(AA) | 6.7 | 0.6 | 23.2 | 2.0 | 0.5 |
| dRnl, dYlm `pullback!`(A) | 3.3 | 9.2 | 6.5 | 17.5 | 1.3 |
| `ET.pullback` total (as used) | 14.7 | 13.2 | 47.4 | 25.5 | 3.5 |
| `_assemble_grad_ed!` | 5.3 | 14.4 | 9.8 | 26.4 | 2.2 |
| **pair basis `evaluate_ed_batched` (splines, allocating)** | **24.4** | **17.9** | **28.3** | **18.7** | **6.0** |
| **`evaluate_ed` total** | **97.6** | **167.6** | **184.4** | **302.9** | **28.4** |
| `evaluate` (energy only) | 56.4 | 73.5 | 92.4 | 126.5 | 14.9 |

The radial splines (many-body + pair) are 65 % (cat_D6), 70 % (emb16_D6),
45 % (cat_D8), 71 % (emb16_D8), 55 % (Si) of the site cost.  The whole tensor
part (A, AA, A2B, both pullbacks) is 25 % / 14 % / 43 % / 15 % / 20 %.

### 2b. Flat sampling profile, `energy_forces_virial`, 256 atoms (inclusive counts)

cat_D6, 2118 samples:

| inclusive | file:line | what |
|---|---|---|
| 28.5 % | `src/models/ace.jl:341` | `evaluate_ed_batched!` -- radial splines |
| 24.7 % | `src/models/ace.jl:384` | pair basis `evaluate_ed_batched` (allocating) |
| 19.2 % | `src/models/ace.jl:365` | `EquivariantTensors.pullback` (A2B', AA pb, A pb) |
| 7.6 % | `src/models/ace.jl:353` | `EquivariantTensors.evaluate` (KA launches + A2B) |
| 5.9 % | `NeighbourLists/src/atoms_base.jl:6` | `PairList` rebuild |
| 3.8 % | `AtomsCalculatorsUtilities/.../neighbourlist.jl:44` | `get_neighbours` (incl. 3 % in `get_id`/`atomic_number`) |
| 3.4 % | `src/models/ace.jl:379` | `_assemble_grad_ed!` |
| 2.5 % | `AtomsCalculatorsUtilities/.../assembly.jl:84` | force accumulation with Unitful |
| 1.4 % | `src/models/ace.jl:345` | Ylm |
| 9.6 % (self) | `Base/boot.jl:588 GenericMemory` | allocation |

Inside the radial spline cost, `Rnl_splines.jl:38-39` (the transform `T_ij(r)`
and envelope `evaluate(env_ij, r, x_ij)` with `Dual` numbers, both calling
`^` with a run-time integer exponent) are 18.8 % of the *total* call -- as
much as the actual B-spline evaluation at `Rnl_splines.jl:44` (16.6 %).

emb16_D6, 2943 samples: `ace.jl:341` radial 48.7 %, `ace.jl:365` pullback
17.1 %, `ace.jl:384` pair 10.0 %, `ace.jl:379` assemble 8.1 %, `ace.jl:353`
forward tensor 5.0 %, PairList 4.1 %, get_neighbours 2.5 %, allocation
(`GenericMemory` self) 8.4 %.

Full flat and tree profiles: `02_profile.log`.

### 2c. Why the radial basis is so expensive

`SplineRnlrzzBasis` stores, for each species pair, one `Interpolations`
cubic B-spline whose *value* is an `SVector{LEN, Float64}` (LEN = 74 / 224 /
384).  `evaluate_ed` (`Rnl_splines.jl:88`) pushes a `Dual` through the whole
thing, so per edge it builds `SVector{LEN, Dual}` objects (3.5 KB for
LEN = 224), unrolled by StaticArrays with no SIMD, then extracts value and
derivative in two more passes, then copies the rows into `Rnl[j, :]`.
Measured cost 5.5-6.7 ns per (function, edge), i.e. ~1.2 GFlop/s.

More importantly, **almost all of those LEN columns are redundant**.  A
column-factorisation of the spline tables (`fast_efv.jl:_factorise_columns`,
tolerance 1e-12 relative, verified exact to 1e-10 absolute) finds:

| model | LEN | distinct columns per pair (NU) |
|---|---|---|
| cat_D6 | 74 | 7 |
| emb16_D6 | 224 | 6 |
| cat_D8 | 126 | 9 |
| emb16_D8 | 384 | 8 |
| Si_D10 | 37 | 10 |
| pair bases | 30 / 6 / 40 / 8 / 10 | 7 / 2 / 9 / 3 / 10 |

The reason is structural, not numerical.  In `LearnableRnlrzzBasis`
`R_{nl}(r) = sum_q W[(n,l), q] P_q(x)` and both initialisations used here make
`W` one-hot in `q`: `set_onehot_weights!` (`Rnl_learnable.jl:83`, categorical:
`R_{(n,l)}(iz,jz) = P_{n'}(x) delta(z', jz)` with `n = (n'-1)*NZ + z'`) and
`set_embedding_weights!` (`embeddings.jl:200`, `R_{(n,l)}(iz,jz) =
emb[jz,k] P_{n'}(x)` with `n = (n'-1)*d + k`).  Neither depends on `l`, so
every Rnl column is a scalar multiple of one of `maxn'` polynomials -- and
for the categorical model 4/5 of the columns are *zero* on any given edge.
Splinifying afterwards preserves this exactly (the B-spline prefilter is
linear).  The evaluator nevertheless evaluates all LEN columns per edge with
Dual arithmetic, which is what makes the embedded model 1.8x slower than the
categorical one despite having 4x fewer basis functions.

## 3. Type-stability and allocation audit (`03_typeaudit.jl`)

`Base.return_types` on the call chain:

| call | inferred return | status |
|---|---|---|
| `get_neighbours(sys, V, nlist, i)` | `Tuple{Vector{Int}, Vector{SVector{3,Float64}}, Vector, Any}` | **BAD** (Zs eltype, z0 unknown) |
| `radii_ed!` | `Tuple{Vector{Float64}, Vector{SVector{3,Float64}}}` | OK |
| `evaluate_ed_batched!` (spline Rnl) | `Tuple{Matrix{Float64}, Matrix{Float64}}` | OK |
| `evaluate(SplineRnlrzzBasis, r, Zi, Zj)` | `SVector{74, Float64}` | OK |
| `P4ML.evaluate_ed` (Ylm) | `Tuple{Matrix{Float64}, Matrix{SVector{3,Float64}}}` | OK |
| `ET.evaluate!`, `ET.ka_evaluate` (A, AA), `ET.evaluate(tensor)` | concrete | OK |
| `ET.pullback(dBB, tensor, Rnl, Ylm, A)` | `Tuple{Any, Any}` | **BAD** (known; handled by the `_assemble_grad_ed!` barrier of #326) |
| `evaluate_ed(model, Rs, Zs, z0, ps, st)` | `Tuple{Float64, Vector{SVector{3,Float64}}}` | OK |
| `eval_grad_site(V, Rs, Zs, z0)` | `Tuple{Any, Vector{SVector{3,Float64}}}` | **BAD** |
| `energy_forces_virial(sys, m)` | `NamedTuple{..., <:Tuple{Any,Any,Any}}` | **BAD** |

The two remaining instabilities are in the driver, not the kernel: `get_id`
(`AtomsCalculatorsUtilities/.../interface.jl:90` = `atomic_number(sys, i)`)
is not inferred for a `FlexibleSystem`, so `Zs` / `z0` are `Any` at the call
site of `evaluate_ed`, which is dispatched dynamically once per site (cheap)
but leaves `Ei::Any` and the accumulated energy/virial untyped.  The
`ACEPotential` struct itself has untyped `ps`, `st`, `co_ps` fields (`Any`),
so `V.ps`/`V.st` are also runtime-dispatched on every `eval_grad_site` --
again once per site, so ~100 ns each, not the problem, but it means nothing
above `evaluate_ed` is inferable.

Allocation scaling.  `evaluate_ed` allocates **~62 per site + ~6 per edge**
(cat_D6: 122 allocs / 10 neighbours, 543 / 80; emb16_D6 identical counts).
`Profile.Allocs` attributes the per-edge ones to:

* `Rnl_splines.jl:37-39`: `basis.transforms[iz,jz]`, `basis.envelopes[iz,jz]`,
  `basis.splines[iz,jz]` -- indexing the `SMatrix{NZ,NZ}` of non-isbits
  spline / envelope objects boxes the result (2555 + 1341 + 1316 sampled
  allocations of `NormalizedTransform`, `PolyEnvelope2sX`,
  `Interpolations.Extrapolation`);
* `AtomsCalculatorsUtilities/.../interface.jl:90 get_id`: `Atom`,
  `ChemicalSpecies`, `Broadcasted`, `RefValue` -- 4 allocations per edge for
  `atomic_number(sys, j)` on a `FlexibleSystem`;
* `AtomsCalculatorsUtilities/.../assembly.jl:84`: `SVector{3, Quantity}` -- the
  `frc[Js[a]] -= dEi[a] * force_unit(V)` accumulation boxes because `dEi`
  comes out of the un-inferred `eval_grad_site`: 3 allocations per edge.

The per-site ones are the `zeros` for `A`, `dRnl`, `dYlm`, `gradEi`, the KA
`similar` for A and AA, the two `A2B` matvec results, the pair-basis `Rnl` /
`Rnl_d` matrices (`evaluate_ed_batched`, not the `!` version) and
`sum(Rpair, dims=1)[:]`.  Bumper (`@no_escape`/`@withalloc`) already covers
`rs`, `grad rs`, `Rnl`, `dRnl`, `Ylm`, `dYlm`.

Other things noticed in `evaluate_ed` (`ace.jl:322-400`):

* the A basis is computed twice: once explicitly (`ace.jl:350`, for the
  pullback) and again inside `EquivariantTensors.evaluate(model.tensor, ...)`
  (`sparse_ace_basis.jl:102`), which also goes through `ka_evaluate` --
  KernelAbstractions CPU launches (`__thread_run`) that cost 3.3 us vs 0.5 us
  for the plain `evaluate!` on cat_D6, and 8 us vs 1.4 us on emb16_D6;
* the pullback does `dAA = sum(A2Bmaps[i]' * dBB[i])` (allocating sparse
  matvec, `sparse_ace_basis.jl:144`) and B = A2B*AA in the forward pass, then
  `dot(B, WB)`; for a *fitted* model both matvecs can be folded away once by
  precomputing `wAA_z = A2B' * WB[:, z]` (this is exactly what
  `fasteval.jl:FastACEInner` does);
* the A2B / AA / A pullbacks are all sparse (CSC matvec; tuple-spec scatter
  loops); nothing is dense.  `_pb_evaluate_pbAA!` (`sparsesymmprod.jl:174`,
  vector version) has no `@inbounds` and is 23 us for cat_D8 (12.6 % of the
  site).

## 4. The radial-width question and the factorised radial

Fraction of site time in the radial basis (many-body + pair) for the embedded
model: **70 % (deg 6), 71 % (deg 8)**; the A/AA/A2B tensor work is 14-15 %.

The prototype (`fast_efv.jl`) replaces the spline evaluation by:

1. per edge, scalar `Dual` evaluation of transform + envelope, then the cubic
   B-spline position and the 4 value / 4 gradient weights taken from
   `Interpolations.positions/value_weights/gradient_weights` (so the
   arithmetic is the same as production's, up to fp reassociation);
2. the NU distinct columns of the coefficient table evaluated for all edges
   with plain SIMD loops over a `(npad, NU, NZ^2)` array;
3. expansion `Rnl[j, n] = c[n, pair_j] * P[j, u[n, pair_j]]` (and the same for
   the derivative).

This is the `P[:, n'] * emb[z, k]` factorisation that acejax exploits, and it
applies unchanged to the one-hot categorical basis (`c in {0, 1}`).  Radial
cost per site (80 neighbours), production -> prototype:
cat_D6 39.6 + 24.4 (pair) -> 9.1 + 5.4 us; emb16_D6 99.3 + 17.9 -> 16.5 + 3.1
us; emb16_D8 196 + 19 -> 28 + 3 us.  What is left of the radial cost is
dominated by the *expansion* stores (2 x LEN x nX doubles per site, ~16 us for
LEN = 224) and 3.3 us of per-edge transform/envelope scalars, not by the
splines themselves (step 2 is < 1 us).

## 5. `fasteval.jl`

`fast_evaluator` (`src/models/fasteval.jl`) is a per-element evaluator that
folds `WB` through `A2B` into `wAA` (`A2Bmap' * wB[:, iz]`), prunes the AA
functions with zero weight, rebuilds a reduced `PooledSparseProduct` /
`SparseSymmProd` for each element, optionally generates a static polynomial
for `AA . wAA` (`aa_static`, models under 1200 AA functions), and splines the
pair basis contracted with `Wpair` into one scalar spline per pair.  It uses
the same `evaluate_ed_batched!` spline radial basis as production, so it
would not touch the dominant cost.  It does not currently work: its kernels
call `P4ML.evaluate!` / `P4ML.pullback!` on the `EquivariantTensors`
`PooledSparseProduct` and `SparseSymmProd` objects (no such methods since the
ET migration, `fasteval.jl:185,197,205,245,257`) and `get_nnll_spec` on the tensor,
which is why the tests in `test/test_fast.jl` are skipped.  Its two useful
ideas -- folding `WB` through `A2B`, and a scalar pair spline -- are both in
the prototype below (the first one exactly, the second as the factorised pair
table; contracting with `Wpair` first would make the pair cost ~1 us).  The
`aa_static` StaticPolynomials trick is only viable for tiny models.

## 6. Prototype results (`fast_efv.jl`, `04_opt.jl`, `06_threads.jl`)

The scratch evaluator `FastEFV(m)` implements, in ~330 lines outside `src/`:
R1 spline kernel, R2 column factorisation (many-body and pair), T1 `wAA`
folding (no B, no A2B matvecs, no KA launches), T2 A basis once, W1 reusable
per-task workspaces (site loop allocation-free: 2 allocations per 32 sites),
L1 direct `PairList` reads with species pre-gathered once per call and
unit-free accumulation, L2 `Threads.@spawn` over chunks, plus a
reduce-first `_assemble_grad_fast!` (radial part reduced to one scalar per
edge before the SVector update; 3.3-3.6x faster than `_assemble_grad_ed!`).

Exactness against production, every model and cell (`04_opt_t1.log`):
|dE| <= 1.4e-12 eV (energies -400..-3500 eV), max |dF| <= 4e-14 eV/A,
max |dV| <= 6e-11 eV.

One thread (`04_opt_t1.log`), ms per call / atom-steps/s:

| model | natoms | production | fast (nlist+ws per call) | fast (ws reused) | fast (ws + nlist reused) | speed-up |
|---|---|---|---|---|---|---|
| cat_D6 | 32 | 4.01 / 8.0e3 | 1.59 / 2.0e4 | 1.59 | 1.27 / 2.5e4 | 2.5x (3.2x) |
| cat_D6 | 256 | 30.2 / 8.5e3 | 12.0 / 2.1e4 | 11.9 | 9.74 / 2.6e4 | 2.5x (3.1x) |
| emb16_D6 | 32 | 6.14 / 5.2e3 | 1.98 / 1.6e4 | 1.97 | 1.66 / 1.9e4 | 3.1x (3.7x) |
| emb16_D6 | 256 | 46.8 / 5.5e3 | 15.2 / 1.7e4 | 15.2 | 13.0 / 2.0e4 | 3.1x (3.6x) |
| cat_D8 | 256 | 50.7 / 5.0e3 | 21.9 / 1.2e4 | 21.8 | 19.7 / 1.3e4 | 2.3x (2.6x) |
| emb16_D8 | 256 | 78.9 / 3.2e3 | 24.4 / 1.0e4 | 24.4 | 22.2 / 1.2e4 | 3.2x (3.6x) |
| Si_D10 | 512 | 21.2 / 2.4e4 | 10.5 / 4.9e4 | 10.2 | 7.62 / 6.7e4 | 2.0x (2.8x) |

Allocations per call drop from ~294 000 / 55-100 MB (256 atoms) to 8 900 /
4.4 MB with a fresh nlist and workspace, and to 1 310 / 0.05 MB with both
reused (the remainder is `atomic_number(sys, i)` on the `FlexibleSystem`, 5
allocations per atom).

Threads (`06_threads_t4.log`, `julia -t 4`, nlist reused for both):

| model | natoms | production 4 thr | fast 1 task | fast 2 | fast 4 | fast-4 vs prod-4 | vs prod-1 |
|---|---|---|---|---|---|---|---|
| cat_D6 | 256 | 8.17 ms / 3.1e4 | 8.37 / 3.1e4 | 4.43 / 5.8e4 | 2.42 / **1.06e5** | 3.4x | 12x |
| cat_D6 | 864 | 27.4 / 3.2e4 | 28.8 / 3.0e4 | 14.8 / 5.8e4 | 7.76 / **1.11e5** | 3.5x | 13x |
| emb16_D6 | 256 | 11.4 / 2.2e4 | 9.50 / 2.7e4 | 4.90 / 5.2e4 | 2.62 / **9.8e4** | 4.3x | 18x |
| emb16_D6 | 864 | 38.0 / 2.3e4 | 31.9 / 2.7e4 | 16.5 / 5.2e4 | 8.52 / **1.01e5** | 4.5x | 18x |

(`04_opt_t4.log` also has 4-task rows, but those were timed interleaved with
the allocating production benchmarks and show only ~2x; `06_threads.jl` is
the clean run.)  The prototype scales 3.5-3.7x on 4 tasks (no shared mutable
state, no allocation in the loop); the production path gets 2.6-3.4x on 4 threads and
was reported to stop at 8 on the 32-core node, which is consistent with the
allocation rate (220 KB and 1150 allocations per site; `GenericMemory` is
already 8-10 % of the single-thread profile).  Removing the allocations is
what makes the site loop scale, as for the basis-path fix.

What remains per site in the prototype (`05_stages_fast.log`, 80 neighbours):
cat_D6 30 us = radial expansion 9.1 + pair 5.4 + AA pullback 6.5 + A pullback
1.8 + assemble 1.6 + Ylm 1.5 + AA 1.2 + A 0.5; emb16_D6 36 us = radial
expansion 16.5 + A pullback 5.8 + assemble 4.0 + pair 3.1 + A 1.3;
cat_D8 67 us = AA pullback 22.8 + radial 13.3 + assemble 3.7 + pair 6.4 + AA
5.0 + Ylm 4.3 + A pullback 4.2.  The wide `Rnl`/`dRnl`/`dRnl-bar` arrays are
now the cost for embedded models: they are written (expansion), read (A
basis), written again (A pullback, with a `fill!`) and read again (assemble).

## 7. Ranked optimisation list

Gains are per site on the 5-element degree-6 models unless stated; "measured"
means demonstrated by the prototype with the exactness check above.

| # | change | gain | effort | where |
|---|---|---|---|---|
| 1 | **Spline kernel on a plain coefficient table** (value + derivative in one pass, transform/envelope as scalar Duals, no `SVector{LEN,Dual}`), with the **column factorisation** `Rnl[:, n] = c[n, pair] P[:, u(n)]` computed at `splinify` time. | measured: radial 64 -> 15 us (cat_D6), 117 -> 20 us (emb16_D6), 215 -> 31 us (emb16_D8); 2.5-3.2x on the whole call | 1-2 days | `Rnl_splines.jl` (`SplineRnlrzzBasis` storage + `evaluate_ed_batched!`), `Rnl_basis.jl:splinify` |
| 2 | **Allocation-free site loop**: fold `WB` through `A2B` once per element (`wAA`), compute A once, use `evaluate!`/`pullback!` into a per-task workspace instead of `ka_evaluate`/`pullback`, non-allocating pair basis, species gathered once per call, unit-free force accumulation, `_assemble_grad_fast!` (reduce-first). | measured together with 1 (allocations 294k -> 1.3k per 256-atom call; tensor part 25 -> 10 us cat_D6); enables 3.5x/4 tasks and, by extension, scaling past 8 threads | 2-3 days | `calculators.jl` (own `energy_forces_virial` for `ACEPotential` instead of the generic SitePotential driver), `ace.jl:evaluate_ed` |
| 3 | **Neighbour-list reuse** across calls (`nlist=` kwarg already exists; MD driver should keep it with a skin). | measured: 5-8 % of production, 15-25 % of the optimised call (2.1 ms per 256-atom rebuild = 8 us/atom) | hours (driver side) | `calculators.jl` / MD integration |
| 4 | **3-factor A basis** `A = sum_j P_{n'}(r_j) Y_lm(r_j) e_k(z_j)` with `e` the one-hot or the frozen embedding row, i.e. `PooledSparseProduct{3}` over `(P, Ylm, E)` with `P` only NU wide. Removes the LEN-wide `Rnl`, `dRnl`, `dRnl-bar` arrays and the expansion/assemble/pullback passes over them (currently ~60 % of the remaining emb16 site time), makes the embedded model cost independent of `d`. `E` has no position gradient, so the pullback is unchanged for the other two factors. | projected: emb16_D6 36 -> ~15 us, cat_D6 30 -> ~18 us; ~2x on top of 1+2 | 1 week (spec generation in `_generate_ace_model`, ET `PooledSparseProduct{3}` already exists) | `ace.jl`, `embeddings.jl`, `ace1_compat.jl` |
| 5 | `@inbounds` + specialised (order-1/2/3) `_pb_evaluate_pbAA!` vector kernel; it is the top item for degree-8 categorical models (23 us / 31 % of the site). | projected 1.5-2x on that stage | hours | `EquivariantTensors/src/ace/sparsesymmprod.jl:174` |
| 6 | Specialise the transform/envelope powers (`s^q`, `(x-x1)^p1`, integer exponents from struct fields, evaluated on Duals) -- 3.3 us per site per basis, i.e. 6-7 us of the optimised site; also drop the duplicate evaluation of edge 1 in `evaluate_ed_batched!` (`Rnl_splines.jl:105`). | ~10 % of the optimised call | hours | `radial_transforms.jl`, `radial_envelopes.jl` |
| 7 | Contract the pair basis with `Wpair` at `splinify` time (one scalar spline per pair, as `fasteval.jl:_make_pair_splines` does). | pair 5.4 -> ~1 us | hours | `Rnl_splines.jl` / `calculators.jl` |
| 8 | Type the `ps`/`st` fields of `ACEPotential` and make `get_id` inferable, so `eval_grad_site` and the driver return concrete types. | negligible time; removes the `Any` results and 3 allocs/edge in the generic driver | hours | `calculators.jl:20`, AtomsCalculatorsUtilities |

Items 1-3 are demonstrated end-to-end by `fast_efv.jl`; 4 is the design
change that would let the Julia evaluator match acejax's structure.

## 8. Can the Julia CPU evaluator reach parity with a well-written CPU kernel?

Yes -- and the demonstrated prototype already does most of it.  The ideal
per-site work at 80 neighbours, NU ~ 7 radial functions, 9 Ylm, ~180 A and
~400-2000 AA functions is a few times 1e5 flops; at 10 GFlop/s per core that
is ~10-30 us, and the prototype is at 30-36 us per site (cat_D6 / emb16_D6)
with the LEN-wide arrays still in the loop, and would be at ~15-20 us with
item 4.

Against the numbers in `acejax/bench/distil/RESULTS.md` (RTX 4000 Ada,
embedded d<=16 degree 6): the JAX GPU kernel does 7.4e4 atom-steps/s in f64
and 1.6e5 in f32.  The Julia prototype does **2.7e4 on one M3 Pro core and
1.0e5 on 4** (f64, exact to 1e-14 vs production), i.e. it already exceeds
the f64 GPU kernel with 4 cores and reaches 60 % of the f32 one.  On the
32-core node, with the site loop now allocation-free, 8-16 threads at
~2.5e4/thread would put it at 2-4e5 atom-steps/s, above the GPU f32 kernel.
The production path, by contrast, is 5.5e3 per core and stops scaling at
~2e4, which is the 8-30x gap the RESULTS table shows.

The gap was never the ACE tensor algebra (A/AA/A2B/pullbacks are 15-25 % of
the site and are sparse, allocation-light and reasonably written); it is
(i) the spline radial basis evaluating LEN Dual-valued columns per edge of
which only NU ~ 6-10 are distinct, (ii) ~1150 allocations per site in the
driver and the `ka_evaluate`/`pullback` conveniences, which also cap the
thread scaling, and (iii) rebuilding the neighbour list every call.

## Files

* `acejax/bench/profile_forward/common.jl` -- models, structures, timer
* `01_measure.jl` (+ `01_measure_t1.log`, `01_measure_t4.log`) -- table in section 1
* `02_profile.jl` (+ `02_profile.log`) -- stage timings and flat/tree profiles
* `03_typeaudit.jl` (+ `03_typeaudit.log`) -- return types, `@code_warntype`, `Profile.Allocs`
* `fast_efv.jl` -- the scratch optimised evaluator
* `04_opt.jl` (+ `04_opt_t1.log`, `04_opt_t4.log`), `04b_allocs.jl` (+ `.log`) -- before/after, exactness, residual allocations
* `05_stages_fast.jl` (+ `.log`) -- stage timings of the optimised site
* `06_threads.jl` (+ `06_threads_t4.log`) -- thread scaling
