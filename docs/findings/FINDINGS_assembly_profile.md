# Profiling the linear-system assembly (and the post-assembly stages) of the distillation fits

Scripts and logs: `acejax/bench/profile_assembly/` (`01_single`, `02_profile`,
`02b_typeaudit`, `03_opt`, `03b_opt_v4`, `04_assemble`, `04b_overhead`,
`05_post`, `06_deg10`; the scratch implementations are in `basis_opt.jl` and
`residuals.jl`).  Nothing in `src/` was modified; every optimisation below is a
standalone function or a method override inside the scripts.

Machine: this Mac (12 cores, 18 GB), Julia 1.13.0, `--project=acejax/julia`
(ACEpotentials 0.10.2 dev, ACEfit 0.3.1, EquivariantTensors 0.4.3,
Polynomials4ML 0.5.10, ForwardDiff 1.4.5).  **Caveat on absolute numbers**: two
other Julia jobs (`acejax/spike_fs/varpro_*.jl`, 4-6 cores each) ran on this
machine throughout and swap was at 12-16 GB of 16-17 GB the whole time.
Single-structure timings (min of 3, section 2a-c) are reasonably clean; the
multi-worker wall times (2d) and the LAPACK GFLOP/s (2e) are pessimistic by an
unknown factor of roughly 1.5-3.  Ratios between variants measured back to back
are reliable.

Test case: 32 structures (32 atoms each, 1248 atoms, 3968 rows) of
`cantor1k_b_mh1.xyz` (CrMnFeCoNi, keys `mace_energy/force/virial`), 78.9
neighbours per site at rcut 6.25 A.  Models:

| model | n_B / element | columns | nR | nY | nA | nAA | pair |
|---|---|---|---|---|---|---|---|
| `cat_D6`   = `ace1_model(order=3, totaldegree=6)`           | 1348 | 6 890  | 74  | 9  | 72  | 1 964  | 30 |
| `cat_D8`   = `ace1_model(order=3, totaldegree=8)`           | 3824 | 19 320 | 126 | 25 | 157 | 7 600  | 40 |
| `cat_D10`  = `ace1_model(order=3, totaldegree=10)`          | 9327 | 46 885 | 191 | 49 | 275 | 25 843 | 50 |
| `emb16_D6` = `ace_embedding_model(..., d_max=16)` degree 6  | 323  | 1 645  | 224 | 9  | 177 | 415    | 6  |
| `emb16_D8` = `ace_embedding_model(..., d_max=16)` degree 8  | 753  | 3 805  | 384 | 25 | 380 | 1 217  | 8  |

## 1. Summary

1. **The slowness is not arithmetic; it is a type instability plus a
   ForwardDiff Jacobian that allocates and copies far more than it computes.**
   `evaluate_basis_ed` (`src/models/ace.jl:663-693`) returns
   `Tuple{Vector{Float64}, Any}` - `dB` is `Any` because
   `ForwardDiff.jacobian` of the closure is not inferred and the result then
   goes through `collect(dB_vec')[:]`, `reinterpret`, `reshape`, `permutedims`,
   `collect`.  The consumer loop in `energy_forces_virial_basis`
   (`src/models/calculators.jl:311-318`) therefore runs
   `F[Js[a], k] -= dv[k, a] * force_unit(calc)` with dynamic dispatch and a
   boxed `SVector{3,Quantity}` per element: ~2.4e8 allocations, 9.6 GB, per
   32-atom structure for `cat_D6` (26 GB for `cat_D8`).  `@code_warntype` on
   the kwarg body shows `dv::Any`, `z0::Any`, `_e0::Any`; the allocation
   profile puts 99 % of the sampled allocations on `calculators.jl:314-315`.
   **Confirmed as the leading cause for the categorical models** (67 % of
   samples in lines 314-315; 28 % in the ForwardDiff Jacobian).  For the
   embedded models (few basis functions, wide radial basis, nR = 384) the
   ForwardDiff Jacobian itself dominates (68 %), mostly Dual-number evaluation
   of `SVector{384}`-valued splines and the copies.
2. **A hand-written forward-mode (pushforward) Jacobian through
   Rnl -> Ylm -> A -> AA -> A2B with `SVector{3}` tangents plus a type-stable
   accumulation loop reproduces the design matrix to 1e-12 absolute (8e-16
   relative) and is 65-150x faster per structure single-threaded
   (`efv_basis_v3`), 105-370x with 4 threads (`efv_basis_v3t`).**  With a
   reusable workspace (`efv_basis_v4`) the per-structure allocation drops
   from 420-1400 MB to 8-18 MB and GC vanishes, giving a sustained 0.075 s
   (`cat_D6`), 0.29 s (`cat_D8`), 0.53 s (`cat_D10`, 46 885 columns) per 32-atom
   structure.  This is 5-40x one `evaluate_ed` call, which is what the
   arithmetic warrants (n_B outputs, 3*nneigh directions).
3. Once the basis path is fast, **`ACEfit.assemble` overhead dominates**: the
   `GC.gc()` after every structure (`ACEfit/src/assemble.jl:44`) costs
   0.15-0.4 s per call (as much as the optimised feature matrix), the
   `pmap` closure captures `basis` so the model is serialised and deserialised
   per task (8-50 ms per task; the `sendto(workers(), basis=basis)` on line 36
   is dead - the closure does not use that global), and `Array(A)` on line 47
   makes a second full copy of the design matrix (for the 3200-structure
   degree-10 case that is 2 x 156 GB).
4. The post-assembly stages in `fit_distilled.jl` are cheap by comparison.
   `compute_errors` on the 200-structure test set costs 3-4 s per lambda
   (1 thread), i.e. 50-70 s for the 19-lambda sweep - under 2 % of the current
   pipeline.  The design-matrix residual (`residuals.jl`) reproduces
   `compute_errors` to 1e-15 relative and costs 2-25 ms per lambda, but it
   needs the test design matrix assembled once (cheap with the fast path).
   `serialize` runs at 0.5 GB/s (a 15 GB cache takes ~30 s), the in-place
   scaling is memory-bound (0.3 s for 0.6 GB), and `TikhonovFactor`'s `svd(R)`
   is O(n^3): it is already ~25 s at n = 4000 and will be hours at n = 46 635
   (section 2e).
5. Projection for the 3200-structure degree-10 categorical assembly on a
   40-core node with 12 workers: **~5-15 min** with items 2-3 applied, versus
   the ~13 h projected today (section 5).

## 2. Measurements

### 2a. Single process, single 32-atom structure (`01_single.log`, `03_opt.log`)

| | cat_D6 | emb16_D6 | cat_D8 | emb16_D8 |
|---|---|---|---|---|
| `PairList` | 0.4 ms | 0.3 ms | 0.3 ms | 0.4 ms |
| `energy_forces_virial` (fitted weights, 1 thread) | 4.2 ms, 6.5 MB | 6.4 ms | 6.6 ms | 17.5 ms |
| `energy_forces_virial_serial` | 4.1 ms | 6.2 ms | 12.9 ms | 10.3 ms |
| `evaluate_basis` (1 site) | 0.10 ms, 0.19 MB | 0.11 ms | 0.13 ms | 0.19 ms |
| `evaluate_ed` (1 site: energy + gradient) | 0.17-0.19 ms, 0.17 MB | 0.23 ms | 0.23-0.29 ms | 0.38-0.72 ms |
| `evaluate_basis_ed` (1 site, ForwardDiff) | 16-19 ms, 102-120 MB | 25 ms, 59 MB | 39-65 ms, 267 MB | 90-136 ms, 111 MB |
| ratio `evaluate_basis_ed` / `evaluate_ed` | ~100x | ~110x | ~170x | ~190x |
| **`energy_forces_virial_basis`** | **6.30 s, 9.6 GB, GC 23 %** | 2.21 s, 3.3 GB, GC 14 % | **16.4 s, 26.4 GB, GC 24 %** | 6.41 s, 7.0 GB, GC 13 % |
| ratio to `energy_forces_virial` | 1545x | 356x | 1273x | 621x |
| of which 32 x `evaluate_basis_ed` | 0.61 s (10 %) | 0.85 s (39 %) | 2.07 s (13 %) | 4.34 s (68 %) |
| of which accumulation loop `calculators.jl:311-318` | ~5.7 s (90 %) | ~1.4 s | ~14 s (87 %) | ~2.1 s |
| `ACEfit.feature_matrix(d, model)` | 5.73 s | 2.23 s | 17.0 s | 6.22 s |
| wrapper overhead over `energy_forces_virial_basis` (2c) | within noise (-9 %..+4 %) | +1 % | +4 % | -3 % |

Allocation count (Profile.Allocs, sample rate 1e-4, `cat_D6`): 23 712 samples,
i.e. ~2.4e8 allocations per structure; by type 43 % `SVector{3, Quantity}`,
42 % boxed `Int64` (the loop indices passed to the dynamic `getindex`), 14 %
`SVector{3, Float64}`; by site 49 % `calculators.jl:314`, 49 % `calculators.jl:315`.
The per-call allocation count scales as `length_basis x nneigh x natoms`
(boxed per element), not as a few big buffers.

### 2b. Profile of `energy_forces_virial_basis` (`02_profile.log`)

Flat profile, `cat_D6`, 3419 samples at 1 ms (excerpt; `Count` is inclusive,
`Overhead` is self):

```
 Count  Overhead File                                 Line Function
  3419        48 ACEpotentials/src/models/calculators.jl  288 energy_forces_virial_basis
  1312       922 ACEpotentials/src/models/calculators.jl  314 energy_forces_virial_basis   F[Js[a], k] -= dv[k, a] * force_unit
   996       705 ACEpotentials/src/models/calculators.jl  315 energy_forces_virial_basis   F[i, k]     += dv[k, a] * force_unit
   949         0 ACEpotentials/src/models/calculators.jl  309 energy_forces_virial_basis   evaluate_basis_ed
    95        49 ACEpotentials/src/models/calculators.jl  317 energy_forces_virial_basis   _site_virial(dv[k, :], Rs)
   598         0 ACEpotentials/src/models/ace.jl          684 evaluate_basis_ed             ForwardDiff.jacobian
   476         0 ForwardDiff/src/jacobian.jl              202 chunk_mode_jacobian
   286         0 ACEpotentials/src/models/ace.jl          690 evaluate_basis_ed             collect(permutedims(reshape(...)))
   232         0 Base/array.jl                            782 _collect_indices             (the copy above)
   597       562 Base/array.jl                            963 getindex                     dynamic dv[k, a]
   516       516 Base/boot.jl                             588 GenericMemory                (allocation)
```

Top-3 inclusive hotspots, `cat_D6`: (1) `calculators.jl:314-315` accumulation
loop, 67 % (all dynamic dispatch and boxing on `dv::Any`); (2)
`ace.jl:684` ForwardDiff Jacobian, 17 % - inside it `Rnl_splines.jl:44/60/77`
(spline evaluation in `Dual{12}` arithmetic) is 29 % of the Jacobian,
`extract_jacobian_chunk!` 20 %, and allocation (`GenericMemory`) 57 % of the
self time; (3) `ace.jl:689-690` the transpose/reshape/permutedims copies of the
(`length_basis` x 3 nneigh) Jacobian, 10 %.  Note that the Jacobian is taken
over the **full** `length_basis` vector (`evaluate_basis` returns
NZ x n_B entries of which 4/5 are structurally zero for the site's species),
so the ForwardDiff output, the copies and the accumulation loop are all 5x
larger than needed.

`emb16_D8` (3616 samples): `calculators.jl:309` (the Jacobian) 69 %, of which
`Rnl_splines.jl:77 evaluate_batched` 46 % (1660 samples: `SVector{384}` cubic
splines through `Interpolations` in `Dual{12}` arithmetic,
`Interpolations.jl:307/314`), ForwardDiff partials arithmetic ~25 %, copies
at `ace.jl:689-690` 3 %; the accumulation loop is 25 %.

Neighbour list, spherical harmonics (`P4ML.evaluate!`), the A basis, AA basis
and A2B products do not appear above the 1 % noise floor in either model.

### 2b'. Type-stability audit (`02b_typeaudit.log`)

| function | inferred return type | `::Any` lines |
|---|---|---|
| `evaluate_basis` | `Vector{Float64}` | 0 |
| **`evaluate_basis_ed`** (`ace.jl:663`) | **`Tuple{Vector{Float64}, Any}`** | 11: `dB_vec`, `dB1`, `dB` all `::Any` |
| `evaluate_ed` | `Tuple{Float64, Vector{SVector{3,Float64}}}` | 4: `dRnl`, `dYlm` `::Any` from `EquivariantTensors.pullback` (already handled by the `_assemble_grad_ed!` barrier) |
| `EquivariantTensors.evaluate(tensor, ...)` | `Tuple{Vector{Float64}}` | 0 |
| `get_neighbours` | `Tuple{Vector{Int}, Vector{SVector{3}}, Vector, Any}` | `Zs::Vector`, `z0::Any` (AtomsCalculatorsUtilities `get_id`) |
| **`energy_forces_virial_basis` kwarg body** | `NamedTuple{..., <:Tuple{Vector, Matrix, Vector}}` | **31**: `_e0::Any`, `z0::Any`, `dv::Any`, and every `getindex`/`*`/`-` of the inner loop |
| `ACEfit.feature_matrix` | `Matrix{Float64}` | 0 (it just copies the result) |

The instability is created at `ace.jl:684-690` and consumed at
`calculators.jl:309-318`; nothing in the AtomsData wrapper adds to it.
`evaluate_ed` (the force path used by `energy_forces_virial`) does not suffer
because of the `_assemble_grad_ed!` function barrier at `ace.jl:288`.

### 2c. `ACEfit.feature_matrix` versus `energy_forces_virial_basis`

Identical within noise (table 2a).  The wrapper's `ustrip.(_f_mat(...))`,
the virial loop over basis functions and the `Array{Float64}` allocation cost
0.03-0.6 s per structure, 1-4 % - negligible today, but after the fix they
would be ~50 % of the per-structure cost, hence `feature_matrix_from_efv` in
`basis_opt.jl` writes the unit-less result straight into the row layout.

### 2d. `ACEpotentials.assemble` on 32 structures (`04_assemble_p0.log`, `04_assemble_p4.log`, `04_assemble_p4_opt.log`, `04b_overhead_*.log`)

| path | 1 process | 4 workers | speed-up | worker-s / structure |
|---|---|---|---|---|
| original, `cat_D6` | 524 s (16.4 s/structure)* | 171 s (5.35 s/structure) | 3.1x | 21 |
| original, `emb16_D8` | 252 s (7.9 s/structure)* | 149 s (4.65 s/structure) | 1.7x | 19 |
| pushforward `v3`, `cat_D6` | 15.9 s (0.50 s/structure) | 14.4-14.8 s (0.45 s/structure) | 1.1x | 1.8 |
| pushforward `v3`, `emb16_D8` | 5.3 s (0.16 s/structure) | 10.9-12.2 s (0.34 s/structure) | 0.5x | 1.4 |
| pushforward + workspace `v4`, `cat_D6` | - | 5.0 s (0.157 s/structure) | - | 0.63 |
| pushforward + workspace `v4`, `emb16_D8` | - | 4.1 s (0.127 s/structure) | - | 0.51 |

\* the 1-process original run overlapped the heaviest external load (an
8-structure run earlier on a quieter machine gave 7.0 s/structure for
`cat_D6`, consistent with 2a).  The optimised paths give the same `A` as the
original to 9.7e-13 (`cat_D6`, max |A| = 842) and 4.4e-12 (`emb16_D8`, max |A| = 1390);
`Y` and `W` are bit-identical.

Where the remaining per-structure time goes once `feature_matrix` is fast
(`04b_overhead_p0.log`, serial, `cat_D6`, `v3` path):

| variant of the `pmap` body | wall (32) | `feature_matrix` sum | `GC.gc()` sum |
|---|---|---|---|
| as in ACEfit (`GC.gc()` per task, model captured) | 13.4-17.4 s | 4.8-5.9 s | 7.9-10.2 s (0.25-0.32 s per call) |
| no `GC.gc()` | 12.7 s | 11.7 s (the GC now happens inside) | 0 |
| no `GC.gc()`, model from a worker global | 12.8 s | 11.8 s | 0 |

So with `v3` the GC is intrinsic (0.4 GB allocated per structure) and the
explicit `GC.gc()` only moves it; `v4` (workspace reuse) removes the
allocations (8 MB per structure, GC 0 %, `03b_opt_v4.log`: 32 structures
back-to-back 5.97 s -> 2.39 s for `cat_D6`, 25.9 s -> 9.3 s for `cat_D8`), and
then the explicit full `GC.gc()` per task (0.15-0.4 s, growing with the heap
size of the worker) becomes the single largest item.

Distributed mechanics (`04_assemble_p4*.log`, `04b_overhead_p4.log`):

* The model **is** shipped per task: `pmap(packets) do p ... feature_matrix(p.data, basis)` captures the local `basis`, so every `remotecall` serialises it (2.4 MB for `cat_D6`, 7.9 MB `emb16_D8`, 8.8 MB `cat_D10`).  Measured round trip for a trivial closure capturing the model: 8-50 ms per call, versus 0.3-1.7 ms with the model as a worker global.  `sendto(workers(), basis = basis)` on `assemble.jl:36` defines that global but the closure never reads it.  Cost today: 3200 tasks x ~10-50 ms = 0.5-3 min of master time per assembly - small next to 13 h, ~10-25 % of the optimised assembly.
* `pmap` uses `batch_size = 1`; with 12 workers and 0.5-1 s tasks that is fine (per-task overhead is the serialisation above, not the scheduling).
* SharedArray writes and the progress meter are not measurable at this scale (the 219 MB `A` for 32 structures is written in well under a second).
* `Array(A)` at `assemble.jl:47` copies the whole design matrix out of the SharedArray: transient 2x memory.
* The 4-worker speed-up of the *original* path is only 1.7-3.1x on this (contended) machine; the original path is memory-allocation bound (9-26 GB allocated per structure), so workers compete for memory bandwidth and the GC - this is the same reason 12 workers on a 32-core node deliver ~65 worker-s per structure.

### 2e. Post-assembly stages (`05_post.log`)

Synthetic (A, Y, W), 20 000 x 4 000 (0.6 GiB; anything larger swapped on this
machine while the other jobs ran):

| stage | time | scaling |
|---|---|---|
| (i) `serialize((A, Y, W))` to disk | 1.35 s (0.47 GB/s); `deserialize` 0.80 s | linear: a 15 GB cache is ~30 s to write, a 156 GB one ~5 min, and the same again to read |
| (ii) in-place `A ./= P'; A .*= W` | 0.33 s | memory-bound, ~2 GB/s here: 156 GB is ~1-2 min |
| (iii) `TikhonovFactor` 10 000 x 4 000 | 28.7 s (qr 8.5 s at 37 GFLOP/s, the rest is `svd(R)` + `Q'y`) | |
| (iii) `TikhonovFactor` 20 000 x 4 000 | 36.6 s (qr 11.7 s at 55 GFLOP/s; `svd` ~20 s) | qr O(m n^2), svd O(n^3) |
| `tikhonov_solve` per lambda | 0.03-0.04 s | O(n^2) |

The `svd(R)` term is the one to watch: at n = 4000 it is already 2x the QR;
at n = 19 120 it is (19120/4000)^3 = 110x larger (~40 min at this machine's
contended rate, minutes on a 40-core node at a few hundred GFLOP/s), and at
n = 46 635 it is 1600x larger (~2e15 flops: 2-3 h even at 200 GFLOP/s).
For the QR, m = 3200 x 130 rows x 46 635 columns is 2 m n^2 = 1.8e15 flops -
the same order.  I could not measure these sizes here (memory).

`compute_errors` versus residual-from-design-matrix, 100 structures, 1 thread:

| model | `compute_errors` (100) | assemble 100 once (`v4`, serial, incl. ACEfit overhead) | residual per lambda | 19 lambdas x 200 test: `compute_errors` | same via design matrix |
|---|---|---|---|---|---|
| cat_D6   | 1.35 s | 38 s | 16 ms | 51 s | 77 s + 0.6 s |
| emb16_D6 | 1.40 s | 25 s | 2.5 ms | 53 s | 51 s + 0.1 s |
| cat_D8   | 1.92 s | 51 s | 23 ms | 73 s | 102 s + 0.9 s |
| emb16_D8 | 1.78 s | 29 s | 7 ms | 68 s | 58 s + 0.3 s |

Agreement on the 32 structures (`rmse_from_design` in `residuals.jl` versus
`compute_errors(...)["rmse"]["set"]`): |diff| <= 3.6e-15 absolute, <= 1.4e-15
relative for E, F and V, for all four models (the 1e-10 target is met by five
orders of magnitude).  Extrapolated to 19 lambdas x 1000 structures:
`compute_errors` 260-360 s single-threaded (and it is threaded over sites by
default, so less on the node).  Conclusion: **the per-lambda `compute_errors`
is ~1 min per sweep, i.e. under 2 % of the current fit; not the problem**.
The design-matrix route is still the right structure (one test assembly,
residuals for free, and the same `A_test` serves every model-selection
question), but it only pays once assembly is fast, and it adds the test
matrix to the cache footprint.

### 2f. Degree 10 categorical, one 32-atom structure (`06_deg10.log`)

`cat_D10`: 9327 basis functions per element, 46 885 columns, nAA = 25 843,
A2Bmap 9327 x 25 843 with 2.8 nonzeros per row; serialised model 8.8 MB.
`efv_basis_v4`: **0.53 s per 32-atom structure (16.5 ms per site, 42 MB
allocated)** = 40x one `energy_forces_virial_serial` (13 ms).  The original
`evaluate_basis_ed` costs 0.23 s and 668 MB **per site** (7 s per structure
before the accumulation loop, which at 46 885 x 79 x 32 = 1.2e8 dynamically
dispatched iterations adds another ~40-60 s): consistent with the ~65
worker-s per structure seen on the node at degree 8 and the 13 h projection
at degree 10.

## 3. Optimisations, ranked

Measured on this machine; "gain" is per structure of the assembly unless said
otherwise.

| # | change | where | gain | effort | status |
|---|---|---|---|---|---|
| 1 | **Type-stable accumulation**: function barrier taking `v::Vector, dv::Matrix{SVector{3}}` (as `_assemble_grad_ed!` already does for `evaluate_ed`), no `dv[k, :]` copies, units applied once at the end, loop only over the species block + pair block of the basis (`get_basis_inds`, `get_pairbasis_inds`) | `calculators.jl:307-318` | 6.2-6.6x (`cat_D6/D8`), 1.6-1.9x (embedded); allocation 9.6 GB -> 3.3 GB | 1 hour | demonstrated: `efv_basis_v1` in `basis_opt.jl`, `03_opt.log` |
| 2 | **Hand-written pushforward Jacobian** (`SVector{3}` tangents through `evaluate_ed_batched` for Rnl, `P4ML.evaluate_ed!` for Ylm, then A, AA, A2B) replacing `ForwardDiff.jacobian` over the full `length_basis` vector | `ace.jl:663-693` | with #1: **65-150x** single-threaded (`cat_D6` 4.6 s -> 0.041 s; `cat_D8` 13.5 s -> 0.21 s; `emb16_D8` 5.3 s -> 0.036 s); per site 1-4 ms versus 16-136 ms; exact to 6e-15 relative | 1 day incl. tests (the kernels are ~80 lines; `EquivariantTensors._jacobian_X` at `sparse_ace_basis.jl:236` and `sparseprodpool.jl:567`/`sparsesymmprod.jl:349` is the same algorithm but takes `promote_type(Float64, SVector{3}) = Any`, so it cannot be called with vector tangents as is - a 3-line fix upstream would let ACEpotentials use it directly, batched over all sites of a structure) | demonstrated: `basis_ed_pf`, `efv_basis_v3` |
| 3 | **Workspace reuse** (preallocate dRnl, dYlm, dA, dAA, dB once per model and max-neighbour count) | new | allocation 420 MB -> 8 MB (`cat_D6`), 1.4 GB -> 18 MB (`cat_D8`) per structure; sustained throughput 2.5-2.8x over #2 (5.97 s -> 2.39 s per 32 structures `cat_D6`; 25.9 s -> 9.3 s `cat_D8`); makes the per-task `GC.gc()` unnecessary | half a day (Bumper `@no_escape` would do the same with less code) | demonstrated: `efv_basis_v4`, `03b_opt_v4.log`, `04_assemble_p4_opt.log` |
| 4 | **Threads over sites inside a worker** (`Threads.@threads :static`, per-thread E/F/V accumulators summed at the end) | `calculators.jl` | 1.5-2.7x with 4 threads on this loaded machine (`cat_D6` 0.041 -> 0.022 s; `emb16_D6` 0.013 -> 0.005 s) - memory-bound at F (natoms x n_B SVectors per thread) so expect ~2x for the 2-3 spare cores per worker on the nodes; exact | 1 hour on top of #2 | demonstrated: `efv_basis_v3t`; thread x worker equality checked to 0 |
| 5 | **ACEfit.assemble**: drop the per-task `GC.gc()` (or make it `GC.gc(false)`), read the model from the worker global that `sendto` already creates instead of capturing it in the closure, write `feature_matrix` straight into the SharedArray view, avoid `Array(A)` (return the SharedArray or `sdata`) | `ACEfit/src/assemble.jl:36-47` | 0.15-0.4 s per structure (GC) + 10-50 ms per task (serialisation) - with #1-3 that is 50-80 % of the remaining per-structure time; `Array(A)` removal halves peak memory (156 GB vs 312 GB at degree 10) | 2 hours, in ACEfit | measured, not implemented (`04b_overhead.jl` shows the variants) |
| 6 | Jacobian over the species block only + fixed `Chunk` for ForwardDiff (if #2 is not taken) | `ace.jl:684` | **negative here**: `basis_ed_fd` in `basis_opt.jl` was 1.3-6x *slower* than the original Jacobian (Dual arithmetic through the `Interpolations` splines and the `JacobianConfig` dominate; the original's `@withalloc P4ML.evaluate!` is cheaper than my `P4ML.evaluate`).  Not worth pursuing; #2 is simpler and 30-100x better | - | demonstrated negative, `03_opt.log` |
| 7 | `fit_distilled.jl`: residuals from a test design matrix instead of `compute_errors` per lambda | script | 50-70 s per sweep -> <1 s, but +1 test assembly (~1 min with #1-3 on 200 structures; ~15 min today at degree 8) and +25 % cache; under 2 % of the current wall time | 1 hour | demonstrated: `residuals.jl` (`rmse_from_design`, 1e-15 agreement) |
| 8 | `fit_distilled.jl`: `TikhonovFactor` - the `svd(R)` is O(n^3) and at n = 46 635 is hours; alternatives: per-lambda `qr([R; lambda*I])` (O(n^3) each but a much smaller constant than SVD and only R, not A), or cap the sweep to ~6 lambdas at degree 10 | script | hours -> minutes at n = 46 635 (not measured at that size) | 2 hours | not demonstrated (could not run n > 4000 here) |
| 9 | Embedded models: `SVector{384}` cubic splines evaluated per neighbour (`Rnl_splines.jl:44`) are 46 % of the *original* Jacobian and still the largest single item inside the pushforward for `emb16_*` (nR = 384 for d = 16, versus 74-191 categorical): evaluate the d-independent radial part once and multiply by the embedding table | `Rnl_splines.jl`, `embeddings.jl` | ~2x on the embedded pushforward (projected) | 1 day | not demonstrated |

Demonstrated combined effect on `ACEpotentials.assemble` (32 structures, 4
workers, contended machine): `cat_D6` 171 s -> 5.0 s (34x), `emb16_D8`
149 s -> 4.1 s (36x); serial `cat_D6` 524 s -> 15.9 s with #1-2 only.  The
per-structure worker time is now 0.5-0.6 s of which ~0.1 s is the basis
Jacobian and the rest is #5.

## 4. What to change first

**Script level (`acejax/bench/distil/fit_distilled.jl`), today, no package changes:**

1. `include` a scratch `basis_opt.jl`-style override of
   `ACEfit.feature_matrix(::AtomsData, ::ACEPotential)` on every process
   (`@everywhere`), using `efv_basis_v4` (+ `Threads.@threads` if the workers
   are started with `-t 3`).  This is the entire 65-150x; the override is ~250
   lines, exact to 1e-12, and needs no change to the cache format or the
   solver.  It is type piracy, so keep it in the script until #2 lands in
   `src/`.
2. Assemble the **test** design matrix once, cache it next to the train one,
   and replace the per-lambda `compute_errors` with `rmse_from_design`
   (`residuals.jl`).  Small saving, but it also removes 19 model re-evaluations
   from the critical path and makes the sweep pure linear algebra on the
   cache.
3. Before the degree-10 run: replace `svd(R)` (n = 46 635) in
   `TikhonovFactor` with something with a small O(n^3) constant or a
   per-lambda solve on `R`, and remember `ACEfit.assemble`'s `Array(A)` doubles
   the 156 GB matrix - that run needs either the SharedArray returned directly
   (#5) or `BIGMEM`-style streaming.

**Package level (ACEpotentials / ACEfit), the right fix:**

1. `src/models/ace.jl`: replace `evaluate_basis_ed` by the pushforward
   (`basis_ed_pf`), returning `(B_block, dB_block::Matrix{SVector{3,T}})` for
   the species block only; export the block indices with it.  Test against
   `ForwardDiff` on random configurations (my check: 4-6e-15 relative on E, F, V).
2. `src/models/calculators.jl:307-318`: function-barrier accumulation over the
   block only, no units inside, optional `Threads.@threads`.
3. ACEfit `assemble`: no per-task `GC.gc()`, model from the worker global,
   `feature_matrix!` into the SharedArray view, return without `Array(A)`.
4. Upstream (EquivariantTensors): make `_jacobian_X` accept `SVector`
   tangents (`T∂A = typeof(zero(TA) * zero(eltype(∂Rnl)))` instead of
   `promote_type`), so ACEpotentials can use the batched, KA-ready kernel
   for all sites of a structure at once - that is also the GPU path.

## 5. Projection: 3200 structures, degree 10 categorical (46 885 columns), 40-core node, 12 workers

Per 32-atom structure, measured here: `efv_basis_v4` 0.53 s (single thread,
contended Mac).  Scale to 32-48 atoms (mean 40): ~0.66 s; a node core is not
faster than this Mac's per core, so keep it.  Add the `feature_matrix`
copy-out (~0.05 s for a 130 x 46 885 block) and `AtomsData` (negligible).

* Basis Jacobians: 3200 x 0.7 s = 2240 worker-s -> **190 s wall on 12 workers**;
  with `-t 3` per worker (#4, ~2x) ~100 s.
* ACEfit overhead as it stands (#5 not done): `GC.gc()` 0.3-0.5 s x 3200 / 12
  = 80-130 s; model serialisation 3200 x ~40 ms on the master = ~2 min
  (serial!); SharedArray fill 156 GB at ~5 GB/s = 30 s; `Array(A)` copy 156 GB
  = 30-60 s and 312 GB peak.  Total ~5-6 min.
* With #5 as well: ~3-4 min.

**Estimate: 5-15 minutes** (versus ~13 h projected today, i.e. 50-150x), with
the run then bounded by the factorisation (`qr` of 416k x 46 885 = 1.8e15 flops,
tens of minutes at node BLAS rates, and `svd(R)` of the same order - item #8)
and by memory (156 GB `A`, plus the QR's own copy unless `inplace = true`).  For
the degree-8 800-structure case (19 120 columns, ~1.2 h today): 800 x 0.35 s
/ 12 = **25 s** of assembly.

## 6. Not determined / caveats

* All wall times were taken with two foreign 4-6-core Julia jobs running and
  swap at 12-16 GB; the multi-worker numbers (2d) in particular are noisy
  (the original path's 1-process run came out 2.3x slower than the same
  per-structure cost measured in isolation).  Ratios between back-to-back
  variants are trustworthy; absolute worker-s per structure on a node should
  be re-measured.
* `TikhonovFactor` could only be measured up to n = 4000 (memory); the
  n = 19 120 / 46 635 figures are O(n^3) extrapolations.
* I did not measure the SharedArray fill or `Array(A)` at production size
  (156 GB); the bandwidth figures are estimates.
* The `ForwardDiff`-on-the-block variant (#6) was slower than expected; I did
  not dig into why (likely `Interpolations` in `Dual` arithmetic plus my
  allocating `P4ML.evaluate`), since #2 supersedes it.
* `EquivariantTensors._jacobian_X` was not exercised end-to-end (it rejects
  `SVector` tangents); the hand-written kernels in `basis_opt.jl` follow its
  structure and are exact against the original, but the batched/GPU variant
  remains to be tried.
* `energy_forces_virial_basis` passes the virial array `V` as the potential to
  `get_neighbours(at, V, nlist, i)` (`calculators.jl:308`); harmless (only used
  for `get_id` dispatch) but it is a latent bug and part of why `z0` infers as
  `Any`.
