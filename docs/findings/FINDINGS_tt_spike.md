# Tensor-train ACE coefficients fitted by alternating linear least squares

Spike for the Stage 2B paragraph *"Tensor trains, and why linear fits in this
space are underexploited"* in `docs/plans/jax_ace_port_plan.md`: compress the
categorical ACE coefficient tensor over the neighbour-species index into a
tensor train (TT / MPS), fit it by ALS sweeps in which every core update is a
linear ACE fit on the **cached categorical design matrix**, and compare with the
frozen-embedding CP model (`ace_embedding_model`) at matched parameters and at
matched per-site flops.

Code: `acejax/spike_tt/` (`ttmap.jl` the map from the TT coefficient space onto
the categorical basis, `tt.jl` the format, ALS and TT-SVD, `oracle_o2.jl`,
`run_tt.jl`, `inspect.jl`, `test_ttmap.jl`, `test_maps.jl`, `probe_o3.jl`; logs
`run_D4.log`, `run_D4_transfer.log`, `run_D5.log`, `run_D5_extra.log`,
`smoke_D4.log`, `oracle_o2_D{4,5,6}.log`, `oracle_o2_D5_solid.log`). Nothing under `src/` or `test/` was
touched. Local only, 12-core M-series Mac, `julia --project=acejax/julia -t 4`
(Julia 1.13), BLAS 4 threads. Same data, split, weights, prior and cached
assemblies as `FINDINGS_fs_spike.md` / `FINDINGS_varpro_spike.md`.

## The answer, in one paragraph

**Yes on the format, and yes on the fitting route, with a clear caveat on
flops.** A TT over the neighbour-species index with tiny bond dimensions, fitted
by ALS on the cached categorical design matrix, beats the frozen-embedding CP
model at matched parameters by a wide margin and beats the *categorical* model
with a fraction of its parameters: at degree 4, TT(1,4,4,4) with **780**
parameters reaches test force RMSE **0.1197** eV/A (categorical 0.1201 with
1 950, CP-16 0.1229 with 1 365, CP-35 "lossless" 0.1214 with 2 125) and
TT(1,5,8,8) with 1 595 reaches **0.1186**; at degree 5, TT(1,4,4,4) with
**1 025** parameters reaches **0.1093** (categorical 0.1119 with 3 795, CP-16
0.1118 with 2 195, CP-35 0.1101 with 3 525) and TT(1,5,8,8) with 2 045 reaches
**0.1079**, i.e. 3.6% below the categorical and 2% below the best CP at 40-60%
of their parameters, with energies and virials 20-35% better as well. The
learned cores transfer: frozen and re-read-out on a disjoint split they keep
the whole gain (TT(1,5,15,16): 0.1185 vs categorical 0.1197 and CP-16 0.1229
on that split, and as good as cores learned on the new split itself). The
"every core update is a linear ACE fit" structure works exactly as advertised:
every ALS step is a small generalised-Tikhonov least-squares solve on
`A_cat * (Phi M_t)`, the objective decreased monotonically at every one of the
2 144 steps logged across all runs (zero violations at 1e-10 relative) once the block solve was
made exact (Julia's pivoted `\` is not a minimiser on these rank-deficient
systems and produced a 1e8 jump in the first attempt), a core step costs 0.2-3 s
against 5 s for the categorical solve at degree 4 (0.5-5 s vs 23 s at degree 5)
and a sweep 3-20 s, and 6-24 sweeps from the CP initialisation are enough.
Both oracles are bit-level: the full-rank TT reproduces the categorical
objective and fit to 3e-13 (O1), and the diagonal-core TT reproduces every
column of `ace_embedding_model(d_max = 16)` to 2e-15 at degrees 4, 5 and 6 (O2,
with unit multiplicity) -- *once the embedded model is built with spherical
rather than its default solid harmonics*, which together with a broken pair
basis in `ace_embedding_model` is a real finding about the branch. **The
caveat is flops**: the TT's per-site cost `S * r_{t-1} * r_t` per one-particle
function per slot is 2-5x CP-16's and 2-3x the categorical's at the ranks
that win on accuracy, so at *matched flops* at S = 5 the categorical basis is
still the cheapest way to that accuracy (TT(1,2,2,2), the only schedule inside
the categorical budget, is 3-4% worse than categorical); TT buys accuracy per
*parameter* and per *training structure* (it regularises: the wide schedules
that can reach the categorical fit do so exactly, train error included, and
gain nothing on test), not per flop. For Stage 2B that argues for the TT
format where parameters / data are the constraint (many species, few
structures) and for the linear-ALS route in either format.
## 1. What was built: the model, precisely

### 1.1 The categorical basis, unfolded

`ace1_model(elements = [Cr,Mn,Fe,Co,Ni], order = 3, totaldegree = D)` folds the
neighbour species into the radial index, `n = (n'-1)·S + z` (`Rnl_learnable.jl:83`,
`S = 5`), so a categorical many-body basis function is

    B_{b,q}(z0) = Σ_M  A2B[(b,q), (b,M)] · Π_t A_{n_t l_t m_t},      A_{nlm} = Σ_j R_{n'l}(r_j) δ_{z, z_j} Y_lm(r̂_j)

where the **block** `b` is a sorted tuple of folded `(n, l)` pairs (ET's
`mb_spec`), `q = 1..num_b` indexes EquivariantTensors' permutation-symmetrised
invariant couplings of that block (`symmetrisation_matrix` -> `A2Bmaps[1]`, rows =
B functions, columns = the sorted `(n,l,m)` products `AA`), and the readout is
per central species `z0` (`get_basis_inds(model, z0)`), plus the species-resolved
pair basis (`npair` per `z0`). Unfolding `n -> (n', z)` gives, per categorical
column, a sorted tuple of triples `(n'_t, z_t, l_t)` and `q`.

Two facts about this basis that matter below (both from `inspect.jl`):

* The ACE1 degree convention `TotalDegree(1.0*S, 1/wL)` is evaluated on the
  *folded* `n`, so the level of a one-particle function is `n' - 1 + z/S + 1.5 l`
  -- it depends on the neighbour species. Consequently the categorical spec is
  **not** "radial/angular tuple x all species assignments": at degree 4, of the
  19 distinct `(n', l)`-tuples `beta` per `z0`, only 10 satisfy the channel-free
  degree `sum(n' + 1.5 l) <= D` that `ace_embedding_model` uses, and the other 9
  ("tail" blocks, present only for some species assignments) hold **160 of the
  370** categorical columns per `z0` (D5: 14 of 30 beta, 304 of 734 columns; D6:
  20 of 45, 513 of 1348). The embedded model at "the same degree" is therefore a
  much smaller model (112 vs 370 columns per `z0` at D4), which is part of why
  it fits worse; see section 3.
* All blocks up to degree 6 have a **single** coupling (`num_b = 1`), because
  the angular content is `l <= 1`; the multiplicity question is therefore about
  the species permutations only.

### 1.2 The TT coefficient space and the map onto the categorical basis (`ttmap.jl`)

Index set (per `z0`): `(beta, eta, zeta)` with `beta` a **sorted** tuple of
`(n', l)` pairs (the coupling block -- radial and angular -- in canonical slot
order), `eta` an invariant coupling of the *ordered* `l`-tuple `L` of `beta` in
that slot order (`O3.coupling_coeffs(0, L, 1:nu; PI = false, basis = real)`,
rows orthonormalised), and `zeta in [S]^nu` an **ordered** species tuple: slot
`t` carries `(n'_t, l_t, zeta_t)`. The site function of one index is

    f_{beta,eta,zeta} = sum_M C^eta_{L,M} * prod_t A_{(n'_t, zeta_t), l_t, m_t}

which is rotation invariant and lies in the span of the categorical B-functions
of the block `b(zeta) = sort((fold(n'_t, zeta_t), l_t)_t)`. Its expansion is
obtained **per block by solving `S_b' x = d`** with `S_b` the block of `A2B` and
`d` the AA-coefficients of `f` (products are accumulated onto the *sorted*
`(n,l,m)` tuple, which is how distinct slot orderings of equal triples merge).
This gives a sparse matrix `Phi` (`nB x N_c` per `z0`; the global map is
block-diagonal over `z0`) with the categorical coefficient vector `x = Phi c`.
Nothing about multiplicities or permutation conventions is assumed; the
least-squares residual of every block solve is the check: max 3.3e-16 at
D4-D6, `Phi` has full row rank (every categorical column is reachable), and it
has exactly one non-zero per column (with `num_b = 1` every `f` is one
categorical column times a scalar). `N_c = 620 / 1160 / 1985` per `z0` at
D4 / D5 / D6 against `nB = 370 / 734 / 1348`: the surplus are the distinct
orderings `zeta != zeta'` of one species multiset for blocks with repeated
`(n', l)` pairs, which map to the same column (their difference is the
antisymmetric part of the coupling, identically zero for these `l`).

Two ET conventions had to be handled: `get_nnll_spec` returns some block tuples
unsorted (`[(2,0),(1,1),(1,1)]`), and the `AAspec` tuples are in the block's
slot order rather than sorted -- both are keyed by the sorted tuple.

### 1.3 The tensor train (`tt.jl`)

Per central species `z0` and per body order `nu` (separate trains per order --
the simplest choice, and the one that matches CP's per-order widths `d_nu`):

    c^{(z0)}[(beta, eta), zeta] = G_1[zeta_1] * G_2[zeta_2] ... G_nu[zeta_nu] * v^{(z0)}_{beta,eta}

    G_t[z] in R^{r_{t-1} x r_t},  r_0 = 1 (the left boundary u is absorbed into G_1),
    v^{(z0)}_{beta,eta} in R^{r_nu}   (the right boundary = the lambda readout, one per coupling block)

The cores are **shared across `z0`** (as the frozen embedding `E[z,k]` is) and
across all `(n', l)` channels of a slot; only the readout is per `z0` and per
block. The TT therefore compresses the *species* tensor of each coupling
block, exactly what the frozen embedding does; the radial/angular structure
stays in the readout. (The spec's literal `G_t^l[n]` with `n` the folded
radial x species index and `v` per `l`-tuple cannot satisfy its own O2 -- with
`r = 16` diagonal cores there is no room to carry the radial index, and the
embedded model's readout is per radial tuple -- so the readout was put per
`(n', l)`-tuple, which is what makes O2 exact.) Bond dimension `r_t <= S^t` is
the most a slot can carry (`G_1[z]` has only `S` distinct rows), so schedules
are written `(1, r_1, r_2, r_3)` with `r_1 <= 5`, `r_2 <= 25`, `r_3 <= 125`;
the order-2 train uses `(1, r_1, r_2)` and the order-1 train `(1, r_1)`.

**Symmetrisation rule, as verified.** The categorical column for a sorted
tuple absorbs every slot ordering of its triples with coefficient +1 each (no
combinatorial factor): every non-zero of `Phi` is exactly `1.0` at D4-D6, and the O2
scale factors came out as `1.0000000000000018`. For a block whose `(n', l)`
pairs are all distinct there is exactly one ordering `zeta` (the slots are
pinned by the sorted `(n', l)`), so **for those blocks the species tensor is a
general `S^nu` tensor, not a symmetric one** -- the categorical basis
distinguishes "Cr on channel (1,0), Fe on channel (2,0)" from the reverse. Only
blocks with repeated `(n', l)` pairs are symmetric in the corresponding slots.
This is why the "lossless" CP width `d_nu = dim Sym^nu(R^S)` (5, 15, 35) is
lossless only for the symmetric part: CP-35 (section 3) does not reach the
categorical fit, and a TT with `r_2 = 25` does.

**CP is a TT.** The frozen-embedding model with `K` channels is
`G_1[z] = E[z, :]` (1 x K), `G_t[z] = diag(E[z, :])`; its TT-SVD ranks are
`(1, min(K,5), min(K,15), K)` (the `(zeta_1 zeta_2 | zeta_3 k)` unfolding of
`sum_k w_k (x) w_k (x) w_k (x) e_k` has rank <= dim Sym^2 = 15). So CP-16 **is**
TT(1,5,15,16) -- the "(1, 5, 15, 16)" schedule of the spec is precisely the CP-16
tensor's own ranks, with zero truncation error -- and CP-35 is TT(1,5,15,35).
Every TT run below is initialised from the CP fit by TT-SVD of the CP species
tensor (`tt_svd`, `init_from_tensor!`; exact when the schedule contains the CP
ranks, a logged truncation otherwise; for schedules wider than CP the padded
directions get 1e-2 relative noise so they are not dead), so the first sweep
can only improve on CP.

### 1.4 The fit (`tt.jl`, `run_tt.jl`)

Objective = the categorical one, pulled back:

    J(theta) = || W (A_MB Phi c(theta) + A_pair p - y) ||^2  +  lambda^2 ( || P_MB Phi c(theta) ||^2 + || P_pair p ||^2 )

with `A` the cached categorical assembly (`asm_cantor_D{4,5}_{train200,test100}.jls`),
`W` the 30/1/1 weights with `1/sqrt(natoms)` on energy and virial rows, `P` the
production `algebraic_smoothness_prior(p = 4)` and `lambda = 1e-7`, the best
value of the categorical fit at degree 4 and 5 in `FINDINGS_fs_spike.md` (the
CP-16 readout fit was re-swept: 1e-6 ... 1e-8 give 0.1228-0.1229 at D4, so 1e-7
is right for it too). The prior is the categorical one so that the full-rank TT
objective *is* the categorical objective (O1). `A_MB Phi` is formed once
(25 544 x 3 100 at D4, 1 s) and reused by every step; the pair coefficients `p`
(100 columns) are solved jointly in every step. Test errors are from the
unweighted test assembly mapped the same way, E and V per atom, as before.

**ALS.** With all cores but `G_t` frozen (for every order nu >= t at once), `c`
is linear in `vec(G_t)`: `c = M_t g` with
`M_t[(z0,beta,eta,zeta), (a,b,zeta_t)] = L(zeta_{<t})_a * R_{beta,eta}(zeta_{>t})_b`,
`L` the left environment `G_1[zeta_1]...G_{t-1}[zeta_{t-1}]` and `R` the right
environment `G_{t+1}[zeta_{t+1}]...G_nu[zeta_nu] v_{beta,eta}`; the readout step
is `c = M_v w`. The coefficients a step does not touch (orders nu < t) are
constants and go to the right-hand side of both the data rows and the
regulariser rows. Each step is the generalised-Tikhonov least-squares problem
`min ||[A_MB Phi M, A_pair] x - y'||^2 + lambda^2 ||[P Phi M, P_pair] x - r'||^2`,
solved **exactly** (thin QR of the augmented matrix, then an SVD of the
triangular factor with a 1e-12 relative cutoff; exactly-zero columns -- dead
bond directions -- are dropped). Julia's `\` (pivoted QR with rank truncation)
is *not* a minimiser on these rank-deficient systems and produced a 1e8 jump
of the objective in the first O1 attempt; with the exact solve every step is
monotone by construction, and the assertion (`NON-MONOTONE` in the logs) never
fired in any run. Sweep = `G_1 -> G_2 -> G_3 -> v -> G_3 -> G_2 -> G_1 -> v`,
objective logged after every step; stop after `NSWEEPS` sweeps (8 at D4, 6 at
D5) or when a sweep improves `J` by less than 1e-4 relative.

**Gauge.** After each core step of the right half-sweep the core is
left-orthogonalised (thin QR of the `(r_{t-1} S) x r_t` unfolding, `R` pushed
into the next core), which leaves `c` and `J` unchanged and keeps the left
environments orthonormal; the left half-sweep is not re-gauged (the readout
absorbs it). The remaining ill-conditioning is handled by the Tikhonov term
and the exact solve.

## 2. Oracles

**O1 -- full rank contains the categorical model** (`run_tt.jl` stage `o1`,
`smoke_D4.log`). Schedule `(1,5,25,125)` with unit cores (`G_1[z] = e_z'`,
`G_2[z]: a -> (a, z)`, `G_3[z]: b -> (b, z)`) is the identity embedding: `c` is
the free species tensor, held in `v`. The categorical solution `x` at
lambda = 1e-7 was mapped to `c` (least-norm `Phi c = x`, `|Phi c - x| = 1.6e-16`)
and placed in `v`:

    J_TT(theta_cat) = 3.092025835876e+02   J_cat = 3.092025835876e+02   rel. diff 3.3e-14
    one readout step (5 975 columns, 20 s):  J = 3.092025835876e+02   rel. diff 5.5e-16, test F 0.120050 = 0.120050
    one G1 step, then one G2 step:           rel. diff 1.3e-15, 3.3e-13

i.e. the TT parameter space at full rank contains the categorical model, its
objective is the categorical objective to roundoff, and the ALS steps leave the
optimum where it is.

**O2 -- diagonal cores reproduce `ace_embedding_model(d_max = 16)`**
(`oracle_o2.jl`, function level, `oracle_o2_D{4,5,6}.log`). With
`E = embedding_rows(emb, Z; d = 16)` (the rows the model itself uses), the CP
columns `x_{beta,k} = sum_zeta Phi[:, (beta,eta,zeta)] prod_t E[zeta_t, k]`
(per-order widths 5/15/16, embedded-spec `beta` only) were evaluated as
categorical site functions on 300 random environments per central species
(8-40 neighbours, random species, `evaluate_basis` -- no derivatives, so this
works at degree 4 where the embedded model has `l_max = 0` and its *assembly*
crashes in SpheriCart), and every embedded column was regressed on the
CP-in-TT columns of the same `(beta, k)`:

| D | embedded columns per z0 | max relative residual (all z0, all columns) | scale factor |
|---|---|---|---|
| 4 | 112 | 1.9e-15 | 1.0000000000000018 |
| 5 | 194 | 1.8e-15 | 0.9999999999999966 |
| 6 | 323 | 2.3e-15 | 0.9999999999999977 |

and the reverse regression (CP-in-TT onto the embedded span) is 1e-15 as well.
So the TT space with diagonal cores *is* the embedded model, column for column,
with unit scale -- the permutation-multiplicity convention is verified with no
factor. Since the columns are identical, the fits are identical for any common
regulariser; the CP rows of section 3 are therefore the `ace_embedding_model`
fits computed inside the harness (with the categorical pair basis, spherical
harmonics and the categorical prior -- see the caveat below).

**Two discrepancies in `ace_embedding_model` exposed by O2** (this branch,
`src/models/embeddings.jl`), both of which make it *not* a reparameterisation
of `ace1_model` as built:

1. It defaults to `Ytype = :solid` while `ace1_model` uses `:spherical`. For
   `l >= 1` the basis functions differ by `r^l`; O2 at degree 5 fails with a 21%
   residual on the `l = 1` blocks until the embedded model is built with
   `Ytype = :spherical` (then 1.8e-15). The embedded-vs-categorical comparisons
   in the distillation work used the default.
2. Its pair basis is not species-resolved and is truncated: the spec is
   `n = 1..pmaxn` with `Winit = :onehot`, which `set_onehot_weights!` reads as
   `(n', z') = (1, n)`, so at S = 5 the embedded model's pair potential has
   **4 functions per `z0`: `P_1(r)` for neighbours Cr, Mn, Fe, Co and nothing for
   Ni**, against `ace1_model`'s `pmaxn * S = 20`. (At degree 4 the embedded model
   also has `l_max = 0`, and `ACEpotentials.assemble` on it crashes in
   `SpheriCart.solid_harmonics_with_grad!`; and on this branch `basis_ed.jl`
   needs `EquivariantTensors.pushforward_rows!` (ET >= 0.5.2) which neither
   `--project=.` nor `--project=acejax/julia` has (both pin ET 0.4.3), so no
   design matrix can be assembled in this checkout at all -- the spike is
   possible only because the categorical assemblies are cached.)

Neither was fixed here (nothing under `src/`); the CP comparator in section 3
uses the categorical pair columns and spherical harmonics, i.e. what the
embedded model *should* be.
## 3. Results

Data: `cantor1k_b_mh1.xyz`, split 0 = `MersenneTwister(0)` 200 train / 100 test
(12 796 test rows), `lambda = 1e-7`, categorical prior and weights as in the
earlier spikes. Units: F eV/A, E and V eV/atom. "params" counts everything that
is fitted: TT cores (shared across `z0`), the readout `v` (per `z0`, per block,
`r_nu` entries; for CP this is the `lambda` readout of `ace_embedding_model`
with per-order widths `min(K, 5/15/35)`), and the 100 (D4) / 125 (D5)
categorical pair coefficients; the frozen embedding of CP is not a parameter.
"flops" is the per-site evaluation cost model of section 3.3. TT rows are
initialised from the CP row with the same `K` (TT-SVD, exact) and ALS'd for the
stated number of sweeps; "init" is their test F after the readout refit at the
initialisation, i.e. the CP number. All TT runs: zero non-monotone steps.

### 3.1 Degree 4 (`run_D4.log`, `run_D4_transfer.log`)

| model | params | flops/site | test F | test E | test V | train F | sweeps (s/sweep) | init F |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| categorical `ace1_model` | 1 950 | 766 | 0.1201 | 0.00622 | 0.0414 | 0.1094 | | |
| CP-4 | 480 | 376 | 0.1252 | 0.00432 | 0.0322 | 0.1266 | | |
| CP-8 | 800 | 740 | 0.1292 | 0.00613 | 0.0411 | 0.1283 | | |
| CP-16 (= `ace_embedding_model(d_max=16)` + tail blocks) | 1 365 | 1 453 | 0.1229 | 0.00531 | 0.0390 | 0.1190 | | |
| CP-16, embedded-spec blocks only (= the shipped model's basis) | 660 | 1 453 | 0.1273 | 0.00509 | 0.0388 | 0.1266 | | |
| CP-35 ("lossless" `d_nu` = 5/15/35) | 2 125 | 3 030 | 0.1214 | 0.00528 | 0.0425 | 0.1162 | | |
| TT(1,2,2,2) | 380 | 806 | 0.1230 | 0.00425 | 0.0318 | 0.1239 | 24 (1) | 0.1277 |
| TT(1,4,4,4) | 780 | 2 684 | 0.1202 / **0.1197** | 0.00495 | 0.0298 | 0.1178 | 8 (9) / 24 (3) | 0.1252 |
| TT(1,5,8,8) | 1 595 | 7 208 | **0.1186** / 0.1188 | 0.00510 | 0.0352 | 0.1141 | 8 (11) / 16 (6) | 0.1292 |
| TT(1,5,15,16) (= CP-16's ranks) | 3 390 | 18 638 | 0.1203 | 0.00631 | 0.0421 | 0.1104 | 8 (19) | 0.1229 |
| TT(1,5,25,16) | 4 500 | 28 445 | 0.1201 | 0.00616 | 0.0418 | 0.1098 | 8 (57) | 0.1222 |
| TT(1,5,25,35) | 8 175 | 53 055 | 0.1201 | 0.00622 | 0.0414 | 0.1094 | 7 (172) | 0.1191 |
| TT(1,5,15,16), random init, seed 1 / 2 | 3 390 | 18 638 | 0.1210 / 0.1207 | 0.00619 / 0.00621 | 0.0416 / 0.0410 | 0.1110 / 0.1113 | 8 (33) | 0.1260 / 0.1300 |
| TT(1,5,8,8), random init | 1 595 | 7 208 | 0.1209 | 0.00545 | 0.0351 | 0.1154 | 8 (13) | 0.1676 |

(Two entries: 8 sweeps in the main run / longer run in `run_D4_transfer.log`;
the per-sweep wall times were measured with a second Julia process running.)

### 3.2 Degree 5 (`run_D5.log`, `run_D5_extra.log`)

| model | params | flops/site | test F | test E | test V | train F | sweeps (s/sweep) | init F |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| categorical | 3 795 | 1 658 | 0.1119 | 0.00674 | 0.0358 | 0.0911 | | |
| CP-4 | 725 | 632 | 0.1172 | 0.00513 | 0.0321 | 0.1161 | | |
| CP-8 | 1 250 | 1 249 | 0.1170 | 0.00554 | 0.0348 | 0.1127 | | |
| CP-16 | 2 195 | 2 462 | 0.1118 | 0.00533 | 0.0324 | 0.1041 | | |
| CP-16, embedded-spec blocks only | 1 095 | 2 462 | 0.1169 | 0.00487 | 0.0346 | 0.1129 | | |
| CP-35 | 3 525 | 5 160 | 0.1101 | 0.00560 | 0.0330 | 0.0994 | | |
| TT(1,2,2,2) | 515 | 1 310 | 0.1167 | 0.00449 | 0.0318 | 0.1169 | 6 (3) | 0.1216 |
| TT(1,3,3,3) | 755 | 2 631 | 0.1108 | 0.00384 | 0.0287 | 0.1079 | 24 (4) | 0.1192 |
| TT(1,4,4,4) | 1 025 | 4 396 | 0.1110 / **0.1093** | 0.00430 | 0.0279 | 0.1050 | 6 (14) / 24 (8) | 0.1172 |
| TT(1,5,8,8) | 2 045 | 11 954 | 0.1084 / **0.1079** | 0.00514 | 0.0305 | 0.0992 | 6 (37) / 24 (15) | 0.1170 |
| TT(1,5,8,8), cores per `z0` (not shared) | 5 225 | 11 954 | 0.1100 | 0.00607 | 0.0342 | 0.0935 | 12 (44) | 0.1170 |
| TT(1,5,15,16) | 4 220 | 31 334 | 0.1099 | 0.00596 | 0.0351 | 0.0943 | 6 (70) | 0.1118 |
| TT(1,5,25,16) | 5 350 | 48 025 | 0.1104 | 0.00629 | 0.0361 | 0.0934 | 6 (87) | 0.1109 |
| TT(1,5,25,35) | 9 775 | 90 195 | 0.1118 | 0.00674 | 0.0361 | 0.0913 | 6 (208) | 0.1089 |

### Reading the tables

1. **At matched parameters TT beats CP everywhere, and by a lot.** D4: TT(1,4,4,4)
   0.1197 with 780 vs CP-8 0.1292 with 800 (-7%) and CP-16 0.1229 with 1 365;
   TT(1,5,8,8) 0.1186 with 1 595 vs CP-16 0.1229 (-3.5%) and CP-35 0.1214 with
   2 125. D5: TT(1,4,4,4) 0.1093 with 1 025 vs CP-8 0.1170 with 1 250 (-6.6%);
   TT(1,5,8,8) 0.1079 with 2 045 vs CP-16 0.1118 with 2 195 (-3.5%) and CP-35
   0.1101 with 3 525. The learned 4- or 8-dimensional species subspace per
   order is worth more than the frozen 16- or 35-dimensional one. The
   qualitative difference: CP-K spans a K-dimensional subspace of the
   *symmetric* tensors fixed by the MACE table; the TT with slot-specific cores
   spans a learned `r_nu`-dimensional subspace of the *full* `S^nu` tensors,
   which is what the categorical basis actually uses for blocks with distinct
   `(n', l)` pairs (section 1.3).
2. **TT beats the categorical model with 20-40% of its parameters, and the
   small schedules beat it outright.** D4: 0.1197 (780 params) vs 0.1201
   (1 950); D5: 0.1093 (1 025) and 0.1079 (2 045) vs 0.1119 (3 795), with 25-35%
   lower energy and virial errors. The wide schedules that *can* reach the
   categorical fit do so -- TT(1,5,25,35) at D4 reproduces the categorical
   test *and* train errors to 4 digits (its objective 309.24 vs the categorical
   309.20; it is a full-rank order-2 train plus a 35-dimensional order-3
   subspace), and at D5 it goes from an initial 0.1089 *up* to the categorical
   0.1118 while its train F falls to 0.0913 = categorical -- and gain nothing
   on test. ALS minimises the training objective; the held-out benefit of the
   small TT is regularisation by rank, which this 200-structure training set
   rewards. That also reads as the answer to "matched parameters": the
   comparison is really TT-rank against the categorical's data hunger, and
   the useful ranks are far below CP's `d_max = 16`.
3. **Sharing the cores across `z0` is right.** Per-`z0` cores (5x the core
   parameters) fit the training set harder (train F 0.0935 vs 0.0992) and test
   worse (0.1100 vs 0.1079) at D5.
4. **The tail blocks matter and the embedded model does not have them.** The
   frozen-embedding readout on the categorical block set (CP-16, 0.1229 at D4)
   is 3.5% better than the same readout restricted to the channel-free-degree
   blocks that `ace_embedding_model` enumerates (0.1273; that model's own
   basis, up to its pair-basis and harmonics bugs). At D5 the gap is 4.4%
   (0.1118 vs 0.1169). A "degree-D embedded model" is not the degree-D
   categorical model with a compressed species index; it is a smaller model
   whose readout is a subset -- which is part of why the distillation runs
   found the embedded models behind.
5. **Init sensitivity.** From the CP initialisation the first sweep does most
   of the work (D4 TT(1,5,8,8): J 455.7 -> 355.0 in sweep 1, then 348.9, 346.3,
   344.7, 343.6, 342.8, 342.2, 341.6; test F 0.1292 -> 0.1186 after 8 sweeps).
   Random cores (seeded) reach 0.1207-0.1210 for (1,5,15,16) against 0.1203
   from CP-16, and 0.1209 for (1,5,8,8) against 0.1186 from CP-8, after the
   same 8 sweeps, still descending: the landscape has many nearby optima and
   the CP-derived start is the better one, as with the VarPro densities. The
   random-init (1,5,8,8) landed at a visibly different point (V 0.0351 vs
   0.0352 but E 0.00545 vs 0.00510), so the optima are not unique.
6. **Convergence.** Every step decreases `J`; per-sweep relative change decays
   from 1e-1 to 1e-3 over 8 sweeps for the small schedules (they were still
   improving at 8; 24 sweeps gave another 0.5-1.5% in test F at D5) and to
   1e-4 by sweep 6-7 for the wide ones (which are close to the categorical
   optimum, a convex problem). The readout step is the largest single gain in
   every sweep; `G_1` (the `1 x r_1` slot, at most 5 x 5 = 25 numbers) barely
   moves after sweep 1.

### 3.3 Matched flops

Per-site cost model (multiply-adds), all on the same single-channel product
DAG (the sorted `(n', l, m)` prefixes of all admissible products: `n_A1 = 7` /
`11` one-particle functions and `[4], [5, 11], [2, 6, 12]` / `[5], [6, 19],
[5, 12, 26]` DAG nodes per order and depth at D4 / D5):

* CP-K: species embedding of every one-particle function once, `n_A1 * S * K`;
  one multiply per DAG node per channel, `K * n_nodes`; readout `min(K, d_nu)`
  per block.
* TT: embedding *per slot and per order* (the cores differ), `n_A1 * S * r_{t-1} * r_t`;
  a `(1 x r_{t-1}) * (r_{t-1} x r_t)` product per DAG node at depth `t`; readout `r_nu` per block.
* categorical: ET's actual product count, `|AAspec| = 396 / 924` products plus
  the `370 / 734` readout (the species-resolved DAG evaluates one multiply per
  `AA` element).

| D | budget | categorical | best CP at the budget | best TT at the budget |
|---|---:|---|---|---|
| 4 | ~800 | **0.1201** (766) | CP-8 0.1292 (740) | TT(1,2,2,2) 0.1230 (806) |
| 4 | ~1 500 | | CP-16 0.1229 (1 453) | TT(1,2,2,2) 0.1230 (806); TT(1,4,4,4) needs 2 684 for 0.1197 |
| 4 | ~3 000 | | CP-35 0.1214 (3 030) | TT(1,4,4,4) **0.1197** (2 684) |
| 5 | ~1 300-1 700 | **0.1119** (1 658) | CP-8 0.1170 (1 249) | TT(1,2,2,2) 0.1167 (1 310) |
| 5 | ~2 500 | | CP-16 0.1118 (2 462) | TT(1,3,3,3) **0.1108** (2 631) |
| 5 | ~5 000 | | CP-35 0.1101 (5 160) | TT(1,4,4,4) **0.1093** (4 396) |

So at matched flops TT is level with or slightly ahead of CP from ~2 500
flops/site up, behind it at the smallest budgets, and **the categorical basis
is the cheapest of the three at S = 5** -- consistent with
`FINDINGS_embedding_spike.md`'s crossover at S ~ 7 for CP. The TT's flop
overhead is specific and structural: the slot-specific species embedding
`S * r_{t-1} * r_t` per one-particle function per slot (3 300 of TT(1,4,4,4)'s
4 396 at D5) and the `r_{t-1} * r_t` matvec per DAG node instead of `K`
multiplies. Both shrink relative to the rest as `S` grows (the categorical
count is `S^t` per node) and the first one would halve with cores shared
across orders (prefix sharing), which was not tried.

### 3.4 Transfer (`run_D4_transfer.log`, degree 4, split 1)

Split 1 = `MersenneTwister(1)` 200 / 100 from the 700 structures unused by
split 0 (`asm_cantor_D4_split1_*`, from the VarPro spike). Cores learned on
split 0 frozen, only the readout `v` (and the pair coefficients) refitted on
split-1 train -- one convex step, 4 s -- and tested on split-1 test:

| model | split-1 test F | E | V |
|---|---:|---:|---:|
| categorical, fitted on split 1 | 0.1197 | 0.00648 | 0.0407 |
| CP-8 / CP-16, readout on split 1 | 0.1320 / 0.1229 | 0.00655 / 0.00574 | 0.0428 / 0.0385 |
| TT(1,2,2,2), split-0 cores, readout on split 1 | 0.1249 | 0.00439 | 0.0341 |
| TT(1,4,4,4), split-0 cores | 0.1211 | 0.00479 | 0.0337 |
| TT(1,5,8,8), split-0 cores | 0.1193 | 0.00492 | 0.0332 |
| TT(1,5,15,16), split-0 cores | **0.1185** | 0.00526 | 0.0362 |
| TT(1,5,15,16), cores learned on split 1 (5 sweeps, in-sample) | 0.1195 | 0.00614 | 0.0387 |
| TT(1,5,15,16), split-1 cores, readout on split 0, tested on split-0 test | 0.1189 | 0.00642 | 0.0357 |

The gain survives the move: the frozen split-0 cores with a fresh linear
readout beat CP-16 by 3.6% and the categorical fit by 1% on the new split
(with 20-40% lower E and V errors), and they are as good as -- slightly better
than -- cores learned on the new split itself. As for the VarPro densities,
the cores are a pre-trainable, transferable object and the readout is the
per-dataset convex fit. The ordering between schedules is not identical on the
two splits ((1,5,15,16) transfers best here although (1,5,8,8) was best
in-sample): differences of 0.5% between schedules are within the split-to-split
noise; the 3-8% gaps to CP are not.

### 3.5 Cost of a step (`COST` lines; D4 and D5 measured with a second process running)

| | D4 | D5 |
|---|---|---|
| categorical solve (augmented QR, 1 950 / 3 795 columns) | 5.1 s (`TikhonovFactor` 3.7 s) | 23.4 s (18.4 s) |
| readout step (1 265 / 2 070 columns) | 4.0 s | 11.3 s |
| core step `G_3` (1 200 columns = 5 x 15 x 16, shared) | 2.9 s | 5.0 s |
| core step `G_2` (750 columns) | 1.3 s | 2.6 s |
| core step `G_1` (75 columns) | 0.2 s | 0.5 s |
| forming `A_cat * Phi` (once) | 1 s | 3 s |

A core step's design matrix has `S * r_{t-1} * r_t` columns per order that
uses the slot (independent of the degree: it is 1 200 at both degrees for
`G_3` of (1,5,15,16), 80 for (1,4,4,4)); its cost is the `m x N_c x n_g`
gemm that forms `A_cat Phi M_t` plus a QR of `m x n_g`, so it scales with
the number of rows and with `N_c ~ n_cat`, not with `n_cat^2` as the
categorical QR does. The readout step is a categorical-sized solve with
`sum_beta r_nu` columns per `z0` instead of `n_B` -- 65% / 55% of the
categorical column count at these degrees (it would be a small fraction at
large S). A full sweep is 2 x (3 core steps + readout) = 8-20 s at D4 and
30-40 s at D5 for the useful schedules, i.e. 2-4 categorical solves per
sweep, and 6-24 sweeps. At the scale of the phase-18 assemblies the same
arithmetic applies with the assembly cached once: ALS is 10-50 categorical
solves' worth of linear algebra on a matrix that was assembled once, which is
cheap next to the assembly.

## 4. What did not work, and what was not done

* **`Base.\` for the core steps.** With rank-deficient `[A Phi M_t; lambda P Phi M_t]`
  (dead bond directions, duplicate orderings mapping to one column) Julia's
  pivoted-QR `\` returned a truncated basic solution whose residual was
  *larger* than the current iterate's -- `J` jumped from 309 to 1.2e11 in the
  O1 check. The exact QR + SVD solve fixed it and the monotone assertion has
  been silent since; the ALS-as-linear-fits route needs a genuine
  least-squares minimiser (or a real ridge on the free block), not a
  rank-revealing solver, and this is the one practical trap.
* **The literal spec** ("cores `G_t^l[n]` over the folded radial x species
  index, readout per `l`-tuple") is not self-consistent with its own O2 (the
  radial index has nowhere to live in diagonal rank-16 cores); the readout was
  put per `(n', l)`-tuple. A TT that also compresses the radial index is a
  different, larger question and was not attempted.
* **Degree 6** was not run (the degree-5 sweeps of the wide schedules were
  already 70-200 s each with the two degrees running concurrently; the
  useful small schedules would take ~10 min at D6 and could be run).
  Cores shared across body orders, DMRG-style two-site updates with adaptive
  rank, and BLR / POPS in the core steps were not tried.
* **`ace_embedding_model` could not be assembled** in this checkout: at degree
  4 SpheriCart's `l_max = 0` gradient kernel crashes, and at any degree this
  branch's `basis_ed.jl` calls `EquivariantTensors.pushforward_rows!`, which the
  pinned ET 0.4.3 does not have. O2 was therefore done at the site-basis level
  (no derivatives), which is the stronger check anyway, and the CP comparator
  was fitted inside the harness (identical columns, section 2).
* **Timing** was measured with two Julia processes sharing the machine, so
  the absolute seconds are pessimistic by up to 2x; the ratios core step :
  categorical solve are what matter.
* Lambda was fixed at 1e-7 for all TT fits (re-verified for CP-16); no
  per-schedule sweep. The test set selects nothing here except in the CP-16
  lambda check, so the TT numbers are not test-tuned.

## 5. Conclusions for Stage 2B and for ET

1. **Does TT beat CP at matched parameters?** Yes, decisively, on this data:
   -3.5% to -7% test force RMSE at equal or fewer parameters at both degrees,
   with better energies and virials, and it beats the categorical model with
   20-40% of its parameters. The reason is structural: the categorical
   coefficient tensor over species is not symmetric for blocks with distinct
   radial/angular channels, CP with a frozen table spans only symmetric rank-1
   directions, and a TT with slot-specific cores spans a learned subspace of
   the full tensor -- the learned 4-dimensional subspace is worth more than
   the frozen 35-dimensional one.
2. **At matched flops?** No at S = 5: the categorical basis is the cheapest
   route to a given accuracy, CP-K next, and TT is level with CP only from
   ~2 500 flops/site. TT's cost is the per-slot species embedding and the
   `r_{t-1} r_t` matvec per node; it pays off in parameters and data, not in
   evaluation time, exactly as the plan's caveat (ii) predicted. The
   crossover in S is where CP's was (~7) or later and has not been measured.
3. **Is ALS-on-the-cached-matrix practical?** Yes: it is a handful of small
   linear fits per sweep on `A_cat * (Phi M_t)`, monotone, 2-4 categorical
   solves per sweep, 6-24 sweeps, with the CP fit as a free exact
   initialisation and a convex readout as the last step -- so everything the
   linear toolkit offers (priors, BLR, POPS, D-optimal selection, closed-form
   lambda) applies step by step. The learned cores transfer across splits
   with a readout-only refit, which is the "pre-train the format, then the
   convex fit at scale" pattern Stage 2B wants. It needs an exact
   least-squares solve, a mapping `Phi` derived (not assumed) from the
   symmetrisation, and care with dead bond directions.
4. **Format choice.** For the parameter-limited / data-limited regime (many
   species, modest training sets, foundation-model distillation on hundreds
   of structures) TT with small ranks is the better format and CP with a
   frozen table is not a good approximation of it. For the flop-limited
   regime at S = 5 neither beats categorical. An ET `formats/tt/` next to
   `formats/cp/` is justified by these numbers if the CP layer's evaluation
   (per-channel products) is generalised to per-slot `r_{t-1} x r_t` cores
   with the species embedded per slot -- the same `EquivLinearL`-style
   mixing, once per slot -- and the fitting story is the ALS above, not
   gradient descent. The cheapest next measurement is S: repeat the D4 table
   on a 10-species set, where the categorical count grows as `S^3` and the
   TT's as `S`.
5. **Side findings that need action on the branch.** `ace_embedding_model`
   (i) defaults to solid harmonics while `ace1_model` uses spherical, so it
   is not a reparameterisation for `l >= 1`; (ii) has a species-blind,
   truncated pair basis (4 functions per `z0` at S = 5, Ni neighbours
   missing); (iii) enumerates only the channel-free-degree blocks, 40-45% of
   the categorical columns short at the same nominal degree. All three
   depress every embedded-vs-categorical comparison made so far.
