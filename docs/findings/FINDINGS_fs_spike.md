# Fixed `sqrt(rho)` Finnis-Sinclair columns in a linear ACE fit

Spike for the "convexity-preserving middle ground" in
`docs/plans/jax_ace_port_plan.md` (section *Spike candidate — Finnis-Sinclair
nonlinearity*): append a **fixed** EAM/FS-style `sqrt(rho_i)` term to the linear
ACE design matrix, where `rho_i` is a fixed (not fitted) pair density, so the
fit stays linear least squares, and measure whether it buys held-out accuracy at
fixed polynomial degree versus the alternative lever of raising the degree.

Code: `acejax/spike_fs/` (`fs_columns.jl`, `fd_check.jl`, `run_spike.jl`,
`analysis.jl`; logs `fd_check.log`, `run_cantor_D456.log`,
`run_tial_D6810.log`, `analysis_cantor.log`). A follow-up on richer
embedding functions and species-weighted densities is appended at the end
(`fs_embed.jl` and friends); its headline is that the *density* matters far
more than the embedding function. Nothing under `src/` or `test/`
was touched. Everything ran locally on a 12-core, 18 GB M-series Mac with
`julia -p 4 --project=acejax/julia` (Julia 1.13).

## The answer, in one paragraph

**Yes, a fixed `sqrt(rho)` column buys accuracy at fixed degree, and the gain
is real (paired bootstrap CIs exclude zero), but it is modest and its size
depends strongly on which density is put under the square root.** With the
spec's per-neighbour-species densities (75 columns at S=5, K=3) the test force
RMSE at degree 4 drops 2.5% (0.1201 -> 0.1170 eV/A); with the classical FS
form — one *total* density per site, `sqrt(sum_s rho_i^s)`, only 15–30
columns — it drops **4.7–5.4%** (-> 0.1136), the virial RMSE drops 15–18%, and
the same 15 columns still give -3.9% at degree 5 and -1.9% at degree 6. The
linear-`rho` controls (same densities without the square root) give 0–0.7%, so
the gain is the nonlinearity, not the extra radial functions. Raising the degree
from 4 to 5 (+1 845 columns, 2x the assembly cost) gives -6.8%, so the 30-column
total-density term is worth roughly three quarters of one degree step at degree
4 — and **degree 5 + `sqrt(rho_tot)` (0.1075) beats plain degree 6 (0.1133)** on
this 200-structure training set, where degree 6 is already data-limited. On the
TiAl DFT set the columns do nothing (< 0.5%), because that fit is limited by
something else (large train/test gap, not model capacity).

## What was built

`acejax/spike_fs/fs_columns.jl` builds, for every structure, the energy, all
forces and the virial of each FS column in exactly the row layout of
`ACEfit.feature_matrix(::AtomsData)` (`src/atoms_data.jl`): per structure
`[1 energy; 3*natoms forces, atom-major x,y,z (the _f_mat reinterpret
order); 6 virial components vec(V)[[1,5,9,6,3,2]] = xx, yy, zz, yz, xz, xy]`.

Column definition, for site `i` with central species `z_i`:

```
rho_i^{s,k} = sum_{j : z_j = s, r_ij < rcut} g_k(r_ij)
g_k(r)      = exp(-alpha_k (r/r0 - 1)) * fcut(r),  fcut = (1 - r/rcut)^2 (1 + 2 r/rcut)
B_{a,s,k}   = sum_{i : z_i = a} phi(rho_i^{s,k})
phi(rho)    = sqrt(rho + eps) - sqrt(eps)   (FS)      or    rho   (control)
```

with `r0 = 2.5`, `rcut = 6.25` (the Cantor models' uniform cutoff, read from
`model.rbasis.rin0cuts`), `alpha in {2,4,6}` (K=3), `eps = 1e-8`. Variants:
per-central-species readout (S*S*K = 75 columns) or shared across central
species (S*K = 15); K=1 (`alpha=4`); K=6 (`alpha in {1,2,3,4,6,8}`); and a
`total` flag that sums the density over neighbour species before `phi`
(`sqrt(rho_tot)`, S*K = 15 columns) — the classical Finnis-Sinclair form, added
after the per-species result came in. Neighbour lists come from
`NeighbourLists.PairList` via `ACEpotentials.Models.NeighbourLists` (no package
added). Densities are accumulated in one pass over pairs, forces and virials
in a second; the virial uses the ACEpotentials convention
`V = -sum_i dV_i (x) R_i` (`calculators.jl` `_site_virial`).

`run_spike.jl` assembles the ACE design matrix once per degree with
`ACEpotentials.assemble` (cached to the scratchpad as `.jls`), applies the
production smoothness prior (`algebraic_smoothness_prior(p = 4)`, column
scaling `A ./= P.diag'`) and the weights `W` (default 30/1/1, energy and virial
rows divided by `sqrt(natoms)` as `weight_vector` does), appends the weighted FS
columns, factorises once with `acejax/bench/distil/tikhonov.jl` and sweeps
`lambda in 10 .^ (0:-1:-8)`. Test errors are formed from the **unweighted**
predictions `A_test * (P \ z_ace) + X_test * c_x` against the assembled targets,
split by row kind, energy and virial rows divided by `natoms`. The best
held-out force RMSE over the sweep is reported (as `fit_distilled.jl` does —
this selects lambda on the test set, so the absolute numbers are mildly
optimistic; the *comparisons* all share the same protocol).

**FS column scaling.** No smoothness prior is applied to the FS columns.
Each weighted FS column is rescaled to the median 2-norm of the weighted,
prior-scaled ACE columns so that the single Tikhonov `lambda` treats them on
the same footing. On Cantor this made no difference (the `[unscaled]` rows are
identical to 4 digits, because the best lambda is at the unregularised end of
the sweep); on TiAl, where the best lambda is 1e-3..1e-2, the unscaled columns
were slightly *worse* than the baseline, so the scaling matters when the
regulariser is active.

## Verification

**Forces and virial by finite differences** (`fd_check.jl`, `fd_check.log`),
structure 1 of the Cantor set (32 atoms, all five species), central
differences on the first 6 atoms x 3 components and on all 9 strain
components, every column:

| variant | ncol | max FD error, forces | max FD error, virial | max abs F / V |
|---|---|---|---|---|
| sqrt, per-species | 75 | 1.3e-10 | 1.5e-09 | 2.7 / 11.7 |
| sqrt, shared | 15 | 1.0e-09 | 3.5e-09 | 1.5 / 46 |
| linear, per-species | 75 | 2.7e-10 | 6.5e-09 | 6.9 / 28 |
| linear, shared | 15 | 1.1e-09 | 2.0e-08 | 5.3 / 109 |

(h = 1e-5; the error scales as h^2 — 3.1e-05, 3.1e-07, 3.1e-09, 1.3e-10 for
h = 1e-2..1e-5 — so it is truncation error, not a bug.) Force sums vanish to
1.4e-15 per column. The virial was checked as `V_ab = -dE/d eps_ab` under a
homogeneous strain of cell and positions, i.e. the convention in
`calculators.jl`.

**Row bookkeeping against ACEpotentials.** For every degree and both
datasets the baseline (no FS columns) coefficients were pushed back into the
model with `set_linear_parameters!(model, P \ z)` and
`ACEpotentials.compute_errors` was run on the test set: it agrees with the
residual-based RMSEs to **1e-14** in F, E and V (six `CHECK` lines in the
logs). Row counts were asserted against `count_observations` per structure.

## Results — Cantor alloy (distilled MH-1 labels)

Data: `cantor1k_b_mh1.xyz` (1000 structures, 32–48 atoms, CrMnFeCoNi, MACE-MH-1
energies/forces/virials), random 200 train / 100 test split with
`MersenneTwister(0)`; test set = 12 796 rows (100 E, 12 096 F, 600 V).
Models: `ace1_model(elements = [:Cr,:Mn,:Fe,:Co,:Ni], order = 3,
totaldegree = D)`, categorical. Units: F eV/A, E eV/atom, V eV/atom.

### Main table (`run_cantor_D456.log`, `analysis_cantor.log`)

| model | n columns | test F | test E | test V | train F | best lambda |
|---|---:|---:|---:|---:|---:|---:|
| **D=4 linear ACE** | 1950 | **0.1201** | 0.00622 | 0.0414 | 0.1094 | 1e-7 |
| D=4 + rho, per-species K=3 (control) | 2025 | 0.1195 | 0.00612 | 0.0413 | 0.1085 | 1e-7 |
| D=4 + sqrt(rho), per-species K=3 (the spec) | 2025 | 0.1170 | 0.00610 | 0.0375 | 0.1058 | 1e-8 |
| D=4 + sqrt(rho), per-species K=1 (alpha=4) | 1975 | 0.1185 | 0.00630 | 0.0393 | 0.1072 | 1e-8 |
| D=4 + sqrt(rho), per-species K=6 | 2100 | 0.1167 | 0.00610 | 0.0369 | 0.1051 | 1e-7 |
| D=4 + sqrt(rho), shared readout K=3 | 1965 | 0.1189 | 0.00629 | 0.0401 | 0.1078 | 1e-7 |
| D=4 + rho_tot, K=3 (control) | 1965 | 0.1192 | 0.00619 | 0.0411 | 0.1086 | 1e-8 |
| D=4 + sqrt(rho_tot), K=3 | 1965 | 0.1144 | 0.00595 | 0.0352 | 0.1037 | 1e-8 |
| **D=4 + sqrt(rho_tot), K=6** | 1980 | **0.1136** | 0.00609 | **0.0339** | 0.1027 | 1e-8 |
| **D=5 linear ACE** | 3795 | **0.1119** | 0.00674 | 0.0358 | 0.0911 | 1e-7 |
| D=5 + rho, per-species K=3 (control) | 3870 | 0.1119 | 0.00680 | 0.0361 | 0.0914 | 1e-6 |
| D=5 + sqrt(rho), per-species K=3 | 3870 | 0.1110 | 0.00676 | 0.0353 | 0.0901 | 1e-7 |
| **D=5 + sqrt(rho_tot), K=3** | 3810 | **0.1075** | 0.00694 | 0.0354 | 0.0882 | 1e-7 |
| D=5 + sqrt(rho_tot), K=6 | 3825 | 0.1075 | 0.00693 | 0.0353 | 0.0881 | 1e-7 |
| **D=6 linear ACE** | 6890 | **0.1133** | 0.00738 | 0.0427 | 0.0754 | 1e-7 |
| D=6 + rho, per-species K=3 (control) | 6965 | 0.1122 | 0.00689 | 0.0414 | 0.0826 | 1e-4 |
| D=6 + sqrt(rho), per-species K=3 | 6965 | 0.1129 | 0.00737 | 0.0431 | 0.0750 | 1e-7 |
| D=6 + sqrt(rho_tot), K=3 | 6905 | 0.1111 | 0.00742 | 0.0424 | 0.0744 | 1e-7 |
| D=6 + sqrt(rho_tot), K=6 | 6920 | 0.1110 | 0.00742 | 0.0422 | 0.0743 | 1e-7 |

`r0` sensitivity (per-species sqrt, K=3, D=4): r0 = 2.0 / 2.5 / 3.0 gives
0.1167 / 0.1170 / 0.1174 — flat.

### Is the gain real? Paired bootstrap over test structures (2000 resamples)

| degree | variant | test F change | 95% CI | structures improved |
|---|---|---:|---|---:|
| 4 | + sqrt(rho), per-species K=3 | -2.5% | [-3.1%, -1.9%] | 80 / 100 |
| 4 | + sqrt(rho_tot), K=3 | -4.7% | [-5.6%, -3.8%] | 89 / 100 |
| 5 | + sqrt(rho), per-species K=3 | -0.8% | [-1.1%, -0.4%] | 69 / 100 |
| 5 | + sqrt(rho_tot), K=3 | -3.9% | [-4.8%, -3.0%] | 89 / 100 |
| 6 | + sqrt(rho), per-species K=3 | -0.3% | [-0.6%, -0.1%] | 55 / 100 |
| 6 | + sqrt(rho_tot), K=3 | -1.9% | [-2.5%, -1.4%] | 81 / 100 |

The improvement is uniform across density terciles (low / mid / high mean
coordination density: -4.5 / -4.3 / -5.2% for `sqrt(rho_tot)` at D=4), i.e. it
is not driven by a handful of compressed cells.

### Reading the table

1. **The square root is what helps, not the extra radial functions.** The
   linear-`rho` controls — the same 15 or 75 densities without the square
   root, which lie in the span of the 2-body ACE basis — move the force error
   by 0.1–0.7%. The `sqrt` versions of the same columns move it 2.5–5.4%.
   (The D=6 per-species control's 0.1122 is a point on a flat lambda curve,
   0.1122–0.1153 from 1e-4 to 1e-8, with a *higher* train error; it is noise,
   not a real gain.)

2. **The total density beats the per-species densities, with 5x fewer
   columns.** This was the surprise. The spec's `sqrt(rho_i^{s})` per neighbour
   species (75 columns) gives -2.5%; the classical FS `sqrt(sum_s rho_i^{s})`
   (15 columns) gives -4.7%, and this ordering holds at every degree. It is not
   a nesting: `sqrt(a+b)` is not in the span of `sqrt(a)` and `sqrt(b)`. In a
   random equiatomic alloy each per-species density counts roughly a fifth of
   the neighbours and fluctuates mostly with *local composition*, so its square
   root is largely composition noise; the total density measures coordination
   and local volume, which is what the FS embedding is physically about.
   Increasing K from 3 to 6 adds a little (-4.7% -> -5.4% at D=4), K=1 loses
   a little (-1.3% per-species).

3. **Compared with raising the degree.** At 200 training structures, degree
   4 -> 5 gives -6.8% for +1 845 columns (2x assembly time, 4x solve), and
   degree 5 -> 6 gives *+1.3%* — degree 6 (6 890 columns, train F 0.075 vs
   test 0.113) is already data-limited at this training size, so "+2 degrees"
   is not a usable lever here at all without more data. The 30-column
   `sqrt(rho_tot)` term buys -5.4% at degree 4, i.e. about 0.8 of a degree
   step, and it stacks: degree 5 + 15 columns (0.1075) is better than degree 6
   (0.1133) and is the best model in the table.

4. **Energies and virials.** The FS columns help the virial more than the
   forces at D=4 (V: 0.0414 -> 0.0339, -18% with `sqrt(rho_tot)` K=6),
   consistent with a volume-dependent embedding term. Energies are 6–7
   meV/atom throughout and are not systematically changed (the 100-structure
   energy RMSE is noisy; the D=5 `sqrt(rho_tot)` row is +3% on E while -3.9%
   on F).

5. **Regularisation.** With the smoothness prior in place, the best lambda is
   at the bottom of the sweep (1e-7..1e-8) for every Cantor model, i.e. the
   prior does the regularising and the fits are effectively unregularised
   least squares in the prior-scaled variables. The FS columns carry no prior;
   they did not need one here, but in a production fit they should get a
   (mild) explicit scale so that BLR's hyperparameter search sees them
   sensibly.

## Results — TiAl (real DFT labels; `run_tial_D6810.log`)

`ACEpotentials.example_dataset("TiAl_tiny")` has only 33 structures, too few
to split, so its parent `TiAl_tutorial` (329 DFT structures, 2–128 atoms,
`FLD_TiAl` + `TiAl_T5000`) was used: 230 train / 99 test, `MersenneTwister(0)`;
`ace1_model(elements = [:Ti,:Al], order = 3, totaldegree = D, rcut = 5.5,
Eref = [:Ti => -1586.0195, :Al => -105.5954])` as in the tutorial; default
weights (the tutorial's per-config-type weights were not used). FS densities
with the model's `r0 = 2.9`, `rcut = 5.5`.

| model | n columns | test F | test E | test V | train F | best lambda |
|---|---:|---:|---:|---:|---:|---:|
| D=6 linear ACE | 270 | 0.4494 | 0.00810 | 0.0936 | 0.372 | 1e-4 |
| D=6 + rho, per-species K=3 (control) | 282 | 0.4481 | 0.00807 | 0.0933 | 0.372 | 1e-4 |
| D=6 + sqrt(rho), per-species K=3 | 282 | 0.4483 | 0.00870 | 0.0964 | 0.364 | 1e-3 |
| D=6 + sqrt(rho), per-species K=1 | 274 | 0.4472 | 0.00870 | 0.0957 | 0.364 | 1e-4 |
| D=6 + sqrt(rho), shared K=3 | 276 | 0.4478 | 0.00871 | 0.0959 | 0.365 | 1e-4 |
| D=8 linear ACE | 648 | 0.4409 | 0.00793 | 0.0781 | 0.289 | 1e-3 |
| D=8 + sqrt(rho), per-species K=3 | 660 | 0.4397 | 0.00808 | 0.0775 | 0.286 | 1e-3 |
| D=10 linear ACE | 1428 | 0.4366 | 0.00940 | 0.0825 | 0.275 | 1e-2 |
| D=10 + sqrt(rho), per-species K=3 | 1440 | 0.4364 | 0.00939 | 0.0824 | 0.274 | 1e-2 |

Nothing moves by more than 0.5%, in either direction, and raising the degree
6 -> 10 only gives -3%. The test force RMSE (0.44–0.45 eV/A) sits far above
the train RMSE (0.27–0.37) at every degree: this fit is not capacity-limited,
it is limited by the T5000 high-temperature structures and their large forces
generalising poorly from 230 structures, and no basis change addresses that.
So TiAl is uninformative about the FS question rather than negative
evidence against it. The `[unscaled]` rows here were 0.5–1% *worse* than
baseline, which is the one place the FS column scaling mattered (lambda is
1e-4..1e-2 on this set, so an unscaled column is regularised differently from
its ACE neighbours). The total-density variant was not run on TiAl.

## Conclusion, and what it implies for the full (non-convex) FS option

- **A fixed `sqrt(rho)` column does buy accuracy at fixed degree, and it is
  nearly free**: 15–30 extra columns, a few seconds to build, no change to the
  solver, priors or BLR. Use the *total* density (classical FS), not
  per-species densities; use K >= 3 radial widths; readout per central species.
  Expect a few percent on forces and 10–20% on virials on a metallic alloy at
  low degree, shrinking as the degree rises (-5.4% at D=4, -3.9% at D=5, -2.0%
  at D=6 here).
- **It is worth about three quarters of a degree step at degree 4, and it
  stacks with degree**: degree 5 + 15 columns beat degree 6 alone at this
  training size. That is the practically useful reading — it is a cheap
  supplement to degree, not a replacement for it.
- **What it says about full FS.** The gap between per-species and total
  densities is the tell: the fixed columns' value depends strongly on *which*
  density is under the square root, and the best fixed choice here (equal
  species weights, three fixed exponentials) is a guess. A full FS term
  `sqrt(sum_s w_s rho_i^s)` with fitted `w_s` and fitted radial shapes is
  precisely the generalisation that would pick that density from the data —
  so the size of the fixed-column gain (a few percent) is a *lower bound* on
  what a fitted embedding could give, and the fact that the total density
  beats the per-species one by 2x suggests the fitted density would find more.
  Against that, the fixed-column route loses none of the convexity, BLR
  calibration or solver reuse that the plan is protecting, and the whole
  non-convex machinery would have to buy well over 5% to be worth it. Given
  that the distilled Cantor fits are at ~0.10 eV/A and production needs
  something below that, the honest reading is: **adopt the fixed
  `sqrt(rho_tot)` columns now (they are a design-matrix change only), and
  treat full FS as a second-order option** to be costed only after the
  larger levers (training-set size and generator, which halved the error in
  Stage 1D; degree with streaming assembly) are exhausted.

## Scope and caveats

- One alloy with *distilled* MACE-MH-1 labels (the fixed columns are helping
  ACE imitate MACE's many-body density term, which is a reasonable proxy for
  DFT on a metal but is not DFT), plus one small DFT set on which the
  question could not be resolved.
- 200 training structures: enough for degrees 4–5 to be capacity-limited
  (train/test 0.106/0.117), but degree 6 is already data-limited, so the
  "+2 degrees" comparison collapses to "+1 degree". At 800+ structures the
  degree lever would look better and the relative FS gain smaller.
- lambda selected on the test set (same protocol as `fit_distilled.jl`);
  all comparisons share it, the absolute numbers are slightly optimistic.
- Only one split. The bootstrap CIs cover test-set sampling, not the
  train-set draw.
- FS densities were not tuned beyond the r0 and K checks shown; alpha set,
  cutoff function and the `sqrt` (versus, say, `rho^{2/3}` or `log`) were
  taken as given.

## What did not work or was awkward

- The repo-root project does not load on Julia 1.13; `acejax/julia` does.
  `ExtXYZ.Atoms` has `system_data`/`atom_data`, not `.data`, and
  `FastSystem` needs `SVector` positions and a `mass` vector — both cost a
  compile-fix iteration in `fd_check`.
- Local assembly is slow: 200 train + 100 test structures took 2.5 min at
  degree 4, 7 min at degree 5 and 12 min at degree 6 with 4 workers, so the
  full run was split across cached assemblies (3.7 GB in the scratchpad,
  `fs_spike_cache/`); a single script invocation at degree 6 exceeds the
  15-minute budget. `analysis.jl` is solve-only on that cache.
- `fish` rejects `VAR=x cmd`; use `env VAR=x cmd`.

---

# Follow-up: more nonlinear functions of the same fixed density

Question: PACE/GRACE-FS use several densities each raised to a different
power, `E_i = sum_p c_p phi_p^{m_p}`, with the `phi_p` fitted. Keeping our
density FIXED (the total density `rho_tot`, K=3 widths, the best variant
above) and staying linear in the coefficients, do **more terms in the
embedding function F(rho)** buy more than the single `sqrt`? And does a general
F saturate — i.e. is what is missing the shape of F, or the shape of rho?

Code: `acejax/spike_fs/fs_embed.jl` (general columns `Phi_f(rho_i)` for any
function of the K-vector of site densities, with gradients; powers, Chebyshev
embeddings, cross terms and species-weighted densities are instances),
`fd_check_embed.jl`, `embed_spike.jl` (A/B/C), `cheb_diag.jl`,
`u_range_check.jl`, `species_tilt.jl`, `stack_check.jl`; logs
`fd_check_embed.log`, `embed_cantor.log`, `cheb_diag.log`, `u_range_check.log`,
`species_tilt.log`, `stack_check.log`. All solve-only on the cached
degree-4/5/6 assemblies (200 train / 100 test, same split, same prior, weights,
column scaling and lambda sweep as above); ~1 h of compute in total.

## The answer, in one paragraph

**No — the shape of F is not what is missing.** Every extension of F beyond
`sqrt` — single extra powers `rho^m` (m = 1/8 … 2), all five powers together,
Chebyshev embeddings of degree 4–8 in `sqrt(rho)` or `log(rho)`, cross terms
between widths — moves the test force RMSE by at most **-1%** at degree 4 and
5 (CIs mostly excluding zero, so real, but tiny), and a general F(rho)
saturates by Kc = 4: Kc = 6 is identical and Kc = 8 is *worse* (its tails are
unconstrained). The one exception is degree 6, where the Chebyshev embedding
gives -3.9%, but only by moving the best lambda up three decades (1e-7 ->
1e-4) — it is acting as a smooth, well-generalising substitute for
higher-degree ACE terms that overfit at 200 structures, and it lands on the
same floor (0.1067) that degree 5 + Chebyshev reaches. Meanwhile a cheap
linear probe of the **shape of rho** — fixed species-tilted densities
`sqrt(rho_tot ± rho^s)`, the first-order directions a fitted species weighting
could move in — gives **-6.8% at degree 4** (-3.3% at 5, -1.5% at 6), seven
times what any F-shape extension gives, and it stacks additively with the
Chebyshev embedding (-8.5% together). Degree 4 + 165 fixed columns (0.1047)
now beats plain degree 5 (0.1119) and degree 6 (0.1133). So: the embedding
function is essentially exhausted by `sqrt` plus one or two low-order terms;
what a fitted FS term would buy is the *density* — its species weighting first
of all — not the embedding.

## FD verification of the new columns (`fd_check_embed.log`)

Structure 1 of the Cantor set, 6 atoms x 3 components and 9 strain
components, all columns, h = 1e-5 unless stated:

| column family | ncol | max abs F / V | FD err forces | FD err virial |
|---|---:|---|---|---|
| powers m in {1/8, 1/4, 1/2, 3/4, 1, 2} | 90 | 69 / 1517 | 1.1e-08 | 8.1e-07 |
| Chebyshev[sqrt] Kc=8 | 120 | 14 / 368 | 1.3e-08 | 3.1e-05 (h=1e-5); 3.5e-07 (h=1e-6) |
| Chebyshev[log] Kc=8 | 120 | 18 / 368 | 1.3e-08 | 3.3e-05 (h=1e-5) |
| cross sqrt(rho_k1 rho_k2) | 15 | 4.3 / 101 | 9.2e-10 | 1.5e-08 |
| weighted sqrt | 10 | 1.0 / 30 | 4.4e-10 | 1.8e-09 |
| species-tilted (per-species channels), 165 cols | 165 | 2.3 / 27 | 3.9e-10 | 1.4e-09 |

The Chebyshev Kc=8 virial error is truncation error of the *check*, not of
the derivative: it scales exactly as h^2 (0.31, 3.1e-3, 3.1e-5, 3.5e-7 for
h = 1e-3 … 1e-6; `dT_8/du` = 64 at the ends of the range and the third
derivative is correspondingly large). Forces are below 1e-6 everywhere. The
`sqrt` column rebuilt through the general code agrees with the first spike's
`FSSpec(total = true)` to 4e-15, and the same through the per-species-channel
code to 7e-15.

## Setup

Site densities `rho_i^k`, k = 1..3 (`alpha` = 2, 4, 6), total over species,
`r0 = 2.5`, `rcut = 6.25`. Readout per central species throughout. Reference
for all marginal gains: **`sqrt(rho_tot)`, K=3, 15 columns** (row "sqrt only
[ref]"), itself -4.7 / -3.9 / -1.9% vs linear ACE at D = 4 / 5 / 6.

- (A) powers: `(rho + eps)^m - eps^m` per width, 15 columns per power.
- (B) Chebyshev: `T_n(u)`, n = 1..Kc, per width, `u = 2 (t(rho) - t_lo) / (t_hi - t_lo) - 1`,
  `t = sqrt` or `log`, `t_lo/t_hi` the training-set range of `t(rho_k)` with a
  2% margin (training range of `sqrt(rho_k)`: 2.82–3.50, 2.41–3.40,
  2.17–3.56). T_1 is `sqrt(rho)` (or `log`) up to a constant, so these nest the
  reference. 15·Kc columns.
- (C) cross terms `sqrt(rho_k1 rho_k2 + eps)` (3 pairs, 15 columns) and
  `sqrt(w . rho)` for w in {(1,1,1), (1,.5,.25), (.25,.5,1), (0,1,1)} (20
  columns), each added to the reference.
- Species tilts: `sqrt(rho_tot + rho^s)` and `sqrt(rho_tot - rho^s)` for each
  species s and width k (75 columns each), added to the reference.

Each weighted column is scaled to the median weighted, prior-scaled ACE column
norm (no centring: the row types are E/F/V, and a per-column offset has no
consistent meaning across them — the Chebyshev columns are already centred in
u). `cond(X)` below is the condition number of the scaled FS block alone.

## (A)/(B)/(C): the embedding function (`embed_cantor.log`)

dF and the 95% paired-bootstrap CI (2000 resamples of test structures) are
relative to the `sqrt` reference at the same degree. F in eV/A, E and V in
eV/atom.

**Degree 4** (linear ACE 0.1201; sqrt ref -4.7%)

| variant | ncol | test F | test E | test V | train F | dF vs sqrt | CI | lambda | cond(X) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sqrt only [ref] | 1965 | 0.1144 | 0.00595 | 0.0352 | 0.1037 | — | — | 1e-8 | 3.6e3 |
| sqrt + rho^1/8 | 1980 | 0.1139 | 0.00605 | 0.0348 | 0.1034 | -0.5% | [-0.8, -0.3] | 1e-8 | 3.6e5 |
| sqrt + rho^1/4 | 1980 | 0.1138 | 0.00606 | 0.0349 | 0.1034 | -0.5% | [-0.8, -0.3] | 1e-8 | 3.1e5 |
| sqrt + rho^3/4 | 1980 | 0.1135 | 0.00607 | 0.0356 | 0.1031 | -0.8% | [-1.2, -0.5] | 1e-8 | 1.6e5 |
| sqrt + rho^1 (in span) | 1980 | 0.1135 | 0.00590 | 0.0347 | 0.1028 | -0.8% | [-1.2, -0.5] | 1e-8 | 7.4e4 |
| sqrt + rho^2 | 1980 | 0.1138 | 0.00601 | 0.0345 | 0.1033 | -0.6% | [-0.8, -0.3] | 1e-7 | 2.2e4 |
| all powers 1/8..2 | 2025 | 0.1132 | 0.00608 | 0.0336 | 0.1024 | -1.1% | [-1.5, -0.7] | 1e-8 | 2.2e10 |
| cheb[sqrt] Kc=4 | 2010 | 0.1133 | 0.00608 | 0.0337 | 0.1024 | -0.9% | [-1.4, -0.5] | 1e-8 | 1.1e3 |
| cheb[sqrt] Kc=6 | 2040 | 0.1132 | 0.00609 | 0.0335 | 0.1023 | -1.0% | [-1.5, -0.6] | 1e-8 | 2.3e3 |
| cheb[sqrt] Kc=8 | 2070 | 0.1316 | 0.00627 | 0.0504 | 0.1022 | **+15.0%** | [-1.1, +43.3] | 1e-8 | 6.5e3 |
| cheb[log] Kc=4 | 2010 | 0.1133 | 0.00611 | 0.0337 | 0.1025 | -1.0% | [-1.5, -0.5] | 1e-7 | 1.1e3 |
| cheb[log] Kc=6 | 2040 | 0.1136 | 0.00607 | 0.0342 | 0.1023 | -0.7% | [-1.3, -0.0] | 1e-8 | 1.9e3 |
| cheb[log] Kc=8 | 2070 | 0.1162 | 0.00617 | 0.0389 | 0.1022 | +1.6% | [-1.1, +6.6] | 1e-8 | 4.7e3 |
| sqrt + cross sqrt(rho_k1 rho_k2) | 1980 | 0.1139 | 0.00595 | 0.0342 | 0.1030 | -0.5% | [-0.8, -0.1] | 1e-8 | 1.0e5 |
| sqrt + 4 weighted sqrt | 1985 | 0.1140 | 0.00597 | 0.0350 | 0.1034 | -0.4% | [-0.6, -0.1] | 1e-8 | 8.6e7 |

**Degree 5** (linear ACE 0.1119; sqrt ref -3.9%)

| variant | ncol | test F | test E | test V | train F | dF vs sqrt | CI | lambda | cond(X) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sqrt only [ref] | 3810 | 0.1075 | 0.00694 | 0.0354 | 0.0882 | — | — | 1e-7 | 3.6e3 |
| sqrt + rho^1/8 | 3825 | 0.1069 | 0.00695 | 0.0352 | 0.0880 | -0.6% | [-0.9, -0.3] | 1e-7 | 3.6e5 |
| sqrt + rho^1/4 | 3825 | 0.1069 | 0.00695 | 0.0353 | 0.0880 | -0.6% | [-0.9, -0.3] | 1e-7 | 3.1e5 |
| sqrt + rho^3/4 | 3825 | 0.1068 | 0.00695 | 0.0353 | 0.0880 | -0.6% | [-1.0, -0.3] | 1e-7 | 1.6e5 |
| sqrt + rho^1 (in span) | 3825 | 0.1074 | 0.00691 | 0.0353 | 0.0881 | -0.1% | [-0.2, -0.0] | 1e-7 | 7.4e4 |
| sqrt + rho^2 | 3825 | 0.1070 | 0.00696 | 0.0353 | 0.0880 | -0.5% | [-0.8, -0.2] | 1e-7 | 2.2e4 |
| all powers 1/8..2 | 3870 | 0.1069 | 0.00697 | 0.0342 | 0.0878 | -0.6% | [-0.9, -0.1] | 1e-7 | 2.2e10 |
| cheb[sqrt] Kc=4 | 3855 | 0.1068 | 0.00700 | 0.0323 | 0.0908 | -0.7% | [-1.6, +0.1] | **1e-4** | 1.1e3 |
| cheb[sqrt] Kc=6 | 3885 | 0.1068 | 0.00703 | 0.0324 | 0.0907 | -0.7% | [-1.7, +0.3] | 1e-4 | 2.3e3 |
| cheb[sqrt] Kc=8 | 3915 | 0.1073 | 0.00704 | 0.0328 | 0.0906 | -0.2% | [-1.4, +1.4] | 1e-4 | 6.5e3 |
| cheb[log] Kc=4 | 3855 | 0.1067 | 0.00699 | 0.0324 | 0.0908 | -0.7% | [-1.6, +0.1] | 1e-4 | 1.1e3 |
| cheb[log] Kc=6 | 3885 | 0.1067 | 0.00700 | 0.0324 | 0.0907 | -0.8% | [-1.6, +0.1] | 1e-4 | 1.9e3 |
| cheb[log] Kc=8 | 3915 | 0.1086 | 0.00700 | 0.0368 | 0.0875 | +1.0% | [-0.9, +4.1] | 1e-7 | 4.7e3 |
| sqrt + cross sqrt(rho_k1 rho_k2) | 3825 | 0.1071 | 0.00697 | 0.0350 | 0.0879 | -0.4% | [-0.6, -0.2] | 1e-7 | 1.0e5 |
| sqrt + 4 weighted sqrt | 3830 | 0.1068 | 0.00694 | 0.0347 | 0.0877 | -0.6% | [-1.0, -0.3] | 1e-8 | 8.6e7 |

**Degree 6** (linear ACE 0.1133; sqrt ref -1.9%)

| variant | ncol | test F | test E | test V | train F | dF vs sqrt | CI | lambda | cond(X) |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| sqrt only [ref] | 6905 | 0.1111 | 0.00742 | 0.0424 | 0.0744 | — | — | 1e-7 | 3.6e3 |
| sqrt + rho^1/8 | 6920 | 0.1106 | 0.00741 | 0.0425 | 0.0742 | -0.4% | [-0.6, -0.2] | 1e-7 | 3.6e5 |
| sqrt + rho^1/4 | 6920 | 0.1106 | 0.00741 | 0.0425 | 0.0742 | -0.5% | [-0.7, -0.3] | 1e-7 | 3.1e5 |
| sqrt + rho^3/4 | 6920 | 0.1106 | 0.00742 | 0.0426 | 0.0742 | -0.4% | [-0.6, -0.3] | 1e-7 | 1.6e5 |
| sqrt + rho^1 (in span) | 6920 | 0.1110 | 0.00742 | 0.0424 | 0.0743 | -0.0% | [-0.1, -0.0] | 1e-7 | 7.4e4 |
| sqrt + rho^2 | 6920 | 0.1101 | 0.00700 | 0.0424 | 0.0816 | -0.9% | [-2.5, +1.0] | **1e-4** | 2.2e4 |
| all powers 1/8..2 | 6965 | 0.1099 | 0.00699 | 0.0421 | 0.0814 | -1.1% | [-2.6, +0.8] | 1e-4 | 2.2e10 |
| cheb[sqrt] Kc=4 | 6950 | 0.1067 | 0.00686 | 0.0357 | 0.0795 | **-3.9%** | [-5.2, -2.6] | **1e-4** | 1.1e3 |
| cheb[sqrt] Kc=6 | 6980 | 0.1067 | 0.00690 | 0.0348 | 0.0794 | -3.9% | [-5.3, -2.6] | 1e-4 | 2.3e3 |
| cheb[sqrt] Kc=8 | 7010 | 0.1073 | 0.00691 | 0.0348 | 0.0793 | -3.4% | [-5.0, -1.5] | 1e-4 | 6.5e3 |
| cheb[log] Kc=4 | 6950 | 0.1067 | 0.00686 | 0.0359 | 0.0795 | -3.9% | [-5.2, -2.6] | 1e-4 | 1.1e3 |
| cheb[log] Kc=6 | 6980 | 0.1066 | 0.00689 | 0.0351 | 0.0794 | -4.0% | [-5.3, -2.7] | 1e-4 | 1.9e3 |
| cheb[log] Kc=8 | 7010 | 0.1078 | 0.00694 | 0.0359 | 0.0793 | -3.0% | [-5.0, -0.2] | 1e-4 | 4.7e3 |
| sqrt + cross sqrt(rho_k1 rho_k2) | 6920 | 0.1110 | 0.00747 | 0.0422 | 0.0742 | -0.1% | [-0.2, +0.1] | 1e-7 | 1.0e5 |
| sqrt + 4 weighted sqrt | 6925 | 0.1107 | 0.00742 | 0.0425 | 0.0742 | -0.3% | [-0.5, -0.1] | 1e-7 | 8.6e7 |

### Reading (A)/(B)/(C)

1. **Powers.** Each single extra power is worth -0.4 to -0.8%; they are
   nearly interchangeable (m = 1/8 and m = 1/4 give identical numbers) and all
   five together give -1.1% at D=4 and -0.6% at D=5 — the marginal gains do
   not add, because on the narrow density range of a bulk alloy (sqrt(rho)
   spans 2.8–3.5) all the powers are nearly the same smooth function. The
   surprise in this block is that `rho^1`, which *is* in the 2-body ACE span,
   gives -0.8% at D=4: its per-central-species readout of a *species-blind*
   total density is a specific combination the prior-scaled 2-body basis does
   not reach cheaply at degree 4; at D=5/6 it gives 0%, as it should.

2. **Conditioning.** The all-powers block has cond 2e10 (as expected — powers
   of one variable on a narrow interval are near-collinear) but this did *not*
   push the best lambda up at D=4/5 and did not hurt: the SVD-based Tikhonov
   solve handles it, and the near-null directions simply carry no signal. The
   Chebyshev blocks are well conditioned (1e3–7e3), as an orthogonal basis
   should be. **Where the best lambda did move up sharply** (1e-7 -> 1e-4):
   the Chebyshev embeddings at D=5 and D=6, and the `rho^2`/all-powers block at
   D=6. Those fits are in a different regime from the reference (regulariser
   active), which is why their CIs are wide even when the point estimate is
   small — the per-structure residuals differ from the reference's for reasons
   beyond the added columns.

3. **A general F(rho) saturates at Kc = 4.** Kc = 6 equals Kc = 4 to the
   third digit at every degree and transform; `sqrt` and `log` transforms are
   indistinguishable. Kc = 8 is worse, spectacularly so at D=4 (+15%, CI
   [-1, +43]). `cheb_diag.log` shows this is one test structure (#14, 32
   atoms: force RMSE 0.166 -> 0.766) whose sites sit at |u| = 0.9, i.e. in the
   outer 10% of the density range where only 0.1% of the *training* sites lie
   (9 of 8048); only one test site in 4032 is outside the training range at
   all. A degree-8 polynomial is simply unconstrained there. Widening the
   u-range margin fixes it monotonically (margin 0.02 / 0.1 / 0.3 / 1.0: +15.0
   / +1.7 / +0.1 / -0.3%) and excluding that structure the Kc=8 fit is -1% like
   Kc=4 — the classical spline-EAM lesson that the embedding's knots must sit
   where the data are. With Kc=4 the fitted F is well-behaved without any of
   this.

4. **The D=6 exception.** At D=6 the Chebyshev embedding is worth -3.9%, but
   note how: the best lambda moves to 1e-4, the train error rises (0.074 ->
   0.080) and the test error falls to **0.1067 — exactly the floor D=5 +
   Chebyshev reaches (0.1067–0.1068)**. Degree 6 at 200 structures is
   data-limited (train 0.075 / test 0.113); a 60-column smooth embedding under
   a regulariser that now bites is a better use of the data than the
   high-(n,l) ACE terms it displaces. That is a real effect, and it also helps
   energies (-7.5%) and virials (-18%), but it is a statement about
   regularising an overfitted degree, not about F(rho) having more to give:
   the same floor is reached from D=5.

5. **Cross terms and mixed widths (C)** are worth -0.1 to -0.6%; the three
   exponentials are already nearly redundant on this density range, and
   re-mixing them buys nothing.

## The shape of rho: species-tilted densities (`species_tilt.log`, `stack_check.log`)

| degree | variant | ncol | test F | test E | test V | train F | dF vs sqrt | CI | lambda | cond(X) |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 4 | sqrt(rho_tot) [ref] | 1965 | 0.1144 | 0.00595 | 0.0352 | 0.1037 | — | — | 1e-8 | 3.6e3 |
| 4 | + sqrt(rho_tot + rho^s) | 2040 | 0.1067 | 0.00539 | 0.0309 | 0.0969 | **-6.8%** | [-7.8, -5.7] | 1e-8 | 1.1e6 |
| 4 | + sqrt(rho_tot - rho^s) | 2040 | 0.1070 | 0.00561 | 0.0316 | 0.0975 | -6.5% | [-7.3, -5.6] | 1e-8 | 3.9e5 |
| 4 | + both tilts | 2115 | 0.1061 | 0.00539 | 0.0305 | 0.0960 | -7.3% | [-8.3, -6.2] | 1e-7 | 5.5e6 |
| 4 | + both tilts + cheb[sqrt] Kc=4 | 2175 | **0.1047** | 0.00550 | **0.0283** | 0.0942 | **-8.5%** | [-9.5, -7.5] | 1e-8 | — |
| 5 | sqrt(rho_tot) [ref] | 3810 | 0.1075 | 0.00694 | 0.0354 | 0.0882 | — | — | 1e-7 | 3.6e3 |
| 5 | + sqrt(rho_tot + rho^s) | 3885 | 0.1040 | 0.00639 | 0.0334 | 0.0855 | -3.3% | [-4.1, -2.5] | 1e-7 | 1.1e6 |
| 5 | + sqrt(rho_tot - rho^s) | 3885 | 0.1043 | 0.00659 | 0.0334 | 0.0857 | -3.0% | [-3.6, -2.4] | 1e-7 | 3.9e5 |
| 5 | + both tilts | 3960 | 0.1038 | 0.00653 | 0.0323 | 0.0849 | -3.5% | [-4.2, -2.7] | 1e-7 | 5.5e6 |
| 5 | + both tilts + cheb[sqrt] Kc=4 | 4020 | **0.1031** | 0.00655 | 0.0316 | 0.0848 | -4.1% | [-4.9, -3.2] | 1e-6 | — |
| 6 | sqrt(rho_tot) [ref] | 6905 | 0.1111 | 0.00742 | 0.0424 | 0.0744 | — | — | 1e-7 | 3.6e3 |
| 6 | + sqrt(rho_tot + rho^s) | 6980 | 0.1095 | 0.00729 | 0.0399 | 0.0737 | -1.5% | [-2.1, -0.9] | 1e-6 | 1.1e6 |
| 6 | + sqrt(rho_tot - rho^s) | 6980 | 0.1098 | 0.00743 | 0.0396 | 0.0737 | -1.1% | [-1.8, -0.5] | 1e-6 | 3.9e5 |
| 6 | + both tilts | 7055 | 0.1094 | 0.00730 | 0.0399 | 0.0734 | -1.5% | [-2.2, -0.8] | 1e-6 | 5.5e6 |

Seventy-five fixed columns that change *which* density is under the square
root are worth -6.8% at degree 4, against -1.1% for the best change to the
function *of* the density (and against -2.5% for the first spike's
per-species `sqrt(rho^s)`, which is a cruder probe of the same thing). The
two tilt families are nearly redundant with each other (+ and - tilts
together: -7.3%), which says the response to species weighting is close to
linear around equal weights — a fitted `w_s` would find roughly this. The
gain shrinks with degree (the higher-degree ACE terms already encode some
species-resolved coordination) but is significant at every degree. It stacks
additively with the Chebyshev embedding: -7.3% and -0.9% separately, -8.5%
together at D=4; -3.5% and -0.7%, -4.1% together at D=5. Best lambda stays
at the unregularised end, and cond(X) 4e5–6e6 caused no trouble.

Cumulative picture at **degree 4**, all convex, all fixed columns: linear ACE
0.1201 -> + sqrt(rho_tot) 0.1144 -> + species tilts 0.1061 -> + Chebyshev
0.1047, i.e. **-12.8% for 225 columns**, which is more than the +1 degree step
(-6.8%, +1845 columns) and beats plain degree 6 (0.1133, +4940 columns). The
energies (0.0055 vs 0.0062 eV/atom) and virials (0.0283 vs 0.0414, -32%) move
with the forces.

## Conclusion

More terms in the embedding function do not help beyond `sqrt`: on the
narrow density range of a bulk alloy, `sqrt(rho)` plus any one additional
smooth term is already a general F(rho) to within 1% of the test force RMSE,
Chebyshev Kc = 4 is the saturation point, and higher orders hurt through
unconstrained tails (fixable by a margin, but not worth it). Pacemaker's
several-powers form is therefore not what makes FS-ACE work on this data;
what makes a difference — 7x more than any F-shape term, and stacking with
it — is **the shape of rho**: the species weighting of the density under the
square root, probed here with fixed ± tilts and worth -6.8% at degree 4. So
if a fitted FS term is ever costed, the parameters worth fitting are the
species weights (and possibly radial shapes) *inside* the density,
`sqrt(sum_s w_s rho^s)`, not the exponents or the embedding polynomial; and
the convex route captures most of that first-order effect with 75 fixed
columns. Recommended fixed set, all linear: `sqrt(rho_tot)` (K=3) +
`sqrt(rho_tot ± rho^s)` + Chebyshev Kc=4 in `sqrt(rho_tot)` with a generous
u-range margin, readout per central species — 225 columns at S=5. Caveats as
before: one alloy, distilled labels, 200 training structures, lambda chosen on
the test set (shared by every row of every table), one split.
