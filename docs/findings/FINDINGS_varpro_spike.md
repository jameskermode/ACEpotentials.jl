# Learning the species weights inside the FS density by variable projection

Spike for the Stage 2B redefinition in `docs/plans/jax_ace_port_plan.md`
("basis pre-training that feeds 2A": learn the non-linear part once, cheaply,
by variable projection; freeze; convex fit at scale). The earlier FS spike
(`FINDINGS_fs_spike.md`) showed that with a *fixed* `sqrt(rho)` column added
to the linear ACE design matrix, the shape of the embedding function `F` is
exhausted by `sqrt`, and that what matters is the species weighting *inside*
the density: hand-chosen tilts `sqrt(rho_tot ± rho^s)` gave -6.8% test force
RMSE at degree 4 (75 extra columns) against -4.7% for `sqrt(rho_tot)`. This
spike asks whether those weights can be **learned** by VarPro with everything
else fixed, whether that beats the hand tilts, and — the gate — whether the
learned density **transfers**: frozen density, fresh convex fit on a disjoint
split.

Code: `acejax/spike_fs/varpro_core.jl` (column builder, projected inner solve,
Kaufman gradient — analytic and ForwardDiff — L-BFGS driver),
`varpro_gradcheck.jl` / `varpro_gradcheck2.jl` (Exp. 1), `varpro_learn.jl`
(Exps. 2–4), `varpro_assemble_split1.jl` + `varpro_transfer.jl` (Exp. 5),
`varpro_alpha_D5.jl` (Exp. 6); logs `varpro_*.log`; learned parameters
`varpro_learned_*.jls`. Nothing under `src/` or `test/` was touched. All local
(12-core M-series Mac, Julia 1.13, `--project=acejax/julia`); the degree-4/5
assemblies for the MersenneTwister(0) split were reused from the earlier
spike's cache, one new degree-4 assembly was made for the transfer split
(2 x 90 s with `-p 4`). Total compute about 1 h of which most was waiting on a
shared CPU (see "What did not work").

## The answer, in one paragraph

**Yes on all three counts.** VarPro over 15 species-by-width weights, starting
from equal weights (i.e. from the `sqrt(rho_tot)` model itself), converges
monotonically in ~60 L-BFGS iterations (14 s) and takes the degree-4 test
force RMSE from 0.1144 to **0.1078 (-5.7% vs `sqrt(rho_tot)`, -10.2% vs linear
ACE) with the same 15 columns** — already within 1% of the 90-column hand-tilt
model (0.1067). Two and three learned densities (30 / 45 columns) reach
**0.1056 / 0.1051 (-12.0% / -12.4% vs linear)**, beating the hand tilts
(-11.1%) and the earlier 165-column "both tilts" stack (0.1061) with 2–6x
fewer columns, and matching the earlier best all-fixed 225-column stack
(0.1047, which also had Chebyshev terms). Learning theta on 150 structures
with early stopping on 50 gives the same theta and the same test error as
learning on all 200 — there is no over-fitting to control at this parameter
count. **Frozen and moved to a disjoint 200/100 split, the learned densities
keep the whole gain**: -9.7 / -11.2 / -12.1% vs the linear baseline on that
split (in-sample it was -10.2 / -12.0 / -12.4%), they beat the hand tilts
there too (-10.6%), and they do as well as or better than densities learned on
the new split itself. They also carry to degree 5 unchanged (D5 + frozen
D4-learned P=3: 0.1040 vs D5 hand tilts 0.1044 and D5 linear 0.1120). The
learned weights are interpretable and stable in the part that matters (Mn and
Fe up, Ni and Co down, more so at short range) but not unique: the loss has
several equal-value optima with different weights, and each of them
transfers. That is good news for the "pre-train basis, then convex fit" route
at this scale, with the caveat that 15–45 parameters is not `Wnlq`.

## Setup

**Data, split, ACE part.** `cantor1k_b_mh1.xyz` (1000 CrMnFeCoNi structures,
32–48 atoms, MACE-MH-1 labels), split 0 = `MersenneTwister(0)` 200 train /
100 test as in the earlier spikes; degree-4 categorical `ace1_model(order = 3,
totaldegree = 4)`, 1950 basis functions, cached assembly (25 544 train rows,
12 796 test rows), `algebraic_smoothness_prior(p = 4)` column scaling, 30/1/1
weights with energy and virial rows divided by `sqrt(natoms)`. Test errors
from unweighted predictions, E and V per atom, as before. The baseline
numbers reproduce the earlier spike to 4 digits (linear 0.1201,
`sqrt(rho_tot)` 0.1144, five `+` tilts 0.1067).

**Extra columns.** Per-species, per-width fixed site densities
`rho_i^{s,k} = sum_{j: z_j = s} g_k(r_ij)`, `g_k(r) = exp(-alpha_k (r/r0 - 1))
fcut(r)`, `alpha = (2, 4, 6)`, `r0 = 2.5`, `rcut = 6.25` (the same
`fs_embed.jl` densities; the builder was checked against `fs_embed.jl`'s
columns to 2e-14). A learned density is `rho_i = sum_{s,k} w_{s,k}
rho_i^{s,k}` with `w = exp(theta)`, the column is `sqrt(rho_i + eps) -
sqrt(eps)` summed over sites, read out per central species. Two
parametrisations, both with S·K = 15 weights per density:

- **per-width** (primary): for each density `p` and width `k` a separate
  column `sqrt(sum_s w^{(p)}_{s,k} rho^{s,k})` — 15 columns per density.
  `theta = 0` is *exactly* the 15-column `sqrt(rho_tot)` reference, and every
  hand tilt is a point in this family, so the comparison is nested.
- **mixed** (the literal spec): one density per `p` mixing all 15 channels,
  `sqrt(sum_{s,k} w_{s,k} rho^{s,k})` — 5 columns per density. `theta = 0` is
  `sqrt(rho^1 + rho^2 + rho^3)`, which is *not* the reference (0.1163 vs
  0.1144 fixed).
- **full**: per-width columns but each mixing all 15 channels (45 weights per
  density) — a superset of per-width, run once as a check.

Weighted columns are rescaled to the median weighted, prior-scaled ACE column
norm, with the scale frozen at the initial theta (the readout coefficient
absorbs any rescaling; at lambda = 1e-8 it is immaterial).

**Inner solve and lambda.** lambda = **1e-8** throughout — the best value of
the `sqrt(rho_tot)` fit at degree 4 in the earlier spike. The lambda sweep was
re-run on every frozen column set reported below and 1e-8 is the best value
for every one of them (the "best F" column in the logs equals the fixed-lambda
column). The inner Tikhonov problem `min ||[A X] c - y||^2 + lambda^2 ||c||^2`
is solved by factorising the ACE block once — thin QR of `[A; lambda I]`
(27.5k x 1950, **4.6 s**, thin `Q` stored explicitly, 430 MB) — and projecting
the extra columns out of its range each iteration (`Q' X` is a gemm), then a
small least-squares in the 15–45 extra coefficients and a triangular solve for
`c_A`: **0.16 s per inner solve** for 10–45 columns, against 3.9 s for a fresh
`TikhonovFactor` of `[A X]` (QR + SVD). The two agree in the objective to
1.1e-11 relative (`varpro_gradcheck.log`); the coefficient vectors differ by
2e-3 relative because `[A X]` has singular values down to 4e-13 — the
prior-scaled degree-4 basis has an (almost) exact linear dependence — and the
two solvers resolve that null direction differently, without effect on the
residual.

**Outer gradient.** Kaufman/VarPro: at `c = c*(theta)` the derivative of the
reduced objective is `dL/dtheta = 2 r' (dX/dtheta) c_X`, which is exact
(`dL/dc = 0` at `c*`). Implemented twice: ForwardDiff through the column
builder (the plan's preferred route; 1–3 s per gradient for 15–45 parameters
because it rebuilds every column in dual numbers) and an analytic version
that exploits the linearity of `rho` in `w` — per structure it needs the
per-site channel sums `rho_ch[i, (s,k)]` and one residual-contracted pair sum
`Dch[i, (s,k)]`, after which the gradient is a 15 x (columns) contraction;
**0.01 s**, 100–300x faster, and it extends to `d/d log alpha` with two more
pair sums. L-BFGS from Optim 1.13.3 (Hager–Zhang line search), objective
logged every iteration.

## Exp. 1 — gradient check (`varpro_gradcheck.log`, `varpro_gradcheck2.log`)

ForwardDiff/Kaufman gradient against central finite differences of the
reduced objective (which re-solves the inner problem at each perturbed theta)
at a P = 2 mixed-width theta (30 parameters), three random components:

| component | analytic | FD, h = 1e-4 | FD, h = 1e-5 | rel. err. (h = 1e-5) |
|---|---:|---:|---:|---:|
| theta[3] | +8.01945698e-01 | +8.01945540e-01 | +8.01944040e-01 | 2.1e-06 |
| theta[20] | +4.37830891e-01 | +4.37830560e-01 | +4.37825628e-01 | 1.2e-05 |
| theta[26] | -7.53327690e-02 | -7.53327083e-02 | -7.53301265e-02 | 3.5e-05 |

The h = 1e-4 column agrees to 2e-7; h = 1e-5 is worse because the objective
is ~285 and 1e-13 roundoff over 2e-5 is 3e-6 — the disagreement is the FD's.
All 30 components: `|g - g_FD| / |g| = 5.0e-07`. The gradient along the
scale-invariant direction of a density (`sqrt(c rho) = sqrt(c) sqrt(rho)`,
absorbed by `c_X`) is 9e-7 against `|g| = 34`, as it should be. The analytic
gradient agrees with ForwardDiff to **4e-14 … 4e-13** relative for all five
parametrisations tested (per-width P = 1/2/3, mixed P = 2 with alpha, full
P = 1), alpha components included. Every optimisation run reported below has
**zero non-monotone steps** at a 1e-12 relative tolerance, except one
(the mixed-width P = 1 run) with four increases of 1e-10 … 1e-13 relative
in its last 20 converged iterations — Hager–Zhang's approximate Wolfe
acceptance at the roundoff floor, not a gradient error.

## Exps. 2–4 — degree 4, split 0 (`varpro_learn_D4.log`, `varpro_learn_D4_full.log`)

F in eV/A, E and V in eV/atom. "ncol" includes the 1950 ACE columns; "nth" is
the number of learned parameters; dF is the change in test force RMSE against
linear ACE / `sqrt(rho_tot)` / the five hand tilts; the CI is a paired
bootstrap (2000 resamples of test structures) of dF vs `sqrt(rho_tot)`. All
rows at lambda = 1e-8, which is also the sweep optimum for every row.

| variant | ncol | nth | train F | test F | test E | test V | vs lin | vs sqrt | vs tilts | CI (vs sqrt) | it | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| linear ACE | 1950 | 0 | 0.1094 | 0.1201 | 0.00622 | 0.0414 | — | +4.9% | +12.5% | | | |
| `sqrt(rho_tot)` K=3 [ref; = per-width P=1 at theta=0] | 1965 | 0 | 0.1037 | 0.1144 | 0.00595 | 0.0352 | -4.7% | — | +7.2% | | | |
| ref + tilt +Mn (best single hand tilt) | 1980 | 0 | 0.0997 | 0.1098 | 0.00555 | 0.0313 | -8.5% | -4.0% | +2.9% | [-4.8, -3.3] | | |
| ref + tilt +Co / +Ni / +Fe / +Cr | 1980 | 0 | | 0.1124 / 0.1124 / 0.1132 / 0.1137 | | | | -1.7 … -0.7% | | | | |
| ref + all 5 hand tilts `sqrt(rho_tot + rho^s)` | 2040 | 0 | 0.0969 | 0.1067 | 0.00539 | 0.0309 | -11.1% | -6.8% | — | [-7.8, -5.7] | | |
| **VarPro per-width P=1, init equal** | 1965 | 15 | 0.0983 | **0.1078** | 0.00523 | 0.0329 | **-10.2%** | -5.7% | +1.1% | [-6.7, -4.9] | 63 | 14 s |
| VarPro per-width P=1, init random 1 (sd 0.5) | 1965 | 15 | 0.0983 | 0.1078 | 0.00523 | 0.0329 | -10.2% | -5.7% | +1.1% | [-6.7, -4.9] | 79 | 19 s |
| VarPro per-width P=1, init random 2 (sd 0.5) | 1965 | 15 | 0.0994 | 0.1094 | 0.00525 | 0.0330 | -8.8% | -4.4% | +2.6% | [-5.6, -3.2] | 86 | 18 s |
| VarPro mixed P=1 (literal spec), init equal | 1955 | 15 | 0.1007 | 0.1111 | 0.00538 | 0.0329 | -7.4% | -2.9% | +4.2% | [-3.9, -1.9] | 151 | 40 s |
| (mixed P=1 at init, fixed) | 1955 | 0 | 0.1053 | 0.1163 | 0.00594 | 0.0367 | -3.1% | +1.7% | | | | |
| VarPro full P=1 (45 weights), init per-width identity | 1965 | 45 | 0.0998 | 0.1099 | 0.00536 | 0.0327 | -8.4% | -3.9% | +3.0% | [-5.0, -3.0] | 151 | 32 s |
| **VarPro per-width P=2, init equal + tilt Mn** | 1980 | 30 | 0.0963 | **0.1056** | 0.00503 | 0.0312 | **-12.0%** | -7.7% | -1.0% | [-8.7, -6.7] | 121 | 39 s |
| VarPro per-width P=2, init equal + random 1 | 1980 | 30 | 0.0960 | 0.1056 | 0.00508 | 0.0310 | -12.1% | -7.7% | -1.1% | [-8.8, -6.8] | 151 | 47 s |
| VarPro per-width P=2, init equal + random 2 | 1980 | 30 | 0.0962 | 0.1058 | 0.00505 | 0.0307 | -11.9% | -7.6% | -0.9% | [-8.6, -6.5] | 151 | 46 s |
| **VarPro per-width P=3, init equal + tilts Mn, Co** | 1995 | 45 | 0.0955 | **0.1051** | 0.00520 | 0.0301 | **-12.4%** | -8.1% | -1.5% | [-9.2, -7.1] | 121 | 48 s |
| VarPro per-width P=3, init equal + random | 1995 | 45 | 0.0959 | 0.1056 | 0.00509 | 0.0306 | -12.1% | -7.7% | -1.0% | [-8.9, -6.7] | 151 | 58 s |
| P=1 learned on 150, early-stopped on 50 (it 21/31), c refit on 200 | 1965 | 15 | 0.0984 | 0.1079 | 0.00525 | 0.0331 | -10.1% | -5.7% | +1.1% | [-6.6, -4.9] | 21 | 6 s |
| P=1 learned on 150, final iterate, c refit on 200 | 1965 | 15 | 0.0984 | 0.1078 | 0.00523 | 0.0330 | -10.2% | -5.8% | +1.1% | | 31 | 6 s |
| P=2 learned on 150, early-stopped (it 50/60), c refit on 200 | 1980 | 30 | 0.0967 | 0.1061 | 0.00505 | 0.0316 | -11.6% | -7.3% | -0.6% | [-8.4, -6.2] | 50 | 16 s |
| P=2 learned on 150, final iterate, c refit on 200 | 1980 | 30 | 0.0967 | 0.1061 | 0.00506 | 0.0315 | -11.6% | -7.3% | -0.6% | | 60 | 16 s |
| P=3 learned on 150, early-stopped (it 26/36), c refit on 200 | 1995 | 45 | 0.0958 | 0.1055 | 0.00521 | 0.0310 | -12.2% | -7.8% | -1.2% | [-9.0, -6.8] | 26 | 12 s |
| P=3 learned on 150, final iterate, c refit on 200 | 1995 | 45 | 0.0957 | 0.1054 | 0.00521 | 0.0309 | -12.2% | -7.9% | -1.2% | | 36 | 12 s |

For reference, the earlier all-fixed stacks at degree 4 (`FINDINGS_fs_spike`):
both tilt families (165 extra columns) 0.1061; both tilts + Chebyshev Kc=4
(225 columns) 0.1047. Like for like (sqrt only): VarPro P=3 with 45 columns
(0.1051) beats the 165-column tilt stack (0.1061); with the Chebyshev
embedding added it would presumably reach the same ~0.104 floor.

Wall times are for the outer loop only (the 4.6 s ACE factorisation is
shared); one L-BFGS iteration is 1–2 objective evaluations at ~0.2–0.4 s
(0.14 s to build the columns for 200 structures, 0.16 s inner solve) plus a
0.01 s gradient. P = 1 converges to `|g| < 1e-4` in 60–90 iterations. P = 2
and 3 hit the 120/150-iteration cap with `|g|` ~ 0.4–9 but the objective is
flat there (P = 2: 2.3918e2 at iteration 100, 2.3906e2 at 120, 2.3900e2 at
150 — 0.05% in the last 50 iterations; test F unchanged to 4 digits).

### Reading Exps. 2–4

1. **P = 1: learning the 15 weights is worth 5.7% on its own, with no extra
   columns.** The same 15 `sqrt` columns as the reference, only the species
   weights inside them changed, go from 0.1144 to 0.1078 — more than any
   single hand tilt (best: +Mn, 0.1098) and within 1% of all five tilts
   together (0.1067, 90 columns). The virial improves 7%, the energy 12%.
2. **P = 2, 3 beat the hand tilts** (0.1056 / 0.1051 vs 0.1067) with a third
   / half the columns, and beat the 165-column two-family tilt stack (0.1061).
   The third density adds only 0.5% over the second; the marginal value of
   more densities is small, consistent with the earlier finding that the
   response to species weighting is close to linear around equal weights.
3. **Init sensitivity.** From equal weights and from a random perturbation
   (seed 1) P = 1 lands on the identical optimum (species shares agree to
   3 decimals); seed 2 finds a different local optimum, 1.5% worse
   (0.1094), with visibly different shares (max difference 0.24). For
   P = 2 all three inits (tilt, two random) give 0.1056–0.1058; for P = 3 the
   tilt init is 0.5% better than the random one. The landscape has multiple
   optima of nearly equal value; a structured init (equal weights, or equal +
   the best single tilt) is the safe choice, and the differences between
   optima are at the 0.5–1.5% level, not the 5–8% level of the gain itself.
4. **The literal mixed-width spec is a worse family.** `sqrt(rho^1 + rho^2 +
   rho^3)` with one column per species starts 1.7% *above* the reference and
   after learning reaches only 0.1111, because it gives up the separate
   readout per width. The **full** 45-weight mixing, which contains per-width
   as a subset, ends *worse* than per-width (0.1099 vs 0.1078) from the
   identity init: the extra cross-width freedom is not useful on this
   density range (the three exponentials are nearly collinear on it, as the
   earlier spike found) and it makes the optimisation land in a poorer
   optimum. Per-width is the right parametrisation.
5. **No over-fitting of theta at this size.** The 50-structure validation
   force RMSE decreases monotonically to within a few iterations of the end
   (best at 21/31, 50/60, 26/36 iterations), the early-stopped and final
   thetas give test errors equal to 3 digits, and theta learned on 150
   structures is as good as theta learned on all 200 (0.1079 / 0.1061 / 0.1055
   vs 0.1078 / 0.1056 / 0.1051). 15–45 parameters on 25k rows leave nothing
   to regularise.

### The learned weights (P = 1, equal init; normalised to max = 1 per width; `varpro_learn_D4.log`)

| width (alpha) | Cr | Mn | Fe | Co | Ni | species share Cr/Mn/Fe/Co/Ni |
|---|---:|---:|---:|---:|---:|---|
| k=1 (2, long range) | 0.79 | 1.00 | 0.98 | 0.79 | 0.65 | 18.8 / 23.7 / 23.3 / 18.8 / 15.3% |
| k=2 (4) | 0.71 | 1.00 | 0.83 | 0.55 | 0.43 | 20.2 / 28.4 / 23.6 / 15.6 / 12.3% |
| k=3 (6, short range) | 0.78 | 1.00 | 0.88 | 0.55 | 0.38 | 21.7 / 27.9 / 24.5 / 15.3 / 10.6% |

Equal weights would be 20% each. The learned density **up-weights Mn and Fe
and down-weights Ni and Co**, with Cr near neutral, and the tilt is stronger
at short range (Ni's weight falls from 0.65 to 0.38 from the widest to the
narrowest exponential). This is the same ordering the hand tilts found — the
single +Mn tilt was worth -4.0%, +Co/+Ni/+Fe/+Cr -1.7 … -0.7% — and it is
physically plausible for MACE-MH-1's picture of the Cantor alloy: Mn and Fe
neighbours carry the largest "embedding" contribution, Ni the smallest, i.e.
the effective density a Ni neighbour contributes is about 40–60% of a Mn
neighbour's. With P = 2 (tilt init) the two densities specialise: density 1
becomes mildly Cr-heavy (Cr share 26–27% at every width) and density 2
strongly Mn/Fe-heavy at long range (k = 1: Mn 35% + Fe 32%, Ni 7%), while at
short range (k = 3) the two densities converge to almost the same pattern (Cr 1.00 / Mn 0.85 / Fe 0.76 / Co 0.65 /
Ni 0.5 in both) — the extra density is used to separate species at long
range, and the short-range weighting is unique.

## Exp. 5 — transfer (`varpro_transfer.log`)

Split 1: `MersenneTwister(1)` shuffle of the **700 structures not used by
split 0**, 200 train / 100 test — disjoint from everything the densities
were learned on (the brief said a new random split of the 1000; a disjoint
draw is the stricter test and was chosen deliberately). New degree-4
assembly (`varpro_assemble_split1.jl`, cached as
`asm_cantor_D4_split1_*.jls`). All densities below are **frozen** as learned
on split 0; only the convex fit is redone. "in-split" rows learned theta on
split 1's own training set (150 iterations) as the upper reference. lambda
= 1e-8 (again the sweep optimum for every row); CI is a paired bootstrap vs
the linear baseline on split 1.

| variant on split 1 | ncol | train F | test F | test E | test V | vs lin | vs sqrt | vs tilts | CI (vs lin) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| linear ACE | 1950 | 0.1082 | 0.1197 | 0.00648 | 0.0407 | — | +4.6% | +11.9% | |
| `sqrt(rho_tot)` K=3 | 1965 | 0.1025 | 0.1144 | 0.00607 | 0.0375 | -4.4% | — | +7.0% | [-5.2, -3.6] |
| ref + 5 hand tilts | 2040 | 0.0958 | 0.1070 | 0.00560 | 0.0341 | -10.6% | -6.5% | — | [-11.7, -9.5] |
| **split-0-learned P=1, frozen** | 1965 | 0.0972 | **0.1080** | 0.00558 | 0.0355 | **-9.7%** | -5.6% | +1.0% | [-10.8, -8.6] |
| **split-0-learned P=2, frozen** | 1980 | 0.0953 | **0.1063** | 0.00554 | 0.0341 | **-11.2%** | -7.1% | -0.6% | [-12.3, -10.0] |
| **split-0-learned P=3, frozen** | 1995 | 0.0944 | **0.1052** | 0.00530 | 0.0336 | **-12.1%** | -8.0% | -1.6% | [-13.2, -10.9] |
| split-0 150/50 early-stopped P=1 / P=2 / P=3, frozen | | | 0.1082 / 0.1068 / 0.1057 | | | -9.6 / -10.8 / -11.7% | | | |
| split-1-learned P=1 (in-split) | 1965 | 0.0972 | 0.1089 | 0.00533 | 0.0372 | -9.0% | -4.9% | +1.8% | [-10.1, -7.9] |
| split-1-learned P=2 (in-split) | 1980 | 0.0951 | 0.1063 | 0.00558 | 0.0349 | -11.2% | -7.1% | -0.7% | [-12.3, -10.1] |
| split-1-learned P=3 (in-split) | 1995 | 0.0949 | 0.1060 | 0.00543 | 0.0342 | -11.5% | -7.4% | -0.9% | [-12.6, -10.3] |

And the reverse direction, split-1-learned densities frozen on split 0:
P = 1 / 2 / 3 give 0.1087 / 0.1060 / 0.1060 (-9.5 / -11.7 / -11.7% vs linear),
against 0.1078 / 0.1056 / 0.1051 for the densities learned on split 0 itself.

**The gain transfers in full.** Relative to the linear baseline on the new
split the frozen densities give -9.7 / -11.2 / -12.1% where in-sample they
gave -10.2 / -12.0 / -12.4%; the bootstrap CIs on the two splits overlap
almost entirely; the frozen columns beat the hand tilts on the new split by
the same margin as on the old one; and — the surprising part — the
split-0-learned P = 1 density is *better* on split 1 (0.1080) than the one
learned on split 1 (0.1089), although both reach the same split-1 training
error (0.0972). The two are different optima with different weights (max
species-share difference 0.17: split 1's P = 1 puts Mn at 31% share at long
range but only 11–12% at the shorter widths, where split 0's puts it at 28%),
so **the learned density is not unique but every optimum found transfers**:
what is being learned is a broad, well-generalising direction in density
space, not a data-specific one. The energy and virial gains transfer as well
(split 1: E 0.00648 -> 0.00530, V 0.0407 -> 0.0336 with P = 3).

## Exp. 6 — learned alpha, and degree 5 (`varpro_alpha_D5.log`)

**Radial exponents learned too** (theta includes `log alpha_k`; degree 4,
split 0):

| variant | test F | test E | test V | train F | learned alpha | it / wall |
|---|---:|---:|---:|---:|---|---|
| per-width P=1, alpha fixed (2, 4, 6) | 0.1078 | 0.00523 | 0.0329 | 0.0983 | — | |
| per-width P=1 + alpha, init equal / (2, 4, 6) | 0.1065 | 0.00522 | 0.0327 | 0.0974 | 7.90, 7.85, 7.50 | 151 / 47 s |
| per-width P=1 + alpha, init learned w / (2, 4, 6) | 0.1065 | 0.00522 | 0.0327 | 0.0974 | 7.68, 7.59, 7.71 | 151 / 107 s |
| per-width P=2, alpha fixed | 0.1056 | 0.00503 | 0.0312 | 0.0963 | — | |
| per-width P=2 + alpha, init learned P=2 / (2, 4, 6) | 0.1051 | 0.00518 | 0.0309 | 0.0958 | 2.50, 6.55, 6.66 | 151 / 307 s |

Learning alpha is worth another -1.2% at P = 1 and -0.5% at P = 2, from both
inits. The interesting part is *how*: with a single density all three
exponents run to the same value, alpha ~ 7.7 — sharper than any of the fixed
ones — so the three "widths" collapse to one short-range density with three
(now redundant) species weightings; with two densities one width stays broad
(2.5) and the other two sharpen to 6.6. The fixed (2, 4, 6) set is therefore
not what the data want — a sharper density is — but the gain from fixing that
is small, and a K = 1 sharp density would presumably do as well as this K = 3
collapse. Not pursued further.

**Degree 5** (3795 ACE columns, split 0, ACE factorisation 35 s):

| variant | ncol | train F | test F | test E | test V | vs D5 linear | vs D5 sqrt |
|---|---:|---:|---:|---:|---:|---:|---:|
| D5 linear ACE | 3795 | 0.0910 | 0.1120 | 0.00668 | 0.0358 | — | |
| D5 `sqrt(rho_tot)` | 3810 | 0.0881 | 0.1077 | 0.00686 | 0.0353 | -3.8% | — |
| D5 + 5 hand tilts | 3885 | 0.0854 | 0.1044 | 0.00632 | 0.0333 | -6.8% | -3.1% |
| D5 + **D4/split-0-learned P=1, frozen** | 3810 | 0.0867 | 0.1062 | 0.00640 | 0.0345 | -5.2% | -1.4% |
| D5 + **D4/split-0-learned P=2, frozen** | 3825 | 0.0855 | 0.1045 | 0.00651 | 0.0334 | -6.7% | -3.0% |
| D5 + **D4/split-0-learned P=3, frozen** | 3840 | 0.0853 | **0.1040** | 0.00638 | 0.0333 | **-7.1%** | -3.4% |
| D5 VarPro P=1 (learned at D5, 47 it, 18 s) | 3810 | 0.0865 | 0.1059 | 0.00631 | 0.0344 | -5.4% | -1.7% |
| D5 VarPro P=2 (learned at D5, 151 it, 139 s) | 3825 | 0.0853 | 0.1042 | 0.00631 | 0.0330 | -7.0% | -3.2% |

(The earlier spike's D5 numbers with the sweep-best lambda = 1e-7 are 0.1119 /
0.1075 / 0.1040; here lambda = 1e-8 is fixed for comparability with the
VarPro rows, which costs 0.0002 — the "best lambda" column in the log gives
the 1e-7 values.) The densities learned at degree 4 **transfer across degree
unchanged**: frozen at degree 5 they match the densities learned at degree 5
(0.1062 vs 0.1059 for P = 1, 0.1045 vs 0.1042 for P = 2), and P = 3 frozen
(45 columns, 0.1040) matches the 90-column hand tilts (0.1044) and the earlier
D5 "both tilts" stack (0.1038, 165 columns); the earlier best D5 all-fixed
model with Chebyshev was 0.1031. As at degree 4, the learned species weights
at degree 5 (P = 1: shares 19/19/23/22/18% long range, 21/23/25/18/14% short
range) are within 0.06 of the degree-4 ones; the gain over the reference is
smaller at degree 5 (-1.7% vs -5.7%) because the higher-degree ACE terms
already encode some species-resolved coordination, exactly as the hand tilts'
gain shrank (-6.8% -> -3.3% -> -1.5% at D = 4/5/6).

## Conclusion

1. **Learning the density by VarPro beats the hand-chosen tilts**, with far
   fewer columns: 15 learned weights turn the 15 `sqrt(rho_tot)` columns from
   -4.7% into -10.2% (vs linear ACE, degree 4), 30 / 45 columns with two /
   three learned densities reach -12.0 / -12.4% against -11.1% for 90
   hand-tilt columns and -11.7% for the 165-column two-family tilt stack.
   The learned weights are interpretable (Mn, Fe up; Ni, Co down; stronger at
   short range) and reproduce the ordering the hand tilts had found; the
   remaining gain from more densities or learned radial exponents is
   0.5–1.2%.
2. **The learned density transfers when frozen** — to a disjoint 200/100 split
   (-9.7 / -11.2 / -12.1% vs -10.2 / -12.0 / -12.4% in-sample, beating the hand
   tilts there too, and matching or beating densities learned on the new
   split itself), and across degree (D4-learned densities at D5 equal
   D5-learned ones). Learning on 150 structures with early stopping changes
   nothing, so at this parameter count there is nothing to over-fit.
3. **What this says about the "pre-train basis, then convex fit" route.** The
   gate passes at this scale, and the mechanics are as the plan hoped: one
   factorisation of the fixed convex block (4.6 s at degree 4, 35 s at degree
   5), an inner solve per outer iteration that is a projection (0.16 s),
   Kaufman's gradient at fixed `c*` verified against finite differences to
   5e-7, a monotone L-BFGS trace, and the whole pre-training done in 15–60 s.
   The frozen result is a set of ordinary columns that the production
   assembly, prior, BLR and QP see as such. Two caveats for extrapolating to
   `Wnlq` and the element embedding: (a) here theta has 15–45 entries, and
   the non-identifiability already visible (several equal-loss optima with
   different weights, 0.5–1.5% apart in test error, all of which transfer)
   will be much larger with thousands of basis parameters — the hold-out and
   regularisation the plan asks for are needed there even though they were
   not here; (b) the family learned here is a *small* non-linear addition to
   a fixed linear basis, so "transfer" is the statement that a broad, smooth
   direction in density space generalises, which is a weaker claim than that a
   fully data-chosen basis would. Within those caveats: adopt learned FS
   densities in the per-width parametrisation (equal-weight init, P = 2 or 3,
   K = 3, ~100 L-BFGS iterations) as the first pre-trained object, and use the
   same VarPro machinery — projection inner solve, analytic gradient where the
   structure allows it — for the next one.

## What did not work, and caveats

- **The literal mixed-width parametrisation** in the brief
  (`rho^{(p)} = sum_{s,k} w_{s,k} rho^{s,k}`, one column per species per
  density) is a worse family than per-width: it does not contain the
  `sqrt(rho_tot)` reference (its equal-weight point is 1.7% worse) and after
  learning it reaches only 0.1111 vs 0.1078. The per-width form (same 15
  parameters, but a separate `sqrt` per width) is what was used for all
  headline numbers, and it nests the reference and the hand tilts exactly.
- **Freeing the cross-width mixing** (45 weights per density) made the
  result worse than its 15-weight subset from the identity init (0.1099 vs
  0.1078): a poorer local optimum, and no evidence the extra freedom is useful.
- **Multiple optima.** One random init of P = 1 (seed 2) converged to a
  1.5%-worse optimum with clearly different weights; densities learned on
  split 1 differ from split 0's by up to 0.17 in species share at equal
  training loss. Report weights as "an" optimum, not "the".
- **ForwardDiff through the column builder** (the brief's preferred route)
  works and was used for the gradient check, but at 1–3 s per gradient it made
  P = 3 cost 5 s per iteration (a first run with it, `varpro_learn_D4_fdgrad_
  partial.log`, with a 100-iteration cap, gives the same numbers to within
  0.0004 in test F and was abandoned at the early-stopping block for time). The analytic gradient — which exists because `rho` is linear
  in `w` — is 100–300x faster and agrees to 1e-13; all reported runs use it.
- **A shared CPU and 12 OpenBLAS threads do not mix.** Another Julia job was
  running on the box for part of the session (load average > 100); with
  `BLAS.set_num_threads(12)` the projected inner solve slowed by 10x (10 s per
  iteration instead of 0.3 s). Four BLAS threads fixed it; the small gemms here
  do not need more. The first `varpro_gradcheck` timing of 38 s for the QR
  (later 4.6 s) is the same effect.
- **P = 2 and 3 did not reach the P = 1 gradient tolerance** in 120–150
  iterations (`|g|` ~ 0.4–9 vs 5e-5); the objective is flat there (0.05% over
  the last 50 iterations) and the test error unchanged, so they were not run
  longer.
- lambda was fixed a priori at 1e-8 and happens to be the sweep optimum for
  every fit here (Cantor fits are effectively unregularised in the
  prior-scaled variables, as before), so there is no test-set lambda selection
  in this spike; the only test-set leakage is in the choice of "Mn" as the
  P = 2 tilt init, and the random inits reach the same result.
- Same scope caveats as the earlier spikes: one alloy with distilled MACE
  labels, 200 training structures, degree 4 (and 5), 100-structure test sets
  (bootstrap CIs cover test sampling only). The transfer test is one
  additional disjoint split.
