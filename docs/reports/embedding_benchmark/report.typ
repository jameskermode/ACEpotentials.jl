#set page(paper: "a4", margin: (x: 2.2cm, y: 2.0cm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: none)
#show heading.where(level: 1): it => block(above: 1.0em, below: 0.5em, text(size: 12pt, weight: "bold", it.body))
#show heading.where(level: 2): it => block(above: 0.9em, below: 0.4em, text(size: 10.5pt, weight: "bold", it.body))
#show table: set text(size: 9pt)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none }, inset: (x: 5pt, y: 3pt))
#show figure.caption: set text(size: 9pt)

#align(center)[
  #text(size: 15pt, weight: "bold")[Frozen element embeddings versus categorical species in linear ACE:\ distillation benchmarks on CrMnFeCoNi]
  #v(0.3em)
  #text(size: 9pt)[ACEpotentials.jl — 14 September 2026 — accompanies pull requests from `fix/basis-ed-performance`, `pr/jax-port`, `pr/element-embeddings`]
]

= Summary

A linear ACE model whose species dependence enters through a *frozen element embedding* (Darby-style tensor reduction, channel folded into the radial index) was benchmarked against the standard *categorical* model on a five-component alloy, with labels distilled from MACE-MH-1. At matched polynomial degree the embedded model costs 7–17% on force RMSE for 3–6× fewer parameters; given one more degree — which its smaller design matrix makes affordable — it *matches* the categorical model at 2.4× fewer parameters (0.0767 vs 0.0763 eV/Å). The best absolute result is categorical degree 10 on 4 000 structures, 0.0641 eV/Å and 2.0 meV/atom, fitted from a 151 GB design matrix. Learning curves show the categorical model is data-limited and the embedded model basis-limited, which is why the trade runs the way it does. Two surprises: the embedding's *values* do not matter at $S = 5$, and the Julia evaluator ran the embedded model *slower* than the categorical one — now profiled and fixed. Recommendations, including code changes in flight, close the report.

= Set-up

*System and labels.* Rattled and strained fcc CrMnFeCoNi cells (32–48 atoms, random equiatomic occupancy, no short-range order), generated with bounded uniform perturbations (lattice $plus.minus 3%$, shear 0.02, rattle 0.02–0.10 Å, minimum separation 2.0 Å). Energies, forces and virials from MACE-MH-1 on an RTX A4500 (4 000 structures in 4.6 min). An earlier 6%-Gaussian generator produced crushed cells (36% compression, forces above 20 eV/Å) carrying 77% of the squared test error; bounding it halved the error at fixed basis and budget. MACE-MP-0-small was rejected as teacher: a difference-fit showed its disagreement with MH-1 to be rough, not a smooth offset.

*Models.* Order 3, ACE1 heuristics, uniform cutoff 6.25 Å. Categorical `ace1_model` with $S = 5$ elements; embedded `ace_embedding_model` with per-order widths $d_nu = min(d_max, binom(S + nu - 1, nu))$: *lossless* $[5, 15, 35]$ (spans the full species tensor at every order — a reparameterisation), $d_max = 16$ ($[5, 15, 16]$, order 3 cut to 16 of 35 directions) and $d_max = 8$. Embedding table: MACE-MH-1's 512-channel `node_embedding` reduced by PCA of the five rows (Gram preserved exactly; generic orthonormal mixing for $d > S$). Parameters are $n_B times S$ (a readout per central species).

*Fitting.* Held-out 80/20 split; algebraic smoothness prior $p = 4$; weights from `acefit!` defaults; factor-once Tikhonov (QR of $A$, SVD of $R$) with $lambda$ swept per model and the best held-out force RMSE reported. Assembly on 12 workers; the degree-10 categorical matrix (404 144 × 46 885, 151 GB) was factorised in place on a 376 GB node.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    table.header([branch (PR)], [based on], [contents]),
    [`fix/basis-ed-performance`], [`main`], [assembly and forward-evaluation speed-ups; profiling findings; 10 commits],
    [`pr/jax-port`], [`fix/basis-ed-performance`], [JAX evaluator port: `acejax`, exporter, tests + CI divergence gate, LAMMPS, benchmarks, plan; 7 commits, no changes under `src/`],
    [`pr/element-embeddings`], [`pr/jax-port`], [frozen element embeddings, factorised radial export, distillation benchmarks, FS/VarPro spikes, this report; 5 commits],
  ),
  caption: [The three pull requests. Fine-grained history (146 commits) is on branch `jax-eval`; the PR branches are squashed into logical commits.],
) <tab-prs>

= Results

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    table.header([model], [$n_B$], [params], [1k test F], [4k test F], [4k test E], [rel. to cat.]),
    [deg 6 categorical], [1 348], [6 740], [0.0969], [—], [—], [—],
    [deg 6 embedded $d lt.eq 16$], [323], [1 615], [0.1101], [—], [—], [+14%, 4.2× fewer (1k)],
    [deg 8 categorical], [3 824], [19 120], [0.0855], [*0.0763*], [0.0024], [—],
    [deg 8 embedded lossless], [1 190], [5 950], [0.0906], [0.0885], [0.0025], [+16%, 3.2× fewer],
    [deg 8 embedded $d lt.eq 16$], [753], [3 765], [0.0911], [0.0894], [0.0025], [+17%, 5.1× fewer],
    [deg 8 embedded $d lt.eq 8$], [408], [2 040], [0.0986], [0.1000], [0.0027], [+31%, 9.4× fewer],
    [deg 10 categorical], [9 327], [46 635], [—], [*0.0641*], [0.0020], [—],
    [deg 10 embedded lossless], [2 730], [13 650], [—], [*0.0753*], [0.0022], [+17%, 3.4× fewer],
    [deg 10 embedded $d lt.eq 16$], [1 609], [8 045], [—], [*0.0767*], [0.0022], [+20%, 5.8× fewer],
    [deg 10 embedded $d lt.eq 8$], [850], [4 250], [—], [0.0895], [0.0025], [+40%, 11× fewer],
  ),
  caption: [Held-out force RMSE (eV/Å) and energy RMSE (eV/atom) against MH-1 labels; 1k = 800/200 split, 4k = 3 200/800. "rel. to cat." compares with the categorical model at the same degree on 4k.],
) <tab-main>

#figure(
  placement: auto,
  image("fig_accuracy_vs_params.png", width: 92%),
  caption: [Test force RMSE against parameter count. Point labels are the polynomial degree. On the 4k set the degree-10 embedded $d lt.eq 16$ model (8 045 parameters) sits at the accuracy of degree-8 categorical (19 120).],
) <fig-acc>

Three things follow from @tab-main and @fig-acc. First, at matched degree the embedding costs 14–20% on forces (energies are unaffected: 2.2–2.7 meV/atom throughout), for 3–6× fewer parameters. Second, *lossless and $d lt.eq 16$ are indistinguishable at every degree* — truncating order 3 from 35 to 16 species directions costs nothing measurable at $S = 5$ — whereas $d lt.eq 8$, which also truncates order 2, costs about one degree. Third, per basis function the embedded model wins at every point: degree-10 $d lt.eq 16$ matches degree-8 categorical at 2.4× fewer parameters, and it was affordable (26 GB) where the categorical alternative needed a 376 GB node.

#figure(
  placement: auto,
  image("fig_learning_curves.png", width: 92%),
  caption: [Learning curves on the 1k set (test solid, train dashed). Degree 6 is basis-limited for both models; at degree 8 the categorical model is still data-limited (gap 0.022 at N = 800) while the embedded one has converged.],
) <fig-lc>

The learning curves (@fig-lc) explain the table. Both degree-6 models saturate — train and test meet — by 400 structures; the embedded model gets there by 200 and is the better model below that. At degree 8 the categorical model still has slope at 800 and duly gains 11% from 4× the data, while the embedded models gain 2%. *The embedded basis wants degree, not data*; the categorical basis wants data. This is the mechanism behind "accurate per structure vs accurate per parameter".

== The embedding's values do not matter at $S = 5$

A 2×2 of embedding source (MP-0-small, MH-1) × reduction (first-$d$ column truncation, PCA) at degree 6 gave test F within 0.001 eV/Å at $d lt.eq 16$ and 0.005 at $d lt.eq 8$, although the truncated tables' element-similarity structure correlates only 0.5–0.7 with the full tables'. At $S = 5$ the frozen embedding acts as a generic tensor reduction, consistent with Darby et al.'s random-projection result; transferred chemistry can only show once $d_max < S$ truncates order 1 itself. PCA remains the right construction (it was also what exposed that a DCT mixing frame spans only 9/15 and 13/35 of the species tensors — a genericity check now in the tests).

== Evaluation cost

#figure(
  placement: auto,
  image("fig_throughput.png", width: 78%),
  caption: [Throughput of the degree-6 $d lt.eq 16$ student on one host (lestrade: 32 cores, RTX 4000 Ada), 256–384-atom cells, f64 (solid) and f32 (hatched). The JAX kernel runs 32–38× the teacher in f32 and about 100× in f64; a single Julia core matches the teacher on a GPU. Julia is f64 only. Neither card is a production GPU: fp64 runs at 1/64 of fp32 on both the RTX 4000 Ada and the A4500.],
) <fig-thr>

The JAX evaluator's GPU kernel runs 32–38× MACE-MH-1's in f32 at 40–1 300 atoms (6–11× end-to-end with the host neighbour list) and *7.4–8.3 × 10#super[4] atom-steps/s in f64* — about 100× MH-1's f64 rate of 7.6–8.6 × 10#super[2], because the ACE kernel is gather-bound and barely notices the card's 1/64 fp64 rate while MACE's dense matmuls do; the same holds on the A4500 (4.8 × 10#super[4] f64 at 40 atoms). MH-1 does not fit a 3 000-atom cell on 20 GB in either precision. The Julia evaluator, however, ran the *embedded* model 1.8× slower than the categorical one despite 4× fewer basis functions, and both far below a well-written CPU kernel. Profiling found the cause: the splined radial tables are LEN wide (224 columns per edge for the embedded model), yet because the embedding weights make every $R_(n l)$ a scalar multiple of $P_(n')$, only 6–8 columns are distinct — the evaluator computes them all, as `SVector{LEN, Dual}`. A prototype exploiting the factorisation ($R_(n l) = c_(n,"pair") P_(n')$, the structure acejax already uses) with an allocation-free site loop is exact to $10^(-12)$ and 12–18× faster on four cores, above the JAX fp64 GPU kernel.

= Stage 2 proposal, for discussion

The plan (`docs/plans/jax_ace_port_plan.md`, Stage 2) has been revised against these results. Its shape, and the decisions it asks for:

*2A — linear fit at scale: an open choice.* Two routes. (i) *Stay in Julia*: assembly and solve where the model, the priors, the parameter-gradient rrules and the data path already live — a single source of truth — with *phase 18* (row-block assembly feeding TSQR, which still yields $R$ for BLR, or ScaLAPACK on the same operator per the `gap_fit` recipe; LSQR beyond the $R$-factor floor) for what no single node holds. (ii) *Port assembly to JAX*: the design matrix built on the GPU through the same differentiable descriptor the evaluator uses, so fitter and evaluator cannot diverge, with the solve in whichever runtime holds $A$. The case for (ii) was strongest when Julia assembly was 100× slower than its arithmetic warranted; the performance branch removed that premise — the 151 GB degree-10 matrix now assembles in 25 min on CPU, and basis pre-training (2B) needs nothing Julia lacks. What remains for (ii) is real but smaller: GPU bandwidth (~10× on a gather-bound assembly, not 100×), the shared descriptor, and MACE's own move to JAX. Sizing either way: at 4 000 structures the categorical degree-12 matrix is 330 GB and needs phase 18 on any route; the embedded one is 50 GB. Rust and PyTorch were assessed and set aside; the plan records the reasoning. _Decision sought: (i) or (ii) — or (i) now with (ii) held open, using the S = 25 gate experiment (a ≈150 GB assembly) as the test of whether the Julia route suffices at the scale that matters._

*2B — basis pre-training, then convex fit.* Everything non-linear (radial mixing $W_(n l q)$, the element embedding $E$, Finnis–Sinclair densities as linear combinations of the order-1 basis) is learned once on a subset by Kaufman variable projection — the gradient at fixed $c^*$, which the existing `grad_params` / `pullback_2_mixed` / `energy_forces_virial` rrules already provide in Julia — then frozen, and the coefficients are fitted convexly at scale. The spike passed its gate: learned densities beat hand-chosen ones with fewer columns and transfer frozen to a disjoint split. This is the GRACE-FS / PACE function class split into two stages; pacemaker, gracemaker and MACE all learn their radials, and ACEpotentials' fixed heuristic radials are the outlier. _Decision sought: make pre-trained radials and densities the default path rather than an option; two Julia items to build (a structured $W_(n l q)(E)$ and the FS term at the hook already marked in `ace.jl`)._

*2C — the convexity demonstration against GRACE-FS.* Neither pacemaker (linear, categorical) nor gracemaker (embedded, gradient-descent) fits a tensor-reduced ACE convexly; that is the differentiator, and it has to be shown on the data the competition used — HEA25S, teacher GRACE-2L-OMAT (public), their extended-distillation recipe. Five measurable benefits: calibrated UQ for free (BLR, and Swinburne–Perez POPS for the misspecified regime a distilled student is always in) against a GRACE-FS ensemble; exact D-optimal data selection (error vs $N$, selected vs random); one-shot deterministic fits with closed-form $lambda$; delta-learning on DFT as one more linear solve against GRACE's fine-tuning curve; and hard physical constraints as convex QPs (equilibrium, elastic constants, phase ordering, repulsion). _Gate first_: accuracy at matched MD cost at $S = 25$ on a 5 000-structure subset (≈150 GB, fits the 376 GB node today) — the first regime where $d_max < S$ and the embedding's values can matter. If the linear model does not reach GRACE-FS accuracy there, items 1–5 are wins on a worse model. _Decision sought: approve the gate experiment as the next substantial run, ahead of phase 18._

*Upstream items.* Multi-tangent `pushforward!` for `PooledSparseProduct` and `SparseSymmProd` in EquivariantTensors (replacing the three interim kernels in `basis_ed.jl`; ET's own commented-out `_pfwd` is most of it); `SparseSymmProdDAG` for the CPU evaluator; ACEfit's per-task serialisation, `GC.gc()` and `Array(A)`.

= Recommendations

+ *Use the embedded model where parameters, memory or many elements are the constraint, and give it one more degree than the categorical model would get.* Degree-10 $d lt.eq 16$ is the current best trade (0.0767 eV/Å, 8 045 parameters, 26 GB matrix on 4k). Where accuracy per structure is all that matters and a large node is available, categorical degree 10 (0.0641) still wins; categorical degree 12 (330 GB) needs the distributed assembly of Stage 2 phase 18, embedded degree 12 (≈50 GB) does not.
+ *Keep $d_max = 16$; do not use $d_max = 8$* (it truncates order 2 and costs a full degree).
+ *Treat the embedding source as immaterial at $S lt.eq d_max$*, and test the transferred-chemistry claim only at $S = 10$–25, where the GRACE-FS distillation on HEA25 is the comparison to beat.
+ *Code changes, as three stacked pull requests against `main`* (branches listed in @tab-prs). (a) `fix/basis-ed-performance`: the design-matrix path — `evaluate_basis_ed` replaced by a type-stable pushforward Jacobian and the boxed accumulation in `energy_forces_virial_basis` removed — 174–218× per structure, exact to $10^(-12)$; and the forward path — factorised radial spline tables with a plain-Float64 kernel, an allocation-free site loop with $w_(A A) = A 2 B^T W_B$ folding, neighbour-list reuse — 9–12× on four threads, exact to $10^(-12)$; 1 300 existing tests plus 471 new ones pass. The 151 GB degree-10 assembly above took 25 min on it instead of a projected 14 h. Its three pushforward kernels reimplement EquivariantTensors' product structure and should move upstream as multi-tangent `pushforward!` methods (a follow-up PR to EquivariantTensors); the ACEfit follow-ups (per-task model serialisation, `GC.gc()` per task, the `Array(A)` copy) are listed in the PR text. (b) `pr/jax-port`: the JAX evaluator, exporter, LAMMPS bundle, benchmarks and the CI matrix that re-exports fitted models from the ACEpotentials under test and requires the evaluator to reproduce them. (c) `pr/element-embeddings`: everything in this report — `ace_embedding_model`, the PCA reduction with its rank check, the factorised radial export, the distillation pipeline, the spikes and the findings.
+ *After the radial fix, revisit the recursive (DAG) AA products for the CPU evaluator.* The Phase-8 spike (`acejax/spike_recursive/`) measured subproduct sharing at *0.99×* on energy+forces for the JAX GPU evaluator in f64 (1.19× in f32; the reverse pass must materialise ≈25 000 intermediate columns, and AA is only 18% of `site_basis` on the GPU once measured against an oracle rather than in isolation — so it is a dead end there), but *2.30×* on CPU, where AA is 75% of the energy+force cost. With the radial splines no longer dominant in Julia, the AA products become its largest stage, which is the CPU regime the spike measured; EquivariantTensors already provides `SparseSymmProdDAG`, so the trial is a swap of the tensor's `aabasis` on the fixed kernels.
+ *Retire the design-matrix cache* (it exhausted the storage quota at 200 GB; assembly is now minutes) and *replace `svd(R)`* in the factor-once solver with a Cholesky per $lambda$ or a threaded SVD — at $n = 46 685$ it is the remaining post-assembly cost.
+ *Generator before basis.* Bounding the perturbations halved the error; adding short-range order and the unary/binary structures of GRACE's "extended distillation" are the next data-side steps and cost minutes of GPU.
+ *Next accuracy lever, convexly:* fixed $sqrt(rho)$ Finnis–Sinclair columns with learned (VarPro) species weights inside the density gave $-12%$ at degree 4 and transfer frozen to a disjoint split; the shape of the embedding function beyond $sqrt(dot)$ does not matter. This is the like-for-like answer to GRACE-FS and should be included in any $S > 5$ run.

#v(0.5em)
#text(size: 8.5pt)[Sources: `acejax/bench/distil/RESULTS.md`; `docs/findings/FINDINGS_assembly_profile.md`, `FINDINGS_forward_profile.md`, `FINDINGS_fs_spike.md`, `FINDINGS_varpro_spike.md`, `FINDINGS_codes_survey.md`; plan `docs/plans/jax_ace_port_plan.md` (Stage 1D–E, 2B–2C).]
