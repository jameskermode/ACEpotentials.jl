# Language and algorithm choices in pacemaker and gracemaker (surveyed 2026-09-13)

Sources: `ICAMS/python-ace` and `ICAMS/grace-tensorpotential` repositories
(languages, trees, `pyproject.toml`, `tensorpotential/potentials/presets.py`,
`tensorpotential/tensorpot.py`, `cli/train.py`, `cli/distribute.py`,
`export.py`), the pacemaker and gracemaker docs, and the PACE (npj 2021),
efficient-parametrisation (PRM 2022), active-learning (PRM 2023), GRACE
(PRX 2024) and GRACE foundation-model (npj 2026) papers.

## Stack

| | pacemaker (python-ace) | gracemaker (grace-tensorpotential) |
|---|---|---|
| Evaluator | **C++** `ace-evaluator` (1.9 MB C++, the bulk of the repo), Cython bindings (`pyace`); LAMMPS `pair_style pace` + Kokkos | GRACE-FS: **C++**, CPU/MPI, in `python-ace` (`feature/grace_fs`); GRACE-1L/2L/3L: LAMMPS **Kokkos** pair styles reading `.npz` weights (no TF at runtime); or TF SavedModel |
| Fitter | Python, **TensorFlow** (`tensorpotential`), autograd for forces/virials | Python, **TensorFlow ≤ 2.20** + `tf_keras` (legacy Keras API), `tf.function(jit_compile=True)` → XLA; no JAX/PyTorch |
| Model definition | YAML potential spec → C++/TF | an **instruction-graph DSL** (`TPInstruction` subclasses: `RadialBasis`, `LinearRadialFunction`, `SphericalHarmonic`, `ScalarChemicalEmbedding`, `SingleParticleBasisFunctionScalarInd`, `ProductFunction`, `FunctionReduce`, `FSOut2ScalarTarget`) |
| Optimisers | **BFGS / L-BFGS-B** (scipy, full-batch, TF gradients) — the default; Adam | **Adam** (batched, XLA) and **BFGS** (scipy `minimize` over a `distributed_bfgs_train_step`); early stopping; LR schedules |
| Multi-device | single GPU | `MultiWorkerMirroredStrategy` with NCCL forced, also on a single host |
| Data pipeline | pandas pickles | sharded/streaming datasets, `dense_nbr` option, per-structure weighting |
| Export | YAML (`.yace`) for the C++ evaluator | `.tdyaml` for GRACE-FS C++; `.npz` for Kokkos; SavedModel |
| Linear / convex mode | **none** — even `fs_parameters: [1,1]` is fitted by BFGS with learned radials | **none** — the `LINEAR` preset has a linear readout but learned radials and embedding, trained by gradient descent |

## Model construction (from `presets.py`)

- **Radial**: a fixed base (`SBessel`, `n_rad_base = 8`) mixed by a learned
  linear map into `n_rad_max = 24` functions per `l` (`LinearRadialFunction`,
  `crad_init="random"`). This is exactly our `Wnlq`.
- **Species**: `ScalarChemicalEmbedding(embedding_size)` — a **learned** table
  `z(μ)` of size 32 (LINEAR), 64 (FS), 72 (FS large), 128 (1L/2L). The
  one-particle basis is `A = Σ_j R_nl(r_ij) Y_lm(r̂_ij) z_j`
  (`SingleParticleBasisFunctionScalarInd`, "indicator = z"). Radials are
  **not** per element pair. This is the Darby tensor-reduced construction; ours
  is the same object with `z` frozen (PCA of a foundation-model table) or, in
  Stage 2B, learned.
- **Products**: `ProductFunction` forms `A⊗A` (`AA`), `AA⊗A` (`AAA`),
  `AA⊗AA` (`AAAA`) with Clebsch-Gordan coupling, `lmax`/`Lmax` pruning and
  `keep_parity`; then `FunctionReduce` mixes the scalar (`L=0`) outputs with a
  learned map that is **central-atom-type dependent**. So the many-body
  basis is a *dense* CG product tree with learned channel mixing, not a
  sparse symmetrised basis with a fixed coupling matrix. Their "channel"
  index (`n_rad_max`) plays the role of our folded `(n', k)` radial index.
- **FS readout**: `fs_parameters = ((1,1),(1,0.5),(1,2),(1,0.75))` — four
  densities `φ_p = Σ c_p·B`, energy `Σ_p a_p φ_p^{m_p}` with exponents
  1, ½, 2, ¾ (`FSOut2ScalarTarget`). Densities are learned linear
  combinations of the basis; the exponents are fixed. An MLP embedding
  (`mlp_embedding`) exists but is "not yet supported for export". GRACE-1L/2L
  replace FS with an MLP readout over `n_mlp_dens` densities.
- **Cutoff**: one `rcut` (6 Å default) with optional per-bond `cutoff_dict`;
  radial basis with a `p=5` polynomial envelope; `avg_n_neigh` normalisation.

## Fitting algorithms

- **pacemaker**: loss `(1-κ)·E + κ·F` with `kappa: auto` from the data
  spread; L1/L2 on coefficients; **BFGS** full batch is the recommended route
  ("non-linear optimisation greatly benefits from autogradients"), batch size
  100–1000 with automatic reduction; **ladder fitting** — grow the basis in
  steps (`ladder_step`, by body order or power order), warm-starting each
  stage from the last; **active learning** via **MaxVol D-optimality on the
  B-basis values** (`pace_activeset`; extrapolation grade γ from the active
  set; PRM 2023). Note the active-set machinery is *linear* in the basis even
  for the non-linear FS model — it linearises. A convex fit gets the exact
  version (Fisher information `AᵀA` of the actual model).
- **gracemaker**: Adam with XLA-jitted train step, optional BFGS; multi-GPU
  via MWMS/NCCL; fine-tuning presets (`-ft`), LoRA (`functions/lora.py`);
  distillation in the foundation-model paper is plain re-training of FS on
  teacher labels (`compat/pace` supplies the FS-compatible path). Uncertainty:
  a GMM-based "uncertainty score" exported with the model (README release
  note: "carried along, so it's available during LAMMPS runs").

## What this means for us

1. **Same function class, different fitting philosophy.** GRACE-FS =
   learned radial mixing + learned chemical embedding + learned densities
   with fixed exponents + linear readout per central species. Our
   Stage 2B (VarPro pre-training) + convex fit is that class split into
   "learn the basis on a subset, then solve". The exponent set
   `{1, ½, 2, ¾}` is what spike 1b tested with the density fixed: the extra
   exponents bought ≤ 1% — the value is in the learned densities, which is
   consistent with their design putting the learnable part there.
2. **No convex mode anywhere in either code**, and no calibrated-posterior
   UQ; their UQ is a GMM density score, their data selection a linearised
   MaxVol. Those are the Stage 2C gaps.
3. **Everyone learns the radials.** Pacemaker, gracemaker and MACE all fit
   `Wnlq`-equivalents by gradient descent; ACEpotentials' fixed
   ACE1-heuristic radials are the outlier. That is an argument for making
   the VarPro pre-training the default path rather than an option.
4. **Their evaluator/fitter split is ours.** C++/Kokkos evaluator with a
   YAML/npz export, TensorFlow fitter — the same division as
   Julia-or-JAX fitter / acejax-LAMMPS evaluator, with the same divergence
   risk handled by an export contract and (in our case) CI.
5. **TensorFlow is a liability for them** (`tensorflow<=2.20`, legacy Keras
   flag, CUDA ≥ 12.8 for Kokkos builds, a pinned memory-leak version). Their
   dense CG product tree is XLA-friendly; our sparse symmetrised basis is the
   gather/scatter-bound path (Phase 15). Neither is BLAS-bound in assembly.
