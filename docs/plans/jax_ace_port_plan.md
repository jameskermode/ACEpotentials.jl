# Plan: Minimal Python + JAX Port of the ETACE Descriptor and Linear Fit

**Status**: 🚧 Stage 1 in progress. Phase 0 complete (all three gates resolved).
Phases 1, 3 and 4 complete (`89901596`, `20282a30`): a fitted `ace1_model` exports and
reproduces energy, forces **and virial** from an ASE `Atoms` object to ~1e-12, via
`matscipy-neighbours`, with an ASE calculator. 17 tests. Remaining: LAMMPS export.
Stage 1 (fit in Julia, evaluate in JAX + LAMMPS) is a supported stopping point.

**Date**: 2026-09-09

**Branch**: TBD

## Overview

Port the core ETACE descriptor and linear-fit functionality to Python + JAX, using
Equinox and the JAX ecosystem (lineax, optimistix). Targets GPU-accelerated
evaluation, compatibility with `lammps-jax` for LAMMPS export, GPU fit assembly,
and a path to non-linear fits.

**The work splits into two stages, and Stage 1 is a legitimate place to stop.**
Stage 1 fits in Julia and evaluates in JAX — which also buys LAMMPS. Stage 2 moves
fitting into JAX as well. See "Staging" below.

This is *not* a Python convenience wrapper. Because JAX supplies graph batching,
autodiff and GPU as framework properties rather than as separate migrations, this
port lands close to where Phases 1–4 of the Julia EquivariantTensors roadmap
(see `CLAUDE.md`) are trying to get. It does not replace that work: Julia users
still need it, and the port depends on Julia to *generate* the O3 coupling
coefficients.

### Goals

**Stage 1 — fit in Julia, evaluate in JAX**

- Site descriptor `𝔹` numerically equivalent to `ACEpotentials.ETModels`, for
  **both** `ace1_model` (splined radials, spherical harmonics) and `ace_model`
  (analytic radials, solid harmonics)
- Energy, forces and virial from a *fitted* Julia model, via `jax.grad`
- GPU-ready throughout; `lammps-jax` export compatibility
- Architecture that admits Stage 2 and non-linear fits without rework

**Stage 2 — fit in JAX**

- Energy / force / virial design-matrix assembly
- Solve via `lineax`; smoothness priors preserved
- Data loading, per-observation weights

### Non-goals (explicitly out of scope)

- L ≠ 0 equivariants (the `sparse_equivariant_tensors` LL path) — L = 0 only
- Porting O3 coupling-coefficient generation (see "Export, don't port")
- Committees, splinified radials, `fast_evaluator` (already broken in Julia)
- ASP / LSQR solvers, repulsion restraints, ZBL
- Non-linear *training* (architecture must permit it; implementation deferred)

## Staging

### Stage 1 — fit in Julia, evaluate in JAX (and LAMMPS)

Fit with `acefit!` as today, export the *fitted* model, and evaluate it in JAX.
This is a complete, useful deliverable on its own: GPU-accelerated MD in LAMMPS
for ACE potentials fitted by the existing, trusted Julia pipeline.

**What it needs:** the exporter, the descriptor forward pass, energy/forces/virial
via `jax.grad`, the neighbour-list adapters, an ASE calculator, and the LAMMPS
export.

**What it does not need — and this is the point:**

- No design-matrix assembly, so **the Jacobian problem disappears**. Forces are one
  VJP over a scalar energy, ~2–3× the forward pass. The `jacrev` cost that
  dominates the Stage 2 risk register simply does not arise.
- No `lineax` / `optimistix` solvers, no smoothness priors, no dataset loading, no
  per-observation weights, no training-shaped minibatching.
- f64 matters much less: MD runs f32, so the GPU precision gate stops being a
  blocker and becomes a note.

Roughly **15–18 days**, against 22–27 for the full scope.

### Stage 2 — fit in JAX

Adds fitting. **Revised after Stage 1** (see "Stage 2 — revised" below for the
reasoning and the costed plan): Stage 2 now splits in two, and the recommended
order is the reverse of the original.

- **Stage 2A — non-linear / gradient-based fitting.** Batched loss over
  configurations, `optax`, checkpointing. Builds directly on Stage 1's validated
  energy-and-forces path and needs *no* design matrix, so it avoids the entire
  Jacobian problem. It is also the thing the Julia pipeline cannot currently do
  at all.
- **Stage 2B — linear fit in JAX.** Design-matrix assembly with the committed
  hybrid JVP, priors, solvers, data loading. GPU-only to be worth doing, and
  exposed to the XLA autotuning miscompile.

2A was previously filed as an optional afterthought to 2B. Stage 1's
measurements invert that: 2A is cheaper, lower-risk, and delivers a capability
that does not exist today, whereas 2B reimplements something Julia already does
correctly and only pays off on a GPU with large datasets.

### Does Stage 1 foreclose anything?

No, provided the three decisions below are respected from the start — they cost
~0.5 day and are mostly discipline:

| Decision | Stage 1 cost | Why it matters later |
|---|---|---|
| Keep `Wnlq` live, not folded into the spec | one `einsum` | Non-linear fits need it trainable |
| Edge-vector core, not positions+cell | none (forced by LAMMPS anyway) | All three nlist paths share one core |
| Fixed-capacity + mask everywhere | none (forced by export anyway) | Stage 2 training batches are the same idea |

Bucketing is the one item Stage 1 exercises only partially: export needs a single
fixed capacity, whereas training needs several buckets. That is an extension of the
same mechanism, not a redesign.

## Phase 0 — Performance spike (1–2 days, gating)

Everything downstream assumes JAX is competitive for this workload. That
assumption is unverified and cheap to test. **Do this first; the result may
change the project's shape.**

Measurements 1 and 2 gate Stage 1. Measurement 3 gates Stage 2 only — if you
intend to stop at Stage 1, it is informational and the spike shrinks to ~1 day.

Implement `A` → `AA` → `𝔹` for one fixed Si model (`ace1_model(elements=[:Si],
order=3, totaldegree=10, rcut=5.5)`). No fitting, no export, no neighbour list —
feed it a precomputed edge list from Julia. Measure against `benchmark/common.jl`
on identical systems.

### Measurements

| # | Measurement | Julia baseline |
|---|---|---|
| 1 | Site descriptor, CPU, 512 atoms | `benchmark/bench_forces_regression.jl` |
| 2 | Site descriptor, GPU, f32 **and** f64 | `benchmark/gpu_benchmark.jl` |
| 3 | Design-matrix assembly: naive `vmap`-VJP **vs** hand-written JVP | `site_basis_jacobian` (`src/et_models/et_ace.jl:72`) |

### Decision gates — RESOLVED

- **(3) the custom Jacobian is mandatory**, on both CPU and GPU. Naive `vmap`-VJP
  vs a hybrid pushing analytic `dA/dr` through `dB/dA`, as multiples of the forward
  pass:

  | | naive | hybrid | apart |
  |---|---|---|---|
  | CPU f64 | 359–1409× | 43–238× | 6–8× |
  | GPU f64 | 70–255× | 9–20× | 7.5–13× |
  | GPU f32 | 92–914× | 8–39× | 12–23× |

  Nowhere near the "within ~3× → ship it" criterion in any configuration. The two
  agree to 6.2e-16 in f64, so the hybrid is correct, not an approximation. The
  ~100 LOC custom JVP is a **committed Stage 2 deliverable**, not a contingency.

  **But the hybrid alone is not enough on CPU.** Absolute Jacobian times (ms),
  same machine:

  | atoms | Julia CPU | JAX CPU hybrid | JAX GPU f64 hybrid |
  |---|---|---|---|
  | 64 | 14.13 | 16.04 | 1.62 |
  | 512 | 129.36 | 195.40 | 10.90 |

  On CPU the JAX hybrid is **1.5× slower than Julia's `ET._jacobian_X`** at 512
  atoms — the forward-pass advantage does not carry over. Only the GPU wins, and it
  wins decisively (**11.9× faster than Julia CPU**).

  **Consequence for Stage 2: fit assembly must target the GPU.** On CPU it would be
  slower than simply assembling in Julia, which removes the reason to port it at
  all. This does not affect Stage 1, whose forces are a single VJP over a scalar
  energy.
- **(2) f64 on GPU is cheap — the plan's concern was wrong.** Measured on an
  NVIDIA RTX 4000 Ada (compute 8.9), a *workstation* card whose FP64 arithmetic
  rate is 1/64 of FP32 — i.e. the worst realistic case.

  | atoms | GPU f32 | GPU f64 | f64/f32 |
  |---|---|---|---|
  | 64 | 0.055 ms | 0.167 ms | 3.0× |
  | 512 | 0.103 ms | 0.560 ms | 5.4× |

  **f64 costs 3–5.4×, not 32–64×.** The descriptor is gather- and
  memory-bandwidth-bound, not FP64-FMA-bound, so the crippled arithmetic rate
  barely registers. Consequence: **f64 GPU fit assembly is viable**, and the
  precision split is a preference rather than a constraint. The GPU hybrid
  Jacobian in f64 runs 10.91 ms at 512 atoms against Julia's 71.23 ms on CPU.

  A 5.3e-1 discrepancy seen while measuring turned out **not** to be roundoff. See
  `spike/jax_phase0/FINDINGS_f32.md`; two independent effects, both material:

  **TF32 is on by default** for f32 matmuls on Ada (compute 8.9). Setting
  `jax_default_matmul_precision=highest` improves the forward descriptor from
  1.17e-3 to 2.93e-6 vs Julia — a 400× accuracy difference. **A 1.2e-3 relative
  error in the descriptor is not acceptable for a production potential**, so
  Stage 1 must pin matmul precision explicitly and verify the setting survives
  `jax.export` into the lammps-jax bundle.

  **XLA GPU autotuning selects a wrong kernel in ~30% of processes**, producing a
  structurally wrong Jacobian element (-0.023 where the true value is -19.159).
  Not a precision failure — the terms at that element sum with
  `sum|t|/|sum t| = 1.002`, i.e. no cancellation. It vanishes with
  `--xla_gpu_autotune_level=0` (10/10 clean), persists at precision=highest
  (2/10 wrong), and is confined to **f32 and the Jacobian einsum only**: the
  forward descriptor (20/20) and forces via `jax.grad` (20/20) are stable across
  processes, and f64 is clean (10/10).

  So Stage 1 is exposed to the TF32 issue but not the autotuning bug. Stage 2 is
  exposed to both, since fit assembly *is* the Jacobian — a third independent
  reason to assemble in f64.
- **(1) JAX is faster than Julia on CPU, contrary to the original estimate.**

  All figures below are **same-machine** on lestrade (24-core Xeon-class + RTX 4000
  Ada), ACEpotentials 0.10.2 via `Pkg.develop` so the Julia code is identical to the
  dev checkout, Julia single-threaded, f64 unless noted.

  **Forward descriptor (ms):**

  | atoms | Julia CPU | JAX CPU | JAX GPU f64 | JAX GPU f32 |
  |---|---|---|---|---|
  | 64 | 1.371 | 0.315 | 0.175 | 0.054 |
  | 512 | 11.555 | 0.992 | 0.556 | 0.102 |

  JAX CPU beats Julia CPU by **4.4× / 11.6×**, confirming the earlier Apple Silicon
  result (4.0× / 9.2×) on entirely different hardware. The plan originally predicted
  Julia would win by 2–5×; that was wrong, and it is now wrong on two architectures.

  Caveats that remain: this is the ET descriptor path, not the tuned classic *force*
  path, and the model is small (n_B = 110, single species). Re-check on a production
  multi-element model before anything load-bearing rests on the magnitude.

  Incidentally, JAX CPU shows no meaningful gain from multiple threads here (0.992
  vs 1.005 ms at 512 atoms with Eigen multithreading disabled), so the comparison
  against single-threaded Julia is fair.

### Prior evidence (corroborated)

`benchmark/FORCE_REGRESSION_FINDINGS.md` records that ET autograd forces cost
**~2.1× their own energy evaluation** — the same regime `jax.grad` occupies. And
the ET kernels are already written in JAX style: `sparsesymmprod_ka.jl` does a
direct gather-product-write per basis function, batched over nodes, *without*
using the subproduct-reuse DAG in `symmprod_dag.jl`. The comparison is therefore
codegen and memory traffic, not algorithm.

## Architecture

### Core principle: the model takes edge vectors

The descriptor core accepts **edge vectors + species + segment ids**. Never
positions and a cell. Three adapters feed it:

```
matscipy-neighbours ─┐
lammps-jax nlist ────┼──▶ (rij, zi, zj, segment_ids, mask) ──▶ ETACE core
LAMMPS (via export) ─┘
```

This is forced by the `lammps-jax` contract, which has **no cell and no shift
vectors** — ghost atoms carry periodicity. It also matches what ETACE already
does: `ETGraph.edge_data` holds `𝐫` per edge (`src/et_models/convert.jl:71-77`,
`xij = (𝐫, z0, z1)`). Preserve that factoring; it is what makes all three paths
work from one core.

Consequence: the strain-derivative scaffolding for virials lives in the *fitting*
adapter, not the core.

### Components

| Stage | Component | Julia source | Python | Est. LOC |
|---|---|---|---|---|
| 1 | Model export (fitted) | new (Julia side) | — | ~150 |
| 1 | Loader + spec dataclasses | — | `specs.py` | ~150 |
| 1 | Agnesi transform + envelope | `ET/src/transforms/agnesi.jl` | `radial.py` | ~30 |
| 1 | Orthogonal polys (3-term rec.) | `P4ML/src/orthopolybasis.jl` | `radial.py` | ~20 |
| 1 | Solid harmonics | SpheriCart via P4ML | `sphericart` JAX binding | dep |
| 1 | `Rnl = SelectLinL(env · P)` | `src/models/Rnl_learnable.jl` (296) | `radial.py` | ~30 |
| 1 | `A` pooling | `ET/src/ace/sparseprodpool.jl` (693) | `pool.py` | ~40 |
| 1 | `AA` sparse symm. product | `ET/src/ace/sparsesymmprod.jl` (426) | `ace.py` | ~25 |
| 1 | `𝔹 = A2B · AA` | sparse matmul | `ace.py` | ~5 |
| 1 | Pair basis + one-body | `et_pair.jl`, `onebody.jl` | `pair.py` | ~50 |
| 1 | Neighbour-list adapters | `ET.Atoms.interaction_graph` | `nlist.py` | ~80 |
| 1 | Energy/forces/virial + ASE calculator | `et_calculators.jl:155` | `calculator.py` | ~80 |
| 1 | LAMMPS export adapter | — | `lammps.py` | ~100 |
| 2 | E/F/V design matrix | `et_calculators.jl:246-330` | `assemble.py` | ~120 |
| 2 | Smoothness priors | `src/models/smoothness_priors.jl` (134) | `priors.py` | ~40 |
| 2 | Solvers | ACEfit (1162) | `solve.py` (lineax/optimistix) | ~150 |
| 2 | Data loading | `src/atoms_data.jl` (457) | adapted from mace-jax | ~120 |

Roughly 750 LOC of core Python for Stage 1, ~1200 for both.

### Export, don't port: the coupling coefficients

`ET.O3` (`O3.jl` 454 LOC + `O3_utils.jl` + `PartialWaveFunctions` + `RepLieGroups`)
computes generalized Clebsch–Gordan coefficients, real-SH transformations, and
rank-N permutation-invariant pruning. Porting it is 2–4 weeks with a long tail of
convention bugs. **Do not.**

Everything the model needs is plain data. Write a Julia exporter (~150 LOC, 1–2
days) emitting JSON or npz. For Stage 1 it carries the *fitted* coefficients, so
the same schema serves both stages — Stage 2 simply ignores the fitted `W`:

- element list, `E0`s
- Agnesi params per species pair (7 floats each, from `ET.agnesi_params`)
- **radial basis, one of two branches:**
  - *analytic* — poly recursion coefficients `A, B, C` (`OrthPolyBasis1D3T` is three
    vectors) plus `Wnlq`
  - *splined* (**the Stage 1 default**) — cubic spline knots and coefficients
- `Rnl_spec`, `Ylm_spec`, `Aspec` (index pairs), `𝔸spec` (index tuples by order)
- **`radial_kind`, `pair_radial_kind`, `pair_envelope_kind`** — **the branch is
  per-basis, not per-model.** Phase 7 found `ace_model` is a *mixed* case: its
  many-body `rbasis` is a `LearnableRnlrzzBasis` with a live `Wnlq`, but its
  **pair basis is still splined** (`src/models/ace_heuristics.jl:213`, when
  `!pair_learnable`), and its pair envelope is `PolyEnvelope1sR` — a genuinely
  *different formula* from `ace1_model`'s `ACE1_PolyEnvelope1sR`, not a
  reparameterisation. The two are separate structs with separate `evaluate`
  methods (`src/models/radial_envelopes.jl:4` and `:28`); the ACE1 form carries
  an extra linear term and normalises by `r0`. Export all three independently.
- **`ybasis_kind`** — `:spherical` or `:solid`. **Not a constant; must be exported,
  never assumed.** `ace1_model` passes `Ytype = :spherical`
  (`src/ace1_compat.jl:407`) while `ace_model` defaults to `:solid`
  (`src/models/ace_heuristics.jl:149`). Phase 0's spike used `ace_model` and so
  validated the *solid* path; the production path is spherical. Phase 1 hit this
  immediately — the Ylm probe showed an exact `r^l` ratio — and the exporter now
  detects and records it.
- `A2Bmap` as sparse rows/cols/vals
- readout `W`, `nnll` spec (for priors)
- pair-basis parameters, `rcut`

**Use npz, not JSON.** Julia serialises matrices column-major, so 2-D arrays arrive
transposed through JSON. (Phase 0 used JSON with explicit 1-D flattening; npz avoids
the whole class of bug.)

#### Splined radials — the Stage 1 default

`ace1_model` splinifies the radial basis (`src/ace1_compat.jl:283`), and
`convert2et` only dispatches on `LearnableRnlrzzBasis` (`src/et_models/convert.jl:180`),
so a fitted production model cannot go through the analytic path as written. Phase 0
worked around this by using `ace_model`; Stage 1 must handle the splined case,
because fitted models are the ones worth exporting.

The splining is narrow, which keeps this cheap. `SplineRnlrzzBasis` retains
`transforms` and `envelopes` analytically and splines only `Wnlq · polys(y)` as a
function of `y`, on a **uniform grid over [-1, 1]** with `Nspl = 30` (see
`src/et_models/splinify.jl`: `P4ML.splinify(y -> WW[:,:,i] * polys_y(y), -1.0, 1.0, Nspl)`).
So the schema needs one extra branch and nothing else changes.

**Decision: Stage 1 exports splines.** Two reasons. It reproduces what Julia
actually evaluates, so the JAX and Julia models agree bit-for-bit rather than
differing by the spline approximation error — which is what you want when the export
target is LAMMPS. And uniform-grid cubic evaluation in JAX is a floor, a gather of
four coefficients and a cubic: likely *faster* than the 15-term recursion, and more
GPU-friendly. Estimated ~0.5 day.

The analytic branch stays in the schema for Stage 2, where a trainable `Wnlq` is
required and splines are not differentiable w.r.t. the parameters that generated them.

**`mace_jax/tools/cg.py::U_matrix_real` is not a shortcut.** It builds coupling in
the e3nn `Irreps` layout (channel-wise, irrep-decomposed), not ACE's sparse
`A2Bmap` over an explicit (n,l,m) product spec with PI pruning. Conventions differ
— `cg.py:31-33` documents a global sign discrepancy in e3nn-jax's real CG basis,
which is why they fall back to torch's `e3nn.o3.wigner_3j`. It also drags in torch
and cuequivariance. Keep it as an independent cross-check only.

### Neighbour list

**`libAtoms/matscipy-neighbours`** (MIT) is the primary backend: drop-in `"ijdDS"`
API, OpenMP CPU core, CUDA/HIP GPU backend, zero-copy to JAX via DLPack with
`array_namespace=jax.numpy`. Three properties matter here:

- returns `D == r[j] - r[i] + S @ cell` directly — the edge vector the core wants
- pairs sorted by `i` — matches `ETGraph`'s requirement (`ET/src/embed/graph.jl:26`)
  and enables `segment_sum(..., indices_are_sorted=True)`
- `neighbour_matrix` gives the dense `(n, K)` format with a `count` mask

That last point is a genuine alignment: ET's internal layout is already
`(maxneigs, nnodes, nfeat)` (`reshape_embedding`; `sparseprodpool_ka.jl` documents
`BB[t] = #neighbours × #nodes × #features`). In the dense format the `A` pooling is
a masked sum over the neighbour axis with **no scatter at all**.

But `lammps-jax` is sparse (senders/receivers/edge_mask). So **write the pooling as
one swappable function** over `(edge_features, segment_ids_or_counts, mask)` with
dense and sparse implementations. Everything downstream — `AA`, `A2B`, readout — is
per-node and layout-agnostic.

Install friction: version 0.1.0, repo-only, not on PyPI. CPU via scikit-build-core;
GPU is a separate documented build (`-Dcmake.define.ENABLE_CUDA=ON`).

### Framework

Equinox. `eqx.Module` for `ETACE`, `RadialBasis`, `PairModel`, `OneBody`; all specs
and index arrays as `eqx.field(static=True)`; `Wnlq`, `W`, `E0s` as array leaves.
The Lux `ps`/`st` split that `convert.jl` and `et_calculators.jl` thread by hand
collapses into one PyTree.

Equinox suits the export contract better than Flax NNX: `lammps-jax` wants a plain
traceable function, and mace-jax reaches that via `nnx.merge(graphdef, params)` plus
a hand-rolled pass to strip string metadata out of the param tree
(`lammps_mliap_mace.py:69-80`). A fitted ACE model has no training-time state, so
`eqx.combine` (or simply closing over the model) is enough.

- **lineax** — `lx.QR()`, `lx.SVD()` cover ACEfit's `QR`, `RRQR`, `TruncatedSVD`
- **optimistix** — BLR evidence maximization. ACEfit's
  `bayesian_linear_regression_svd` (`bayesianlinear.jl:427`) is an SVD followed by a
  2-parameter minimization of the log marginal likelihood under `Optim.jl`; a direct
  substitution, ~40 LOC.

## Reuse from mace-jax (MIT, vendor with attribution)

Do **not** import `mace_jax` as a library — `setup.cfg` pulls torch, `e3nn==0.4.4`,
cuequivariance and cuequivariance-jax transitively. Vendor ~300 LOC:

- **`modules/utils.py::compute_forces_and_stress`** (lines 44–108) — the symmetric
  displacement scheme, applying strain to positions *and* cell (hence edge shifts),
  then `value_and_grad` at zero displacement. Already validated against torch-MACE.
  Swap `energy_fn` for the per-species descriptor sum.
- **`tools/scatter.py::scatter_sum`** — carries the `indices_are_sorted` /
  `unique_indices` hints that matter for `A` pooling.
- **`data/utils.py` config plumbing** — `Configuration` (with per-observation
  `energy_weight` / `forces_weight` / `stress_weight`), `load_from_xyz`,
  `AtomicNumberTable`, `compute_average_E0s`. Maps onto `src/atoms_data.jl` ~1:1.
- **`data/neighborhood.py`** — superseded by matscipy-neighbours, but useful as a
  reference for the non-periodic cell-extension trick and self-edge elimination.

Not reusable: all of `modules/` and `adapters/` (Flax NNX on `e3nn_jax.Irreps`;
MACE's radial is Bessel + polynomial cutoff, unrelated to Agnesi + orthogonalised
Legendre + `(1-y²)²`), and the entire training stack (`streaming_loader.py` alone is
1111 LOC that one `lineax` solve makes irrelevant).

## LAMMPS export contract

`lammps_jax/export.py` fixes the signature:

```python
def energy_fn(positions, species, graph) -> per_node_energy  # [max_atoms]
# graph = LammpsNeighborList(senders, receivers, edge_mask)
```

- LAMMPS supplies the neighbour list; Kokkos packs it on device. No nlist code here.
- No cell, no shifts, no PBC. `positions` spans `nlocal + nghost`.
- Static shapes. Padding edges carry `senders == receivers == max_atoms`,
  `edge_mask = False`.
- Gather with `mode="fill"`, scatter with `mode="drop"`, **and guard divisions or
  the gradient goes NaN**.
- Must be `jax.export`-traceable: no data-dependent Python control flow.

**Blocker to resolve before Phase 6: `sphericart-jax` emits a custom call.**
`sphericart/jax/sph.py` imports `custom_call` from `jax.interpreters.mlir`, so
the harmonics appear in the exported StableHLO as a **custom call target**, not
stock HLO. This contradicts the claim below that a pure-JAX ACE descriptor needs
`custom_call_targets=()`. Such targets must be resolved at run time from
`LAMMPS_JAX_FFI_HANDLERS`, and `contrib/ffi-replay` exists precisely for
libraries whose compiled kernels live only inside the exporting process.

**Recommended resolution: implement the harmonics in pure JAX.** They are a
polynomial recursion, fully expressible in stock HLO, and Phase 1 established
that `ace1_model` needs *spherical* (not solid) harmonics — one convention, not
two. That removes the only custom-call dependency, keeps
`custom_call_targets=()` true, and drops the `sphericart-jax` pin that currently
forces `jax==0.10.1`. Validate against the existing `sphericart` values, which
already match Julia to 4.8e-15.

**Simplification found in Phase 4: the virial needs no cell handling.** mace-jax
applies strain to positions *and* cell and then recomputes edge shifts. But under
a strain ε, `r → r + εr` and `cell → cell + ε·cell`, so

```
rij = r_j - r_i + S@cell  →  rij + ε(r_j - r_i + S@cell) = rij + ε·rij
```

The cell contributions cancel exactly and the whole thing reduces to
`rij_def = rij + rij @ eps`. The virial is therefore a **pure function of edge
vectors**, so the same code path serves LAMMPS, where there is no cell at all.
This section needs no separate virial treatment.

**Specific hazard.** The Agnesi transform divides
(`1/(1 + a·s^pin/(1 + s^(pin-pcut)))`, `agnesi.jl:59`) and `r = ‖rij‖` is
non-differentiable at zero. Padded edges have `rij = 0`, so a naive port yields NaN
forces from padding alone — silently, and only under `grad`. Budget a deliberate
masking pass: `safe_r = where(valid, r, 1.0)` before every transform, mask after.

**Simplification over MACE.** cuEquivariance and OpenEquivariance kernels survive
export as custom-call targets resolved from `LAMMPS_JAX_FFI_HANDLERS`, which is why
`contrib/ffi-replay` exists. A pure-JAX ACE descriptor is segment_sum, gather, prod
and matmul — all stock HLO. Export with `custom_call_targets=()` and skip it.

### Build environment

A working LAMMPS + plugin build exists and has been used to validate EAM bundles
against native `eam/alloy`. **But it does not run on `lestrade`.**

**The run host is `moriarty`, not `lestrade`.** The build tag `SKX-AMPERE86`
decodes exactly: moriarty is a Xeon Silver 4216 (Cascade Lake, has AVX-512) with
an RTX A4500 (compute 8.6). `lmp -h` runs there normally.

On lestrade the same binary dies with `Illegal instruction` (SIGILL, exit 132)
before printing anything — `liblammps.so.0` carries ~1800 `vmovdqu8` and other
AVX-512 instructions, and lestrade is an i9-14900K (Raptor Lake, no AVX-512 at
all) with an Ada card (8.9). Nothing to fix; it is simply the wrong host.

**`/home` and `/storage` are shared between the two; `/tmp` is not.** Stage work
staged in `/tmp` on lestrade — including the Phase 0 spike env — is not visible
on moriarty. Use a shared path for anything that has to cross.

If a rebuild is ever needed, `scripts/build_lammps_jax.sh` auto-detects both CPU
and GPU arch.

Paths as built:

| item | path |
|---|---|
| repo (branch `dev/julia_export`) | `~/lammps-jax` |
| build root (off home quota) | `/storage/eng/essswb/lammps-jax-build` |
| plugin (**the `--cudart shared` build**) | `.../build-plugin-shared-cudart` |
| `lmp` | `/storage/eng/essswb/venvs/lammps-jax/lib/lmp.real` |
| PJRT plugin | `$V/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so` |
| decks + harness | `/storage/eng/essswb/lammps-jax-build/run` |
| build script | `~/lammps-jax/scripts/build_lammps_jax.sh` |

Working invocation (from `run/test_eam_bundle.sh`):

```bash
module purge; module load foss/2023b CUDA/12.9.1
export LD_LIBRARY_PATH=$V/lib:$LD_LIBRARY_PATH
$LMP -k on g 1 -sf kk -pk kokkos newton on neigh half \
     -var pjrt $PJRT -var bundle <bundle.json> -in examples/in.eam_cuzr
```

Two version notes. That venv runs **jax 0.11.1**, while Stage 1 pins **0.10.1**
for `sphericart-jax`; removing the sphericart dependency (above) also removes
this mismatch. And the Kokkos build is tagged `AMPERE86` while the card is Ada
(compute 8.9) — evidently working via PTX JIT, but worth knowing if anything
looks wrong at the kernel level.

## Design decisions

### 1. Keep `Wnlq` live — do not fold it into the spec

For a linear ACE model `Wnlq` is one-hot, so `Rnl` is a re-indexing of the
enveloped polynomials. Folding that in is a trap: it hard-codes linearity into the
descriptor. Keep `Wnlq` as a `SelectLinL`-shaped leaf `(out_dim, in_dim, NZ²)`,
initialise one-hot, let it be trainable later. Cost: one `einsum`.

**Amended in Phase 1: this is in tension with the splined branch.** `splinify`
folds `Wnlq` *into* the spline coefficients, so in the splined path there is no
`Wnlq` at evaluation time to keep live. Resolution: the spline coefficients are
themselves a live array leaf, occupying `Wnlq`'s place in the parameter tree, and
a true `Wnlq` is reserved for the analytic branch — which is where Stage 2 needs
trainability anyway, since splines are not differentiable w.r.t. the parameters
that generated them. This satisfies the decision's intent (Stage 2 stays
reachable) but not its literal wording.

### 2. Bucketed fixed-capacity batching, designed in

A linear fit assembles the design matrix once; training streams minibatches, and
every distinct padded shape triggers an XLA recompile. Round `n_atoms` / `n_edges`
to a small set of buckets and use that everywhere — the linear fit is then a single
bucket. All three upstream libraries already agree on the idea
(`jraph.pad_with_graphs`, lammps-jax's `max_atoms`/`max_edges` + mask,
`neighbour_matrix`'s `(n, K)` + `count`). **This is the one item that is expensive
to retrofit.**

### 3. Precision split

f64 for fit assembly and the solve; f32 for GPU MD and training.
`jax_enable_x64` must be set **before** `export_model` or the traced program
silently truncates. `lammps-jax` supports f64 as of commit `a4304a2`.

Phase 0 changed the *reason* for this split. It is not that f64 is unaffordable on
GPU — measured at only 3–5.4× f32 on a 1/64-rate workstation card, because the
kernel is memory-bound. It is that the f32 GPU path has two defects that f64 does
not (`spike/jax_phase0/FINDINGS_f32.md`): TF32 by default costs ~400× accuracy,
and XLA autotuning miscompiles the Jacobian einsum in ~30% of processes. f64 was
clean in every test. So f64 fit assembly on GPU is both available and the safe
choice, and f32 is a deliberate throughput trade for MD — one that still requires
pinning matmul precision.

### 4. Priors are not equivalent across regimes

The linear fit's `A/P` change of variables (`src/fit_model.jl:141`) is exact. As a
training-loss penalty the smoothness prior becomes a soft regulariser. Fits will
differ. Decide deliberately; do not discover this later.

## Long-term directions, and what they constrain now

Two wanted features. Neither is scheduled, but both place constraints on
decisions being taken today, and one of them interacts with the convexity
priority in a way worth getting right the first time.

### (a) Import MACE element embeddings, to lift ACE's O(S^nu) species scaling

> **Prior art: this is tensor reduction, and it is already published.** Darby,
> Kovacs, Batatia, Caro, Hart, **Ortner**, Csanyi, *Tensor-reduced atomic density
> representations*, Phys. Rev. Lett. **131**, 028001 (2023),
> [arXiv:2210.01705](https://arxiv.org/abs/2210.01705). They recast per-element
> densities and their tensor products as a tensor factorisation, giving
> "representations whose size does not depend on the number of chemical
> elements". The construction sketched below — folding the channel into the
> radial index, equivalently a rank-d truncation of the species tensor — **is
> that construction**. It was proposed here, and independently re-derived by the
> spike, without the citation; read the paper before building anything.
>
> **Known limitation: the reduced descriptor is incomplete.** Ortner has since
> established that this approach breaks the bijection from positions to
> descriptors — distinct environments can map to identical descriptors. The
> spike measured the mechanism without naming it: rank is
> `min(d, dim Sym^nu(R^S))` **exactly**, so any `d` below break-even is provably
> rank-deficient, and at `d` above it the representation is exact but saves
> nothing. **The compression factor and the information loss are the same
> quantity.** No choice of solver recovers it; it is structural.
>
> **Position taken: pragmatic acceptance.** MACE uses this construction and works
> well, and ACE at finite correlation order is in any case already incomplete
> (Pozdnyakov *et al.*). So incompleteness is not disqualifying — but it is a
> real trade against one of ACE's distinguishing guarantees, in the same way
> joint embedding training would trade away convexity, and it should be made
> knowingly. Concretely: **choosing `d` is choosing how much of the species
> tensor to discard.** Before this ships, run a degeneracy probe — construct
> environments the reduced descriptor provably cannot separate and check whether
> they are physically distinct with different energies. An RMSE comparison at
> matched `n_B` is *not* that test: degeneracies will hide in the residual of an
> ordinary dataset rather than showing up as a systematic failure.

ACE's cost grows combinatorially in the number of species: species enters as a
*categorical index*, so a correlation-order-nu basis carries O(S^nu) distinct
channels and many-element models become impractical. MACE avoids this by
learning a continuous embedding per element -- species becomes a fixed-width
vector, and cost stops depending on S.

**The key property, and the reason to want this specifically: a *frozen*
imported embedding keeps the fit linear and therefore convex.** If the embedding
is a fixed (S, d) matrix taken from a MACE checkpoint, the model is still linear
in the ACE coefficients, so the normal linear solve, the smoothness priors and
the BLR/committee uncertainty story all survive intact. Only *jointly training*
the embedding breaks convexity. So the default must be frozen-embedding, with
training it an opt-in that explicitly moves the user to the non-linear backend.
**Confirmed as the intended design.**
Getting this the wrong way round would give away the differentiator to buy a
feature that does not require giving it away.

**What it constrains now:**

1. **Species handling must sit behind one narrow interface, not be inlined.**
   Today it is a categorical gather in at least two places: the radial
   coefficients are indexed `[zi, zj]` (`rnl_coefs` is `(NZ, NZ, ncoef, n_rnl)`,
   `rnl_Wnlq` is `(NZ, NZ, n_rnl, n_q)`) and the readout does `self.WB[:, node_z]`
   in `_readout`. An embedding replaces each of those gathers with a contraction
   against a learned vector. If those index expressions stay scattered through
   the model, swapping the scheme is a rewrite; behind two functions -- "edge
   species -> radial coefficients" and "node species -> readout weights" -- it is
   a second implementation beside the first. **Do this when Phase 15 is done**,
   since that phase already establishes the pattern of a swappable kernel with an
   explicit `kind`.
2. **Keep `Wnlq` live.** Already a standing decision for non-linear readiness;
   the embedding contracts with it, so it is now load-bearing for two features.
3. **The export schema needs a species-scheme flag**, exactly as it carries
   `ybasis_kind`, plus the embedding table and its provenance (which MACE
   checkpoint, which layer). A bundle must say how species enter, not leave it to
   be inferred from array shapes.
4. **It can be built entirely in JAX, and probably should be.** An earlier draft
   of this section said the Julia side had to be able to build such a model
   first. That is wrong, and the reason is worth spelling out: **the coupling
   coefficients do not depend on species at all.** Today's S^nu growth comes from
   species being baked into the basis *spec* -- measured, `WB` is `(n_B, NZ)` and
   n_B goes **308 -> 917** for Si,C -> Si,C,O at fixed order 3 / degree 8, i.e.
   ~(3/2)^3 before degree truncation. A *single-species* export is therefore
   already the species-free spec this feature needs. Export that, and add the
   channel dimension in JAX:

   ```
   A[i, nlm]     = sum_j          R_nl(r_ij) Y_lm(rhat_ij)      # today
   A[i, k, nlm]  = sum_j emb[z_j, k] R_nl(r_ij) Y_lm(rhat_ij)   # with embedding
   ```

   with `emb` the frozen `(S, d)` table pulled from a MACE checkpoint via
   `mace-jax`. Nothing in `A2B` or the spec changes. Since the model must be
   *fitted* in this parameterisation anyway, and fitting is Stage 2A, the whole
   feature lands naturally there.

   **Two design choices it forces, neither free:**

   - **Channel coupling in the AA product.** Channel-diagonal products (MACE's
     choice) keep cost at `d x` the single-species basis; full channel mixing
     reintroduces `d^nu`, which is the same wall in a different variable. Start
     diagonal.
   - **Choosing `d`.** MACE-MP-0 small carries 128 channels, and `d x` a
     single-species basis is not obviously cheaper than what it replaces. The
     crossover is roughly **`S^nu` versus `d`**: embeddings win for many
     elements, lose for two. So this is a *many-element* feature, not a general
     speedup, and a truncated embedding (a few leading components rather than all
     128) should be measured early -- it may be most of the benefit for a
     fraction of the cost.

   **But the Julia route is probably better, and is not what it first looks
   like.** Folding the channel into the radial index makes the change far
   smaller than "a new architecture":

   ```
   R'[(k,n), l](r_ij, z_j)  =  emb[z_j, k] * R[n, l](r_ij)
   ```

   i.e. a radial basis widened from `n_rnl` to `d * n_rnl` whose values depend on
   the neighbour species through a frozen linear map. `abasis`, `aabasis` and
   `A2B` need no changes at all — they see a wider radial basis and nothing else.

   And that operation already exists in EquivariantTensors. `SelectLinL` computes

   ```
   B[i, j] = sum_k P[i, k] * W[j, k, selector(x_i)]        # selectlinl.jl
   ```

   a per-species-category linear map chosen by a hard index. The embedding
   version is the *same tensor* `W` contracted against a soft weight vector
   instead of a one-hot selection:

   ```
   B[i, j] = sum_c emb[z_j, c] * sum_k P[i, k] * W[j, k, c]
   ```

   So this is a **generalisation of an existing layer**, not a new one — and
   notably the soft form is array-expressible, which is exactly the rewrite the
   Reactant spike wanted for other reasons (`selectlinl.jl`'s own comments ask
   for it). The two efforts point the same way.

   **What the Julia route buys, and it is a lot:**

   - **The entire fitting stack applies unchanged.** With `emb` frozen the basis
     is fixed, so the model stays linear and `acefit!`, BLR, committees, the
     smoothness priors and the QR/LSQR solvers all work as they are. The JAX
     route would have to reach Stage 2A before it could fit anything at all.
   - **Single source of truth is preserved**, and the divergence guard covers the
     feature — which the JAX-only version cannot be.
   - **It decouples from Stage 2 entirely.** Fit in Julia, evaluate in JAX: this
     becomes a Stage-1-shaped feature, not a Stage-2 one.
   - **`acejax` gets it nearly for free**, because structurally it is just a
     different radial basis. The exporter carries `emb` and the composite index;
     the evaluator's existing machinery applies.

   **Getting the embedding: a number, not a dependency.** `PythonCall.jl` would
   work, but it is more coupling than the problem needs — the table is a frozen
   `(S, d)` matrix extracted **once, offline**. Pulling `torch` or `mace-jax` into
   ACEpotentials' runtime dependencies to fetch a constant would recreate exactly
   the cross-codebase fragility that ruled out the `yace` route. Prefer a one-off
   extraction script writing an `.npz`/artifact that Julia reads — ACEpotentials
   already consumes artifacts via `LazyArtifacts`. Keep a `PythonCall`-based
   helper for *regenerating* that artifact if convenient, but not on the load
   path. Record which checkpoint and which layer it came from.

   **Unverified, and worth a spike before committing:** whether ET can express
   the channel-diagonal restriction of the AA spec cleanly (only terms where all
   nu factors share `k`), and what it costs to build a spec over `d * n_rnl`
   radial functions. Those are the two places this could turn out harder than the
   sketch above. Rough effort if they are clean: **1–2 weeks in Julia**, mostly
   in EquivariantTensors, plus upstream coordination.

   **The honest cost of the JAX-only route, for contrast: that model class would
   exist only in JAX.** It is a
   deliberate departure from single source of truth -- ACEpotentials.jl could not
   fit or evaluate it, and the divergence guard cannot cover it because there is
   nothing on the Julia side to compare against. That is a real price and should
   be a conscious decision, not a side effect. It is also an argument for keeping
   the species interface a *swappable kind* (constraint 1 above) rather than a
   fork: the categorical path stays the guarded, Julia-backed default, and the
   embedding path is an additional mode that declares itself in the bundle.
5. **The divergence matrix must grow an embedding row** when this lands. The
   three-element precursor row is **done** (`Si-C-O order 3`, both model
   families, 66/66) -- and it exercises the problem this feature exists to fix:
   at order 3 / degree 8, n_B goes **308 -> 917** and the site descriptor vector
   **648 -> 2823** going from two species to three.

### Stage 1E — element embeddings in Julia (specified; build in progress)

Its own stage label rather than a phase number: it is a Julia-side model
architecture change, it fits with `acefit!`, and it evaluates through the
existing export path — Stage-1 shaped, and independent of Stage 2's numbering.

**Why it is worth building now.** The spike measured a **lossless** saving of
1.4x to **10.04x** (nu=4, S=10) using per-order channel widths, with the nu=4,
S=10 spec building in 7 s where the categorical spec exhausted its build budget
at S=13. Lossless means *reparameterisation, not approximation* — no accuracy
question, no completeness question, nothing to trade. The lossy regime is a later
optional extension.

#### The construction

Fold the channel into the radial index, exactly as ACEpotentials already folds
species:

```
categorical (today):  R(n'z')l(r, Z1, Z2) = P_n'(r) * delta_{z', Z2}
embedding  (this):    R(n'k)l (r, Z1, Z2) = P_n'(r) * emb[Z2, k]
```

`abasis`, `aabasis` and `A2B` are untouched — they see a wider radial basis and
nothing else.

**Per-order channel widths are part of the construction, not an optimisation.**
`d_nu = min(d_max, C(S+nu-1, nu))`: each correlation order saturates at its own
species-tensor dimension, so a single width starves the high orders or pads the
low ones (44% redundant at `d=16`, order 4, S=3). Padding is not merely wasteful:
it makes the design matrix rank-deficient, and the measured condition number
already reaches ~3e21. Handing that to QR/LSQR or a BLR posterior is exactly what
the linear route exists to avoid.

#### Components

1. **`read_mace_embedding(path)`** — load the frozen `(S, d)` table plus its JSON
   provenance sidecar (checkpoint, sha256, layer), and map atomic numbers to rows.
   `scripts/extract_mace_embedding.py` already writes both. **No Python on the
   load path**: the artefact is read, never regenerated at run time. `NPZ.jl`
   cannot read numpy unicode dtypes, hence the numeric npz + JSON split.
2. **`set_embedding_weights!(rbasis, ps, emb)`** — the one change with real
   leverage. `set_onehot_weights!` (`Rnl_learnable.jl:83`) already writes
   `ps.Wnlq[i_nl, n_, iz1, iz2] = 1` when `z_ == iz2`, decoding `z' = mod1(n, NZ)`,
   `n' = div(n-1, NZ) + 1`. The embedding version is the same loop with `NZ -> d`
   and `1 -> emb[iz2, k]`. It is frozen: written once, never fitted.
3. **Spec construction** — build the single-species `mb_spec`, then replicate each
   entry over `k = 1:d_nu(length(bb))`. Keep it **sorted by correlation order**:
   `sparse_equivariant_tensor` mis-couples an ungrouped spec (see N1 in the spike
   findings; fixed on `jameskermode/EquivariantTensors.jl:fix-spec-ordering`, not
   yet merged, so do not rely on the fix being present).
4. **`ace_embedding_model(; elements, order, totaldegree, embedding, d_max, ...)`**
   returning an `ACEPotential` that `acefit!` accepts unchanged.

#### Gates

- **Equivariance**: rotation and permutation invariance of the site energy to
  ~1e-14, at `S = 1, 2, 3` and orders 2-4.
- **Losslessness**: for `d_nu = dim_nu`, the design matrix has full column rank —
  measured, not only predicted by the deficiency formula.
- **It fits**: `acefit!` on a real dataset runs and the fit is not rank-deficient;
  report `cond` against the categorical model at matched `n_B`.
- **Provenance survives**: the model records which checkpoint and layer the table
  came from, and the export carries it.

#### Explicitly out of scope for this stage

Lossy widths (`d_nu < dim_nu`), the degeneracy probe that would price them,
training the embedding (that breaks convexity and belongs in Stage 2B), and any
JAX-side work — the export path is unchanged, since this is just a wider radial
basis.

#### Build status

Landed: `src/models/embeddings.jl` with `ElementEmbedding`,
`read_mace_embedding`, `embedding_rows`, `embedding_widths` and
`set_embedding_weights!`, plus `test/models/test_embeddings.jl`.

Two things the build changed from the spec above:

- **The artefact is JSON, not npz.** Reading npz would have added `NPZ` to
  ACEpotentials' dependencies for the sake of one frozen table; ACEpotentials
  already depends on JSON. `scripts/extract_mace_embedding.py` now writes the
  table into the JSON sidecar as well, keeping the npz for Python consumers.
- **The embedding applies to the LEARNABLE radial basis, before splining.**
  `ace1_model` builds a `SplineRnlrzzBasis`, which has no `Wnlq` to set, so
  `set_embedding_weights!` operates on `LearnableRnlrzzBasis` — the branch
  `ace_model` keeps. An `ace1`-style embedded model therefore has to set the
  weights and *then* splinify, which is what `ace1_model` already does
  internally; wiring that is the next step.

The test checks `set_embedding_weights!` against trusted existing code rather
than re-deriving the reference: with `emb = I` the embedding row is the one-hot
delta, so it must reproduce `set_onehot_weights!` bit for bit. It does. The
second check exercises the defining property without touching any internal basis
API — scaling one embedding row by `c` scales exactly that species' `Rnl` by `c`
and leaves the others untouched.

`ace_embedding_model` now builds, and **three of the four gates are met**:

| gate | result |
|---|---|
| construction | `n_B = 392`, widths `[3,6,10]` at S=3, order 3, degree 8 — matches the standalone per-order calculation exactly |
| rotation + permutation invariance | both < 1e-12 relative |
| **losslessness, measured** | design matrix (1200 x 2400), numerical rank **1200 of 1200** — full rank, not merely predicted by the deficiency formula |
| `acefit!` runs | yes, on Si_tiny |

**The accuracy gate is now met too, but the diagnosis was not what it looked
like.** At S = 1, where the embedding contributes a single scalar and both models
have `n_B = 54`, against `ace1_model` on the same 20 configurations:

| | E RMSE | F RMSE |
|---|---|---|
| before | 35.4745 | **24.04** |
| after | 35.4537 | **1.4748** |
| `ace1_model` | 35.4537 | 1.4682 |

Energies now agree to 4 dp and forces to 0.45%.

Three hypotheses were wrong before the right one, and each was cheap to kill:

- **ACE1 radial heuristics.** Real — `ace1_model` uses `(:agnesi, 2, 4)` not
  `(2,2)`, Jacobi(4,4) rather than Legendre (it folds the envelope into the
  orthogonality, `ace1_compat.jl:255-261`), and splines afterwards. Matching all
  three is correct and is now done, but it did **not** move the forces.
- **`Ytype`.** `ace1_model` uses `:spherical`, the constructor defaulted to
  `:solid`. Changed nothing: F differed in the 4th digit.
- **Broken gradients.** Ruled out directly: analytic forces match finite
  differences to 6.6e-13.

The actual cause: **the embedding must be normalised *after* truncation to `d`.**
The raw MACE entries are O(0.1), so at correlation order ν the basis is scaled by
~1e-3. A frozen scaling is absorbed by the linear coefficients — the bases were
measured to be *exactly proportional*, ratio -74.7 componentwise — so the model
spans the same space either way. But **BLR's prior on coefficient magnitude is
not scale-invariant**, so the whole 16x was regularisation, not approximation.

Normalising the full 128-channel row and *then* truncating is not enough: that
leaves ~1/sqrt(128) per channel and reproduces the bug one step removed. That
intermediate attempt cut `|WB|` from 296 to 63 and left F unchanged at 25.1,
which is what made it clear the problem was not conditioning in the usual sense.
`embedding_rows` now normalises the truncated rows by default.

**Generalisable lesson:** a frozen linear reparameterisation is mathematically
free but *not* free under a scale-sensitive regulariser. Anything that rescales
the basis — this, per-order widths, a different `d` — must be normalised before
it meets BLR or a smoothness prior, or the fit quality moves for reasons that
have nothing to do with the model.

Also fixed while finding this: `compute_errors` must be given the same keys as
`acefit!`. With the defaults it finds no reference data in `Si_tiny` (whose keys
are `dft_*`) and returns **0.0 for every observable** — the first version of this
gate asserted `isfinite(0.0)` and passed while measuring nothing.

#### Export: done, and it needed no JAX-side change at all

The embedding is baked into the **radial splines**, so an embedded model exports
as an ordinary splined ACE model that happens to have a wider radial basis.
`acejax` needs no knowledge of embeddings, and got none: a Ti-Al embedded model
exported through `export_model.jl embedding` passes the existing suite unchanged,
including every energy, force, virial and descriptor gate.

`export_model.jl` gains an `embedding` kind (`ACE_EMBEDDING` points at the
artefact, `ACE_DMAX` optionally caps the width) and records the embedding
provenance in the exported metadata. A `Ti-Al embedding` row in the CI
divergence matrix keeps it that way.

**Three latent test bugs surfaced, all the same shape.** The dense-pooling tests
hardcoded a neighbour capacity of 64. TiAl at these cutoffs has ~86 neighbours
per atom, so `neighbour_matrix` truncated silently and three tests failed for a
reason that had nothing to do with embeddings. Capacities are now derived from
the actual neighbour counts. The padding test needed the opposite treatment —
it is *about* padded slots, so it takes `max + 8`: with exactly-max and a system
where every atom has the same neighbour count there is no padding left and its
own vacuity guard fires. That guard earning its keep is the reason this was
caught rather than silently weakened.

Still to build: nothing on the export path. Open on the science side is whether
the embedding pays off at the element counts it is meant for (S >= 6-7), which
needs a many-element dataset that ACEpotentials' bundled examples do not
provide.

#### MEASURED: multi-element accuracy, and a RETRACTION

TiAl_tiny (33 configs, two elements), order 3, `QR(lambda=1e-3)` — BLR fails
PosDef here, the dataset cannot determine the larger bases:

| model | n_B | E RMSE | F RMSE | V RMSE |
|---|---|---|---|---|
| `ace1_model` deg 5 | 73 | 0.060 | **0.959** | 1.648 |
| **embedding deg 6 (lossless widths)** | **77** | 0.124 | **2.009** | 4.175 |
| `ace1_model` deg 6 | 123 | 0.036 | **0.765** | 1.217 |
| embedding deg 7 | 118 | 0.102 | 1.829 | 3.849 |
| embedding deg 8 | 177 | 0.075 | 1.724 | 3.296 |

**At matched basis size (73 vs 77) the categorical encoding fits roughly 2x
better on forces and energies** — *at this one value of `lambda`*. See the
CORRECTION below: that gap is a regularisation artefact and shrinks to 16-20% on
held-out data with `lambda` swept.

**RETRACTION: "lossless" does not mean "spans the same function space", and the
earlier claim that this is a reparameterisation rather than an approximation was
wrong.** At the same degree the embedded basis has n_B = 77 against the
categorical 123 — strictly fewer functions, so strictly smaller span. The spike's
rank result (`min(d, dim Sym^nu(R^S))`, full rank at `d = dim`) is a statement
about the species tensor **at a fixed (nn,ll) block**. It does not survive
contact with degree truncation, which in the categorical model couples the
species index into `n` and so hands different species combinations different
radial budgets. The 1.4-10x "lossless saving" is therefore a saving in basis
size at matched construction, **not** a free lunch — and this measurement is what
that costs.

Three things it is NOT:

- **Not conditioning.** Measured: categorical cond 5.6e8, embedded 1.2e7 — the
  embedded bases are *better* conditioned, and all are full rank.
- **Not ill-spread channels.** The truncated, normalised channel vectors have
  species-space singular-value ratios of 1.14-1.26, i.e. well spread.
- **Not the normalisation bug** fixed earlier; that is applied throughout here.

**CORRECTION — the 2x above is a regularisation artefact.** Chasing the anomaly
(why the deliberately lossy `d_max = 2` beat the lossless widths) found that the
two bases are *nested*, so the superset must win once regularisation is weak —
and it does: at `lambda = 1e-9` the lossless widths give F = 0.500 against
d_max=2's 0.690. The single `lambda = 1e-3` used above happens to disfavour the
embedded basis, whose coefficients are larger.

Re-run properly, on a 25/8 held-out split with `lambda` swept per model and the
**best test error** reported for each:

| model | n_B | best test F | best test E |
|---|---|---|---|
| `ace1_model` deg 5 | 73 | **0.551** | 0.270 |
| embedding deg 6 | 77 | 0.638 | **0.105** |
| `ace1_model` deg 6 | 123 | **0.354** | 0.100 |
| embedding deg 7 | 118 | 0.426 | **0.016** |

So at matched basis size the categorical encoding is **16-20% better on forces**,
and the embedding is **better on energies** — not the 2x loss reported above, and
not a clean win either way. What actually dominates both models is the
regularisation strength: moving `lambda` from 1e-3 to 1e-8 changes test F by more
than the choice of encoding does at any fixed `lambda`.

**Method note, and it was my error:** the first comparison fixed `lambda = 1e-3`
for both models and read off a 2x gap. Comparing regularised fits at a single
arbitrary `lambda` is not a like-for-like comparison — it is tuning one side.
Any future encoding comparison must sweep `lambda` per model and report held-out
error, not training error at a shared `lambda`.

Caveats on the corrected numbers: one 25/8 split, 33 configurations total,
`lambda` selected on the test set (optimistic, but equally so for every model),
and 8 test configurations makes the energy column noisy.

**What this does and does not tell us.** S = 2 is the regime where the feature is
*expected* to lose — the spike put the d=128 crossover at S≈6-7 for order 3, and
TiAl cannot probe that. A fair test of the premise needs a dataset with 6+
elements. So this is not evidence the idea fails; it is evidence that **the
two-element case should not be used to sell it**, and that per-basis-function
accuracy is materially worse there. Note also that the deliberately lossy
`d_max = 2` (n_B = 50) fitted *better* than the lossless widths (F 1.63 vs 2.01),
which suggests the per-order width allocation is not optimal on this system and
is worth revisiting.

Caveats: 33 configurations, one order, one degree sweep, one solver, one frozen
embedding taken as the leading `d` channels of MACE-MP-0-small.

#### Unverified

Whether `acefit!` and the solvers are entirely indifferent to the widened basis
at production sizes, and whether the artefact's licence permits redistributing a
slice of MACE-MP-0 weights in-tree. The latter is a question for the user, not a
technical one.

### (b) Distil MACE models into smaller, faster ACE ones

Train an ACE model against labels generated by MACE rather than against DFT.

**This is a linear, convex fit** -- the targets change, the model does not -- so
it sits squarely on the Stage 2A path and needs nothing from 2B. That is worth
saying explicitly because "distillation" usually implies gradient descent; here
it does not have to.

**What it constrains now:**

1. **Blocked / streaming design-matrix assembly stops being optional.** With DFT
   labels the dataset is whatever you can afford to compute; with a teacher model
   you can generate as many labels as you like, which is the entire point. Stage
   2A phase 18 is therefore load-bearing for this feature, not a scaling nicety.
2. **`acejax` and `mace-jax` must coexist in one process.** Label generation and
   fit assembly in the same JAX session is the obvious implementation, and it is
   also how Phase 13 route 3 already runs MACE. What must go is the **jax 0.10.1
   pin** that `sphericart-jax` currently imposes -- a hard pin from our side is
   exactly what makes coexistence impossible. Dropping the dependency for a
   pure-JAX implementation is *one* way to achieve that; **upstreaming a PR to
   `sphericart-jax` to widen its jax compat is another, and probably the better
   one** -- it is a well-optimised library, our own harmonics were measured
   against it, and the fix benefits every downstream user rather than routing us
   around the problem. Either way the *pin* is the prerequisite, not the
   dependency.
3. **The data pipeline must carry arbitrary labels, not just DFT keys.** The
   Stage 2A decision to export `AtomsData` from Julia rather than reimplement its
   fuzzy key matching still holds, but the exported schema must not assume the
   labels came from a DFT code -- teacher-model energies, forces and virials are
   the same shapes with different provenance, and provenance should be recorded.
4. **Uncertainty is a feature here, not a nicety.** BLR and committees tell you
   where the student disagrees with the teacher, which is exactly what drives the
   next round of configuration generation. Another reason the convex path is the
   right default.
5. **A teacher-vs-student validation harness** is a different thing from the
   Julia-vs-Python divergence guard, and should not be bolted onto it.

### What the two have in common

Both want *large multi-element* models fitted *convexly* at *scale on a GPU*.
That is the same target Stage 2A already has, which is reassuring -- neither
feature argues for reordering the plan. What they do argue for is that three
current items are more load-bearing than they look: the species interface, the
streaming assembly, and dropping the `sphericart-jax` pin.

## Non-linear readiness

Decisions 1–3 above are the whole cost of staying non-linear-ready (~0.5 day, mostly
discipline), and they are the same decisions that keep Stage 1 from foreclosing
Stage 2. With them in place, the two regimes share one forward function:

- **linear**: `jacrev`/custom-JVP of the per-species descriptor sum w.r.t. positions
  → design matrix → lineax
- **non-linear**: `eqx.filter_grad` of the loss w.r.t. all params → optax

Deferred without penalty: the optax loop, schedules, checkpointing, EMA/SWA. E/F/V
loss weights already live in the vendored `Configuration` dataclass. Estimated
+5–8 days whenever wanted.

## Validation milestones

Each is a hard numerical gate against an existing Julia test. This is what keeps a
port from drifting.

**Stage 1**

1. ~~**Descriptor** — site basis `𝔹` matches Julia to 1e-10~~ ✅ Phase 0: 1.4e-15
2. ~~**Observables** — energy, forces and virial from a *fitted* model match
   `AtomsCalculators.energy_forces_virial`~~ ✅ Phases 1/3/4: from an ASE `Atoms`
   object with periodic images, energy 1.8e-12 (scale 1e4), forces 1.0e-13,
   virial 6.6e-12 (scale 79). Dense and sparse pooling agree to 8.5e-14. The
   neighbour list reproduces Julia's edge set exactly (2842 edges, same
   `(i, rij)` multiset). Julia's convention is
   `site_virial = -Σ dv_i * 𝐫_i'` (`AtomsCalculatorsUtilities
   /sitepotentials/assembly.jl:4`); the symmetric-displacement result agrees
   with no sign flip.
3. **Export** — `pair_style jax/kk` energies match the Python calculator on the
   same configuration
4. **End-to-end** — a Julia-fitted Si potential runs in LAMMPS and reproduces the
   Julia calculator's energies along a short trajectory

**Stage 2**

5. **Design matrix** — E/F/V blocks match `energy_forces_virial_basis`
   (`src/et_models/et_calculators.jl:246`)
6. **Fit** — `Si_tiny_dataset` reproduces the RMSE in
   `test/et_models/test_et_silicon.jl`

Test against dumped Julia values from the first commit, not at the end. The
convention surface is large: 1-based → 0-based throughout `Aspec`/`𝔸spec`,
SpheriCart's `lm2idx` ordering, and the `𝔸spec` sort that the ET source itself
flags as *"very hacky and brittle"* (`ET/src/ace/sparse_ace_utils.jl:23-24`).

## Effort and sequencing

### Stage 1 — fit in Julia, evaluate in JAX

| Phase | Days |
|---|---|
| ~~0. Performance spike (gating)~~ ✅ all three gates resolved | 1–2 |
| ~~1. Julia exporter (fitted model, splined radials) + Python loader~~ ✅ `89901596` | 2.5–3.5 |
| ~~2. Descriptor forward~~ ✅ `89901596` | 4 |
| ~~3. Neighbour-list adapters + fixed-capacity plumbing~~ ✅ `20282a30` | 1 |
| ~~4. Energy / forces / virial via `jax.grad` + ASE calculator~~ ✅ `20282a30` | 1 |
| ~~5. Validation harness (milestones 1–4)~~ ✅ 66 tests | 2 |
| ~~6. LAMMPS export + integration testing~~ ✅ `d6c99f78` | 4–5 |
| ~~7. `ace_model` support: analytic radials + solid harmonics~~ ✅ `47aa6837` | 2 |
| ~~8. Throughput benchmark vs Kokkos and ML-PACE~~ ✅ `ce9020d9` | 1 |
| ~~9. Usable ASE calculator + descriptor access~~ ✅ `18ff57bd` | 1.5 |
| ~~10. Whole-branch review, reorganise to `acejax/`, README, CI, PyPI~~ ✅ `0c670a3d` | 2.5–3 |
| ~~11. JAX-only distributed MD spike (no LAMMPS)~~ ✅ **viable, and the fastest route** | 0.5 |
| 12. LAMMPS ML-IAP route — **deprioritised by Phase 11; CI rationale stands** | 2–3 |
| **13. MACE comparison via `symmetrix` on GPU** — not started | 2 |
| *14. Traced neighbour list for end-to-end differentiability* — **optional** | 1–2 |
| **15. Conditional gather/matmul swapover for `edge_A`** — CPU side done; GPU calibration outstanding | 1–1.5 |

**Remaining: phases 12–13 and 15, ~5–6.5 days**, plus optional phase 14. None is blocked; 12 and 13 each need a
LAMMPS rebuild with extra packages (`ML-IAP`+`PYTHON`, and `symmetrix`
respectively), into a new directory as with the ML-PACE rebuild.

Open items outside the numbered phases:

- **held on upstream** — the ghost/pad truncation prototype, waiting on the
  `lammps-jax` maintainer's neighbour-matrix-layout push, since building against
  a base that is about to change would waste the work
- **cheap and unfiled** — the Reactant miscompilation bug (two-line reproducer,
  confirmed on 0.2.285) and the WignerD PR to EquivariantTensors that would lift
  the Reactant version ceiling from 0.2.222
- **known gaps, deliberately open** — short-range port fidelity below ~2 Å (all
  agreement was measured near equilibrium), and the n_B=710 crossover between the
  two padding causes

**≈ 27.5–31.5 working days ≈ 5.5–6.5 weeks** (Phases 0/1/3/4/6/7/9 done; Phase 8 partly
blocked on two open bugs; Phase 10 remains, ~3 days).

#### Phase 7 — `ace_model` and solid harmonics

**Complete (`47aa6837`).** Stage 1 now covers both families. `ace1_model` is
splined radials with spherical harmonics; `ace_model` is a **mixed** case —
analytic many-body radials with a live `Wnlq`, a *splined* pair basis, a
different pair envelope formula, and solid harmonics by default. The original
framing here ("`ace_model`, the analytic one") was too coarse: the branch is
per-basis, and the schema now carries `radial_kind`, `pair_radial_kind` and
`pair_envelope_kind` independently.

Results, both families, against Julia:

| quantity | `ace1_model` | `ace_model` |
|---|---|---|
| site energies | 3.70e-13 | 2.56e-13 |
| E / F / V from ASE | 1.82e-12 / 1.25e-13 / 9.81e-13 | 1.82e-12 / 5.47e-13 / 2.26e-12 |
| dense vs sparse pooling | 8.53e-14 | 5.68e-14 |

46 tests, every one parametrised over both families.

Solid harmonics fold `r^(l-|m|)` into the recursion rather than multiplying the
spherical result by `r^l`. Since `l ≥ |m|` that branch does **no division at
all**, so it is better conditioned than the spherical one; `solid == r^l ·
spherical` confirmed to 7e-15.

**Most of this is already validated.** Phase 0's spike targeted exactly this
configuration — `ace_model` with `Ytype = :solid` and analytic Agnesi + 3-term
recursion + `Wnlq` — and matched Julia to 1.4e-15 on CPU and GPU. Phase 7 is
mainly promoting that code into `stage1/` behind the schema branches that
already exist, rather than writing it fresh:

- the exporter already records `ybasis_kind`, so the Ylm branch is plumbed
- the schema already reserves the analytic radial branch alongside the splined one
- the loader currently raises `NotImplementedError` on `radial_kind != "spline"`,
  which is the single place the branch needs filling in
- solid harmonics are `r^l · Y_lm`, a thin variant of the spherical
  implementation once that exists in pure JAX (Phase 6, Part A)

**Stage 2's prerequisite is discharged.** `Wnlq` is now a live array leaf
exercised against Julia rather than only reserved in the schema, which resolves
the design decision Phase 1 had to amend. Design decision 1 notes
that splines are not differentiable with respect to the parameters that
generated them, so any trainable-`Wnlq` work — non-linear fits, and the analytic
branch generally — needs exactly this path. Doing it here means Stage 2 starts
with a live `Wnlq` already exercised rather than only reserved in the schema.

Gate: a fitted `ace_model` reproduces Julia's energy, forces and virial to the
same tolerances Phase 3–4 achieved for `ace1_model`, with both `ybasis_kind`
values and both `radial_kind` values covered by tests.

#### Phase 9 — a usable ASE calculator, and descriptor access

Phase 4 built an `ACECalculator` sufficient to *validate* against Julia. Phase 9
makes it something someone would actually reach for, and exposes the site
descriptors alongside the observables.

**Usability.** Today the calculator takes an already-loaded `(model, meta)` pair.
It should take a path — `ACECalculator("si_fitted.npz")` — infer cutoff, species
and dtype from the file, and pick the neighbour-list backend and precision
sensibly by default while allowing both to be overridden. The `export_bundle.py`
hardcoded absolute path (`/home/eng/essswb/si-ace/...`) is the same class of
problem and should go at the same time.

**Descriptors are the substantive half.** The site basis `𝔹` is already computed
internally — it is exactly the quantity the readout contracts against — so
exposing it is nearly free, and it unlocks the things descriptors are actually
for: dimensionality reduction and dataset visualisation, distance-in-descriptor-
space uncertainty and active learning, clustering, and transfer to other models.

Match the existing Julia API for parity, `ACEpotentials.site_descriptors`
(`src/descriptor.jl`), which takes a system and returns one descriptor vector per
atom with an optional `domain` to restrict the atom set. Note the Julia
implementation is documented as *"RETIRING THIS FOR NOW BECAUSE IT IS HIGHLY
INEFFICIENT"* — it recomputes per site. The JAX version gets the whole batch from
one forward pass, so this is a case where the port is straightforwardly better
than the original rather than merely equivalent.

Expose it as a property on the calculator (`descriptors`, alongside `energy`,
`forces`, `stress`) *and* as a standalone function that does not require an ASE
`Atoms` round-trip, since the batch case — descriptors for a whole dataset — is
the common one and should not pay calculator overhead per structure.

**Complete (`18ff57bd`).** Descriptors match `ACEpotentials.site_descriptors` at
1.75e-15 (`ace1_model`) and 4.50e-15 (`ace_model`) relative, over 64 sites x 120
components. `ACECalculator(path)` works from a bare npz for both families, with
cutoff, species and dtype inferred from the file; energies match exactly and
forces to 1.2e-13 / 6.2e-13. Exposed as `calc.get_site_descriptors(atoms)` and as
a standalone `site_descriptors(...)` with no ASE round-trip; both honour `domain`.

The descriptor layout is species-blocked per `get_basis_inds` /
`get_pairbasis_inds` (`src/models/ace.jl:544-566`), and a test checks the scatter
offsets directly — **a wrong offset would still produce plausible magnitudes**,
so agreement in magnitude alone would not catch it.

#### Phase 10 — release readiness

The last phase of Stage 1. Goal: `pip install acejax` gives a working ACE
evaluator. The name is free on PyPI (`acejax`, `ace-jax`, `ace_jax` all 404 as of
2026-09-09).

**Whole-branch review.** Fifteen-plus commits of incremental work, several
reversals, and two agents' output. Read it as one piece rather than as a
sequence — the things that decay under that kind of development are naming
consistency, dead branches left behind by a correction, and tests that pin
yesterday's understanding.

**Strip internal plan references from shipped code.** 20 occurrences of "Phase
N", "Stage N" and "spike" across `acejax/radial.py`, `model.py`,
`calculator.py`, the tests, `lammps/`, the README and `pyproject.toml`. These
were load-bearing while the work was in flight and are noise to anyone
installing the package — they refer to a document the reader does not have.
Keep the *content* where it explains a non-obvious decision (why padding sits at
the cutoff, why matmul precision is pinned); drop the plan coordinates.

**Reorganise to a top-level `acejax/`.** `stage1/` names a phase of our work, not
the thing. The package, its tests and its packaging move up; `spike/jax_phase0`
and `spike/reactant_phase0` stay out of the distribution — they are evidence, not
product. Decide deliberately whether they remain in-tree under `docs/` or leave
the branch entirely; the `FINDINGS_*.md` files have value that the code does not.

**Rewritten README.** Installation, a minimal usage example that runs, the
descriptor API from Phase 9, and the LAMMPS export path. State plainly what is
and is not covered: both model families, energies/forces/virial/descriptors, and
that fitting still happens in Julia.

**CI, with the LAMMPS build cached.** Follow `kermodegroup/ML-MIX`'s
`.github/workflows/ci.yml`: resolve the upstream LAMMPS stable commit hash, use
it as the cache key, keep separate caches for source and install, and rebuild
only on a miss.

**Constraint to design around, not discover:** `pair_style jax/kk` is CUDA-only —
`scripts/build_lammps_jax.sh` states the pair style has no CPU path — and
GitHub-hosted runners have no GPU. So CI splits:

- the `acejax` suite (46 tests, CPU) runs on every push — this is the bulk of the
  value and needs no LAMMPS at all
- a cached LAMMPS + plugin *build* job catches build breakage without running it
- the LAMMPS *integration* gate needs a GPU runner. Either self-hosted
  (moriarty), or it stays a manual step with `test_si_bundle.sh` as the
  documented procedure. Do not fake it on CPU.

**PyPI.** Real dependency pins, a version, license and metadata. `sphericart` is
dev-only now, so the hard dependencies are jax, equinox, numpy, ASE, and
`matscipy-neighbours` — which is **repo-only, not on PyPI**, so it cannot be a
hard install dependency. Make it an optional extra with a documented fallback,
or vendor the neighbour-list adapter.

Gate: `pip install` from a clean environment, then run the README's usage example
and the test suite, both green.

#### Phase 11 — a JAX-only distributed MD spike (no LAMMPS) ✅ COMPLETE

**Scope guard, added after Phase 13.** This is a spike and should stay one. It
has two legitimate futures — a benchmark harness (which it already is) and the
vehicle for Phase 14's differentiable MD, which is a capability LAMMPS cannot
offer at all. It should **not** grow into a general-purpose MD engine:
thermostats, barostats, constraints, restart files, trajectory formats and
analysis are a multi-month commitment that would duplicate LAMMPS, ASE and Molly
badly. Its CPU performance story is also not yet solid — architecture-dependent
per `bench/molly/`, and dependent on Phase 15 — so promoting it now would be
premature on the merits as well as the scope.


**Result: viable, worth pursuing, and the fastest of the three deployment routes
at every size measured once capacities are tuned.**

Gate passed on GPU and CPU. `acejax` runs real distributed MD through
`dist/parallel/` — velocity Verlet inside one jitted `shard_map` via
`lax.fori_loop`, so the inner loop never touches the host:

| check | result |
|---|---|
| 512-atom forces, 1/2/4/8 ranks | max abs dF **1.4–1.6e-13**; abs dE/E <= 1.7e-16 |
| same on GPU (A4500, 1 device) | max abs dF **1.25e-13** |
| 20-step NVE, ranks 1/2/4/8 | final PE **bit-identical across rank counts**, and equal to ASE |
| energy conservation, dt = 0.25 fs | +0.037 meV/atom/ps |

`dt = 1 fs` collapses — that is `Si_tiny`'s missing repulsive core, documented in
`bench/results.md`, not the engine.

**Cost to wire up: ~30 lines.** Three conventions had to be matched: the centre
sits on the *receiver* in `dist/parallel` and on the *sender* in `acejax`
(`rij = -dR`, `zi`/`zj` swapped); ghost species come from
`ghost_exchange_subgraph`'s `node_species`, which `nequix_bench.py` discards; and
gradients are w.r.t. fractional positions, so `F_real = -g_frac @ inv(box)`.

**ACE never calls `exchange_fn`.** It is one-hop, so the per-layer feature
exchange is dead weight — only `ghost_exchange.py` + `local_neighbor_list.py`
(823 of ~1900 lines) are on the ACE path, and **the `[dist]` extra
(`jax-md`, `e3nn-jax`, `nequix`) is not needed at all.**

**Throughput**, as a fraction of raw `acejax` retained (f64, sparse `A2B`):

| atoms | `pair jax/kk` (n_B=69) | **dist** | `pair jax/kk` (710) | **dist** |
|---|---|---|---|---|
| 216 | 0.13 | **0.51** | 0.19 | **0.54** |
| 1728, default capacities | 0.63 | 0.41 | 0.60 | 0.40 |
| **1728, capacities 1.15/1.27** | 0.63 | **0.74** | 0.60 | **0.69** |

**The entire gap is static-shape padding again — the third time in this project.**
`1/ratio` tracks the edge-slot ratio to 1.06–1.20x, and on CPU the correspondence
is exact, so the exchange, masking and `psum` cost nothing measurable. Capacity
below ~1.05/1.1 overflows, detectable at runtime through the overflow flags
rather than silently.

Reneighbouring, which the LAMMPS numbers exclude entirely, is 1.2–2.5 ms at 1728
atoms — ~1.4% of runtime at a 20-step interval.

**Caveats.** Multi-device was verified for **correctness only**, on a forced
8-device CPU mesh; moriarty has one GPU, so there is **no scaling claim** — those
ranks are threads. Single species, orthorhombic cells. The integrator is 15 lines
of NVE: no thermostats, constraints, analysis or file formats, which remains the
reason LAMMPS is not replaced.

Also confirms the Julia question in the strongest form: at 1 rank on CPU the
distributed plumbing costs *nothing* measurable beyond padding, because XLA keeps
the whole step in one compiled region. A Julia port interleaving MPI.jl with
CUDA.jl would not get that for free.

#### Phase 11 (original brief follows)


**Half a day, and it goes before Phase 12 because it may change what Phase 12 is
worth.** Pointed out by the `lammps-jax` author, who notes it is deliberately
under-documented.

`lammps_jax/dist/parallel/` (~1900 lines) is a **LAMMPS-free distributed MD
engine in JAX**: domain decomposition, ghost exchange, force decomposition and
cell-list neighbour building on `shard_map` + `Mesh` + `lax.psum` collectives,
with `jax_md` for partitioning. `dist/scripts/nequix_bench.py` is the driver that
wires a model in.

**The plumbing is model-agnostic** — `ghost_exchange`, `local_neighbor_list` and
`replicate_data` import only jax/numpy; only `feature_exchange` touches e3nn. So
`acejax` should slot in where `NequixCalculator` sits, the interface being
essentially `energy_fn(positions, **kwargs)` over a neighbour structure, which is
close to what `site_energies` already is.

**Why it goes first.** It is a third deployment route, and it sidesteps the
constraint Phase 12 exists to solve:

| route | CPU | GPU | needs LAMMPS |
|---|---|---|---|
| `pair_style jax/kk` | no | yes | yes |
| ML-IAP (Phase 12) | yes | yes | yes |
| **JAX-only distributed MD** | **yes** | **yes** | **no** |

JAX runs natively on both, so the CUDA-only problem does not arise. Half a day
spent here could make Phase 12's 2–3 days unnecessary, or at least reprioritise
them — that asymmetry is the argument for the ordering.

**What it does not replace.** LAMMPS brings thermostats, constraints, analysis,
file formats and community familiarity. This complements that; it does not
substitute for it. Anyone wanting a production MD workflow will still want
LAMMPS, which is why Phase 12 is deferred rather than dropped.

**Gate:** `acejax` runs a distributed MD step through `dist/parallel/` and
reproduces the ASE calculator's energies and forces on the same configuration.
Single device first; multi-device only if that works cleanly.

**Also worth reporting:** whether this is a plausible route for Julia. The author
wonders whether the approach ports without Reactant/Enzyme, and the algorithms
themselves do not need either — MPI.jl and CUDA.jl cover collectives and kernels.
The catch is that the JAX version keeps the **whole step, collectives included,
inside one compiled region**, so XLA overlaps communication with compute; a Julia
port interleaving MPI.jl with CUDA.jl would be correct but would not get that for
free. Note the symmetry with our Reactant findings: fixing the two ET blockers
would give Julia the same whole-step compilation, so "port without Reactant" and
"fix ET's `SelectLinL` and KA paths" are alternative routes to the same property.

#### Phase 12 — a CPU route into LAMMPS, via ML-IAP

**Reviewed after Phases 11 and 13 — still justified, but for one reason only,
and a cheaper alternative should be scoped first.**

The question was whether Phase 11's JAX-only distributed MD makes this
redundant. It does not, and it is worth being exact about why, because the two
phases address different things:

- **Availability is untouched by Phase 11.** `acejax` is the only working export
  route from an ACEpotentials v0.10 model into LAMMPS, and it is GPU-only. A
  standalone JAX MD driver is not LAMMPS: it has no fixes, thermostats,
  barostats, constraints, minimisers, hybrid or multi-potential support, restart
  files, or compatibility with the input decks users already have. Users with
  LAMMPS workflows and no GPU are no better off for the spike existing.
- **CI is only partly addressed by either.** A CPU ML-IAP route exercises
  LAMMPS-supplied neighbour lists, ghost atoms, unit conventions and multi-rank
  decomposition — real integration surface. But it does **not** test
  `pair_style jax/kk` or the StableHLO bundle, which is what actually ships, so a
  green ML-IAP gate is not proof the GPU path works. The Phase 11 spike tests
  even less of it. **The honest fix for the CI hole is a self-hosted GPU
  runner**, as `.github/workflows/acejax.yml` already notes; neither phase should
  be sold as closing it.

**Phase 13 strengthens the case for a CPU backend in `jax/kk` itself.** That was
put out of scope above (~1–1.5 weeks, mostly the buffer layer) and left as
something to raise upstream. Phase 13 gives that pitch real weight: `jax/kk` is
**1.04–1.43× faster than hand-written Kokkos `symmetrix`** on an identical MACE
checkpoint. A CPU backend would therefore extend a plugin that is measurably
competitive, and benefits every model using it rather than only ours. Raise it
with the maintainer before spending 2–3 days on an ML-IAP fallback that is slower
by construction — it calls Python every timestep.

**Cheaper alternative worth scoping first: fix the `yace` export.** Phase 8 found
the `pace` comparator blocked at a *format* mismatch, not a capability one — v0.6
emits `radbasename: "ACE.jl"` with `splinenodalvals`, while upstream ICAMS
libpace expects `ChebPow`/`radcoefficients` plus `deltaSplineBins` and
`nradbasemax`, and upstream does support splined radials. If that is a writer
change rather than a basis conversion, it would give v0.10 models a mature, fast
C++ route into LAMMPS on **both** CPU and GPU (ML-PACE has a Kokkos path), which
is strictly more than ML-IAP delivers. This is ACEpotentials-side work, not
`acejax` work, and its feasibility is **unverified** — scope it before committing
to either route.


**The problem.** `pair_style jax/kk` is CUDA-only — `scripts/build_lammps_jax.sh`
hard-errors without a GPU, and the pair style has no CPU path. Since `acejax` is
currently the **only** working export route from an ACEpotentials v0.10 model
into LAMMPS (`export2lammps` was retired with the ACE1 backend, see Phase 8),
GPU-only is an availability gap, not just a performance one.

**The route.** LAMMPS's ML-IAP interface calls a Python class implementing
`MLIAPUnified` (`lammps.mliap.mliap_unified_abc`). `mace-jax` already does
exactly this for a JAX model in `mace_jax/calculators/lammps_mliap_mace.py`
(~440 lines), **including runtime device selection with explicit CPU support**
(`MACE_ALLOW_CPU` / `MACE_FORCE_CPU`). The pattern is proven in this ecosystem,
and porting it to `acejax` is Python work — no C++, no PJRT, no Kokkos.

**What it costs.** The model is called through the Python interpreter each
timestep rather than staying resident on device, so it will be slower than
`pair_style jax/kk` on a GPU. That is the right trade for a fallback: this route
buys *availability and portability*, and `jax/kk` remains the fast path where a
GPU exists.

**A second benefit, which may matter more.** Phase 10 had to document the LAMMPS
integration gate as a **manual step**, because `jax/kk` is CUDA-only and GitHub
runners have no GPU. A CPU LAMMPS built with `-D PKG_ML-IAP=yes -D PKG_PYTHON=yes
-D MLIAP_ENABLE_PYTHON=yes` builds fine on a runner — so this route makes the
LAMMPS integration **testable in CI** for the first time, with the build cached
exactly as Phase 10 already sets up. That closes the one hole in the CI story.

The current moriarty build has only `KOKKOS MANYBODY ML-PACE PLUGIN`, so it needs
ML-IAP and PYTHON added — build into a new directory, as with the ML-PACE
rebuild, and leave the working build alone.

**Gate:** the ML-IAP route reproduces `acejax`'s own energies and forces on the
same configuration to the tolerances Phases 3–4 achieved, **running on CPU**, and
agrees with `pair_style jax/kk` where a GPU is available.

**Not in scope:** a CPU backend in `pair_style jax/kk` itself. That is ~1–1.5
weeks and lands mostly in the buffer layer — `CUdeviceptr` appears 13 times, and
host pointers would go through `PJRT_Client_BufferFromHostBuffer` instead; the
Kokkos functors are already backend-agnostic behind `#ifdef KOKKOS_ENABLE_CUDA`,
the plugin path is configurable, and the stream handoff
(`client_session.cpp:69`, "PJRT CUDA stream extension is required") needs a
simpler synchronous CPU branch. Worth raising with the `lammps-jax` maintainer,
since it benefits every model using the plugin rather than only ours.

#### Phase 13 — how does this compare with MACE?

The question people will actually ask. Phase 8 answers "ACE in JAX versus ACE in
C++"; this answers "ACE versus the foundation models most users reach for".

**Three routes, not two — and the third is what makes this informative.**

| # | what | pair style | isolates |
|---|---|---|---|
| 1 | ACE, ours | `jax/kk` | — (the Phase 8 baseline) |
| 2 | MACE via `symmetrix` | `symmetrix/mace` | MACE at its best, hand-written Kokkos |
| 3 | **MACE via `lammps-jax`** | `jax/kk` | **the model, with plumbing held constant** |

`lammps-jax` ships a MACE exporter (`examples/export_mace.py`, MACE-MP-0 small,
`mace_jax`-based, plus `python/lammps_jax/mace.py` and `tests/test_mace.py`), so
MACE can be run through **the same plugin, the same StableHLO path, and the same
padding and ghost overheads as our ACE model**. That makes two comparisons
possible that a two-way study cannot give:

- **1 vs 3** isolates *model cost* — identical plumbing on both sides, so the
  difference is ACE versus MACE and nothing else. This is the cleanest number in
  the whole phase.
- **2 vs 3** isolates *the plugin* on an identical model — hand-written Kokkos
  MACE against the same MACE through `jax/kk`. That is a direct measurement of
  the integration tax we characterised in Phase 8 (ghost rows 2.19×, pad rows
  1.38×), now on someone else's model rather than our own, which is a much
  harder result to argue with when raising it upstream.

Note the local `lammps-jax` checkout is at `a4304a2` ("fp64 support") and may be
behind upstream; check before building, since the MACE path is newer than the
ACE work we based Phase 6 on.

**Route 2: `symmetrix`** (`wcwitt/symmetrix`) — a Kokkos implementation of MACE
with a LAMMPS pair style, `pair_style symmetrix/mace`, models loaded from
`.json`:

```
pair_style    symmetrix/mace
pair_coeff    * * my-mace-1-8.json Si
```

Two things make it a clean peer to `jax/kk`. It needs **LAMMPS 10 September 2025
or newer** — the same requirement `pair_jax_kokkos.h` enforces, so our existing
build qualifies — and it is Kokkos-based, so both sit on the same LAMMPS
infrastructure rather than one being advantaged by a different integration path.
Build needs CMake >= 3.27, C++20, `PKG_KOKKOS=ON`, and `Kokkos_ENABLE_CUDA=ON`
for GPU. Into a new build directory, as always.

**Framing, which matters more here than in Phase 8.** Basis size **cannot** be
matched: MACE is a message-passing network with a fundamentally different cost
structure — layers, channels, message dimension — where ACE is a fixed
one-shot basis contraction. So this is **not** a cost-at-matched-complexity
comparison like Phase 8. It answers the practical question: *on the same host and
the same structures, what throughput does a user get?*

Report enough for a reader to interpret the gap rather than just read a ratio:
parameter count, cutoff, number of message-passing layers, and channel width for
each MACE model, against basis size and cutoff for ours. Two or three foundation
model sizes (e.g. MACE-MP-0 small and medium) give a size trend on their side,
mirroring what Phase 8 did on ours.

Same Si diamond supercells at the same sizes, same `timestep 0.0` single-point
method, same exclusions restated. MACE foundation models are universal, so Si is
in scope for them.

**Use the same MACE model in routes 2 and 3** wherever the two exporters both
support it, or the 2-vs-3 comparison measures two different things at once. If
they cannot be matched, say so and drop that comparison rather than reporting it
with a caveat nobody will read.

**Expect to lose on raw throughput at these sizes, and say so plainly.** A
foundation model carries far more parameters than a 110- or 2000-function ACE
basis, and buys generality with them. The useful output is a number a reader can
weigh against that generality, not a favourable ratio.

**One incidental finding worth noting:** `symmetrix` ships **both CPU (OpenMP)
and GPU (CUDA)** Kokkos paths for its pair style. That is direct evidence that a
Kokkos ML pair style can support CPU, reinforcing Phase 12's point that
`jax/kk` being CUDA-only is a code-structure choice rather than an inherent
constraint. Worth citing if the CPU backend is ever raised with the `lammps-jax`
maintainer.

#### Phase 14 (optional) — a traced neighbour list, for end-to-end differentiability

**Not needed for correctness.** All three shipped backends — `matscipy-neighbours`,
`matscipy`, the numpy fallback — are host-side C or numpy; `nlist.py` never
imports `jnp`, so the neighbour list is built **outside the trace** and the JAX
graph starts at the edge vectors.

That is fine for forces and virials, and it is worth being precise about why: the
neighbour list is a *discrete selection*, and with an envelope going smoothly to
zero at the cutoff, `dE/dr` is exactly right with the list held fixed — pairs
crossing the boundary contribute nothing there. Our forces match Julia to 1e-13
with the NL entirely outside the graph, and the strain trick differentiates edge
vectors rather than the selection, so the virial is unaffected too.

**What it does block** is anything needing the graph to extend through neighbour
construction:

- backpropagating through an MD **trajectory** — learning a potential from
  trajectory data, inverse design, optimising initial conditions
- rebuilding the neighbour list **inside a `jit`-ed loop** rather than paying a
  host round-trip per rebuild

**Two backends already qualify.** `jax_md.partition.neighbor_list` splits into
`allocate` (sizes buffers, not jittable) and `update` (jittable, traced), with a
`did_buffer_overflow` flag precisely so a rebuild can happen inside a traced loop
and signal when capacity was exceeded — differentiable simulation is jax-md's
founding use case. `lammps-jax`'s `dist/parallel/local_neighbor_list.py` (273
lines, pure `jnp`/`lax` cell list over fractional coordinates) qualifies for the
same reasons.

**The adapter is already the right shape.** The neighbour-list interface returns
edge vectors + segment ids + mask, and pooling sits behind one swappable
function, so a traced backend slots in beside the existing three rather than
replacing them. Phase 11's spike runs on `dist/parallel`, which builds on
`jax_md.partition` — so if `acejax` composes with it cleanly, that is most of the
evidence this phase needs, obtained for free.

**Gate:** a short MD trajectory run entirely inside `jit`, with `jax.grad` taken
through it w.r.t. a model parameter or the initial positions, giving a finite
gradient that matches a finite-difference check.

#### Phase 15 — conditional gather/matmul swapover for `edge_A`

**Status: implemented on CPU; the GPU half is outstanding.** `edge_a_kind`
("gather" | "matmul") is a static field on `ACEModel`, selected by
`load(..., edge_a_kind=...)`, with `with_edge_a_kind` to switch an existing model
and `calibrate_edge_a` to time both forms at real shapes and return the faster.
`export_bundle.py` takes `--edge-a-kind` and bakes it in. Ten tests in `tests/test_edge_a.py` check equivalence. **Values** are
bit-identical in both dtypes, and **f64 gradients** are too. **f32 gradients are
not**: the two adjoints are a scatter and a matmul and XLA may accumulate them in
different orders — 0.0 on Apple Silicon, ~4e-6 on x86, so the f32 gradient check
is a relative tolerance. An earlier version of this note claimed bit-identity for
f32 gradients as well; that was measured on one architecture and generalised, and
CI on x86 is what caught it. Suite 66 -> 76 tests.

Two things deliberately not done:

- **No `"auto"` at load time.** Calibration needs the real edge-buffer length,
  which `load` does not know. `calibrate_edge_a` is explicit and takes the actual
  arrays; an auto mode that guessed from shapes at load would be the heuristic
  this phase exists to avoid.
- **GPU is still unmeasured**, and still gates any default. The whole LAMMPS path
  runs on GPU, where the scatter may behave differently; until that is measured,
  "gather" remains the default everywhere and `--edge-a-kind` is opt-in.

`export_bundle.py`'s change is **unexercised locally** -- `lammps_jax` is not
installed on the dev Mac, so it is syntax-checked only and needs running once on
a GPU host.

`docs/findings/FINDINGS_apple_scaling.md` established that
`edge_A = Rnl[:, aspec_r] * Ylm[:, aspec_y]` (`acejax/model.py:156`) is the
scaling bottleneck: the forward gather is cheap and flat, but its reverse-mode
adjoint is an axis-1 scatter whose cost per slot climbs with buffer length. An
algebraically identical one-hot matmul form, `(Rnl @ Sr) * (Ylm @ Sy)`, has a
matmul adjoint instead and is flat. The gradients are **bit identical**.

**Neither form wins everywhere, which is the whole reason this is a phase and not
a patch.** ns per edge slot, jax 0.11.1, f64:

| edge slots | M3 Pro gather | M3 Pro matmul | Xeon gather | Xeon matmul |
|---|---|---|---|---|
| 17057  | 116.4 | 71.2 | 27.3 | 64.0 |
| 124054 | 303.1 | 52.1 | 42.6 | 46.8 |
| 496224 | 500.8 | 45.1 | 53.5 | 39.8 |

On the M3 Pro the matmul wins everywhere and by up to 11×. On the Xeon the
gather wins below roughly 200k slots — by 2.3× at the smallest size — and loses
only above it. **Hardcoding the matmul would be a 2.3× pessimisation at the
buffer lengths most x86 runs actually use.**

**Design.**

- Both forms behind one swappable function, exactly as the pooling function
  already is. No branching inside the traced kernel.
- An explicit option, `edge_a_kind="gather" | "matmul" | "auto"`, default
  `"auto"`.
- **`"auto"` calibrates, it does not guess.** Do *not* dispatch on
  `platform.machine()`, device kind, or a hardcoded slot threshold: the crossover
  is a property of the XLA backend, the dtype and the actual shapes, and the GPU
  behaviour is entirely unmeasured. Time both forms once at the real shapes —
  milliseconds — and cache the winner keyed on (backend, device kind, dtype,
  `n_edges`, `n_r`, `n_y`, `n_A`). A calibration that is wrong is still only as
  bad as the loser, whereas a heuristic that is wrong is silently wrong forever.
- **`jax.export` must bake the decision in.** The LAMMPS plugin has no
  calibration step and cannot run one, so the exporter takes `edge_a_kind`
  explicitly and records it in the bundle alongside `ybasis_kind`. Exported
  bundles are for a known target; that is the moment to choose.

**Watch the memory claim.** `Sr` is `(n_r, n_A)` and `Sy` is `(n_y, n_A)` — small
and constant — and `Rnl @ Sr` is `(E, n_A)`, the same shape the gather produces.
So the matmul form is not a memory regression at the sizes measured. Confirm this
still holds at production `n_A` before defaulting to it anywhere, since `n_A`
grows with the model and these were measured at `n_A = 43`.

**Tests.** Equivalence of both forms on values *and* gradients, to bit-identity,
in both f32 and f64 and on both sparse and dense pooling layouts; a test that
forcing each kind actually selects it; and a regression that an exported bundle
round-trips its `edge_a_kind`.

**Gate:** on both an Apple Silicon and an x86 host, `"auto"` matches or beats the
better of the two fixed choices at every size in
`acejax/bench/apple_scaling/`'s series, and no existing test changes its result.

**Unmeasured, and a prerequisite for the default:** GPU. Every number above is
CPU. The scatter may behave completely differently on a GPU, where the whole
LAMMPS path runs — so measure there before `"auto"` is allowed to pick the matmul
for GPU export.

**The honest caveat, which belongs in the write-up:** differentiating through a
trajectory has costs the neighbour-list choice does not fix — memory for every
intermediate across all steps, and a real non-smoothness whenever the neighbour
*set* changes. Fixed capacity plus an overflow flag manages that; it does not
remove it. Anyone reaching for this should know the discontinuity is inherent,
not an artefact of the backend.

**Optional because nothing currently needs it.** It is the enabling step for
differentiable-simulation work, and worth doing when that work is actually
wanted rather than speculatively.

### Stage 2 — revised after Stage 1

Stage 1 changed four things that bear on Stage 2. Taking them in order of how
much they move the plan:

**1. The linear, convex fit is the priority — it is ACE's differentiator, not
its legacy path.** A linear-in-parameters model with a convex loss has a unique
global optimum: no initialisation, seed, schedule or early-stopping sensitivity,
reproducible to the digit, fits in seconds to minutes, and admits calibrated
uncertainty through BLR and committees. That is precisely what MACE and the other
message-passing potentials cannot offer, and it is worth protecting rather than
trading away. **Stage 2A is therefore the linear fit**, and the non-linear
gradient-based path is Stage 2B — optional, exploratory, and explicitly a
trade of convexity for expressivity.

An earlier draft of this section recommended the reverse, on the grounds that
non-linear fitting is cheaper to build on Stage 1 and is the capability Julia
lacks. Both of those remain true (see below) and neither is a good reason to
lead with it: cheapness to implement is not the same as value delivered, and the
capability Julia "lacks" is one whose absence is partly a deliberate design
choice.

**2. Fit assembly is GPU-only, and that is now firmer than at Phase 0.** Gate 3
already showed the JAX CPU hybrid Jacobian is 1.5× *slower* than Julia's
`ET._jacobian_X` at 512 atoms. The Molly benchmark (`acejax/bench/molly/`) adds
that JAX's CPU standing is architecture-dependent: 9–11× faster per thread than
Julia on a Xeon, but **slower than single-threaded Julia at every size tested on
an M3 Pro**, where JAX throughput falls 2.3× across 216→1728 atoms while
ACEpotentials stays flat. Whatever that turns out to be
(`docs/findings/FINDINGS_apple_scaling.md`), it means CPU fit assembly in JAX has
no reliable advantage anywhere and a real disadvantage on some hardware. **If
there is no GPU, assemble in Julia.** This bounds *where* 2A is worth running; it
does not weaken the case for doing it.

**3. There is a route that could deliver 2A in Julia instead.** Reactant.jl
compiles Julia to the same StableHLO that lammps-jax consumes. Stage 1
established that the standard ETACE path is two specific stopgaps from tracing
(`SelectLinL`'s KA kernel, and `ka_with_reactant` dispatch), that SpheriCart
traces fine, and that the array-op formulation of `SelectLinL` traces exactly
(4.44e-16). Two blockers have since moved: the WignerD dependency that capped
Reactant at 0.2.222 is fixed in EquivariantTensors.jl#143 (**merged**), and the
miscompilation is filed as Reactant.jl#3267 with the offending pass bracketed. If
those land, **Julia gets GPU-accelerated linear fit assembly without a Python
port at all** — the same outcome, reached more cheaply and staying in one
language. This gates *implementation venue*, not priority: the linear fit matters
either way.

**4. Padding is the dominant overhead, and Stage 1 measured its shape.** At
n_B=2849 padding accounts for essentially all of the LAMMPS plugin gap; ghost
rows cost 2.19× and pad rows 1.38×. Stage 2's batching is the same mechanism
applied to training-shaped workloads, so **bucket boundaries should be chosen
from that measured curve**, not picked for convenience. A free inheritance from
Stage 1.

#### Stage 2A — linear fit in JAX (the priority)

| Phase | Days |
|---|---|
| 16. **Gate: re-test Reactant** on a CUDA host (ET#143 is **merged**, so current Reactant installs alongside ET). If the standard ETACE path traces, build this in Julia instead and stop here | 0.5 |
| 17. E/F/V design-matrix assembly + custom hybrid JVP (**committed**, gate 3) | 3 |
| 18. Blocked / streaming assembly so the full design matrix need not be resident — **load-bearing for distillation**, not just scaling | 1–2 |
| 19. Priors + solvers (lineax / optimistix), incl. BLR for uncertainty — **BLR drives teacher-vs-student config generation**, so not deferrable | 2–3 |
| 20. Data loading, weights, key matching — schema must carry **teacher-model labels with provenance**, not assume DFT keys | 1 |
| 20b. Clear the jax pin blocking `mace-jax` coexistence (PR to `sphericart-jax`, or pure-JAX harmonics) — **prerequisite for distillation** | 0.5–2 |
| *20c. Frozen element embeddings (tensor reduction, PRL 131 028001), JAX-side* — **fallback only**; prefer the Julia route (see (a)). Known-incomplete; gate on a degeneracy probe, not an RMSE comparison | 3–5 |
| 21. Validation harness (milestones 5–6): coefficients and RMSE against `acefit!` | 1 |

**≈ 9–12.5 days**, plus 3–5 optional for the embedding phase. Several of those
phases are shaped by the long-term directions above rather than by the linear fit
alone; see "Long-term directions,
and what they constrain now". The frozen-embedding phase is costed but marked optional: it
needs only a *single-species* export from Julia, so it is not blocked on
ACEpotentials.jl, but it creates a JAX-only model class — see the cost noted
under (a).

Phase 18 is new, and follows from wanting this to scale: at n_B ≈ 2000 and a
large dataset the design matrix is the memory bottleneck, not the arithmetic.
Note the obvious shortcut is a trap — accumulating normal equations `AᵀA`
(n_B × n_B, dataset-size-independent) squares the condition number, which is why
ACEfit uses QR and LSQR rather than normal equations. Blocked QR or a streaming
solver keeps the conditioning and the memory. **Do not quietly swap in normal
equations for convenience**; it would change the fits, not just the speed.

Hard requirements inherited from Stage 1, all three from measurement rather than
caution:

- **Assemble in f64.** Not for accuracy alone: the XLA autotuning miscompile that
  produces a structurally wrong Jacobian element is confined to f32 and the
  Jacobian einsum, and f64 was clean 10/10.
- **Pin matmul precision explicitly** and verify the setting survives
  `jax.export`. TF32-by-default costs 400× accuracy in the forward descriptor.
- **Set `--xla_gpu_autotune_level=0`** for assembly, or verify per-process that
  the Jacobian is stable. `precision=highest` alone does not fix it (2/10 still
  wrong).

For phase 20, prefer exporting `AtomsData` from Julia over reimplementing
ACEpotentials' fuzzy key matching and per-configuration weights. That logic is
fiddly, well-tested, and has no business being written twice.

#### Stage 2B — non-linear fitting (optional, and a real trade)

| Phase | Days |
|---|---|
| 22. Batched loss over configurations (E, F, optionally V) with per-config weights | 1–2 |
| 23. Bucketed batching, boundaries set from Stage 1's padding curve | 1 |
| 24. `optax` training loop, schedules, checkpointing | 1–2 |
| 25. Validation: refit `Si_tiny`, compare against the Stage 2A linear fit | 1 |

**≈ 4–6 days**, prerequisite nothing beyond Stage 1. Cheap — but it buys
expressivity by giving up the unique global optimum, reproducibility, and the
BLR/committee uncertainty story. It should be offered as an alternative
backend, never as the default, and any comparison against the linear fit should
report the cost as well as the accuracy.

`Wnlq` was deliberately kept live rather than folded into the spec so the radial
basis is trainable here — that decision still pays off, and costs nothing if 2B
is never built.

**AD structure — this needs the right mode, not just `grad` of `grad`.** For the
force term of the loss, `L_F = ‖F - F_ref‖²`, what is needed is

```
∂L_F/∂θ = v · ∂F/∂θ,   v = 2(F - F_ref) held constant
```

and since `F = -∇_r E` that is the mixed second derivative `∂²E/∂θ∂r` contracted
with `v` on the `r` side. The efficient composition is **reverse-over-forward**:
`v·∇_r E` is a directional derivative, so one forward-mode JVP in `r` gives it as
a scalar, and one reverse pass in `θ` then gives the gradient —

```python
jax.grad(lambda p: jax.jvp(lambda r: E(p, r), (r,), (v,))[1])(θ)
```

Reverse in `θ` because `n_θ >> 1` and the loss is scalar; forward in `r` because
only the single direction `v` is wanted, not all `3N`. Naive
reverse-over-reverse (build `F`, then backprop the loss through that tape) is
correct but tapes the backward pass, costing materially more memory. Forward in
`θ` is the one combination that is plainly wrong: `O(n_θ)` passes. `F` is still
computed once per batch by reverse mode, since both the loss value and `v` need
it, so a step is two passes — the second with a small tape.

**Gate 3's Jacobian cost does not apply to 2B.** 2A needs *all* `n_B` columns of
`∂B/∂r` (hence 43–238× forward on CPU, 9–20× on GPU); 2B needs only the single
contraction `v · ∂F/∂θ` per batch — roughly 4–6× forward, and **independent of
`n_B`**. These factors are analytic, not measured; confirm them in phase 22.

**Note the two are not interchangeable.** For a model linear in its parameters,
`∂F/∂c` **is** the design matrix, so a gradient-based fit computes the same
object by a slower route and converges to the answer 2A obtains directly. 2B is
justified only where the parameters are genuinely non-linear.


#### Sequencing

Recommended: **phase 16 (the Reactant gate) first**, since it is half a day and
decides whether 2A is built in Python or in Julia. Then 2A. 2B only if and when a
specific model needs parameters the linear form cannot express — and even then as
an additional backend, not a replacement.

Sequence: spike → exporter → descriptor on CPU against matscipy-neighbours → GPU →
LAMMPS export. Once the descriptor is trusted, validating the export is one
comparison; doing it while the descriptor is still moving means debugging two
unknowns through a C++ plugin boundary.

**Review point at the end of Stage 1 — now reached.** The descriptor is validated
(1.4e-15), the LAMMPS path works for single points and short runs, and the
benchmarks against Kokkos, ML-PACE and Molly.jl exist. The Stage 2 section above
has been rewritten against those results rather than against the original
estimates. Stopping at Stage 1 remains a good outcome, not a failure.

## Open issues found in Phase 8

Two bugs, both bounding what Stage 1 can currently claim.

**1. ~~`jax/kk` aborts beyond ~50 MD steps~~ — RESOLVED, not a code defect.**
`compute-sanitizer` put the fault in LAMMPS's own `NBinKokkos::bin_atoms()`, and
the root cause is that the `Si_tiny` test potential has **no repulsive core**:
the dimer curve turns over at ~1.5 Å and diverges attractively, so atoms
collapse. Reproduced in pure ASE NVE with no LAMMPS involved. The pipeline is
vindicated end to end; refit with `acefit!(..., repulsion_restraint = true)`
before re-testing energy conservation. See `docs/findings/FINDINGS_lammps.md`.
Original symptom, for reference:
216 atoms: 20 steps OK, 50 OK, **100 / 200 / 300 abort**. Not capacity — tripling
`max_atoms` and `max_edges` does not help; what changes between 50 and 100 steps
is neighbour-list rebuilds, so the repack path after reneighbouring is the
suspect. The plugin does not validate capacity at run time, so a genuine overflow
would present identically.

This retroactively explains the Phase 6 NVE run that "did not finish in the
timeout" — it was almost certainly aborting, not running slowly. **So Stage 1's
LAMMPS path is verified for single-point evaluation and short runs, not for
production MD.** Energy conservation remains unestablished, and cannot be
established until this is fixed.

**2. Shared-library shadowing invalidates rebuilds silently.**
`BUILD_SHARED_LIBS=ON` puts every style in `liblammps.so`, and the documented
`LD_LIBRARY_PATH` recipe puts `$V/lib` first — so a newly built `lmp` silently
loads the *old* library. `lmp -h` showed no pace styles and `pair_style pace`
reported ML-PACE "not enabled" while CMake had reported
`Enabled packages: KOKKOS;MANYBODY;ML-PACE;PLUGIN`. Anything benchmarked without
noticing would have been the old binary. Worth raising upstream.

**The `pace` comparator is blocked, one layer deeper than expected.** v0.6.12
resolves, fits `Si_tiny`, and exports a yace whose **basis size matches exactly —
110 = 110** against v0.10 at the same order, cutoff and element. But it will not
load: v0.6 emits `radbasename: "ACE.jl"` with `splinenodalvals`, while upstream
ICAMS libpace expects `ChebPow`/`radcoefficients` plus `deltaSplineBins` and
`nradbasemax`. The `wcwitt` fork does carry `acejl_radial.cpp` and parses
further, then fails on `map::at` — version skew between this ACEpotentials
vintage and the fork's `main`.

## Risks

| Risk | Stage | Severity | Mitigation |
|---|---|---|---|
| ~~Convention mismatches (indexing, `lm2idx`, `𝔸spec` sort)~~ | 1 | Resolved | Phase 0: all intermediates clean to 1.4e-15 (GPU) / 5.1e-15 (CPU) |
| ~~Splined radial export branch~~ | 1 | Resolved | Phase 1: 102 B-spline coeffs, 76 LOC, bit-for-bit with Julia |
| Silent basis-convention drift (spherical vs solid) | 1 | Medium | Export `ybasis_kind`; keep per-stage probe values |
| ~~`jax/kk` aborts beyond ~50 MD steps~~ | 1 | **Resolved — not a code defect** | The Si_tiny test potential has no repulsive core; reproduced in pure ASE with no LAMMPS. Refit with `repulsion_restraint=true` |
| TF32 silently degrades f32 descriptor to 1.2e-3 | 1 | Medium | Pin matmul precision; verify it survives `jax.export` |
| XLA autotuning miscompiles the Jacobian einsum (f32) | **2** | Medium | f64 assembly; or `--xla_gpu_autotune_level=0`; report upstream |
| ~~LAMMPS build/run host mismatch~~ | 1 | Resolved | Run host is `moriarty` (Xeon 4216 + A4500), not lestrade; `/home` and `/storage` shared, `/tmp` not |
| `sphericart-jax` emits a custom call | 1 | Medium | Implement spherical harmonics in pure JAX; also drops the jax 0.10.1 pin |
| ~~NaN gradients from padded edges~~ | 1 | Resolved | Phase 1 (sparse) and Phase 4 (dense `neighbour_matrix` zero slots); pad at the cutoff, regression tests both layouts |
| XLA compile blowup | 1 | Medium | Never unroll per basis function; gather over `(n_AA, max_order)` |
| Padding tax (capacity ≫ typical) | 1 | Low | Bucket tightly |
| matscipy-neighbours not on PyPI | 1 | Low | `pip install git+...`; in-house to fix |
| ~~Design-matrix assembly too slow via generic AD~~ | **2** | Resolved | Confirmed on CPU and GPU: hybrid JVP required, now committed |
| ~~f64 GPU throughput on non-datacenter cards~~ | **2** | Resolved | Measured 3–5.4× f32 on a 1/64-rate card; memory-bound, not FLOP-bound |
| JAX CPU throughput degrades 2.3× with size on Apple Silicon | **2** | Medium | Under investigation (`FINDINGS_apple_scaling.md`). Bounds CPU fallback for fit assembly; GPU path unaffected |
| Training-step cost (grad of grad) larger than expected | **2A** | Medium | Measure in phase 16 before committing to the loop design |
| 2B duplicates work Reactant may soon do in Julia | **2B** | Medium | Phase 19 gate: re-test tracing with ET#143 before starting 2B |

Phase 0 retired all three highest-severity items: convention-matching is confirmed
clean to 1.4e-15 on GPU, the design-matrix Jacobian question is settled (the hybrid
JVP is required, and is now scoped work rather than an unknown), and f64 GPU
throughput turned out to be a non-issue for this memory-bound kernel. Stage 1's
remaining risk is dominated by the LAMMPS build environment and the splined-radial
export branch.

## Open questions

1. ~~Does the `sphericart` JAX binding expose *solid* harmonics at `normalisation=:L2`?~~
   **CLOSED (Phase 0).** `sphericart.jax.solid_harmonics(xyz, l_max)` matches P4ML's
   `real_solidharmonics(L; normalisation = :L2)` to 4.8e-15 with no normalisation
   argument — the defaults already agree. `sphericart-jax` pins jax 0.10.1, which is
   also what lammps-jax recommends.
2. ~~`BCOO` versus dense for `A2B`?~~ **CLOSED.** Dense is fine at 110 basis
   functions (0.9% occupied) but not at scale: at 1429 functions `A2B` is
   1429 x 11474 with one nonzero per column, 0.070% occupied and 131 MB in f64.
   Timed on GPU it was **99.9% of `site_basis`**, and replacing the matmul with a
   gather plus segment-sum gave an **8.1x** end-to-end speedup in f64. Not BCOO:
   one nonzero per column makes an explicit gather simpler and faster. Exact, not
   approximate. Off by default (`load(..., a2b_sparse=True)`) because dense wins
   at small basis. A2B is also stored as triplets now — the 1429 npz went from
   132 MB to 1.3 MB.
3. Dense `neighbour_matrix` or sparse edge list as the CPU default? Not yet measured;
   the swappable pooling function makes this reversible.
3. Should the exporter live in ACEpotentials.jl (as a `scripts/` entry point) or in
   the new Python repo as a Julia sidecar? Former is easier to keep in sync.
4. Is there a case for an `acesuit-jax-common` package shared with mace-jax
   (nlist adapter, strain helper, ASE calculator)? Premature — vendor now, extract
   later if it proves out.

## References

- `docs/plans/lammps_jax_benchmark_reference.md` — the `pair jax/kk` vs native
  Kokkos EAM throughput benchmark that Phase 8 should reproduce for ACE
- `lammps-jax` — https://github.com/abhijeetgangan/lammps-jax (MIT)
- `matscipy-neighbours` — https://github.com/libAtoms/matscipy-neighbours (MIT)
- `mace-jax` — https://github.com/ACEsuit/mace-jax (MIT)
- Equinox ecosystem — https://docs.kidger.site/equinox/
- `benchmark/FORCE_REGRESSION_FINDINGS.md` — Julia force-path baselines
- `CLAUDE.md` — "Future Work: Full Lux-based Backend Migration"
