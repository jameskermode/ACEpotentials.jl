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

Adds design-matrix assembly, priors, solvers and data loading. Justified when you
want GPU-accelerated fit assembly over large datasets, or as the platform for
non-linear fits.

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
| 0. Performance spike (gating; ~1 day if stopping at Stage 1) | 1–2 |
| 1. Julia exporter (fitted model, splined radials) + Python loader | 2.5–3.5 |
| 2. Descriptor forward | 4 |
| 3. Neighbour-list adapters + fixed-capacity plumbing | 1 |
| 4. Energy / forces / virial via `jax.grad` + ASE calculator | 1 |
| 5. Validation harness (milestones 1–4) | 2 |
| 6. LAMMPS export + integration testing | 4–5 |
| ~~7. `ace_model` support: analytic radials + solid harmonics~~ ✅ `47aa6837` | 2 |
| 8. Throughput benchmark vs Kokkos ⚠️ partly blocked (`59f56d90`) | 1 |
| ~~9. Usable ASE calculator + descriptor access~~ ✅ `18ff57bd` | 1.5 |
| 10. Whole-branch review, reorganise to `acejax/`, README, CI, PyPI | 2.5–3 |
| 11. JAX-only distributed MD spike (no LAMMPS) | 0.5 |
| 12. LAMMPS ML-IAP route: CPU support, and a CI-testable path | 2–3 |
| 13. MACE comparison via `symmetrix` on GPU | 2 |

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

#### Phase 11 — a JAX-only distributed MD spike (no LAMMPS)

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

**Route: `symmetrix`** (`wcwitt/symmetrix`) — a Kokkos implementation of MACE
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

### Stage 2 — fit in JAX (incremental)

| Phase | Days |
|---|---|
| 7. E/F/V design-matrix assembly + custom JVP (**committed**, gate 3) | 3 |
| 8. Bucketed batching for training-shaped workloads | 1 |
| 9. Priors + solvers (lineax / optimistix) | 2–3 |
| 10. Data loading, weights, key matching | 1 |
| 11. Validation harness (milestones 5–6) | 1 |

**≈ 7–9 further days.** Full scope ≈ 22–27 days ≈ 4.5–5.5 weeks.

### Non-linear fits (further, optional)

**≈ 5–8 days** on top of Stage 2: optax loop, schedules, checkpointing.

Sequence: spike → exporter → descriptor on CPU against matscipy-neighbours → GPU →
LAMMPS export. Once the descriptor is trusted, validating the export is one
comparison; doing it while the descriptor is still moving means debugging two
unknowns through a C++ plugin boundary.

**Review point at the end of Stage 1.** By then the descriptor is validated, LAMMPS
works, and the marginal cost and value of Stage 2 are both much better understood
than they are today. Stopping there is a good outcome, not a failure.

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
