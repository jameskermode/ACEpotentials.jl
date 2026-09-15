# Closing the gap to ML-PACE in LAMMPS — design

**Date:** 2026-09-15. **Status:** approved design; implementation plan to follow.

## Problem

Students will not adopt `acejax` + `pair_style jax/kk` at a 3x throughput
deficit to `pair_style pace/kk`. Measured like-for-like (both in LAMMPS, Kokkos
GPU, f64, 1728 Si atoms, moriarty RTX A4500; `acejax/bench/results.md`):

| n_B (ours vs pace) | pace/kk ÷ jax/kk | acejax in Python ÷ pace/kk | retention through the plugin |
|---|---|---|---|
| 69 vs 78 | **3.0x** | 1.9x | 0.63 |
| 710 vs 693 | **2.1x** | 1.3x | 0.60 |
| 2849 vs 2874 | **2.6x** | 0.9x | 0.34 |

At production basis size the kernel already matches pace's in-LAMMPS
throughput; the deficit is the plugin path. Two causes are measured and one
lever is unexploited:

1. **Atom-axis padding.** `lammps/export_bundle.py` passes `positions.shape[0]`
   (= `max_atoms`, sized for nlocal + nghost) as `n_nodes`, so the per-atom
   stages — `AA` products, `A2B` contraction, readout — run over every ghost and
   pad row (3.1x nlocal at 1728 atoms) and are masked afterwards. The padding
   sweep showed retention tracks per-atom work, which is why it is worst at the
   largest basis.
2. **Edge-axis padding** is a flat ~1.4x, and capacity cost is U-shaped (2.4x
   penalty at both ends; optimum is per point).
3. **No coefficient folding.** `_from_pooled` materialises `B = A2B·AA` (largest
   isolated stage, 54% at n_B = 2849) and `_readout` then dots it with `WB`.
   The model is linear in `B`, so `A2Bᵀ·WB` can be precomputed — PACE's C-tilde
   basis is exactly this fold.

Students also run on CPU clusters. `jax/kk` is CUDA-only, and XLA-CPU on this
workload was slower than single-threaded Julia in the assembly tests, so a JAX
CPU backend would lose to ML-PACE's recursive C++ evaluator by more than 3x,
not less. The only realistic CPU parity is running under `pace` from an exact
export — kept optional, in this repo, and CI-gated against a pinned libpace,
so a change at their end fails a test here rather than a student's job.

## Constraints

- Single source of truth: the fitted Julia model. `acejax` and any yace
  exporter are derived from it and live in this repo.
- Not tied to ML-PACE: `acejax` remains the primary deployment path; yace is
  an optional exact route for the models it can express (linear/FS readouts,
  per-pair radials). Embedded models (`ace_embedding_model`) are JAX-only —
  expressing them in yace requires the O(Sᵛ) categorical expansion.
- Exactness is not negotiable: every change is verified against the current
  path to ≤1e-10 on energies, forces and virial before it is timed.
- Attribute by difference, never by isolated stage timing (that rule has been
  earned three times in `results.md`).

## Package 1 — plugin-path fixes (GPU)

All in-repo (`acejax/`), no change to `lammps-jax` needed. Landed and measured
in the order A → B → C so each has a clean difference measurement.

### A. Fold the readout through `A2B`

For a linear readout,
`e_i = Σ_b WB[b, z_i] B_ib + Σ_p Wpair[p, z_i] Apair_ip + E0[z_i]` with
`B = AA · A2Bᵀ`, so `e_i = Σ_a c̃[a, z_i] AA_ia + …` with `c̃ = A2Bᵀ · WB`,
shape `(n_AA, NZ)`, computed once at load.

- `acejax/model.py`: a `fold_readout(model) -> model` transform that stores
  `ctilde` (a live array leaf) and sets a static `folded: bool`. When folded,
  `site_energies` bypasses `_from_pooled`'s `A2B` branch and `_readout`
  contracts `AA` against `ctilde[:, node_z]`.
- `site_basis` and `site_descriptors` are unchanged: fitting and descriptor
  code needs `B`. Folding is applied only in the calculator/bundle path
  (`load(..., fold=True)` default for evaluation; `export_bundle.py` always
  folds).
- The `AA` product is unchanged; only the seed of its reverse pass changes
  (from `A2Bᵀ·WB` computed per call to a constant), and the `A2B`
  gather/`segment_sum` and its adjoint disappear.
- Committees fold per member; Finnis–Sinclair readouts (if/when they land in
  `acejax`) fold per density. Neither is in scope here beyond not precluding it.

**Before implementing, one hour on the deletion oracle:** time `site_energies`
with the `A2B` stage replaced by an identity of the right shape to bound the
gain. If the ceiling is under ~1.15x at n_B = 2849, A drops to last in the
order (it still removes code) and B leads.

**Verification:** `tests/`: folded vs unfolded `site_energies`, `forces`,
`virial` agree to 1e-12 on the three fixtures (`si_s69`, `si_m710`,
`si_l2849`), both `edge_a_kind`s, f64. All current fixtures are single-species,
and the fold's per-species indexing (`ctilde[:, node_z]`) is untested by them,
so A also adds a small two-species fixture exported from Julia (`TiAl` or a
two-element toy fit, order 3, totaldegree 6) with the same parity checks. The
existing Julia-parity tests keep running on the unfolded path.

### B. Local-only node axis in the bundle

`PackNeighborFunctor` (lammps-jax `cpp/pair_jax_kokkos.cpp`) iterates `ilist`
over local rows only for `n_hops = 1`, so every sender is `< nlocal`. The
ghost rows of the node axis are zero-neighbour rows that cost full per-atom
work and contribute nothing.

- `export_bundle.py` gains a second capacity, `max_local` (default sized from
  the structure, see C; must satisfy `max_local ≤ max_atoms`). `energy_fn`
  calls `model.site_energies(..., n_nodes=max_local, node_z=node_z[:max_local])`
  and pads the result to `(max_atoms,)` with zeros, which is the per-atom
  shape `wrap_energy_fn` requires. `positions` stays `(max_atoms, 3)` — ghosts
  are still gathered as neighbours.
- **Overflow guard.** `jax.ops.segment_sum` silently drops segment ids
  `≥ num_segments`, so a sender `≥ max_local` would produce a wrong energy with
  no error. The plugin checks only `max_atoms`. The bundle therefore computes
  `overflow = any(edge_mask & (senders >= max_local))` and returns
  `where(overflow, nan, e)` — a NaN is loud in LAMMPS (`ERROR: Non-numeric
  atom coords` / energy check) and costs one reduction. Documented in the
  exporter's contract notes alongside the existing padding rules.
- No change to `newton="on"`, `force_output="atom-force"`, `n_hops=1`, or the
  autodiff force path.

**Verification:** `lammps/check_vs_python.py` on the existing Si bundles at
216 and 1728 atoms: forces and energy match the Python calculator to the
current tolerance with `max_local` at 1.15x nlocal; a bundle exported with
`max_local < nlocal` produces NaN energy (new negative test in
`lammps/test_si_bundle.sh`).

### C. Size capacities from the structure

Replace the `--max-atoms N --edges-per-atom K` guesswork with
`--size-from structure.xyz [--margin-edges 1.3 --margin-atoms 1.15]`: run the
same neighbour build LAMMPS will (cutoff + a stated skin), count nlocal, nall
and edges, and set `max_local`, `max_atoms`, `max_edges` with the margins.
Print all three and the actual counts so a user quoting a throughput can say
what capacity produced it. Explicit `--max-*` flags still override.

No bucket ladder or runtime bundle selection: that would need plugin changes,
and the NaN guard already makes an undersized bundle fail loudly rather than
silently. Revisit if users with variable system sizes ask for it.

### Measurement protocol (Package 1)

On moriarty, the existing harness (`bench/make_bundles.py`, `run_bench.sh`,
`bench_acejax.py`), same three fixtures, 216 / 1728 / 4096 atoms, f64, sparse
`A2B` where applicable, `timestep 0.0`, 100 steps, with the same two stated
exclusions (no reneighbouring, compile in the loop). Four rows per point:
baseline (current `main`), +A, +A+B, +A+B+C. The pace/kk numbers already in
`results.md` are the yardstick; re-run pace/kk once on the same day to confirm
the host has not drifted. Report retention (LAMMPS ÷ Python) and pace/kk ÷
jax/kk. Repeatability floor for E+F is ≤1% (`results.md`); anything under 3%
is reported as "no change".

**Target:** pace/kk ÷ jax/kk ≤ 1.5x at n_B = 2849 and ≤ 2x at n_B = 69, f64,
1728 atoms. If A + B land under that, C and the f32 tier are the remaining
levers and the plan is updated with the measured numbers before either starts.

### Out of scope for Package 1, deliberately

- **f32 / mixed precision** (measured 1.5–2.3x, pace has no f32): gated on an
  energy-conservation study against f64, which needs a potential with a
  repulsive core (the `Si_tiny` fit has none — `FINDINGS_lammps.md`). Separate
  spike after Package 1's numbers are in.
- **Edge-force export** (`force_output="edge-force"`, hand-structured backward)
  and the **DAG/recursive evaluator** (a real 1.19x only in f32): both depend on
  what A leaves of the reverse pass; measure after A.
- **A CPU backend for `jax/kk`**: rejected on the throughput argument above.

## Package 2 — exact yace export (CPU, and GPU via `pace/kk`)

A **timeboxed spike (2 days)** that answers one question: *can a v0.10
`ace1_model` be written to `.yace` such that upstream ML-PACE reproduces our
forces to 1e-10?* Its output is a recommendation; code it produces is
throwaway unless the answer is yes, in which case it is reclassified and
specified as an exporter with CI.

### What is known

- Basis sizes already match at equal hyperparameters (110 = 110 at v0.6 vs
  v0.10; 2849 vs 2874 at the production shape), so the many-body structure is
  expressible.
- The blocker is the radial basis. Upstream `lammps-user-pace` (the tag
  LAMMPS's CMake fetches, 2023.11.25) rejected the v0.6 export with
  `bad conversion`, wanting `ChebPow`/`ChebExpCos` + `radcoefficients`,
  `deltaSplineBins`, `nradbasemax`; the `wcwitt` fork parses `ACE.jl`
  `splinenodalvals` and then fails on `map::at` (version skew). The findings
  also note the `.ace`-text-named-`.yace` confusion may have contributed to the
  first failure, so it was never a clean test.
- `pyace` produces yace files that load (`bench/` pace potentials), so a
  known-good file exists to diff against.

### Spike steps

1. Diff a pyace-generated yace against the v0.6 export field by field; settle
   what upstream actually parses (`ace_radial.cpp`, `ace_yaml_reader.cpp` at
   the pinned tag), including whether `radbasename: ACE.jl` has any upstream
   support. Half a day.
2. Decide the radial route:
   - **spline nodal values** if upstream (or a fork we would pin) accepts them —
     exact by construction, since v0.10 radials are already splines with `Wnlq`
     folded (`splinify`);
   - otherwise **ChebPow/ChebExpCos coefficients** — exact only if PACE's
     transform and cutoff function can reproduce ours (`rnl_transform`,
     `rnl_envelope`); if not, it is a fit, and the spike reports the residual
     and recommends against.
3. Write `export_yace(model, path)` in Julia (`src/`), from the v0.10 model
   directly — not via v0.6. Emit both `.yace` (YAML) and, if needed, `.ace`
   text, with the extension matching the content.
4. Load under `pair_style pace` (CPU) and `pace/kk` on moriarty; compare
   forces/energies to the Julia calculator on the `Si_tiny` test
   configurations. Pass criterion: max |ΔF| ≤ 1e-10 eV/Å. One day for 3–4.

### If the spike passes

- Exporter stays in `src/` with a test that round-trips through libpace via
  the `pyace`/`lammps` Python bindings where available.
- CI job pinned to a libpace/LAMMPS tag (build from the pinned tag with
  ML-PACE only, cached; conda-forge `lammps` if its ML-PACE build is current):
  fit `Si_tiny` → export → `lmp` single point → compare. Pinning is the
  answer to "if it breaks at their end we're stuck": a bump is a deliberate,
  reviewable change.
- Documented scope: linear `ace1_model` with ACE1-compatible radials and pair
  basis; embedded and nonlinear models raise a clear error at export.

### If it fails

Record why in `docs/findings/FINDINGS_yace.md`, and the CPU story is stated
plainly in the docs: `acejax` is a GPU path; CPU users fit here and deploy
via whichever route the finding recommends.

## Sequencing

1. Package 1 A (oracle first, then implement, test, measure).
2. Package 1 B (implement, negative test, measure).
3. Package 2 spike can run in parallel with 1–2 (different code, different
   host time); its recommendation lands before C is decided.
4. Package 1 C, then the comparison table and an update to
   `acejax/bench/results.md` and `docs/plans/jax_ace_port_plan.md` open items.
5. Only then: f32 tier spike.
