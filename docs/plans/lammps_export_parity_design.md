# `lammps-export`: correctness fixes and CPU parity with ML-PACE — design

**Date:** 2026-09-15. **Status:** approved design; implementation plan to follow.
**Executes on:** moriarty, by a fresh Claude Code instance (see §7).

## Problem

The ACEsuit `lammps-export` branch exports a fitted ETACE model to generated
Julia, compiles it with `juliac --trim` into a shared library, and runs it in
LAMMPS through `pair_style ace`. Two findings on this branch establish where it
stands (`docs/findings/FINDINGS_lammps_export.md`,
`docs/findings/FINDINGS_lammps_export_perf.md`, both against
`origin/lammps-export` at 075d3859):

- `:polynomial` mode is **exact** in LAMMPS (max |ΔF| 6.7e-14 eV/Å on a
  five-element CrMnFeCoNi model) but **7.0x slower** than `pair_style pace
  recursive` per core (522 vs 75 µs/site).
- `:hermite_spline` mode, the branch's recommended one, is **wrong for any
  NZ ≥ 2** (symmetric pair index into asymmetric per-pair tables; 13.5 eV/Å),
  and once fixed inherits ET `splinify`'s 2.7e-4 eV/Å (Nspl=50) / 3e-6
  (Nspl=200) error; it is 2.6x slower than pace.
- The exporter **silently drops `ETPairModel`**: a deployed library is
  one-body + many-body only.
- The library is **not re-entrant** (global `WORK_*` scratch), so the
  plugin's default OpenMP build computes wrong forces; `MAX_NEIGHBORS = 256`
  is a silent cap; CI's multi-species test asserts only `isfinite`.
- Profile: `:polynomial` time is 84 % a dense 74×45 `SMatrix` matvec with
  14–16 nonzeros; Hermite time is dominated by 2 KB-strided stores into
  `[MAX_NEIGHBORS, feature]` arrays, per-neighbour zeroing, and a force
  assembly that reads them back. The AA products are flat (no DAG) and the
  readout is unfolded (B via A2B in both directions).

The review's conclusion: every gap is in the export generator; no
EquivariantTensors or ETModels change is needed; the generated code can be
made isomorphic to `ace_recursive.cpp`, with an expected residue of
~30–40 µs/site against pace's 75 on the same core.

## Goal

A `lammps-export` library that is (1) correct for multi-species models with
the pair term included, (2) re-entrant, and (3) **≤ 1.2x `pair_style pace
recursive` per core** on two reference models, in the exact `:polynomial`
mode, measured by difference at each step.

## Decisions (made with the maintainer, 2026-09-15)

- Acceptance gate: parity, ≤ 1.2x pace recursive, single core, on both
  reference models (§4). All four performance items are in scope.
- `:polynomial` becomes the default and the documented exact mode.
  `:hermite_spline` is kept, fixed, non-default, labelled approximate with its
  Nspl error stated; it is for genuinely learned radials with small N_POLYS.
- Work lands on a branch `export/perf-parity` off `origin/lammps-export`,
  PR into `lammps-export`. No rebase onto `main` (9 commits behind, small
  `src/` diff — deferred to the PR).
- Nothing in `EquivariantTensors` or `ACEpotentials.ETModels` changes.
  `SparseSymmProdDAG` (ET 0.4.3, `src/ace/symmprod_dag.jl`) is consumed as is.
- Plugin changes are limited to the workspace API (§3) and its tests.

## Constraints

- Exactness is checked before anything is timed: generated code vs
  `ETACEPotential` in Julia at 1e-12; library via the Python C API at 1e-12;
  `pair_style ace` in LAMMPS vs Julia at 1e-10 (1 rank, 2 ranks, and
  `OMP_NUM_THREADS=4` vs serial at 1e-12). Hermite is compared to the
  *splinified* model at 1e-12 and its error vs the *fitted* model is reported,
  never hidden by a loose tolerance.
- Each performance step keeps the previous generator's output as its
  reference: a step that changes any force by more than 1e-13 relative is a
  bug, not a speed-up.
- Attribution by difference only: one change per step, whole `pair_style
  ace` step timed, pace re-run the same day on the same core; repeats within
  3 %; no timing block on a contended core (loadavg recorded per block).
- No test tolerance is loosened; every test states which reference it uses.

## 1. Stream A — correctness (lands first)

### A1. Export the pair potential

`export_ace_model.jl:60-90` keeps `ETOneBody` and `ETACE` from a
`StackedCalculator` and ignores `ETPairModel`; `has_pair` is unused. Emit the
pair basis through the same radial path as the many-body radials (its own
transform/envelope parameters per pair, `N_PAIR` functions per species pair,
the fitted pair weights), as an extra energy term and an extra force term in
pass 2 (§2). A stack containing anything else raises. The exported energy
must equal the `StackedCalculator` energy (one-body + pair + many-body) to
1e-12.

### A2. One indexing convention for per-pair data

Hermite tables and any per-pair static data are emitted only for the ordered
pairs the model has, indexed by `pair_idx(iz, jz)` in one place used by both
modes; `zz2pair_sym` is removed. Unused (n,l) radials and rows that are
identically zero for a pair are pruned at export (this is also part of B1).

### A3. Neighbour cap

`MAX_NEIGHBORS` and its `@assert`s go with the per-neighbour kernel (§2);
until then the assert becomes a clear error message with the count.

### A4. Tests

- `export/test/test_multispecies.jl`: NZ ≥ 3 with asymmetric `rin0cuts`,
  both modes, compared to `ETACEPotential` (polynomial, 1e-12) and to the
  splinified `ETACEPotential` (Hermite, 1e-12) on ≥ 5 rattled configurations
  containing every species, energies, forces and virial; plus one config
  with > 256 neighbours (small cutoff, dense cell).
- `test_hermite_accuracy.jl`: keeps its 1e-8 check against the splined
  reference **and** reports the error against the fitted model for
  Nspl ∈ {50, 200} in the test log; `:polynomial` asserts 1e-12 against the
  fitted model.
- `test_lammps.jl`: compare to Julia at 1e-10 (not 1e-6), 1 and 2 ranks,
  with the pair term; add `OMP_NUM_THREADS=4` vs serial at 1e-12 once §3
  lands.
- CI (`.github/workflows/export-ci.yml`): the multi-species test runs both
  modes; the LAMMPS job builds the plugin with OpenMP and runs the threaded
  comparison.

### A5. Defaults and docs

`radial_basis=:polynomial` default in `export_ace_model` and
`build_deployment`; README mode table rewritten from measurements (§4):
Hermite "approximate, error depends on Nspl and N_POLYS; use for learned
radials with small N_POLYS"; the "machine precision" and "3–4x faster" claims
removed.

## 2. Stream B — the generated evaluator (target shape)

Isomorphic to `ace_recursive.cpp`: per site, two neighbour passes and one
tensor pass; no `[MAX_NEIGHBORS, feature]` arrays; all per-neighbour data on
the stack; per-site scratch in a caller-supplied workspace (§3).

**Pass 1, per neighbour.** Transform with integer `pin`/`pcut`; Chebyshev
three-term recurrence for values and derivatives; envelope; **mixing emitted
from `W`'s structure**: if `W` is one-hot (every ACE1-style linear fit;
`init_Wradial = :onehot`), a static index gather of the ~N_POLYS/NZ nonzero
radials for pair (iz,jz), otherwise a vectorised GEMV over the pruned row set
(both selected at export time by inspecting `W`). Solid harmonics as now
(SpheriCart straight-line code). Accumulate `A[k] += R·Y` over the static
per-pair list of A functions in the neighbour species' block only. Nothing
per neighbour is stored.

**Tensor pass.** `AA` by DAG forward (`const DAG_NODES = ((n1,n2),…)` from
`SparseSymmProdDAG`; leaves are A entries), energy `= dot(c̃_iz, AA_nodes)`
with `c̃_iz = projection(A2Bmapᵀ · WB_iz)` precomputed at export for each
centre species; backward seeded with `c̃_iz`, two FMAs per node
(`symmprod_dag_kernels.jl:51-83` pattern), yielding `∂A`. No B, no A2B loops,
no `∂B` copy, no `WB` in the library.

**Pass 2, per neighbour.** Re-evaluate the per-neighbour derivative terms
(or keep the ~24 doubles from pass 1 in the workspace); `f_j = Σ_k ∂A[k]
(dR_k Y_k r̂ + R_k dY_k)` over that species block plus the pair term; write
`f_j` into the caller's force buffer; `R_j ⊗ f_j` once per neighbour, only
when the virial entry is called.

**Order of implementation (each measured by difference, each with the
previous generator as 1e-13 reference):**

| step | change | expected (Cantor, from the profile) |
|---|---|---|
| B1 | sparse/static mixing; prune unused (n,l) and per-pair zero rows; integer powers | `:polynomial` 522 → ~150 µs/site; Hermite tables 4–5x smaller |
| B2 | per-neighbour kernel (passes 1 and 2), species-block A accumulation, forces from `∂A`, one virial outer product, forces into caller buffer; `MAX_NEIGHBORS` removed; workspace API (§3) | stores+zeroing+assembly 130 → ~25 µs |
| B3 | DAG + C-tilde fold | tensor block 28 → ~10 µs; larger at order 4 |
| B4 | OpenMP correctness test enabled; per-thread workspaces in the plugin | correctness; threads are a bonus over MPI-only parity |

Hermite mode receives B1's pruning and B2's kernel (its lookup replaces the
recurrence in pass 1) so that it stays correct and re-entrant; it is not
tuned further.

## 3. C API and plugin

- `void* ace_workspace_new(void)` / `void ace_workspace_free(void*)`: an
  opaque handle sized at export (A, DAG node values and cotangents, `∂A`,
  per-neighbour derivative cache, pair scratch; ~20 KB for the Cantor
  model). Every `ace_site_*` entry gains the handle as its first argument.
  `WORK_*` globals are removed; the library is re-entrant by construction.
- `pair_ace.cpp`: one workspace per OpenMP thread, created in `init_style`,
  freed in the destructor; the virial entry is called only when
  `vflag_global` is set (LAMMPS uses F·r otherwise); the two-pass neighbour
  copy is kept (not a factor). Newton on, full list, as now.
- `ase-ace/` Python calculator: creates one workspace per calculator
  instance; its "single-threaded" note is updated.
- Rejected: per-call allocation (GC traffic, no guarantee under adopted
  threads); transposing the global arrays (fixes strides, keeps the race and
  the cap).

## 4. Reference models and measurement protocol

**Reference models** (both multi-species, asymmetric cutoffs):

1. **Cantor**: the already-fitted CrMnFeCoNi `ace_model`, order 3, level 6,
   rcut 6.25, 1348 many-body functions + pair, BLR + repulsion restraint,
   saved on moriarty under
   `~/ace-potentials-julia-1.2/ACEpotentials.jl/verify_cantor/` (the fit
   took 43 min: do not refit; `lsq_cantor.jld2` is the 1.1 GB design matrix,
   never load it). Its 10 held-out configurations and the v0.6 twin's
   `.yace` (10⁴ nodes) for the pace comparison are under
   `~/si-ace/spike_yace/`.
2. **Large**: TiAl from `ACEpotentials.example_dataset("TiAl_tutorial")`
   (the branch's `benchmark/fit_tial_model.jl` uses it at order 3,
   totaldegree 10, rcut 5.5 ≈ 1166 functions), refitted as an `ace_model` at
   **order 4** with the total degree chosen to give ~1500–2500 many-body
   functions, rcut 5.5, BLR + repulsion restraint; fitted once at the start of
   Stream B and saved. This is where the DAG's value is measured. Its matched
   pace basis for the throughput rows comes from `pyace` as in
   `acejax/bench/make_pace_basis.py` (random coefficients are fine for timing:
   pace evaluates every function regardless).

**Throughput protocol.** `pair_style ace` vs `pair_style pace recursive`
(matched `.yace`), single MPI rank pinned to one idle core (`taskset`,
`OMP_NUM_THREADS=1`), 1728–2000-atom cell, `timestep 0.0`, 100 steps, mean
of two runs agreeing within 3 % (third run and median otherwise), pace
re-run the same day, `/proc/loadavg` recorded per block, no block on a
contended core. Report µs/site and the ratio after each of B1–B4. The gate
is the B3 row on both models.

**Exactness protocol** (every step, both models): generated code vs
`ETACEPotential` (Julia), library via Python C API, `pair_style ace` 1 and
2 ranks, threaded vs serial — tolerances in Constraints.

## 5. Out of scope

- GPU/Kokkos path for the plugin; MPI-only parity is the target.
- Making ET's `splinify` accurate at N_POLYS 45 (needs Nspl ≈ 2000–3000;
  tabulation is the wrong tool); the recurrence is exact and, after B1, cheap.
- Rebase onto `main`; changes to `src/` beyond what the branch already
  carries.
- Compiling the real ET evaluator instead of generated code (would remove the
  second evaluator; ET's kernels are not `--trim`-safe today) — noted as
  future work, not attempted here.

## 6. Risks

| risk | mitigation |
|---|---|
| `juliac --trim` rejects a construct in the new kernel (dynamic dispatch, allocation) | keep the generated code in the same style as today's (static tuples, `SVector`, `@inbounds` loops); compile after every step, not at the end |
| Sparse mixing has to handle a genuinely dense learned `W` | export-time branch on `W`'s sparsity with both paths tested (one-hot model and a learned-radial model) |
| DAG node order vs `A2Bmap` column order | fold `c̃` through `dag.projection` exactly as the acejax spike did (`acejax/spike_recursive/dag.py`, structural check that reconstructs each node's A-multiset) |
| Host contention spoils timings | per-block loadavg gate; pinned cores; same-day pace rows |
| The gate is missed on the large model | B3's DAG is where order-4 gains live; if still > 1.2x, report the profile by difference and stop — do not tune Hermite instead |

## 7. Executor environment (moriarty)

- Checkout: `~/ace-potentials-julia-1.2/ACEpotentials.jl`, currently on the
  throwaway branch `lammps-export-verify` (host-local, carries the two-line
  Hermite fix 39d853dc and a stash of a pre-existing Manifest edit). Create
  `export/perf-parity` from `origin/lammps-export` (075d3859), cherry-pick
  39d853dc, and work there; nothing is pushed until the maintainer says so.
- Julia 1.12.2 (`julia`); `export/` has its own project (`ACEpotentials =
  {path = ".."}`, P4ML pinned v0.5.8, JuliaC 0.3.10); the branch's 43 tests
  pass as-is. juliac takes 30–40 s per library.
- LAMMPS with the plugin: `~/lammps/lammps-22Jul2025/build/lmp` (rebuild the
  plugin with `BUILD_OMP=ON` once §3 lands; until then `BUILD_OMP=OFF`).
  ML-PACE for the pace rows: the `-mlpace` build under
  `/storage/eng/essswb/lammps-jax-build/lammps/` (its build dir must precede
  `$V/lib` on `LD_LIBRARY_PATH`; recipe in `acejax/bench/run_bench.sh` of the
  ACEpotentials `pr/lammps-throughput` branch, also rsync'd at
  `~/si-ace/ACEpotentials/acejax/bench/`).
- Data: `/home/eng/essswb/ACEpotentials-jax/cantor1k_b_mh1.xyz`
  (CrMnFeCoNi, keys `mace_energy/force/virial`); do not touch that checkout's
  git state.
- The host is shared: check `/proc/loadavg`, `nvidia-smi` is irrelevant (CPU
  work), yield to other users, never run timing on a contended core.
