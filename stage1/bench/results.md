# Phase 8 raw results — moriarty, RTX A4500 (compute 8.6), Xeon Silver 4216

Model: fitted `ace1_model(Si, order=3, totaldegree=10)`, n_B = 110, rcut 6.0, f64
unless noted. Bundle capacities sized per point (see `make_bundles.py`).

## acejax in Python, on the GPU

Forces via `jax.grad`, `min` of 10 repeats, compilation excluded.

| atoms | edges | f64 ms/step | f64 atom-steps/s | f32 ms/step | f32 atom-steps/s | f64/f32 |
|---|---|---|---|---|---|---|
| 64 | 2852 | 0.416 | 1.54e5 | 0.205 | 3.13e5 | 2.03 |
| 216 | 9582 | 0.773 | 2.80e5 | 0.447 | 4.83e5 | 1.73 |
| 512 | 22796 | 2.020 | 2.54e5 | 0.532 | 9.63e5 | 3.80 |
| 1000 | 44548 | 3.715 | 2.69e5 | 1.467 | 6.82e5 | 2.53 |
| 1728 | 76920 | 6.784 | 2.55e5 | 2.701 | 6.40e5 | 2.51 |

f64 plateaus near **2.5-2.8e5 atom-steps/s**; f32 near **6.4-9.6e5**. The
**f64/f32 ratio of ~2.5x** corroborates Phase 0's 3-5.4x: the descriptor is
memory-bound, so f64 costs far less than the 1/32 FP64 arithmetic rate implies.

## LAMMPS `pair_style jax/kk`, f64, 20 MD steps

| atoms | loop s | atom-steps/s |
|---|---|---|
| 64 | 0.0280 | 4.57e4 |
| 216 | 0.1583 | 2.73e4 |
| 512 | 0.0778 | 1.32e5 |
| 1000 | 0.1196 | 1.67e5 |
| 1728 | 0.1809 | 1.91e5 |

**These are lower bounds, not steady-state.** XLA compilation happens inside
LAMMPS's timed loop, so at 20 steps it is a large and size-dependent share --
which is why 216 atoms reads slower than 64. Longer runs would amortise it but
abort (below).

At 1728 atoms, jax/kk gives 1.91e5 against acejax-in-Python's 2.55e5 on the same
GPU, so the plugin path costs roughly 25% -- but with compilation still inside
the loop, treat that as indicative only.

## Not obtained

- `pace/kk` comparator: blocked, see FINDINGS_benchmark.md
- `eam/alloy/kk` anchor: no Si eam/alloy potential ships with LAMMPS (only
  meam/edip/mliap for Si); the available eam files are CuZr, so an anchor would
  differ in both physics *and* element. Skipped rather than mislabelled.
- Steady-state LAMMPS throughput and larger sizes: blocked by the abort below.
