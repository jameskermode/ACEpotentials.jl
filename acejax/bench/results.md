# Throughput results — moriarty, RTX A4500 (compute 8.6), Xeon Silver 4216

Model: fitted `ace1_model(Si, order=3, totaldegree=10)`, n_B = 110, rcut 6.0.
Si diamond supercells, single MPI rank, `gpu/aware off`.

The reference chart in `docs/plans/lammps_jax_benchmark_reference.md` is an
RTX 4070 Laptop at 70 W. Absolute numbers do not transfer between the two;
shape and same-host ratios do.

## Scaling

![ACE throughput vs atom count](scaling.png)

Throughput in timesteps/s against atom count, log-log, one panel per precision,
in the shape of the upstream chart in
`docs/plans/lammps_jax_benchmark_reference.md` so the two can be read side by
side. Regenerate with `uv run --with matplotlib python make_plot.py`.

Three things the shape shows that the tables do not:

- **The f64 dip at 216 atoms is visible as a V**, not a smooth curve. It is the
  capacity effect below, and it reproduces to under 1%.
- **f32 and f64 share a y-axis**, so the ~2.2x gap between them is a vertical
  offset rather than a number to compare across two scales.
- **Stillinger-Weber is near-flat** while the ACE curves fall roughly as 1/N —
  SW keeps near-constant wall time over this range, so the visible gap widens
  with system size. That is a scale reference, not a like-for-like comparison:
  it is a classical 3-body potential, different physics entirely.

## Method, and what it excludes

`timestep 0.0` freezes the configuration, so `run 100` is 100 repeated
single-point force evaluations at a fixed geometry. This avoids the `Si_tiny`
potential's missing repulsive core, which otherwise lets atoms collapse.

**Two exclusions, stated rather than buried:**

1. With atoms static, `check yes` never triggers a neighbour-list rebuild, so
   these numbers **exclude reneighbouring cost**, which real MD pays
   periodically.
2. XLA compilation happens once at the first force call. At 100 steps its share
   is small but not zero, so small-system numbers are mild underestimates.

## Bundle capacity is a tuning parameter, not a detail

Exported programs have static shapes, so every padded edge is evaluated. At 216
atoms in f64, holding everything else fixed and varying only `max_edges`
(actual edge count 9880):

| max_edges | x actual | ms/step |
|---|---|---|
| 13888 | 1.4 | 7.00 |
| 17360 | 1.8 | 2.97 |
| 20832 | 2.1 | **2.87** |
| 32984 | 3.3 | 3.79 |
| 65968 | 6.7 | 6.92 |

**U-shaped, with a 2.4x penalty at both ends.** Too tight is as bad as too
loose — the small-capacity case is the *slowest* of the five, which is not what
padding cost alone would predict, so some shapes evidently land on a poor kernel
configuration.

The optimum is **per point, not a constant**: a 1.35x margin is near-pessimal at
216 atoms in f64 but the best of the two tried at 512+, while 2.0x fixes 216 and
costs 25-30% at 1728 and 4096. Both series are given below rather than a single
tuned line. Anyone quoting a single throughput number for this pair style should
say what capacity produced it.

## LAMMPS `pair_style jax/kk` (atom-steps/s)

| atoms | f64, 1.35x | f64, 2.0x | f32, 1.35x | f32, 2.0x |
|---|---|---|---|---|
| 64 | 5.17e4 | 4.67e4 | 1.02e5 | 8.06e4 |
| 216 | 3.05e4 | **7.45e4** | 2.04e5 | 1.78e5 |
| 512 | **1.49e5** | 1.10e5 | **3.13e5** | 1.81e5 |
| 1000 | **1.84e5** | 1.37e5 | **3.26e5** | 2.49e5 |
| 1728 | **2.12e5** | 1.44e5 | **3.86e5** | 2.79e5 |
| 4096 | **1.98e5** | 1.37e5 | **3.53e5** | 2.48e5 |

The f64 216-atom point at 1.35x reproduces to under 1% across five runs
(3.05, 3.05, 3.09, 3.06, 3.07e4), so it is a systematic shape effect, not noise.
The same margin is fine for f32 at that size, consistent with a tiling effect
that depends on element width.

## acejax in Python, same GPU

Forces via `jax.grad`, `min` of 10 repeats, compilation excluded.

| atoms | edges | f64 ms/step | f64 atom-steps/s | f32 ms/step | f32 atom-steps/s |
|---|---|---|---|---|---|
| 64 | 2852 | 0.433 | 1.48e5 | 0.202 | 3.17e5 |
| 216 | 9582 | 0.810 | 2.67e5 | 0.320 | 6.74e5 |
| 512 | 22796 | 2.019 | 2.54e5 | 0.541 | 9.46e5 |
| 1000 | 44548 | 3.832 | 2.61e5 | 1.648 | 6.07e5 |
| 1728 | 76920 | 6.839 | 2.53e5 | 2.830 | 6.11e5 |
| 4096 | 182384 | 16.502 | 2.48e5 | 7.433 | 5.51e5 |

f64 plateaus near **2.5e5 atom-steps/s**, f32 near **5.5-9.5e5**.

**f64 costs only ~2.2x f32** (16.50 vs 7.43 ms at 4096), corroborating the
earlier 3-5.4x measurement on different hardware: the descriptor is
memory-bound, so the 1/32 FP64 arithmetic rate is not what governs. f64 is not
the afterthought here that it is for cheaper potentials.

## Plugin overhead

Best-case LAMMPS against acejax-in-Python at the same size and precision:

| atoms | f64 LAMMPS / acejax | f32 LAMMPS / acejax |
|---|---|---|
| 512 | 1.49e5 / 2.54e5 = 0.59 | 3.13e5 / 9.46e5 = 0.33 |
| 1728 | 2.12e5 / 2.53e5 = 0.84 | 3.86e5 / 6.11e5 = 0.63 |
| 4096 | 1.98e5 / 2.48e5 = 0.80 | 3.53e5 / 5.51e5 = 0.64 |

So the plugin path retains roughly **80% of raw model throughput in f64** and
**~65% in f32** at 1728+ atoms, the gap being LAMMPS's own neighbour list,
communication and integration. The f32 gap is larger because the model cost it
is being compared against is smaller, so the fixed overhead weighs more.

## Scale anchor: Stillinger-Weber (different physics)

`pair_style sw/kk`, same Si structure. **This is a classical 3-body empirical
potential, not an ML potential** — included to show where an ACE model sits on
the same host, not as a like-for-like comparison.

| atoms | sw atom-steps/s | vs jax f64 best |
|---|---|---|
| 64 | 2.74e5 | 5x |
| 512 | 1.76e6 | 12x |
| 1728 | 5.15e6 | 24x |
| 4096 | 1.27e7 | 64x |

SW keeps near-constant wall time as the system grows over this range (0.023 to
0.032 s for 100 steps), so its throughput scales almost linearly with atom count
while the ACE model is already GPU-saturated by ~500 atoms.

## Comparison with `pair_style pace` (ML-PACE), at matched basis size

**This is a cost comparison at matched basis size, not a comparison of two
fits.** `pace/kk` evaluates every basis function regardless of coefficient
values, so a basis of the right *structure* is all that is needed; the pace
potentials here carry random coefficients. That is safe because `timestep 0.0`
means atoms never move, so a physically meaningless potential cannot
destabilise anything. Neither side's energies are meaningful and none are
quoted.

Built with `pyace` (`python-ace` 0.2.8, cp39 wheels only). Note
`BBasisConfiguration.save()` writes the B-basis; `pair_style pace` reads the
C-tilde basis, so the config must go through
`ACEBBasisSet(bc).to_ACECTildeBasisSet()`, and that writes the `.ace` *text*
format — giving it a `.yace` extension makes yaml-cpp reject it.

### What was matched

| | acejax | pace | |
|---|---|---|---|
| **small** basis functions | 110 | **99** | 10% low |
| **large** basis functions | 211 | **211** | exact |
| correlation order | 3 | 3 | matched |
| lmax (small) | 4 | 4 | matched |
| lmax (large) | 6 | 6 | matched |
| cutoff | 6.0 Å | 6.0 Å | matched |
| elements | 1 (Si) | 1 (Si) | matched |

Same Si diamond supercells for both, so neighbour counts match too. pace is
double precision; our f64 column is the like-for-like comparison and f32 is an
option pace does not offer.

### Throughput, atom-steps/s

| atoms | 110: jax f64 | jax f32 | pace (99) | 211: jax f64 | jax f32 | pace (211) |
|---|---|---|---|---|---|---|
| 64 | 5.34e4 | 1.02e5 | 1.77e5 | 2.55e4 | 7.21e4 | 1.36e5 |
| 216 | 3.03e4 | 2.00e5 | 4.61e5 | 1.95e4 | 1.09e5 | 2.55e5 |
| 512 | 1.50e5 | 3.05e5 | 7.18e5 | 6.76e4 | 1.82e5 | 3.42e5 |
| 1000 | 1.85e5 | 3.29e5 | 8.85e5 | 7.57e4 | 1.83e5 | 3.77e5 |
| 1728 | 2.14e5 | 3.89e5 | 1.08e6 | 8.45e4 | 2.06e5 | 3.94e5 |
| 4096 | 1.98e5 | 3.51e5 | 1.14e6 | 9.60e4 | 1.62e5 | 4.01e5 |

### The ratio is size-dependent, and it narrows

At 4096 atoms, where both are saturated:

| | 110 vs 99 | 211 vs 211 |
|---|---|---|
| pace / jax f64 | **5.8x** | **4.2x** |
| pace / jax f32 | 3.2x | 2.5x |

**pace's advantage shrinks as the basis grows.** Doubling the basis costs pace
2.84x (1.14e6 -> 4.01e5) but costs acejax only 2.06x in f64 (1.98e5 -> 9.60e4)
and 2.16x in f32. So pace degrades super-linearly in basis size over this range
while acejax is close to linear.

A plausible reading is that the small case is dominated by fixed overheads and
neighbour handling, where a mature C++/Kokkos kernel wins comfortably, while the
tensor contraction — which grows with the basis — is where the JAX
implementation is relatively stronger. That is a hypothesis consistent with two
points, not an established scaling law: **both models here are small** (110 and
211 functions), and production ACE potentials are often several hundred to a
couple of thousand. Whether the trend continues, flattens or reverses at that
scale is untested.

## Third size, and a large-basis finding that changes the reading

A third matched pair at production-ish scale: **ours 1429 functions vs pace
1551** (8.5% apart), order 3 and lmax 10 on both sides, rcut 6.0. This is the
closest matched pair available — pace's function count moves in coarse jumps, so
1809-vs-1725 would have matched the count to 4.9% but with lmax 12 against 5,
and shape matters more than count. ~2000 with both shapes matched was not
reachable; 1429/1551 is the documented ceiling.

The 1429 model uses **random, unfitted weights** (`ACE_NOFIT=1`). Coefficients do
not affect cost, and a fit of 1429 functions to Si_tiny would be badly
underdetermined. Note `ace1_model` initialises `WB` to *zero*, which would let
XLA fold the readout away and eliminate the descriptor entirely, so the exporter
randomises explicitly.

### The dense `A2B` contraction dominates at large basis

`A2B` is extremely sparse — usually exactly one nonzero per column. At 1429
functions it is 1429 x 11474 with 11474 nonzeros: **0.070% occupied, 131 MB
dense in f64**. Per-stage timing at 216 atoms, f64, on the GPU:

| stage | dense `A2B` | sparse `A2B` |
|---|---|---|
| A2B contraction | **19.658 ms (99.9%)** | 0.10 ms |
| everything else | ~0.78 ms | ~0.69 ms |
| `site_basis` total | 19.684 ms | 0.80 ms |

Replacing the matmul with a gather and segment-sum (one nonzero per column makes
this straightforward) gives, at 1728 atoms:

| | dense | sparse | speedup |
|---|---|---|---|
| f64 | 2672 atom-steps/s | 21550 | **8.1x** |
| f32 | 27960 | 35280 | 1.26x |

The sparse path is exact, not an approximation — it agrees with the dense one to
0.000e+00 (`tests/test_efv.py::test_sparse_a2b_matches_dense`). It is off by
default because dense is faster at small basis; pass `a2b_sparse=True` to
`load()`.

**Any large-basis number taken with the dense contraction measures our
implementation, not the architecture.** The figures below use the sparse path
for the 1429 point and dense for 110 and 211, where it is the faster of the two.

### The three-point trend

pace / acejax throughput at 1728 atoms, f64 (like-for-like; pace is
double-precision):

| basis (ours vs pace) | pace / jax f64 |
|---|---|
| 110 vs 99 | **5.8x** |
| 211 vs 211 | **4.2x** |
| 1429 vs 1551 | **2.7x** |

**The narrowing holds across all three points and continues.** With the dense
contraction the third point would have read 21.6x and inverted the trend
entirely — which is why measuring both mattered.

## Where the remaining gap is: per-stage breakdown

Stages timed in isolation, 1728 atoms, f64, GPU, n_B = 110:

| stage | ms | share of `site_basis` |
|---|---|---|
| **angular (Ylm)** | 1.135 | **51.7%** |
| radial | 0.454 | 20.7% |
| A2B contraction | 0.359 | 16.4% |
| edge_A product | 0.157 | 7.2% |
| A pooling | 0.139 | 6.3% |
| AA products | 0.083 | 3.8% |
| readout | 0.062 | 2.8% |
| sum of stages | 2.389 | 108.8% |
| `site_basis` whole | 2.196 | |

**Method caveat, and it is a real limit.** Timing stages in isolation *over*
counts, because XLA fuses them in the full computation and avoids materialising
intermediates. At n_B = 110 and 1728 atoms the sum is 8.8% over the whole, which
is tolerable. At n_B = 211 it is 105% over, and at 216 atoms it is 55-176% over,
because the individual kernels are too small for the isolated timing to mean
anything. Only the 110-at-1728 column above is quoted for that reason; the
others are recorded in the commit history but not presented as attributions.

So on the one decomposition that is trustworthy:

- **The spherical harmonics are the single largest stage at ~52%.** That is our
  pure-JAX recursion; pace uses hand-optimised C++. This is a larger lever at
  small basis than `A2B`, and it is the concrete place to look next.
- `AA` products are only 3.8%, so the recursive-evaluator hypothesis
  (`ace_recursive.cpp` sharing subproducts) cannot account for much at this
  basis size. It may matter more at large basis, untested.
- **Edge padding**: at the 1.35x capacity margin used, roughly 26% of evaluated
  edges are padding, so this is a ~1.3x effect on the edge-wise stages
  (radial, angular, edge_A) — about 80% of the work at n_B = 110. Real, but not
  the dominant term.
- **Precision is neutral**, as expected: the f64 column is compared against
  pace's double precision throughout.

**Residual.** Padding (~1.3x on edge-wise stages) does not close a 5.8x gap, and
the harmonics being half our time points at implementation quality rather than
architecture. A substantial fraction of the small-basis gap remains
unattributed. That is the honest state: the measurement says where to look
next — the harmonics — not what the answer is.

## Not obtained

**A `pace` comparison of two *fitted* potentials.** The comparison above matches
basis size only. Exporting our own fitted model to `.yace` remains blocked: an
ACEpotentials v0.6.12 fit exports cleanly and matches on basis size (110 both
versions), but upstream ICAMS libpace rejects it (`bad conversion` — it wants
`ChebPow`/`radcoefficients`, not v0.6's `ACE.jl` spline nodal values) and the
`wcwitt` fork, which does carry `acejl_radial.cpp`, fails with
`Exception: map::at`. For a *throughput* comparison this does not matter, since
coefficients do not affect cost. See `docs/findings/FINDINGS_benchmark.md`.

**A fully matched pair at ~2000 functions.** pace's achievable function counts
jump coarsely at fixed lmax, so the largest matched pair is 1429 vs 1551.
