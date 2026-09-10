# Throughput results — moriarty, RTX A4500 (compute 8.6), Xeon Silver 4216

Model: fitted `ace1_model(Si, order=3, totaldegree=10)`, n_B = 110, rcut 6.0.
Si diamond supercells, single MPI rank, `gpu/aware off`.

The reference chart in `docs/plans/lammps_jax_benchmark_reference.md` is an
RTX 4070 Laptop at 70 W. Absolute numbers do not transfer between the two;
shape and same-host ratios do.

> **Read the CORRECTION section at the end before using anything below it.**
> The scaling series was built at an unrealistic model shape (high lmax, low
> correlation order). Two conclusions below — "angular is 51.7% of runtime" and
> "the pace/acejax ratio narrows monotonically" — are artefacts of that choice
> and are superseded there. The absolute throughputs stand, but they are the
> throughputs of a 110-function model and **none of them are what
> `scaling.png` plots**; the figure is built entirely from the realistic-shape
> numbers in the CORRECTION section. Two sections below are marked SUPERSEDED
> for that reason.

## Scaling

![ACE throughput vs atom count](scaling.png)

Throughput in timesteps/s against atom count, log-log, one panel per basis
size, at the realistic shape of the CORRECTION section (n_B = 69, 710, 2849).
Regenerate with `uv run --with matplotlib python make_plot.py`.

**Solid lines are LAMMPS pair styles; dashed lines are acejax called directly
from Python**, which is *not* a pair style: it carries no neighbour list, no
communication, no integration, and no static-shape padding. Same hue = same
implementation, so within a colour the solid/dashed gap is the cost of the
plugin path, and between colours it is the cost of the implementation.

Four things the shape shows that the tables do not:

- **The f64 dip at 216 atoms is a V in the LAMMPS lines and absent from the
  Python ones.** In `pair jax/kk` f64, 216 atoms is *slower* than 64 at
  n_B = 69 and at n_B = 710; in Python, 216 atoms is faster than 64 at all
  three basis sizes. So the dip belongs to the plugin path — bundle capacity
  and kernel shape — not to the model.
- **The plugin gap narrows with system size rather than sitting at a constant
  offset.** The shaded f64 band is at its widest at the left-hand end
  (64-216 atoms) of every panel and at its narrowest at 1728.
- **The f32 pair narrows the same way**, so this is not a precision effect.
- **At n_B = 2849 the dashed blue line crosses above the green one**: the model
  in Python is slightly faster in f64 than `pair pace/kk` at a matched basis
  size. That crossing is not like-for-like (see below) but it locates the
  remaining gap in the plugin path rather than in the kernel.

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

## acejax in Python, same GPU (n_B = 110 — SUPERSEDED)

> **These numbers are for the superseded 110-function model** (order 3,
> lmax 4) and correspond to no series in `scaling.png`. The Python series at
> the current shapes is measured further down, under "acejax in Python at the
> realistic shapes". Do not mix the two.

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

## Plugin overhead (n_B = 110 — SUPERSEDED)

> **Superseded**, for the same reason: both sides of these ratios are the
> 110-function model. The realistic-shape version is "The plugin gap across
> sizes" below, and it is a good deal larger than the ~80% quoted here.

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

# CORRECTION: the series above used an unrealistic model shape

Everything above reached its basis size via **high lmax at low correlation
order** — the 1429-function model is lmax 10, order 3. Real ACE potentials do
not get there that way: they use **lmax 3-6 with correlation order 4-5**. The
exemplar shipped with LAMMPS, `Cu-PBE-core-rep.ace`, is rank 6 / lmax 6 / 742
functions — high body order, moderate lmax. Ours was the opposite shape.

**This invalidated the headline per-stage result.** lmax 10 gives 121 Ylm
channels against 36 at lmax 5, so of course the angular stage dominated — we
made it dominate. The series was rebuilt at **order 4, lmax 5** throughout, with
only the size varying.

## Rebuilt series: order 4, lmax 4-5, rcut 6.0, Si

One detail worth correcting in the sentence above: the rebuilt series is order 4
throughout, but the *small* model is lmax 4, not 5 — `fixtures/si_s69.npz`
carries `lmax: 4`, while `si_m710.npz` and `si_l2849.npz` carry `lmax: 5`. It is
labelled "order 4, lmax 4-5" in `scaling.png` for that reason.

| | acejax | pace | gap | AA terms by rank |
|---|---|---|---|---|
| small | 69 | 78 | 13% | 8 / 39 / 37 / 21 |
| mid | 710 | 693 | 2.4% | 14 / 270 / 1191 / 1481 |
| large | **2849** | **2874** | **0.9%** | 18 / 672 / 7134 / 16292 |

Both sides now match on shape as well as count. The large point is genuinely
production-scale — bigger than the 742-function Cu exemplar. Construction was
fast (0.1-0.5 s); `_auto_nnllmm_spec` was not the bottleneck its comment warns
about at these sizes.

## The per-stage picture changes completely

Same measurement, GPU, 1728 atoms, f64, sparse `A2B`:

| stage | lmax 10 / order 3 (n_B=1429) | **lmax 5 / order 4 (n_B=2849)** |
|---|---|---|
| angular (Ylm) | **51.7%** | **18.9%** |
| AA products | **3.8%** | **37.6%** |
| radial | 20.7% | 28.0% |
| A2B contraction | 16.4% | 54.1% |

**The 51.7% angular figure was a property of the model we built, not of ACE.**
At a realistic shape the harmonics are ~19% and the `AA` products are ~38%.

**And the recursive-evaluator hypothesis is back.** It was dismissed on
"AA is 3.8%, so it cannot explain much" — but that was measured on a shape with
almost no high-rank terms. At order 4 the rank-3 and rank-4 blocks hold 7134 and
16292 terms, `AA` is ~38% of the time, and `ace_recursive.cpp` exists precisely
to share subproducts across those. That dismissal was conditioned on our model
choice and should not have been generalised.

`A2B` remains the single largest stage even sparse. Dense is not an option at
this size at all: 2849 x 24116 is 550 MB in f64 and takes **679 ms/step**
against 6 ms sparse.

## Throughput, realistic shape (atom-steps/s at 1728 atoms)

| basis | jax f64 | jax f32 | pace | pace/f64 | pace/f32 |
|---|---|---|---|---|---|
| 69 vs 78 | 3.15e5 | 6.47e5 | 9.54e5 | **3.0x** | 1.5x |
| 710 vs 693 | 6.45e4 | 1.17e5 | 1.39e5 | **2.1x** | 1.2x |
| 2849 vs 2874 | 1.27e4 | 2.24e4 | 3.30e4 | **2.6x** | 1.5x |

**The monotonic narrowing does not survive the reshape.** The earlier series
gave 5.8x -> 4.2x -> 2.7x; at a realistic shape it is 3.0x -> 2.1x -> 2.6x —
narrower overall, and roughly flat rather than steadily converging. The clean
trend was partly an artefact of varying lmax along with size.

In f32 the two are within 1.2-1.5x throughout.

## acejax in Python at the realistic shapes

The same model, same GPU, called directly from Python — no LAMMPS. Forces via
`jax.grad`, sparse `A2B`, `min` of 10 repeats, compilation excluded. Exact edge
and atom counts: nothing is padded, which is the point of the comparison.

`bench_acejax.py --npz fixtures/si_{s69,m710,l2849}.npz --reps 2 3 4 5 6
--repeats 10 --a2b-sparse [--f32]`

**n_B = 69** (order 4, lmax 4)

| atoms | edges | f64 ms/step | f64 atom-steps/s | f32 ms/step | f32 atom-steps/s |
|---|---|---|---|---|---|
| 64 | 2852 | 0.301 | 2.13e5 | 0.179 | 3.58e5 |
| 216 | 9582 | 0.514 | 4.20e5 | 0.264 | 8.19e5 |
| 512 | 22796 | 0.934 | 5.48e5 | 0.319 | 1.61e6 |
| 1000 | 44548 | 2.002 | 4.99e5 | 0.620 | 1.61e6 |
| 1728 | 76920 | 3.465 | 4.99e5 | 1.480 | 1.17e6 |

**n_B = 710** (order 4, lmax 5)

| atoms | edges | f64 ms/step | f64 atom-steps/s | f32 ms/step | f32 atom-steps/s |
|---|---|---|---|---|---|
| 64 | 2852 | 0.788 | 8.12e4 | 0.282 | 2.27e5 |
| 216 | 9582 | 1.779 | 1.21e5 | 0.536 | 4.03e5 |
| 512 | 22796 | 3.984 | 1.28e5 | 2.003 | 2.56e5 |
| 1000 | 44548 | 8.689 | 1.15e5 | 4.508 | 2.22e5 |
| 1728 | 76920 | 16.062 | 1.08e5 | 8.948 | 1.93e5 |

**n_B = 2849** (order 4, lmax 5)

| atoms | edges | f64 ms/step | f64 atom-steps/s | f32 ms/step | f32 atom-steps/s |
|---|---|---|---|---|---|
| 64 | 2852 | 1.652 | 3.87e4 | 0.551 | 1.16e5 |
| 216 | 9582 | 4.018 | 5.38e4 | 1.922 | 1.12e5 |
| 512 | 22796 | 11.361 | 4.51e4 | 6.236 | 8.21e4 |
| 1000 | 44548 | 25.129 | 3.98e4 | 15.825 | 6.32e4 |
| 1728 | 76920 | 46.624 | 3.71e4 | 31.570 | 5.47e4 |

**Exclusions, restated** — these are the same two the LAMMPS numbers carry, plus
one the LAMMPS numbers do not:

1. **No reneighbouring.** The edge list is built once per size and reused, as
   `timestep 0.0` does on the LAMMPS side.
2. **Compilation excluded.** The first call is run and discarded; the reported
   figure is the `min` of 10 timed repeats after it. Unlike the `run 100`
   numbers, compilation is fully outside the measurement here, not merely a
   small share of it.
3. **No neighbour list, no MPI communication, no integration, and no padding.**
   That is what makes this the right baseline for isolating plugin cost, and
   what makes it *not* a pair style.

**Repeatability.** The whole six-block series was run twice, in separate
processes: once with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and once with JAX's
default preallocation. Of the nine points longer than 5 ms/step, eight agree to
within 1.5% and the ninth to 3.3%; points under 1 ms scatter far more, up to 17%
(0.226 vs 0.264 ms at n_B = 69, 216 atoms), which is what timing a quarter of a
millisecond over the Python/JAX dispatch path costs. There is no systematic
difference between the two settings, so preallocation is not what the LAMMPS
comparison turns on. The table above is the default-preallocation run; both raw
logs are on moriarty at `~/si-ace/acejax/bench/py_series{,_prealloc}.txt`, with
the compute-app samplers beside them.

**GPU exclusivity.** A first attempt failed outright — another process held
15.2 GB of the 20 GB card and every allocation OOM'd, so no numbers were
produced. Both reported runs were taken with `nvidia-smi
--query-compute-apps` verified empty beforehand and a 3-second sampler running
throughout; in neither run does a second PID ever appear.

## The plugin gap across sizes

`pair jax/kk` throughput divided by acejax-in-Python throughput at the same
size, precision and basis — the fraction of raw model throughput that survives
the plugin path. Alongside it, how much each bundle's static shapes over-provision
against what that configuration actually uses.

| atoms | atom slots / atoms | edge slots / edges | 69 f64 | 710 f64 | 2849 f64 | 69 f32 | 710 f32 | 2849 f32 |
|---|---|---|---|---|---|---|---|---|
| 64 | 15.0x | 1.68x | 0.34 | 0.32 | 0.18 | 0.30 | 0.17 | 0.08 |
| 216 | 8.0x | 1.45x | 0.13 | 0.19 | 0.18 | 0.34 | 0.19 | 0.14 |
| 512 | 5.6x | 1.37x | 0.37 | 0.49 | 0.35 | 0.29 | 0.41 | 0.27 |
| 1000 | 4.4x | 1.47x | 0.55 | 0.55 | 0.39 | 0.36 | 0.48 | 0.38 |
| 1728 | 3.7x | 1.40x | **0.63** | **0.60** | **0.34** | 0.55 | 0.61 | 0.41 |

**The overhead is not a constant offset — it shrinks as the system grows.**
Retention rises by 1.8-1.9x between 64 and 1728 atoms in the f64 columns, and by
1.8-5x in f32. On the plot this is the shaded band closing from left to right —
not monotonically, since 216 atoms is a local worst case in f64 at the two
smaller bases, the same V the LAMMPS series shows. This is the thing the ratio
table alone could not say.

**Two mechanisms with exactly known padding factors** — their *effect* on
runtime is a different matter, see the caveat below. The bundles have static
shapes, so every padded slot is evaluated:

- The **atom** axis is padded to `max_atoms`, which covers local atoms *plus
  ghosts* plus a 1.25x margin — 15.0x the local atom count at 64 atoms, falling
  to 3.7x at 1728 as the surface-to-volume ratio does. `export_bundle.py` passes
  `positions.shape[0]`, i.e. `max_atoms`, as the node count to
  `model.site_energies`, so the per-atom stages (`A` pooling, `AA` products,
  `A2B`, readout) run over every slot. In Python they run over exactly `N`.
- The **edge** axis is padded ~1.4x throughout, so the edge-wise stages
  (radial, angular, `edge_A`) pay a roughly constant 1.4x.

That the atom padding falls with N while the edge padding does not is
consistent with the narrowing, and it predicts the second pattern in the table:

**Retention is worst at n_B = 2849** — 0.34 at 1728 atoms against 0.60-0.63 at
the two smaller bases. That is where the per-atom stages dominate (`A2B` 54%,
`AA` 38% of `site_basis` at this shape), so the 3.7x atom-slot padding lands on
most of the work rather than a minority of it.

**What is not separated here.** This measurement gives the *total* cost of the
plugin path; it does not decompose it. Padding, LAMMPS's own neighbour list,
MPI-layer bookkeeping, integration and the PJRT call boundary are all inside
these ratios. The padding factors above are exact counts and the code path is
verified, but no experiment here attributes a share of the lost throughput to
each. Re-running one point with capacities cut to the actual counts would do
that, and has not been done.

## What that says about the pace comparison

At 1728 atoms, `pair pace/kk` against acejax **in Python**:

| basis (ours vs pace) | pace / acejax f64 | pace / acejax f32 |
|---|---|---|
| 69 vs 78 | 1.9x | 0.8x |
| 710 vs 693 | 1.3x | 0.7x |
| 2849 vs 2874 | **0.9x** | 0.6x |

**This is not like-for-like and must not be quoted as "acejax beats pace".**
pace is measured inside LAMMPS and pays the neighbour list, communication and
integration that the Python line does not, so the comparison flatters our side.
The honest reading is a bound in the other direction: **the like-for-like
deficit at n_B = 2849 (2.6x, both in LAMMPS) is larger than any deficit
attributable to the kernel** — in Python the same model in f64 matches pace's
throughput at a matched basis size. Whatever the kernel gap is at production
basis size, it is smaller than the plugin path's contribution.

**f64 against f32 in Python** costs 2.34x at n_B = 69, 1.80x at 710 and 1.48x at
2849, all at 1728 atoms. The penalty *falls* as the basis grows, consistent with
the large-basis stages being bandwidth- and index-bound rather than
arithmetic-bound.

## Our spherical harmonics against sphericart

The harmonics are hand-written because sphericart lowers to an FFI custom call,
which would put a custom-call target in exported StableHLO. Measured on GPU,
1728 atoms, f64:

| lmax | isolated | end-to-end (`site_basis`) |
|---|---|---|
| 4 | sphericart **3.06x faster** | **ours 1.25x faster** |
| 10 | ours 1.5x faster | **ours 1.13x faster** |

**Ours is faster end-to-end at both, despite losing 3x in isolation at lmax 4.**
XLA fuses our recursion into the surrounding computation; sphericart's FFI call
is an opaque barrier that forces intermediates to be materialised. Our isolated
harmonic call at lmax 4 (2.209 ms) is *slower than the whole `site_basis` that
contains it* (1.315 ms), which is only possible with fusion.

So there is no fork to make: the pure-JAX recursion stands on speed as well as
on avoiding the FFI machinery, and no second backend is needed. Folding
`r^(l-|m|)` in — no division at all — is likely part of why.

This is also the clearest demonstration of why isolated stage timings mislead
here, and why the end-to-end delta was the right thing to insist on.

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
