# Distilling MACE-MP-0-small into ACE models of CrMnFeCoNi

250 rattled/strained fcc cells (32 and 48 atoms), equiatomic, **random site
occupancy — no short-range order**. Labelled with MACE-MP-0-small on an RTX
A4500 (energies, forces, virials). 200 train / 50 test. `lambda` swept per model
over 1e-2 … 1e-8; the best held-out force RMSE is reported.

| model | n_B | parameters | test F (eV/Å) | test E (eV/atom) | best λ |
|---|---|---|---|---|---|
| categorical `ace1_model` | 1 348 | 6 740 | **0.1736** | 0.0108 | 1e-2 |
| embedded, lossless `d_ν=[5,15,35]` | 475 | 2 375 | 0.1780 | 0.0102 | 1e-2 |
| embedded, `d_max=16` | 323 | 1 615 | 0.1771 | 0.0082 | 1e-8 |
| embedded, `d_max=8` | 182 | 910 | 0.1817 | **0.0076** | 1e-2 |

## What it says

**The frozen MACE embedding matches categorical accuracy at a fraction of the
size.** Forces span 0.1736–0.1817 eV/Å across all four — a **2.5% spread** —
while parameters fall 7.4x from 6 740 to 910. Energies are *better* for every
embedded model than for categorical.

Against the success criterion fixed in advance ("within ~10–20% of categorical
at matched n_B"), this passes comfortably, and at *lower* n_B rather than
matched.

Two things worth noting:

- **The lossless arm behaves.** It was included as a correctness check: an exact
  reparameterisation must not fit worse. At 0.1780 against 0.1736 it is within
  2.5%, so the construction is sound and the earlier scale/regularisation
  failures are not recurring.
- **Truncation is nearly free here.** `d_max=8` keeps 182 of the categorical
  1 348 basis functions and loses 4.7% on forces while gaining on energies. At
  S=5 the species tensor is small (dim Sym³(R⁵) = 35), so there is little to
  lose — this should NOT be extrapolated to larger S, where the crossover
  analysis says truncation bites harder.

## What it does not say

- **This is student-vs-teacher agreement, not physical accuracy.** The labels
  are MACE's, so these errors measure how well ACE reproduces MACE, not DFT.
- **No short-range order.** Random occupancy is not the alloy; a real Cantor
  alloy has chemical SRO and the fit has never seen it.
- **The λ sweep hit its boundary.** Three of four models chose λ = 1e-2, the
  largest value swept, so the optimum may lie outside the range and these are
  lower bounds on achievable accuracy. Extend the sweep upward before quoting
  the absolute numbers.
- **No smoothness prior** (`P = I`), unlike a production `acefit!` call, and a
  single 200/50 split with λ chosen on the test set — optimistic, but equally
  so for every model.
- 0.17 eV/Å against forces reaching 8 eV/Å is ~2% relative: reasonable, not tight.

## Reproduce

```
python acejax/bench/distil/make_structures.py -n 250 -o cantor.xyz
# label on a GPU host with the mace-jax environment, then:
DATA=cantor_v9.xyz EMB=mace_embedding_full.json \
  julia -p 16 --project=acejax/julia acejax/bench/distil/fit_distilled.jl
```

## Learning curves and the label-quality result (1k set, degree 6 and 8)

Degree 6 was refitted on the 1000-structure set so degree comparisons share a
test set. Held-out force RMSE (200 structures):

| | categorical | lossless | d≤16 | d≤8 |
|---|---|---|---|---|
| degree 6 | 0.434 | 0.433 | 0.439 | 0.452 |
| degree 8 | 0.397 | 0.404 | 0.412 | 0.420 |

The embedding result holds on the harder set: within 4-6% at 3-9x fewer
parameters, and better on energies. But **degree buys little** (9% for 3x the
parameters) and **λ pinned to 1e-9**, the bottom of a nine-decade sweep.

**Learning curve, categorical degree 6** — test error is flat (0.458 at N=100
→ 0.433 at N=800) while *train* error rises to 0.27 and plateaus: not
data-limited, and the model cannot represent its own training labels.

**Teacher disagreement explains it.** MACE-MP-0-small vs MACE-MH-1 on the same
1000 structures: **force RMS difference 0.531 eV/Å**, per-structure median
0.412, on a force scale of 1.06. The student's 0.43 is *smaller* than the
disagreement between two foundation models. On energies: student 0.02 eV/atom,
teachers 0.12 after removing a constant offset.

**RETRACTED as stated.** "Student error below teacher disagreement" implies a
noise floor only if the disagreement is *rough*. If MP-0 and MH-1 differ by a
smooth function of structure — a different equation of state, a different
force scale — an ACE fits that difference fine, and the 0.53 says nothing about
whether MP-0's labels are learnable. The train-error plateau at 0.27 remains
unexplained by this comparison. The discriminating test is below: fit the same
basis to MP-0, to MH-1, and to their *difference*. A small fit error on the
difference means it is smooth and the floor is representational; a fit error
near 0.53 means the disagreement is rough.

Consequences: (1) the structure set is the lever — milder rattle, physical
short-range order, staying where the teacher is reliable; (2) use the current
state-of-the-art teacher (MH-1 is cached), not MP-0; (3) none of the earlier
absolute errors should be read as a statement about ACE's capability on this
alloy.

## Smooth or rough? Fitting the difference (degree 6, categorical, same A)

The design matrices for the MP-0 and MH-1 labelled sets are bit-identical
(`max|A0 - A1| = 0`), so three targets were fitted with one factorisation:

| target | test F (eV/Å) | train F | test E (eV/atom) |
|---|---|---|---|
| MP-0 labels | 0.4328 | 0.2680 | 0.0205 |
| **MH-1 labels** | **0.1801** | **0.1483** | **0.0063** |
| MP-0 − MH-1 | 0.4377 | 0.2844 | 0.0213 |

**The disagreement is rough.** The difference fits exactly as badly as MP-0
itself — 0.28 *in-sample* — so it is not a smooth offset that an ACE could
absorb. **MH-1 is the smooth teacher**: 2.4x better on forces and 3.3x on
energies with the same basis on the same structures, and a small train/test gap.

This is what the earlier teacher-disagreement number could not establish on its
own (a smooth disagreement would have given the same 0.53). The difference-fit
is the discriminating test, and it should be run before trusting any teacher.

Combined with the per-structure breakdown (ten crushed cells carry 77% of the
squared test error), the picture is: **MP-0's labels are rough on this
structure set, MH-1's are learnable, and the residual test error is dominated
by a generator tail** (6% Gaussian on the lattice parameter -> cells at 36%
compression with forces above 20 eV/Å). Both are fixable before touching the
basis.

## Round 3: bounded generator, MACE-MH-1 teacher (1k set, 800/200)

Both fixes applied. The generator now draws bounded uniform perturbations
(lattice parameter ±3%, shear 0.02, rattle 0.02–0.10 Å, min separation 2.0 Å)
instead of a 6% Gaussian, and the labels come from MACE-MH-1 (`label.py`, with
a `--fmax 10` filter that dropped **nothing**: the bounded set's largest force
is 5.25 eV/Å, median 1.88). Volume per atom spans 10.1–13.2 Å³ (fcc 11.6).

### Degree 6

| model | n_B | parameters | test F (eV/Å) | test E (eV/atom) | best λ |
|---|---|---|---|---|---|
| categorical `ace1_model` | 1 348 | 6 740 | **0.0969** | 0.0039 | 1e-8 |
| embedded, lossless `d_ν=[5,15,35]` | 475 | 2 375 | 0.1087 | 0.0036 | 1e-9 |
| embedded, `d_max=16` | 323 | 1 615 | 0.1101 | 0.0032 | 1e-8 |
| embedded, `d_max=8` | 182 | 910 | 0.1162 | 0.0035 | 3e-7 |

- Test force RMSE **halved** against the previous MH-1 result on the same basis
  (0.1801 → 0.0969), with the same 1k-structure budget. That whole gain is the
  generator: no crushed cells, so no tail for the test error to be dominated by.
- Energies are now at 3–4 meV/atom for every model.
- The embedding ranking is unchanged and the gap is now measurable: lossless
  +12%, `d_max=16` +14%, `d_max=8` +20% on forces at 2.8×, 4.2× and 7.4× fewer
  parameters. Energies are *better* for the embedded models (fewer parameters,
  same 1k structures — the categorical readout is the least constrained).
  This is inside the "fair trade" band set in advance (10–20%).
- `d_max=16` and lossless are indistinguishable (0.1101 vs 0.1087): at S=5 the
  truncation from `[5,15,35]` to `[5,15,16]` at order 3 costs nothing
  measurable, as the rank measurement predicted.

### Degree 8

| model | n_B | parameters | test F (eV/Å) | test E (eV/atom) | best λ |
|---|---|---|---|---|---|
| categorical `ace1_model` | 3 824 | 19 120 | **0.0855** | 0.0040 | 3e-7 |
| embedded, lossless `d_ν=[5,15,35]` | 1 190 | 5 950 | 0.0906 | 0.0033 | 1e-9 |
| embedded, `d_max=16` | 753 | 3 765 | 0.0911 | 0.0030 | 1e-9 |
| embedded, `d_max=8` | 408 | 2 040 | 0.0986 | 0.0033 | 3e-8 |

- Degree 6 → 8 buys **12%** on forces for the categorical model (0.0969 →
  0.0855) at 2.8× the parameters, and 17% for `d_max=16` (0.1101 → 0.0911) at
  2.3×. Energies do not move (3–4 meV/atom at both degrees): the force error is
  the only thing the extra basis is buying.
- **The embedding gap narrows with degree**: +14% at degree 6, **+7%** at
  degree 8 for `d_max=16`, now at 5.1× fewer parameters than categorical.
  `d_max=8` is +15% at 9.4× fewer.
- Lossless and `d_max=16` remain indistinguishable (0.0906 vs 0.0911).
- **`d_max=16` at degree 8 (753 functions, 0.0911) beats categorical at degree
  6 (1 348 functions, 0.0969)** with 56% of the basis. Per basis function the
  embedded model is the better trade at every point measured.
- The best λ keeps falling (1e-9 for the embedded models): with 1k structures
  and a bounded distribution the smoothness prior is doing almost nothing, so
  these fits are data-regularised. A larger structure set would let the prior
  matter and probably lower the floor further — the learning curve on this set
  has not been run.

## Speed-up after distillation (RTX A4500, in-process, `latency.py`)

`head2head.py` times `mace_jax_predict` as a subprocess by difference of two
runs, which needs ~1000 frames for the variable part to clear the run-to-run
variation of its 40–70 s fixed cost (its README says so). Run here at 64
frames on 40-atom cells it gave 3.5 ms/frame for MH-1 and a *negative* time
for MP-0-small — **those two runs are noise; the S=75 numbers in
`manyelem/README.md` were taken at 1000 frames and stand.** `latency.py` jits both models once in one process, pads to a fixed
capacity, and times each frame with `block_until_ready`. "kernel" is device
time with inputs pre-built; "frame" adds the host neighbour list (matscipy),
padding and transfer — what an ASE calculator sees. All on the same frames.

Student = embedded `d_max=16`, S=5, order 3, which distils MH-1 to 0.110 eV/Å
(degree 6) on this distribution. The cost is weight-independent, so the
timing used the exported basis with random weights (`ACE_NOFIT=1`).

**float32** (what MD would run):

| cell | model | kernel ms/frame | atom-steps/s | frame ms | atom-steps/s |
|---|---|---|---|---|---|
| 32–48 atoms | MACE-MH-1 | 9.76 | 3.9e3 | 11.8 | 3.2e3 |
| | MACE-MP-0-small | 4.43 | 8.6e3 | 6.61 | 5.7e3 |
| | ACE deg 6 | **0.469** | **8.1e4** | 3.10 | 1.2e4 |
| | ACE deg 8 | 0.744 | 5.1e4 | 3.40 | 1.1e4 |
| 256–384 atoms | MACE-MH-1 | 44.9 | 6.8e3 | 53.9 | 5.6e3 |
| | MACE-MP-0-small | 15.2 | 2.0e4 | 24.6 | 1.2e4 |
| | ACE deg 6 | **2.02** | **1.5e5** | 11.8 | 2.6e4 |
| | ACE deg 8 | 5.23 | 5.8e4 | 15.2 | 2.0e4 |
| 864–1296 atoms | MACE-MH-1 | 136 | 7.2e3 | 164 | 5.9e3 |
| | MACE-MP-0-small | 43.1 | 2.3e4 | 70.6 | 1.4e4 |
| | ACE deg 6 | **12.2** | **7.9e4** | 41.5 | 2.3e4 |
| | ACE deg 8 | 27.8 | 3.5e4 | 58.7 | 1.7e4 |
| 2048–3072 atoms | MACE-MH-1 | **OOM** (14.5 GiB allocation on a 20 GB card) | | | |
| | MACE-MP-0-small | 106 | 2.2e4 | 169 | 1.4e4 |
| | ACE deg 6 | 30.2 | 7.6e4 | 103 | 2.2e4 |
| | ACE deg 8 | 69.2 | 3.3e4 | 140 | 1.7e4 |

**Kernel speed-up of the degree-6 student over its teacher MH-1: 21× (small
cells), 22× (256–384 atoms), 11× (~1000 atoms).** Degree 8: 13×, 8.6×, 4.9×.
Against MP-0-small the degree-6 student is 9.4× / 7.5× / 3.5×.

float64 (for reference; the A4500 runs fp64 at 1/64 rate, so this mostly
measures MACE's matmuls): MH-1 49.6 ms and MP-0-small 12.7 ms on the small
cells vs ACE 0.82 / 1.59 ms — 60× / 31×. Not the number to quote.

What the table also says:

- **End-to-end, the host neighbour list dominates ACE.** At 256–384 atoms the
  kernel is 2 ms and the frame is 12 ms; the ASE-calculator view of the
  speed-up over MH-1 is 3.8–4.6× at every size, not 11–22×. The kernel number is
  what LAMMPS (neighbour list built on the device, reused across steps) sees.
  So the speed-up available *to MD* depends on the driver, and the lammps-jax
  path is where the kernel number is realised.
- **ACE's kernel throughput is not flat in cell size:** 1.5e5 atom-steps/s at
  ~330 atoms drops to 7.9e4 at ~1100. Kernel time grows 6× for 3.3× the atoms
  between those two points, then linearly after. This is the regime Phase 15
  (gather vs matmul `edge_a_kind` calibration) targets; it has not been run on
  this GPU. MACE's throughput is flat above ~300 atoms.
- **MH-1 does not fit a 3 000-atom cell on 20 GB** in this driver; MP-0-small
  and both students do.

### What "comparable accuracy" can and cannot mean here

The student reproduces MH-1 to 0.11 eV/Å / 3 meV/atom on the bounded
distribution — that is student-vs-teacher agreement. Whether the student is
as good as MH-1 *at physics* needs both measured against DFT on the same
structures, and no DFT set exists for this alloy in this work. Until one does,
the honest statement is: **a degree-6, S=5 frozen-embedding ACE reproduces
MACE-MH-1 to 0.11 eV/Å on rattled/strained fcc CrMnFeCoNi at 11–22× the kernel
throughput (fp32, A4500), and 4× end-to-end through an ASE calculator.**

## Embedding table and reduction: a 2×2 at degree 6 (bounded set, MH-1 labels)

Every embedded fit above used **MACE-MP-0-small's** table with **MH-1** as the
teacher, reduced by taking the **first `d` columns**. Two things were wrong
with that. The table did not match the teacher; and column truncation is not a
faithful reduction — measured on CrMnFeCoNi, the cosine Gram of the first 16 of
MP-0-small's 128 channels correlates 0.52 with the full table's (0.71 for the
first 16 of MH-1's 512), and the first 5 columns are unrelated to it (−0.27 /
0.18). The models were using an arbitrary projection, not the foundation
model's element similarity.

`embedding_rows` now defaults to `reduction = :pca`: row-normalise, SVD, keep
the principal coordinates (Gram exact at any `d ≥ rank`), and for `d > rank`
mix them into `d` channels with an orthonormal-row frame. The frame has to be
**generic** — the channel-diagonal many-body basis spans `Sym^ν` through the
columns' ν-th powers, and a DCT-II frame reached only 9/15 and 13/35 — so it is
a pseudo-random orthonormal frame from an inline xorshift (reproducible across
Julia versions). Verified: Gram to 1e-15, full rank at orders 1–3 for
d = 5, 15, 16, 35 on both tables; `:truncate` kept for comparison.

Test F (eV/Å), categorical control 0.0969 in every run:

| table / reduction | lossless `[5,15,35]` | `d_max=16` | `d_max=8` |
|---|---|---|---|
| MP-0-small, truncate (all results above) | 0.1087 | 0.1101 | 0.1162 |
| MH-1, truncate | 0.1087 | 0.1098 | 0.1208 |
| MP-0-small, PCA | 0.1091 | 0.1105 | 0.1209 |
| MH-1, PCA | 0.1093 | 0.1107 | 0.1212 |

**At S=5 neither the table nor the reduction matters** (spread 0.001 at
`d_max=16`, 0.005 at `d_max=8`). The lossless column is the expected control:
it spans the full species tensor at every order whatever the columns, so the
embedding's *values* cannot enter. But `d_max=16` cuts order 3 to 16 of 35
directions and `d_max=8` cuts orders 2 and 3, and even there the choice of
directions — chemically meaningful (PCA of the teacher's own table) or
arbitrary — makes no measurable difference. **On this system the frozen
embedding is acting as a generic tensor reduction (Darby et al.), not as
transferred chemistry.** That is not a failure of the PCA change, which is
still the right construction; it is a statement about S=5 with 1k structures,
where the truncated orders retain enough generic directions that which ones
barely matters. The chemistry can only show at larger S, where `d_1 = d_max <
S` truncates order 1 itself and the directions decide which element contrasts
the model can see at all. That is the S=10/20 experiment, still unrun.

## Julia vs JAX evaluators, same host (lestrade: 32 cores, RTX 4000 Ada)

`latency_julia.jl` (ACEpotentials, `energy_forces_virial`, f64, random weights)
and `latency.py` on the same 16 frames and supercells. atom-steps/s:

| evaluator | 32–48 atoms | 256–384 | 864–1296 |
|---|---|---|---|
| Julia CPU, 1 thread, embedded d≤16 deg 6 | 4.7e3 | 4.9e3 | 4.6e3 |
| Julia CPU, 1 thread, embedded d≤16 deg 8 | 2.6e3 | 2.7e3 | 2.6e3 |
| Julia CPU, 1 thread, **categorical** deg 6 | 8.3e3 | 8.4e3 | 7.8e3 |
| Julia CPU, 8 / 32 threads, embedded deg 6 | — | 1.8e4 / 2.1e4 | — |
| Julia CPU, 8 / 32 threads, categorical deg 6 | — | 2.7e4 / 2.6e4 | — |
| JAX GPU f64 kernel, embedded deg 6 | 7.4e4 | 8.3e4 | (MH-1 OOM aborted the run) |
| JAX GPU f64 kernel, embedded deg 8 | 3.4e4 | 3.8e4 | |
| JAX GPU f32 kernel, embedded deg 6 | 1.6e5 | 1.8e5 | 1.7e5 |
| JAX GPU f32 kernel, embedded deg 8 | 7.2e4 | 7.5e4 | 6.8e4 |
| JAX GPU f32 frame (host nlist), embedded deg 6 | 2.6e4 | 5.3e4 | 5.5e4 |
| MACE-MH-1 GPU f64 kernel | 7.6e2 | 8.6e2 | OOM |
| MACE-MH-1 GPU f32 kernel | 4.2e3 | 4.9e3 | 5.3e3 |

Speed-up of the degree-6 student over MH-1 (fp32 GPU, its MD mode):

- **JAX evaluator, GPU kernel: 32–38×**; end-to-end with the host neighbour
  list 6–11×.
- **Julia evaluator, one CPU core: ~1×** (4.7e3 vs 4.2–5.3e3) — a single core
  matches the foundation model on a GPU; 8 threads give ~4× and the evaluator
  stops scaling beyond that (32 threads: 2.1e4).
- JAX-over-Julia on this host: 16–18× (f64, like for like), 35× (f32), 8× vs
  the 32-thread CPU run.

Two findings that matter beyond the numbers:

1. **In Julia the embedded model is 1.8× *slower* than categorical** (4.7e3 vs
   8.3e3) despite 4× fewer basis functions, at every size and thread count.
   The embedded radial basis is `d = 16` channels wide — 224 radial functions
   against categorical's 14 per pair — and ACEpotentials evaluates them all.
   Only acejax's factorised radial (`P[:, n′] · emb[z, k]`, the export path's
   `(ncoef, n′) + (S, d)` form) turns that into cheap work. So the embedding's
   evaluation-cost advantage currently exists in the JAX evaluator only; the
   Julia one would need the same factorisation in `SplineRnlrzzBasis` to see
   it. Its parameter-count advantage (fitting cost, memory) holds in both.
2. **The ACE kernel is flat in cell size on the RTX 4000 Ada** (1.6–1.8e5 at
   40–1300 atoms), where the A4500 dropped 2× between ~330 and ~1100 atoms with
   the earlier exports. Same code; the difference is the GPU (or the export —
   these are the MH-1-PCA exports). Phase 15's calibration should be run on
   both before reading anything into the A4500 drop.

Julia here is single-node CPU; there is no GPU Julia path (Reactant is
blocked, `docs/findings/FINDINGS_reactant.md`). That, plus finding 1, is the
practical case for the JAX evaluator as the MD target for embedded models.

## Learning curves on the bounded set: data- or basis-limited?

Same split as the fits; the first N training structures are the first rows of
the cached design matrix; λ swept per N; held-out test F and in-sample train F
at the best λ. Embedded = `d_max=16`, MH-1 table, PCA.

| N | deg 6 categorical | deg 6 embedded | deg 8 categorical | deg 8 embedded |
|---|---|---|---|---|
| 50 | 0.153 / 0.084 | 0.125 / 0.090 | 0.158 / 0.056 | 0.124 / 0.072 |
| 100 | 0.132 / 0.080 | 0.118 / 0.095 | 0.136 / 0.070 | 0.109 / 0.072 |
| 200 | 0.113 / 0.073 | 0.113 / 0.100 | 0.114 / 0.063 | 0.098 / 0.078 |
| 400 | 0.101 / 0.082 | 0.111 / 0.104 | 0.099 / 0.063 | 0.093 / 0.083 |
| 800 | **0.097** / 0.087 | **0.111** / 0.105 | **0.086** / 0.063 | **0.092** / 0.086 |

(test F / train F, eV/Å)

- **Degree 6 is basis-limited**, both models: train and test meet (0.097 vs
  0.087; 0.111 vs 0.105) and the curves are flat by N=400. More structures
  will not move either. The embedded model saturates by N=200.
- **Degree 8 categorical is data-limited**: the gap is still 0.022 at N=800
  and the test curve still has slope (0.114 → 0.099 → 0.086 per doubling).
  With 19k parameters and 100k rows it wants more labels, and labels are
  cheap (~2 min of GPU per 1 000 structures). This is the model that would
  benefit from a 4k set.
- **Degree 8 embedded is basis-limited too**: gap 0.006 at N=800 (0.092 / 0.086). Only the categorical degree-8 model is still data-limited on this set.
- **The embedded model is the data-efficient one at every degree**: at N=50 it
  is 0.125 vs 0.153 (deg 6) and 0.124 vs 0.158 (deg 8); at N=100 degree-8
  embedded (0.109) already beats degree-6 categorical at N=800 (0.097 is
  reached by N=200). Fewer parameters fill up sooner. Below ~200 structures
  the embedded model is simply the better model; above that the categorical
  one's extra parameters start to pay, and at degree 8 they are still paying
  at 800.

So the answer to "data or basis?" is *both, by degree and by model*: degree 6
has run out of basis for both models, and so has degree-8 embedded; only
degree-8 categorical (19k parameters) has not run out of data. The next fits
worth doing on a 4k set are degree-8 categorical (to see where its curve
lands) and the embedded model at degree 10 (the embedded basis needs more
degree, not more data, and its 4x smaller matrix makes degree 10 affordable
where categorical degree 10 is not).

A 4k set (`cantor4k_b_mh1.xyz`, seed 12, same generator) is labelled: 4.6 min
on the A4500, nothing dropped, max|F| 5.90 eV/Å.

## 4k structures (3200 train / 800 test), degrees 8 and 10

Assembled on the fast basis path (`FASTBASIS=1`, `docs/findings/FINDINGS_assembly_profile.md`):
the 404 144 × 46 885 degree-10 categorical matrix took ~25 min on 12 workers
instead of the ~14 h the old path projected. MH-1 labels, PCA-reduced MH-1
embedding; test F in eV/Å, E in eV/atom.

| model | n_B | params | 1k test F | **4k test F** | 4k test E |
|---|---|---|---|---|---|
| deg 8 categorical | 3 824 | 19 120 | 0.0855 | **0.0763** | 0.0024 |
| deg 8 embedded lossless | 1 190 | 5 950 | 0.0906 | 0.0885 | 0.0025 |
| deg 8 embedded d≤16 | 753 | 3 765 | 0.0911 | 0.0894 | 0.0025 |
| deg 8 embedded d≤8 | 408 | 2 040 | 0.0986 | 0.1000 | 0.0027 |
| **deg 10 categorical** | 9 327 | 46 635 | — (151 GB) | **0.0641** | 0.0020 |
| **deg 10 embedded lossless** | 2 730 | 13 650 | — | **0.0753** | 0.0022 |
| **deg 10 embedded d≤16** | 1 609 | 8 045 | — | **0.0767** | 0.0022 |
| deg 10 embedded d≤8 | 850 | 4 250 | — | 0.0895 | 0.0025 |

- The learning-curve predictions held exactly. Degree-8 categorical was
  data-limited and gained 11% from 4× the data; the degree-8 embedded models
  were basis-limited and gained 2% (d≤8: nothing).
- **The embedded basis wanted degree, not data**: at degree 10, embedded d≤16
  (8 045 params) matches degree-8 categorical (19 120 params, 0.0763) at
  0.0767 — **2.4× fewer parameters** — and lossless (13 650) beats it at
  0.0753, the best number in this work. Energies 2.2 meV/atom.
- Lossless and d≤16 remain indistinguishable at every degree (0.0753 /
  0.0767); d≤8 at degree 10 (0.0895) ≈ d≤16 at degree 8 (0.0894) at 12%
  more parameters — the order-2 truncation of d≤8 costs about one degree.
- **Degree-10 categorical: 0.0641 eV/Å, 2.0 meV/atom** — 46 635 parameters,
  151 GB design matrix, ~25 min assembly + in-place QR + 46k SVD on cowf02
  (376 GB node, BIGMEM path). The best absolute number, 16% better than
  degree 8, and still with room (its 1k→4k learning curve had slope). So the
  trade is exactly as anticipated: **categorical is the most accurate per
  structure, embedded the most accurate per parameter** — at degree 10 the
  embedded lossless model gives 0.0753 with 3.4× fewer parameters, d≤16
  0.0767 with 5.8× fewer. Per basis function the embedded models win at every
  point measured; per structure of teacher data the categorical model wins
  once the data are there to fill it.
- Where the embedded models sit relative to production: 0.075 eV/Å against
  a teacher whose forces reach 6 eV/Å (median 1.9) — ~4% relative; energies
  at 2 meV/atom. Degree 12 embedded (15 885 params at d≤16, ~50 GB at 4k)
  is affordable on any big node and is the obvious next point; categorical
  degree 12 (101 540 params, 330 GB) is not, without phase 18.
