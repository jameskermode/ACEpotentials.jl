# Why the JAX MD path degrades with system size (worst on Apple Silicon)

Follow-up to `acejax/bench/molly/RESULTS.md`, which measured the JAX MD spike
(`acejax/spike_distmd/ace_md.py`) against ACEpotentials.jl + Molly.jl on two CPUs
and left one result unexplained: on the M3 Pro the JAX per-atom step cost climbs
**2.3x** from 216 to 1728 atoms, while on a Xeon 4216 it is flat and
ACEpotentials.jl is flat on both hosts. That file listed the cause under "Not
obtained" because attributing it needed difference measurements against varied
static capacities, and the spike's are hardcoded.

## The answer, in one line

**It is not a padding effect and capacities are not the cause — but they are half
the cure.** The padded shapes grow strictly linearly with `n_atoms` and the
padding fraction is *identical* (0.491) at every system size. The step cost turns
out to be a function of the padded **edge-buffer length alone**, and the cost *per
padded slot* grows 2.2x over the range the series spans, because reverse-mode
differentiation of one line of `acejax` —

```python
edge_A = Rnl[:, self.aspec_r] * Ylm[:, self.aspec_y]      # acejax/model.py:156
```

— lowers to a **scatter along the inner axis of a tall `(E, K)` array**, and
XLA-CPU's scatter kernel on this machine costs 4.6x more per row at 500k rows
than at 17k. Everything else in the kernel (the splined radial basis, the
spherical harmonics, the segment_sum pooling into nodes) is flat.

Rewriting that one expression as an algebraically identical one-hot matmul —
whose adjoint is a matmul, not a scatter — **removes the size degradation
entirely**: per-atom cost becomes 38.9 / 40.0 / 39.3 / 39.9 µs across
216→1728 atoms, flat to 2.8%. Combining it with a tighter edge capacity gives
**5.1x** end-to-end at 1728 atoms, with bit-identical trajectories.

| atoms | published M3 Pro JAX | this harness, stock | + tight capacity | + gather fix | tight + fix |
|---|---|---|---|---|---|
| 216  | 1.85e4 | 1.94e4 | 3.30e4 | 2.57e4 | **4.13e4** |
| 512  | 1.32e4 | 1.36e4 | 3.11e4 | 2.50e4 | **4.45e4** |
| 1000 | 9.38e3 | 1.01e4 | 2.29e4 | 2.55e4 | **4.16e4** |
| 1728 | 8.15e3 | 8.27e3 | 1.87e4 | 2.51e4 | **4.25e4** |

atom-steps/s, end-to-end (`inner step + rebuild/10`), min of two invocations each
themselves a 6-leg mean. The first column is `bench/molly/logs/m3pro_collected.json`,
not re-measured here; the second is the same configuration under this harness and
agrees with it to 1-8%.

---

## 1. The hypothesis under test, and why it is wrong

The brief's hypothesis was that "the padded shapes grow differently with n_atoms
than the real work does". They do not. At one rank the two hardcoded capacities
reduce to:

- `gx.create_config(..., capacity_mult=1.5)` → `max_owned = ceil(1.5 N)`; with a
  1x1x1 grid the ghost fraction is zero, so `max_ghost` is its floor of 8.
- `gx.subgraph_params(..., capacity_mult=2.0)` → `max_edges =
  ceil(owned_bound × density × (4/3)π(r_cut+skin)³)` with `owned_bound =
  max(2.0 N, max_owned) = 2.0 N`.

Both are exactly proportional to N, and the silicon density and the 7.0 Å reach
are the same at every `rep`. Measured, not argued:

| atoms | real edges | edge slots | slots/atom | **edge fill** | node slots | **node fill** | cells | cell_cap |
|---|---|---|---|---|---|---|---|---|
| 216  | 15218  | 31014  | 143.58 | 0.491 | 324  | 0.667 | 3³ | 16 |
| 512  | 36066  | 73515  | 143.58 | 0.491 | 768  | 0.667 | 3³ | 38 |
| 1000 | 70422  | 143584 | 143.58 | 0.490 | 1500 | 0.667 | 3³ | 75 |
| 1728 | 121762 | 248112 | 143.58 | 0.491 | 2592 | 0.667 | 4³ | 54 |

**The padding fraction is constant to three decimal places across the series.**
Half of every edge slot is padding at 216 atoms and half is padding at 1728. A
constant cannot produce a 2.3x trend. *The padding hypothesis is dead on the
shapes alone, before any timing.*

## 2. What the cost actually depends on

Sweep the edge buffer independently of the system (`--max-edges`, which overrides
only that one shape), and the answer is unambiguous. **Four system sizes, spanning
8x in atoms and 8x in real edges, at one padded buffer length:**

| atoms | real edges | edge slots | ms/step |
|---|---|---|---|
| 216  | 15218  | 248112 | 201.610 |
| 512  | 36066  | 248112 | 205.106 |
| 1000 | 70422  | 248112 | 206.665 |
| 1728 | 121762 | 248112 | 207.778 |

**Within 3%.** The inner step does not care how many atoms there are or how many
of the edges are real. It cares only how long the padded edge buffer is. Since
that buffer is 143.58 × N, "cost versus system size" is really "cost versus buffer
length" wearing a disguise.

The node buffer is nearly irrelevant, checked the same way (sweep C, `--max-owned`
varied 1.0x…6.0x with `--max-edges` pinned): 11.44→12.27 ms at 216 atoms and
205.5→230.9 ms at 1728 atoms, i.e. ≤7% and ≤12% over a 6x change in node slots.

### The per-slot cost is what grows

At a fixed 216-atom system, with the real edge count fixed at 15218:

| edge slots | ms/step | ns per padded slot | XLA temp arena |
|---|---|---|---|
| 17057  | 6.902   | 404.6 | 71.6 MB |
| 21000  | 8.273   | 393.9 | 87.8 MB |
| 25000  | 9.125   | 365.0 | 104.3 MB |
| 31013  | 12.130  | 391.1 | 129.1 MB |
| 37000  | 13.765  | 372.0 | 153.8 MB |
| 44000  | 17.161  | 390.0 | 182.6 MB |
| 52000  | 21.745  | 418.2 | 215.6 MB |
| 62027  | 29.217  | 471.0 | 256.9 MB |
| 78000  | 36.892  | 473.0 | 322.7 MB |
| 93000  | 50.880  | 547.1 | 384.5 MB |
| 110000 | 62.735  | 570.3 | 454.6 MB |
| 124054 | 72.964  | 588.2 | 512.5 MB |
| 160000 | 112.604 | 703.8 | 660.6 MB |
| 200000 | 152.202 | 761.0 | 825.5 MB |
| 248112 | 202.184 | 814.9 | 1023.7 MB |

Flat at ~380 ns/slot to about 44k slots, then a smooth climb to 815 — **2.15x**,
which is the 2.3x of the original report. The arena figure is XLA's own
`memory_analysis().temp_size_in_bytes`, not an estimate from array shapes.

Repeating the sweep in f32 (`--f32`) moves the knee to roughly twice the slot
count — flat to ~124k slots instead of ~44k — so the transition is set by *bytes*,
not by slot count. The collapse onto a single bytes axis is approximate rather
than exact: at matched arena sizes the slowdown factors relative to each dtype's
own flat baseline are 1.24 / 1.09, 1.55 / 1.77, 2.14 / 2.09 (f64 / f32).

## 3. It is not generic XLA-CPU behaviour

`micro_kernel.py` is the same skeleton with none of the ACE in it: an elementwise
chain to an `(E, K)` per-edge array, a `segment_sum` into `(N, K)` nodes with the
padded slots pointing at node 0 exactly as `ghost_exchange_subgraph` fills them, a
quadratic readout, and `value_and_grad` through all of it. On the same Mac:

| edge slots | K=43 ns/slot | K=128 ns/slot | K=256 ns/slot |
|---|---|---|---|
| 17057  | 53.9 | 206.7 | 425.9 |
| 31013  | 61.9 | 217.4 | 431.4 |
| 62027  | 67.2 | 210.5 | 412.4 |
| 124054 | 69.9 | 210.6 | 407.2 |
| 248112 | 67.1 | 207.1 | 388.0 |
| 496224 | 65.8 | —     | —     |

**Flat**, at every byte density, out to a 1.9 GB arena. So the degradation is not
"XLA-CPU on M3 falls off a memory cliff"; a generic kernel of the same shape and
larger footprint does not.

## 4. Which part of the ACE kernel, by difference

`ace_stage_scan.py` times `value_and_grad` of nested **prefixes** of
`Model.site_energies` at a fixed 15218 real edges and a growing buffer. ns per
padded slot, cumulative:

| edge slots | norm | + radial | + angular | + edge_features | + pool | full |
|---|---|---|---|---|---|---|
| 17057  | 4.9 | 114.6 | 121.5 | 201.3 | 224.7 | 261.8 |
| 31013  | 2.1 | 113.7 | 124.7 | 215.3 | 261.7 | 284.3 |
| 62027  | 1.2 | 110.3 | 122.2 | 293.1 | 353.1 | 350.7 |
| 124054 | 0.7 | 104.6 | 113.9 | 370.1 | 403.3 | 502.7 |
| 248112 | 0.6 | 107.7 | 123.7 | 477.2 | 488.8 | 675.8 |

The splined radial basis — transform, envelope, the 4-coefficient spline gather —
is **flat at 105-115 ns/slot** over the whole range. The spherical harmonics add a
flat 7-16. The jump appears the moment `edge_features` closes over them: its
increment goes **+80 → +354 ns/slot**.

> Caveat, stated rather than buried: prefix subtraction is not a clean
> difference measurement, because XLA fuses each prefix differently — the
> `full - pool` increment rises when the work it names is per-node and should be
> constant in E. The reliable readings here are the *totals* (262 → 676 ns/slot,
> matching the MD harness) and the *flatness of the radial and angular prefixes*.
> The attribution below does not rest on this table; §5 re-derives it in isolation.

## 5. The mechanism, isolated from ACE entirely

`edge_features` does one thing the flat stages do not: it gathers along **axis 1**
of a tall array, `Rnl[:, aspec_r]` and `Ylm[:, aspec_y]`, `(E,37)` and `(E,25)`
into `(E,43)`. Reverse mode turns each into a scatter-add along that inner axis.

`gather_probe.py` runs that on plain random arrays — no ACE, no splines — against
an algebraically identical one-hot matmul, `(R @ Sr) * (Y @ Sy)`:

| edge slots | gather ns/slot | matmul ns/slot | ratio | max abs gradient difference |
|---|---|---|---|---|
| 17057  | 116.4 | 71.2 | 1.64 | 0 |
| 31013  | 127.2 | 56.1 | 2.27 | 0 |
| 62027  | 216.0 | 59.9 | 3.60 | 0 |
| 124054 | 303.1 | 52.1 | 5.82 | 0 |
| 248112 | 427.8 | 53.4 | 8.01 | 0 |
| 496224 | 500.8 | 45.1 | 11.11 | 0 |

The gather form degrades 4.3x; the matmul form is flat; the gradients are **bit
identical**. Splitting it three ways (`gather_mechanism.py`), ns/slot:

| edge slots | dup fwd | dup grad | perm fwd | perm grad | matmul fwd | matmul grad |
|---|---|---|---|---|---|---|
| 17057  | 18.6 | 105.9 | 16.8 | 85.5  | 41.3 | 67.6 |
| 62027  | 15.9 | 160.9 | 14.1 | 157.2 | 31.5 | 44.9 |
| 248112 | 13.4 | 453.6 | 12.4 | 385.0 | 30.7 | 45.0 |
| 496224 | 13.2 | 488.7 | 12.4 | 425.0 | 32.3 | 50.2 |

Three things follow, and they are the whole finding:

1. **The forward gather is fine** — 13-19 ns/slot, and it gets *cheaper* per slot
   as the buffer grows. All of the cost is in the adjoint.
2. **The duplicate indices are not the problem.** ACE's `aspec_r` puts 43 entries
   into 37 columns, so its scatter has colliding writes; a pure permutation with
   no collisions degrades essentially as badly (85.5 → 425.0). It is the axis-1
   scatter itself.
3. **It is not streaming.** The matmul form moves the same `(E,37)`, `(E,25)` and
   `(E,43)` arrays and is flat. Only the scatter kernel degrades.

### Xeon control — measured after the fact

`gather_probe.py`, same command, same jax 0.11.1, on `moriarty` (Xeon Silver
4216), pinned with `taskset -c 0`:

| edge slots | gather ns/slot | matmul ns/slot | ratio |
|---|---|---|---|
| 17057  | 27.3 | 64.0 | 0.43 |
| 31013  | 33.8 | 58.8 | 0.58 |
| 62027  | 39.9 | 48.6 | 0.82 |
| 124054 | 42.6 | 46.8 | 0.91 |
| 248112 | 53.2 | 44.3 | 1.20 |
| 496224 | 53.5 | 39.8 | 1.35 |

**This corrects the framing of this document.** The scatter degrades on x86 too —
1.96x over the same buffer range — so "Apple Silicon" is the wrong label for the
*mechanism*. What is Apple-specific is the *severity*: 4.3x versus 1.96x, and in
absolute terms the M3 Pro scatter is 4.3x slower than the Xeon's at 17k slots and
9.4x slower at 496k. On the Xeon the effect is mild enough that it never becomes
a large share of an MD step, which is why that host looked flat.

**Consequence for adopting the fix: it is platform-dependent, and on x86 at
typical MD sizes it is a pessimisation.** On the Xeon the gather form is *faster*
than the one-hot matmul below ~200k edge slots (0.43x at 17k) and only loses
above it. Swapping in the matmul unconditionally would slow down the very sizes
most runs use on x86. Any upstreaming of this into `acejax/` must therefore be
conditional — on measured cost at the actual buffer length, not on a compile-time
choice — or the two forms must be selected per platform. It is not a free win.

**Not a jax-version artefact.** `bench/molly/RESULTS.md` ran jax 0.10.1 on the Mac
and jax 0.11.1 on the Xeon, so the flat-versus-degrading contrast had a version
confound in it. Installing jax 0.11.1 on this Mac and re-running `gather_probe.py`
gives 118.3 / 122.4 / 142.7 / 269.4 / 444.2 / 468.3 ns/slot over the same buffer
range — the same degradation. The confound is real but it is not the explanation.

## 6. Can the 2.3x be recovered? Yes, and more

Two independent changes, each measured across the full series, each verified to
leave the trajectory bit-identical. µs per atom per step, end-to-end:

| atoms | stock, default caps | tight caps only | gather fix only | **both** |
|---|---|---|---|---|
| 216  | 51.67  | 30.27 | 38.91 | **24.20** |
| 512  | 73.39  | 32.16 | 40.00 | **22.47** |
| 1000 | 99.07  | 43.76 | 39.26 | **24.06** |
| 1728 | 120.88 | 53.49 | 39.87 | **23.52** |
| **spread, 216→1728** | **2.34x** | **1.77x** | **1.02x** | **0.97x** |

- **Tighter capacity alone** buys 1.7-2.3x in absolute terms and cuts the
  size-degradation from 2.34x to 1.77x. It cannot remove it, because even a tight
  buffer still grows with N and so still walks up the per-slot curve of §2.
- **The gather fix alone** removes the size-degradation *completely* — 38.9 to
  39.9 µs/atom across an 8x range — while leaving 143.58 slots per atom of padding
  in place.
- **Together**: 22.5-24.2 µs/atom, flat, and **5.1x** faster than the benchmarked
  configuration at 1728 atoms.

**How tight is tight.** 79.0 edge slots per atom (fill 0.892) ran clean at all four
sizes. 72.5 slots/atom (fill ~0.97) **failed** at 216 atoms — `ghost_exchange`'s
overflow flag tripped at leg 4, once the configuration had heated and the real
edge count had grown past the buffer. Nothing between 72.5 and 79.0 was tested.
In `gx.subgraph_params` terms 79.0 slots/atom is `capacity_mult ≈ 1.10`, against
the spike's 2.0; `create_config`'s 1.5 can go to 1.0 with no measurable effect
either way. **A production setting needs headroom for the real edge count to grow
during a leg — this is a benchmark bound, not a recommendation to run at 1.10.**

**What this does to the cross-engine conclusion.** `RESULTS.md` concluded that on
the M3 Pro ACEpotentials.jl on one thread beats the JAX path by 1.35-2.94x, with
the margin growing. Against its Julia 1-thread column (2.49e4 / 2.42e4 / 2.47e4 /
2.40e4 atom-steps/s, **not re-measured here**), the tuned JAX path at 4.13e4 /
4.45e4 / 4.16e4 / 4.25e4 is 1.66-1.84x *ahead* at every size. That reverses the
single-thread half of that finding. Julia's 12-thread column (5.75e4-9.71e4) still
wins; the JAX path remains single-threaded on CPU.

**Only one side was tuned.** This work optimised the JAX path and nothing else.
`RESULTS.md` was explicit that more effort had gone into the Julia side than the
JAX side; that is now reversed, and the comparison above should be read as "what
the JAX path can do", not as a fresh like-for-like verdict.

## 7. Correctness gating

Every timing in this file is gated on a check against `acejax.ACECalculator` at
the final configuration of the same 70-step trajectory, because a capacity too
tight to hold the edge list silently drops edges and looks fast.

- **f64 runs**: max |F − F_ASE| ≤ **2e-13 eV/Å** (observed 1.2-1.7e-13, against
  max|F| ≈ 1.8-2.8 eV/Å), and |E − E_ASE| ≤ 6e-11 eV on ~2e5 eV (mostly exactly 0).
- The final potential energy agrees to all 9 printed decimals **across every
  configuration** — stock vs gather-fix, default vs tight capacity — as does the
  energy drift (e.g. 1728 atoms: −281982.405215895 eV, +0.2825 meV/atom/ps, in all
  four). The two changes are numerically inert.
- `ghost_exchange`'s own overflow flag is asserted on at every leg, which is what
  caught the 72.5 slots/atom failure above.
- **f32 runs** (§2 only, never mixed into an f64 table): |ΔE| = 3.9e-3 eV,
  max|ΔF| = 7.1e-5 eV/Å against the f32 calculator.
- The dtype actually in flight is recorded in every result row (`pos_dtype`,
  `edge_idx_dtype`, `x64`), because `ace_dist.py` calls
  `jax.config.update("jax_enable_x64", True)` at import time and would silently
  override a setting made before it.

## 8. Retractions and things not determined

**Retracted mid-investigation.** The first run of sweep A produced ten points that
looked like a capacity sweep and were not: the `--max-edges` override never
reached `run()` because a patch to `main()` mismatched by one space and applied
silently. All ten ran at the default capacity. They are kept, relabelled, in
`logs/A_BROKEN_override_noop.jsonl` and appear in no table here. (As replicates of
the default at 216 atoms they give a useful noise figure: 11.53-12.56 ms, ±4%.)

**~~Not measured: a Xeon control under this harness.~~ Now measured** — see
"Xeon control" above. It was run from a session that could authenticate, and it
changed the conclusion: the scatter degrades on x86 as well (1.96x), so the
mechanism is not Apple-specific, only its severity is. The end-to-end MD
measurements in this document remain Mac-only; only the microbenchmark has an
x86 control.

**Not determined: the microarchitectural cause of the scatter degradation.** No
performance counters were read (Instruments' CPU-counter templates need GUI/root;
`powermetrics` needs sudo). What is established is what it is *not*: not the
forward gather, not index collisions, not streaming bandwidth, not arena size in
general, and not a jax version. Whether it is a cache-residency, TLB, or
store-buffer effect in XLA-CPU's scatter emitter is open.

**Not measured: whether the fix belongs upstream in `acejax/`.** It is applied
here as a monkeypatch (`--gather-fix`) so both paths could be measured in one
harness. `acejax/acejax/model.py` is untouched. The one-hot matrices are `(37,43)`
and `(25,43)` constants that XLA folds at compile time, and the gradients are bit
identical, but the change has not been run against the acejax test suite, and its
effect on GPU (where scatter is a different kernel entirely) is unknown and could
easily go the other way.

**Not measured: the reneighbour build's own scaling.** It is small here (0.75 →
14.5 ms per 10-step leg, i.e. 3.6% of a tuned step at 1728 atoms) but it is
growing faster than linearly: `subgraph_params` clamps the tile to a minimum of
3 cells per side, so at 216-1000 atoms the 27-cell stencil covers the entire box
and the candidate list is every atom. Once the force kernel is 5x faster this is
the next thing to look at. Not pursued.

**Not measured: multi-rank, and any system larger than 1728 atoms.**

## 9. Reproducing

Everything lives in `acejax/bench/apple_scaling/`. `spike_distmd/` is untouched:
`ace_md_cap.py` is a copy of `ace_md.py`'s `run()` with the two capacities lifted
to flags **whose defaults are the hardcoded values**, so the benchmark behind
`bench/molly/RESULTS.md` stays reproducible from the spike itself.

```bash
cd acejax/bench/apple_scaling
LJAX=/path/to/lammps-jax/python          # the `main` branch: `subgraph_params`
                                         # does not exist on dev/julia_export
export PYTHONPATH=../../spike_distmd:$LJAX
export XLA_FLAGS=--xla_force_host_platform_device_count=1
PY=../../.venv/bin/python

# the unmodified spike, and the same thing with both changes
$PY ace_md_cap.py --model ../../fixtures/si_fitted.npz --rep 6 \
    --ranks 1 --dt 0.25 --outer 7 --inner 10 --quiet
$PY ace_md_cap.py --model ../../fixtures/si_fitted.npz --rep 6 \
    --ranks 1 --dt 0.25 --outer 7 --inner 10 --quiet \
    --max-edges 136458 --gather-fix

# the sweeps, as run (point files in points/, raw results in logs/)
$PY sweep.py A logs/A.jsonl points/pts_A.txt          # capacity x size
$PY sweep.py B logs/B.jsonl points/pts_B.txt          # buffer length at fixed N
$PY sweep.py C logs/C.jsonl points/pts_C.txt          # node buffer
ACEMD_MEMORY_ANALYSIS=1 $PY sweep.py D logs/D.jsonl points/pts_D.txt   # fine, f64
ACEMD_MEMORY_ANALYSIS=1 $PY sweep.py E logs/E.jsonl points/pts_E.txt   # fine, f32
$PY sweep.py F logs/F.jsonl points/pts_F.txt          # the 2x2 of section 6
$PY analyse.py logs/*.jsonl

# the attribution, none of which needs the MD driver
$PY ace_stage_scan.py --model ../../fixtures/si_fitted.npz \
    --edges 17057 31013 62027 124054 248112
$PY gather_probe.py --edges 17057 31013 62027 124054 248112 496224
$PY gather_mechanism.py
$PY micro_kernel.py --n-nodes 216 --k 43 --n-real 15218 \
    --edges 17057 31013 62027 124054 248112 496224
```

**Host**: Apple M3 Pro, 6 performance + 6 efficiency cores, macOS 25.6.0, jax
0.10.1 (and 0.11.1 for the cross-check of §5), f64 unless stated. Single rank,
single-threaded XLA. macOS has no `taskset`; nothing was pinned. Runs were kept
short and serial because this is an interactive machine.
