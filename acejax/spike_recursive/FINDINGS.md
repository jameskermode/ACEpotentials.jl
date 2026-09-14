# Spike: does a recursive/DAG AA evaluator pay off in JAX on GPU?

**THROWAWAY.** One-day timeboxed spike. Everything in this directory is
evidence, not product. Nothing here is imported by `acejax/`.

## Question

At a realistic model shape (order 4, lmax 5, n_B = 2849) the AA products are the
largest single stage of `site_basis` — 37.6% by the per-stage breakdown in
`bench/results.md`. The flat evaluator takes independent products over gathered
index rows, so basis functions sharing subproducts recompute them; at order 4
the rank-3 and rank-4 blocks hold 7134 and 16292 terms. `SparseSymmProdDAG`
(EquivariantTensors) and `ace_recursive.cpp` (LAMMPS) both share subproducts
through a DAG. Does that pay in JAX on GPU?

## Answer

**Not worth implementing — with one qualification that stops it being a flat
no.**

At the configuration that actually matters — the production basis (n_B = 2849,
order 4, lmax 5), 1728 atoms, **f64**, which is the like-for-like comparison
against `pace` — the DAG **does not help. Energy+forces come out at 0.99x,
repeatable to 0.1% across two passes.** The forward pass does exactly what the
theory promises, 1.24x on `site_basis` against an absolute floor of 1.22x, and
the backward pass hands all of it back.

Three things behind that, any one of which would weaken the case on its own:

1. **The prize is about half what the brief assumed.** Measured as a difference
   against a build with the AA products deleted, AA is **18.3% of `site_basis`,
   not 37.6%** — the 37.6% is an isolated-stage number and it over-counts, in
   exactly the way this project's own notes warn isolated timings do. The
   ceiling for *any* AA scheme at the headline configuration is 1.22x on
   `site_basis` and **1.34x on energy+forces**. Even perfection would not close
   a 2.6x gap to `pace`.
2. **The DAG's cost is the buffer, and the buffer is worst exactly where it
   matters.** Sharing subproducts means materialising ~25000 intermediate
   columns and keeping them live for the reverse pass — 349 MB in f64 at 1728
   atoms — where the flat form scatters cotangents into 238 columns. The sign of
   the result tracks that buffer size directly: **f64/1728 atoms (349 MB) loses
   at 0.99x; f32/1728 atoms (175 MB) wins at 1.19x; f64/216 atoms (44 MB) wins
   at 1.12x.** Same DAG, same arithmetic saving, opposite sign.
3. **The obvious fix to the backward made it 4x worse**, not better (0.24x).

**The qualification.** In **f32 at the production basis the DAG is a real,
repeatable 1.19x on energy+forces** (1.182x and 1.192x on the two passes,
agreeing to 0.3%), against a ceiling of 1.58x. That is not nothing: the f32 gap
to `pace` at this basis is 1.5x, and this would take it to about 1.26x. But it
is a gain in the precision `pace` does not offer, it would have to be switched
on by precision and system size (it *regresses* f64 at 1728 atoms, so it cannot
simply be turned on), and it requires the DAG to be threaded through spec
export, serialisation and the LAMMPS bundle. **On a
cost/benefit basis that is not where the next week should go.** The measurement
that should direct that instead is the oracle: at the headline configuration
**everything to do with AA is worth at most 1.34x on E+F**, so the gap to `pace`
is mostly somewhere else.

**Cost of the route, since the brief asked.** The DAG construction — the part
flagged as most likely to eat the day — was **not** expensive: ~90 lines of
numpy, 0.45 s to build at the production shape. The day went on measurement
infrastructure and on the backward pass, not on construction.

## 1. DAG construction is cheap — the day was never at risk there

The brief flagged construction as the thing most likely to eat the day. It did
not: the Julia greedy ports to ~90 lines of numpy and runs in **0.45 s** at the
production shape.

Three constructions were built, because the shape of the DAG turns out to be a
free parameter that matters:

| mode | rule | why |
|---|---|---|
| `julia` | port of `_find_partition`: fewest parts, then lowest node index | what EquivariantTensors does |
| `balanced` | always split a term into two near-equal halves | minimises depth, which is what a levelled GPU evaluator cares about |
| `chain` | `kk = ensure(kk[:-1]) * kk[-1]` | worst depth, but associates exactly as `jnp.prod` does — the exact correctness gate |

At the production shape (`si_l2849`, n_A = 238, n_AA = 24116):

| mode | depth | levels | nodes | aux | buffer cols | multiplies vs flat | index reads vs flat |
|---|---|---|---|---|---|---|---|
| `julia` | 3 | 1438 / 14301 / 9256 | 24995 | 897 | 25233 | 24995 vs 63816 (**2.55x fewer**) | 49990 vs 87932 (1.76x fewer) |
| `balanced` | **2** | 2757 / 23426 | 26183 | 2085 | 26421 | 26183 vs 63816 (2.44x fewer) | 52366 vs 87932 (1.68x fewer) |
| `chain` | 3 | 2111 / 11213 / 16292 | 29616 | 5518 | 29854 | 29616 vs 63816 (2.15x fewer) | 59232 vs 87932 (1.48x fewer) |

Two things worth noting. The **arithmetic saving is real but modest — 2.5x on
multiplies, only 1.76x on gathers**, and the AA stage is gather-bound, not
multiply-bound. And the DAG's value buffer is *larger* than the flat AA output
(25233 vs 24116 columns), because the shared subproducts are extra columns that
must be materialised: the DAG trades arithmetic for storage, which is the
opposite of the trade a memory-bound stage wants.

`balanced` reaching depth 2 matters: it means the whole AA stage is **two**
vectorised gather-multiplies, which is as XLA-friendly as this can get.

## 2. Correctness

`verify_dag.py`, CPU, f64, all three fixtures, all three modes: **all pass.**

Three checks of increasing strength:

1. **Structural (exact, integer).** Reconstruct the multiset of A-indices each
   DAG node computes and check `projection` maps it onto the flat spec row for
   row. **EXACT for every mode and every model.** This is floating-point-free
   proof that the DAG evaluates the right products.
2. **Bitwise.** `chain` associates left-to-right exactly as `jnp.prod` over a
   gathered row does, so its AA output is **bit-identical to flat** (0 ulp), and
   so is the resulting `B`. That gates the entire index pipeline — node list,
   levels, buffer layout, projection, and the projection folded into the A2B
   column index.
3. **ULP.** `julia` and `balanced` re-associate the products, which *is* the
   point of sharing subproducts, so bitwise equality is unattainable by
   construction. Measured: **max 3 ulp, mean 0.31 ulp** on AA at the production
   shape; forces agree with the flat path to **8.4e-16 relative**, i.e. round-off,
   two orders inside the 1e-13 the port is validated to against Julia.

So "reproduces the flat output exactly" holds in the only two senses available:
exactly in structure, and bitwise for the association-preserving variant.

The hand-written backward in `dagjax.make_cvjp` is checked the same way, against
the flat path's forces: 6.2e-16 relative.

## 3. The oracle: how much is even available

Before comparing schemes, an upper bound. `site_basis_oracle` keeps every shape,
the same pooling and the same A2B contraction, but **replaces the AA products
with a plain gather of the same output width** — zero multiplies, no sharing to
be had. Nothing that computes the AA basis can be faster. So
`flat - oracle` is the entire budget any recursive scheme is competing for, and
it is the number that decides whether this is worth doing at all.

## 4. Directional only: CPU

**CPU, not the answer.** Recorded because it is a genuine signal about where the
DAG's advantage comes from, and because it is the opposite of the GPU result.
si_l2849, 216 atoms, f64, 4 cores:

| | site_basis | E+F |
|---|---|---|
| flat | 53.2 ms | 272.5 ms |
| oracle (AA deleted) | 39.9 ms | 68.5 ms |
| **AA budget** | 13.2 ms (24.9%) | **204.0 ms (74.9%)** |
| best DAG (`balanced`/`dus`) | 46.5 ms (1.14x) | **118.4 ms (2.30x)** |

**On CPU the DAG is a large, unambiguous win — 2.3x on energy+forces.** The AA
products are three quarters of the CPU E+F cost, and sharing subproducts takes a
big bite out of it. This is presumably why `ace_recursive.cpp` exists: LAMMPS's
ML-PACE is a CPU-first code and this is exactly the regime where a DAG pays.

It says nothing about GPU, where the flat form is fused into its surroundings
and the whole cost structure differs. That is measured next.

## 5. End-to-end measurements (GPU) — the answer

RTX A4500, exclusive use. Every run refused to start unless
`nvidia-smi --query-compute-apps` was empty; a 5 s sampler ran alongside each
run and across **295 samples saw only this session's own PIDs** on the device,
never a foreign process; zero aborts. The **whole series was run twice**.

**Repeatability, stated first because it decides what counts as a result.**
Pass-to-pass agreement on `energy+forces` is **≤1% almost everywhere** (worst
4.2%, on the 69-function model where the numbers are tiny). Agreement on
`site_basis` is **much worse — up to 10.5%**, because at 1-11 ms it is a short
kernel sequence and the flat baseline itself moved 2.0-10.3% between passes.
**So E+F is the metric that can carry a conclusion here and `site_basis` alone
cannot**, which is worth saying plainly because the brief nominated `site_basis`
as the decider. Both are reported; only differences comfortably outside those
spreads are called effects.

### The AA budget is roughly half what the isolated timing said

The oracle deletes the AA products (plain gather, same width, zero multiplies)
and keeps everything else. Nothing computing the AA basis can beat it, so
`flat - oracle` is the entire budget. Mean of two passes:

| config | AA budget, site_basis | AA budget, E+F | E+F ceiling |
|---|---|---|---|
| **n_B=2849, 1728 atoms, f64** | **18.3%** | **25.4%** | **1.34x** |
| n_B=2849, 1728 atoms, f32 | 32.4% | 36.9% | 1.58x |
| n_B=710, 1728 atoms, f64 | 18.0% | 9.3% | 1.10x |
| n_B=2849, 216 atoms, f64 | 8.5% | 24.5% | 1.32x |
| n_B=69, 1728 atoms, f64 | 3.1% | 1.0% | 1.01x |

**The 37.6% figure in `bench/results.md` over-counts by about a factor of
two.** Measured as a difference against a build with the products removed —
the only way to ask what a stage costs inside the fused whole — AA is 18.3% of
`site_basis` at the production shape. This is the same over-counting that file
already documents for sphericart, now quantified for AA. The ceiling for *any*
AA scheme at the headline configuration is **1.22x on `site_basis` and 1.34x on
E+F**.

### What the DAG delivers (mean of two passes; `concat` forward, autodiff backward)

| config | best DAG, site_basis | best DAG, E+F | ceiling (E+F) | verdict |
|---|---|---|---|---|
| **n_B=2849, 1728 atoms, f64** | **1.24x** | **0.99x** | 1.34x | **regression** |
| n_B=2849, 1728 atoms, f32 | 1.03x | **1.19x** | 1.58x | real gain |
| n_B=710, 1728 atoms, f64 | 1.13x | 1.02x | 1.10x | within noise |
| n_B=2849, 216 atoms, f64 | 1.07x | **1.12x** | 1.32x | real gain |
| n_B=69, 1728 atoms, f64 | 1.09x | 1.00x | 1.01x | nothing to win |

Full per-variant numbers with both passes are in `results_compare.txt`;
the filtered run logs are `results_gpu_pass{1,2}.txt`.

**Forward: the levelled hypothesis in the brief is correct.** At the production
shape the best variant reaches 9.01 ms against a flat 11.20 ms and an
unreachable floor of 9.15 ms — it has taken essentially the *whole* AA budget
out of `site_basis`, 1.24x. Two or three vectorised gather-multiplies are indeed
XLA-friendly; the DAG is not hostile to XLA.

**Backward: it gives all of that back at the headline configuration.** E+F comes
out at 0.99x (0.981/0.989 across passes — repeatable to 0.1%). The forward saves
~2.2 ms and the backward loses about the same.

**But the sign flips with the size of the materialised buffer, and that is the
real finding.** The DAG's benefit is arithmetic (2.5x fewer multiplies); its
cost is that ~25000 intermediate columns must be materialised *and kept live for
the reverse pass*, where the flat form scatters cotangents into a 238-column
array. That cost scales with `n_atoms x n_total x sizeof(dtype)`:

- 1728 atoms, f64 → 349 MB buffer → **0.99x (loses)**
- 1728 atoms, f32 → 175 MB buffer → **1.19x (wins)**
- 216 atoms, f64 → 44 MB buffer → **1.12x (wins)**

Same DAG, same arithmetic saving, opposite sign. The DAG trades arithmetic for
materialised intermediates, and whether that is a good trade depends on whether
the buffer fits the memory system — which at the production size in f64 it does
not.

**The obvious backward fix made things four times worse.** `cvjp` is a
hand-written reverse-level backward with scatter targets pre-sorted at build
time — what a production implementation would reach for. Its forward is the
fastest of all and its gradients are correct (1.3e-15 against the flat path),
but E+F is 197 ms against 46 ms (repeatable to 0.0%). Writing the backward by
hand removes XLA's freedom to fuse it into the surrounding computation, and a
full-width scatter-add per level costs far more than it saves. That is one naive
custom VJP, not a proof that no good one exists — but it does say the backward
is the hard part, and it is where any further effort would have to go.

### Discarded measurements

An earlier GPU run was taken before it was established that another agent was
sharing the device; it is discarded on principle rather than used. For the
record it agreed with the clean numbers to within 1-6%, and the contention ran
the other way — this session's 15.2 GB JAX preallocation is what OOM'd the other
agent, not the reverse.

## 6. Method notes

- Timings are **end-to-end**, never isolated stages: `site_basis` whole, and
  `energy + forces` through `jax.value_and_grad`. This project has twice been
  misled by isolated stage timings (`bench/results.md`, sphericart section:
  an isolated harmonic call measured *slower than the whole computation
  containing it*). An isolated AA number is reported only as a footnote.
- `model.site_basis` is timed alongside the spike's own flat path in every run,
  so the baseline is demonstrably the shipped one and not a strawman.
- The shape is order 4, lmax 5 throughout — the realistic shape from the
  CORRECTION in `bench/results.md`, not the high-lmax/low-order shape that
  produced the retracted 51.7%-angular conclusion.
- moriarty was shared with another agent's GPU work for part of the day. Every
  timing run records `nvidia-smi` occupancy immediately before and after
  (`run_gpu.sh`), and any run that overlapped another process is discarded.
