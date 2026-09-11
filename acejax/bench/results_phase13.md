# Phase 13 — ACE against MACE, three routes

Host: **moriarty** — Xeon Silver 4216 (AVX-512), RTX A4500 (compute 8.6, 20 GB).
Si diamond supercells, **1 MPI rank, 1 GPU, 1 OpenMP thread**,
`-k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware off`. Scripts and
raw data in `phase13/`; the sweep output every table below is built from is
`phase13/results/phase13_sweep.txt`.

| # | route | pair style | isolates |
|---|---|---|---|
| 1 | our ACE model | `jax/kk` | — (the Phase 8 baseline) |
| 2 | MACE via `symmetrix` | `symmetrix/mace[/float32]/kk` | MACE at its best, hand-written Kokkos |
| 3 | MACE via `lammps-jax` | `jax/kk` | the model, with plumbing held constant |

## Headline

At 1728 atoms, f32 on both sides, all three routes in LAMMPS:

- **1 vs 3 — ACE against MACE with the plumbing held constant.** Our
  69-function ACE model is **16.4x** faster than MACE-MP-0 small through the
  same pair style. Our 710-function model is **3.1x** faster. Our
  production-scale 2849-function model is **0.60x** — that is, **MACE-MP-0
  small is 1.7x FASTER than our largest ACE model**. Against the largest MACE
  measured (MACE-MP-0b3 medium) the same three ACE models run 38.0x, 7.2x and
  1.4x.
- **2 vs 3 — the same MACE model through two engines.** `pair_style jax/kk`
  is **1.43x faster** than hand-written Kokkos `symmetrix` on MACE-MP-0b3
  medium and **1.04x** on MACE-MP-0b2 small, both f32. Against symmetrix's
  f64 default the same two figures are 1.74x and 1.28x.

**The second result is the one that changes a Phase 8 conclusion, and it goes
against us.** Phase 8 characterised the `jax/kk` path as costing an
"integration tax" — retention of only 0.34–0.63 of raw model throughput. That
is still true of our own ceiling, but it evidently is not enough to make the
path uncompetitive: on somebody else's model, run through both engines, the
JAX-exported program is the *faster* of the two. See
"What this retracts, and what it does not" below.

**The first result deserves to be read the other way round from the one that
flatters us.** The 2849-function ACE model carries **5,135 learnable
parameters**; MACE-MP-0 small carries **3,847,696** — a **750x** parameter gap
for a 1.7x throughput deficit, and MACE is a universal model that was never
fitted to Si while ours is Si-only. The useful number here is not the ratio but
its size: whatever generality costs, at production ACE basis size it is not
costing MACE throughput on this hardware.

## The models, in enough detail to interpret the gap

Measured with `phase13/facts.py` from each torch checkpoint (route 2 and route
3 both start from these files), and from the `.npz` fixtures for ours.

### MACE

| | MACE-MP-0 small | MACE-MP-0b2 small | MACE-MP-0b3 medium |
|---|---|---|---|
| checkpoint | `2023-12-10-mace-128-L0_energy_epoch-249` | `mace-small-density-agnesi-stress` | `mace-mp-0b3-medium` |
| **parameters** | **3,847,696** | **8,221,984** | **9,063,204** |
| **cutoff** | **6.0 Å** | **5.0 Å** | **6.0 Å** |
| **message-passing layers** | **2** | **2** | **2** |
| **channel width** | **128** | **128** | **128** |
| hidden irreps | 128x0e | 128x0e | 128x0e+128x1o |
| max_ell | 3 | 3 | 3 |
| correlation | 3 | 3 | 3 |
| species in the model | 89 | 89 | 89 |
| first / second block | Residual / Residual | Density / DensityResidual | Density / DensityResidual |
| loadable by `symmetrix` | **no** (see below) | yes | yes |
| loadable by `lammps-jax` | yes | yes | yes |

### Ours

| | n_B = 69 | n_B = 710 | n_B = 2849 |
|---|---|---|---|
| **basis functions** | **69** | **710** | **2849** |
| **cutoff** | **6.0 Å** | **6.0 Å** | **6.0 Å** |
| correlation order | 4 | 4 | 4 |
| lmax | 4 | 5 | 5 |
| learnable parameters | 365 | 1,984 | 5,135 |
| AA terms by rank | 8/39/37/21 | 14/270/1191/1481 | 18/672/7134/16292 |
| species | 1 (Si) | 1 (Si) | 1 (Si) |
| message-passing layers | **0** (one-shot basis contraction) | 0 | 0 |

Learnable parameters = `WB` + `Wpair` + `rnl_Wnlq`. These are the same three
fixtures Phase 8's realistic-shape series used.

## `symmetrix` cannot load MACE-MP-0, so 2-vs-3 uses different models than the plan assumed

The plan named MACE-MP-0 small for both routes. It cannot be: `symmetrix`'s
extractor refuses it.

```
$ venv-p13/bin/python extract_sym.py checkpoints/mace-mp-0-small.model out.json
REJECTED RuntimeError Currently, symmetrix only supports MACE models whose
first interaction is RealAgnosticInteractionBlock or RealAgnosticDensityInteractionBlock.
```

MACE-MP-0 small and medium both use `RealAgnosticResidualInteractionBlock` as
their *first* interaction, and that is not a subclass of either accepted class
(checked: `issubclass(...) is False`). `symmetrix`'s own test suite bears this
out — its cached models are MACE-OFF23-small and **mace-mp-0b3-medium**, not
MACE-MP-0.

**So the shared models are MACE-MP-0b2 small and MACE-MP-0b3 medium**, which
use the Density blocks `symmetrix` accepts and which `mace-jax` also converts.
Both routes load the *same torch checkpoint*, so 2-vs-3 is a genuine
same-model comparison — just not of the model the plan named. MACE-MP-0 small
is carried as a route-3-only point, because it is the model `lammps-jax`'s
exporter is written and tested around, and because it is the cheapest MACE
here and therefore the toughest 1-vs-3 comparison for us.

**One asymmetry, stated rather than buried.** `extract_sym.py` is invoked with
`-Z 14`, which specialises the model to Si and shrinks the species embedding
from 89 to 1; the `lammps-jax` export keeps all 89 and maps LAMMPS types into
them. That favours route 2 by one 89x128 matvec per atom per layer. It is small
against 9M parameters of tensor products, but it is not zero and it is not in
our favour.

## Route 3 runs in `comm` mode because `ghost` mode does not fit on the card

`lammps-jax`'s MACE exporter offers two ways to serve a 2-layer receptive field:

- `--mode ghost` — no in-model exchange; the contract carries
  `n_hops = num_interactions = 2`, so `pair_jax_kokkos` extends the ghost shell
  to `n_hops*cutoff + skin` (13 Å) and sets `num_rows = inum + gnum`. At 216
  atoms in a 16.3 Å box that is ~17.5x the local atom count.
- `--mode comm` — one-cutoff ghost shell plus in-model feature exchange
  (`comm_widths [512]` for the L=1 model), with `--owned-rows` bounding the
  product basis to owned rows.

**Ghost mode OOMs at the smallest size that matters:**

```
ERROR on proc 0: LAMMPS-JAX compute failed: PJRT_Event_Await failed:
Out of memory while trying to allocate 12.13GiB with allocator GPU_0_bfc
```

— 216 atoms, MACE-MP-0b3 medium, `max_atoms` 4725, `max_edges` 239202. Every
route-3 number below is therefore `comm` mode. That is not a handicap
(`comm` is the cheaper of the two by a wide margin) but it *is* the only
configuration that runs, and a reader should know the alternative was tried.

**Route 3 is f32 only.** `examples/export_mace.py`'s `run_export` calls
`load_model_bundle(bundle_dir, "float32")` unconditionally. The like-for-like
comparison against route 2 is therefore f32-vs-f32, and
`pair_style symmetrix/mace/float32/kk` provides it; symmetrix's f64 default is
reported alongside rather than compared against.

## Route 2 was measured at symmetrix's best, not at a mode that suits us

`pair_symmetrix` offers three modes and picks `no_domain_decomposition` itself
on one rank. All three were measured, 1728 atoms, MACE-MP-0b3 medium:

| mode | f64 ms/step | f32 ms/step |
|---|---|---|
| **no_domain_decomposition** (its 1-rank default) | **182.6** | **149.8** |
| mpi_message_passing | 189.7 | 156.0 |
| no_mpi_message_passing | 240.7 | 197.8 |

Its default is its fastest, by 4% over `mpi_message_passing` and 32% over
`no_mpi_message_passing`. Energies are identical to 9 significant figures
across all six. The sweep uses `no_domain_decomposition` throughout.

## The gate: every engine agrees before any timing is quoted

216-atom Si diamond, identical geometry in every run (`max|dx| = 0.000e+00`
between dumps confirms it). Tolerances are `lammps-jax`'s own exporter
preflight thresholds: **1e-4 eV/atom and 5e-3 eV/Å**.

| comparison | energy | forces | verdict |
|---|---|---|---|
| route 1 ACE `jax/kk` f64 vs `acejax` in Python | 1.89e-11 eV (5.3e-14 rel) | **1.03e-13 eV/Å** (1.3e-14 rel) | PASS |
| route 2 symmetrix f64 vs torch MACE f64 | 3.18e-7 eV/atom | 4.28e-5 eV/Å | PASS |
| route 3 `jax/kk` f32 vs torch MACE f64 | 1.55e-7 eV/atom | **1.40e-5 eV/Å** | PASS |
| **2 vs 3: symmetrix f32 vs `jax/kk` f32, same checkpoint** | — | **4.15e-5 eV/Å** | PASS |
| symmetrix f32 vs symmetrix f64 | — | 2.76e-6 eV/Å | PASS |

Force scale is 1.134 eV/Å for the MACE rows and 8.124 eV/Å for the ACE row, so
the MACE agreements are ~3.7e-5 relative and the ACE one 1.3e-14 relative.
Independently, `export_mace.py`'s own preflight reported `dE/atom = +0.00e+00`,
`max|dF| = 9.54e-7 eV/Å` for the adapter against `mace_jax` directly.

Worth noting which way the residuals fall: **route 3 is closer to torch MACE
(1.4e-5) than route 2 is (4.3e-5)**. `symmetrix` splines its radial basis onto
256 points; the JAX export evaluates it. Neither matters at these tolerances,
but it rules out any suggestion that the faster route is the sloppier one.

**One retraction on the gate itself.** The ACE row first read
`max|dF| = 5.04e-9 eV/Å` and *failed* `check_vs_python.py`'s 1e-9 threshold.
That was the dump, not the model: the deck wrote `%.12g` positions, so the
Python side was evaluating a slightly different geometry. Re-dumping at
`%.17g` gave 1.03e-13, in line with Phase 6's 1.37e-13. The deck now writes
`%.17g`. The 5.04e-9 figure was never used for anything, but it is the kind of
number that would have been quoted as a real disagreement.

## Method, and how it differs from Phase 8

Same Si diamond supercells (64 to 1728 atoms, the sizes the realistic-shape
Phase 8 series used), same `timestep 0.0` single-point method, same deck
lineage. Two deliberate differences:

1. **The timed run is the second run** (`run 3` then `run 50`), where
   `in.si_bench` times a single `run` with the one-off XLA compile inside it.
   MACE-MP-0b3 medium compiles for ~30 s; leaving that in would have measured
   the compiler. **Route 1 was therefore re-run under this deck rather than
   quoted from the Phase 8 table.**
2. **`atom_modify map yes`** is set for every style, because
   `symmetrix/mace/kk` requires a global-to-local map. Setting it for all three
   keeps the decks identical.

**Neither change moves route 1.** The re-run reproduces the Phase 8
realistic-shape numbers at 1728 atoms to within 1.2% in f64 and 4.8% in f32:

| n_B | Phase 8 f64 | here f64 | Phase 8 f32 | here f32 |
|---|---|---|---|---|
| 69 | 3.15e5 | 3.188e5 | 6.47e5 | 6.16e5 |
| 710 | 6.45e4 | 6.456e4 | 1.17e5 | 1.17e5 |
| 2849 | 1.27e4 | 1.274e4 | 2.24e4 | 2.24e4 |

**The Phase 8 exclusions carry over unchanged**, plus one that is now stronger:

1. With atoms static, `check yes` never triggers a rebuild, so these numbers
   **exclude reneighbouring cost**, which real MD pays periodically.
2. **Compilation is fully outside the measurement** here, not merely a small
   share of it — the warm run pays for it.
3. Capacities are sized per point (`export_all.py`), so no route is measured at
   an over-provisioned bundle. Phase 8 established that measuring at 16x
   capacity measures padding, not the model.

**Both LAMMPS binaries are `patch_4Jul2026`**, built with the same flags. The
symmetrix build is a *new* directory
(`phase13/lammps/build-SKX-AMPERE86-symmetrix`) from a fresh clone at that tag;
the Phase 6 and Phase 8 builds are untouched. `PKG_PLUGIN` is on, so
`pair_style jax/kk` loads into the same binary and **all three routes ran on one
`lmp`**. Cross-checked: the same ACE bundle gives PotEng `-358.884244998` on
both the new and the Phase 8 binary — bit-identical — and 4.84 vs 5.13 ms/step.

**Shared-library shadowing was checked before timing**, as the Phase 8 findings
demand: `ldd` puts `liblammps.so.0` in the new build directory, and `lmp -h`
lists `symmetrix/mace`, `symmetrix/mace/kk` and `symmetrix/mace/float32/kk`.

**GPU exclusivity.** `sweep.sh` refuses to start if any compute app is
resident, and samples `nvidia-smi --query-compute-apps` every 3 s throughout.
No sample in the 65-run sweep shows two resident processes. `jax` ran with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, so the card was never preallocated.

## The three-way table

Throughput in **atom-steps/s**, 1 rank, 1 GPU. Blank = that precision does not
exist for that route (route 3 exports f32 only; route 1 has no MACE model).

### 1728 atoms

| | route | precision | atom-steps/s | ms/step |
|---|---|---|---|---|
| **ACE n_B = 69** | 1 `jax/kk` | f64 | 3.188e5 | 5.42 |
| | | f32 | **6.16e5** | 2.81 |
| **ACE n_B = 710** | 1 `jax/kk` | f64 | 6.456e4 | 26.77 |
| | | f32 | **1.17e5** | 14.77 |
| **ACE n_B = 2849** | 1 `jax/kk` | f64 | 1.274e4 | 135.62 |
| | | f32 | **2.24e4** | 77.15 |
| **MACE-MP-0 small** | 3 `jax/kk` | f32 | **3.755e4** | 46.02 |
| **MACE-MP-0b2 small** | 3 `jax/kk` | f32 | **4.734e4** | 36.50 |
| | 2 `symmetrix` | f32 | 4.562e4 | 37.88 |
| | 2 `symmetrix` | f64 | 3.699e4 | 46.72 |
| **MACE-MP-0b3 medium** | 3 `jax/kk` | f32 | **1.621e4** | 106.59 |
| | 2 `symmetrix` | f32 | 1.132e4 | 152.59 |
| | 2 `symmetrix` | f64 | 9.296e3 | 185.88 |

### Across sizes, atom-steps/s

| model | route | prec | 64 | 216 | 512 | 1000 | 1728 |
|---|---|---|---|---|---|---|---|
| ACE n_B=69 | 1 | f64 | 7.124e4 | 5.424e4 | 2.058e5 | 2.563e5 | 3.188e5 |
| ACE n_B=69 | 1 | f32 | 1.165e5 | 3.207e5 | 4.839e5 | 5.470e5 | 6.160e5 |
| ACE n_B=710 | 1 | f64 | 2.558e4 | 2.269e4 | 6.172e4 | 6.351e4 | 6.456e4 |
| ACE n_B=710 | 1 | f32 | 4.476e4 | 7.750e4 | 1.029e5 | 1.061e5 | 1.170e5 |
| ACE n_B=2849 | 1 | f64 | 7.007e3 | 9.634e3 | 1.595e4 | 1.568e4 | 1.274e4 |
| ACE n_B=2849 | 1 | f32 | 9.155e3 | 1.587e4 | 2.209e4 | 2.399e4 | 2.240e4 |
| MACE-MP-0 small | 3 | f32 | 1.868e4 | 2.877e4 | 3.653e4 | 3.453e4 | 3.755e4 |
| MACE-0b2 small | 3 | f32 | 1.630e4 | 3.008e4 | 4.006e4 | 4.678e4 | 4.734e4 |
| MACE-0b2 small | 2 | f32 | 1.462e4 | 3.276e4 | 3.818e4 | 4.364e4 | 4.562e4 |
| MACE-0b2 small | 2 | f64 | 1.342e4 | 2.770e4 | 3.250e4 | 3.589e4 | 3.699e4 |
| MACE-0b3 medium | 3 | f32 | 8.180e3 | 1.256e4 | 1.448e4 | 1.553e4 | 1.621e4 |
| MACE-0b3 medium | 2 | f32 | 5.627e3 | 1.014e4 | 1.059e4 | 1.131e4 | 1.132e4 |
| MACE-0b3 medium | 2 | f64 | 5.194e3 | 8.633e3 | 8.816e3 | 9.230e3 | 9.296e3 |

The f64 dip at 216 atoms that Phase 8 documented for `pair jax/kk` is visible
again in the ACE rows (n_B=69 falls from 7.12e4 at 64 atoms to 5.42e4 at 216)
and is **absent from every MACE row**, including the route-3 MACE rows through
the same pair style — consistent with Phase 8's reading of it as a bundle-shape
effect rather than anything in the model.

## Isolation 1 vs 3 — ACE against MACE, plumbing held constant

Both sides `pair_style jax/kk`, both f32, same deck, same structures, same
capacity-sizing rule. **The ratio is ACE throughput ÷ MACE throughput: above 1
means ACE is faster.** 1728 atoms.

| | vs MACE-MP-0 small (3.85M) | vs MACE-0b2 small (8.22M) | vs MACE-0b3 medium (9.06M) |
|---|---|---|---|
| ACE n_B = 69 (365 params) | **16.4x** | 13.0x | 38.0x |
| ACE n_B = 710 (1,984 params) | **3.12x** | 2.47x | 7.22x |
| ACE n_B = 2849 (5,135 params) | **0.60x** | 0.47x | 1.38x |

**ACE does not win everywhere, and the place it loses is the one that matters
most.** At the production basis size Phase 8 went to some trouble to construct
— 2849 functions, order 4, lmax 5, larger than the 742-function `Cu-PBE-core-rep`
exemplar LAMMPS ships — a universal MACE foundation model with 750x the
parameters is **1.7x faster** on the same GPU through the same pair style. Only
against the largest MACE does our largest ACE still lead, and then by 1.38x.

The honest summary is that **ACE's throughput advantage is an advantage of
small models, not of the architecture**: it is 16–38x at 69 functions, 2.5–7x at
710, and gone by 2849. Anyone choosing between them at production accuracy
should expect the throughput argument to be close to neutral on this hardware,
and should decide on fit quality, data requirements and generality instead.

Two things this comparison is *not*:

- **Not matched complexity.** MACE is two rounds of message passing with 128
  channels and equivariant tensor products; ACE is a one-shot basis
  contraction. There is no shared axis along which to equalise them, which is
  why the plan set this phase up as a throughput question rather than a
  cost-at-matched-size question the way Phase 8 did against `pace`.
- **Not a statement about accuracy.** Nothing here was fitted or validated
  against reference data. The n_B=710 and 2849 ACE models carry unfitted
  weights (coefficients do not affect cost); the MACE models are fitted but to
  MPtrj, not to anything Si-specific.

## Isolation 2 vs 3 — one MACE model, two engines

Same torch checkpoint, same LAMMPS binary, same Kokkos settings, same deck,
each engine in its fastest available single-rank configuration. **Ratio is
`jax/kk` ÷ `symmetrix`: above 1 means the JAX-exported program wins.**

### f32 against f32, the like-for-like comparison

| atoms | MACE-0b2 small | MACE-0b3 medium |
|---|---|---|
| 64 | 1.11x | 1.45x |
| 216 | 0.92x | 1.24x |
| 512 | 1.05x | 1.37x |
| 1000 | 1.07x | 1.37x |
| **1728** | **1.04x** | **1.43x** |

### against symmetrix's f64 default, at 1728 atoms

| | `jax/kk` f32 ÷ `symmetrix` f64 |
|---|---|
| MACE-0b2 small | 1.28x |
| MACE-0b3 medium | 1.74x |

**On the L=0 model the two engines are within 10% of each other at every size;
on the L=1 model `jax/kk` wins by 1.24–1.45x throughout.** The gap being larger
for the equivariant model is the one piece of mechanism the data suggests — L=1
hidden features mean wider tensor products and a 512-wide feature exchange,
which is where an XLA-fused program has more to gain over hand-written kernels
— but that is a two-model observation, not an established trend.

### What this retracts, and what it does not

Phase 8 measured the `jax/kk` path retaining only **0.34–0.63** of raw
acejax-in-Python throughput and attributed that, at large basis, entirely to
atom-axis padding. **Those measurements stand.** What does not stand is the
gloss that has accumulated around them — that the plugin path is a tax which
puts `jax/kk` at a structural disadvantage to a hand-written implementation.
Run against `symmetrix`, a mature Kokkos MACE that the MACE foundation-model
paper itself cites for performance, the same plugin path with the same padding
and ghost overheads is **the faster of the two on an identical model**.

The correct reading of Phase 8's retention figures is therefore that they
bound *our own* headroom — what we would gain by evaluating site energies on
local rows only, or by tightening `max_atoms` — and not that the integration
route is uncompetitive. Those two things were not distinguished before, and the
weaker one was the one being implied.

**One caveat that keeps this from being a pure plumbing measurement.** The two
engines serve MACE's 2-layer receptive field differently: route 3 uses
in-model feature exchange with an owned-row bound, route 2 uses
`no_domain_decomposition`. Both are each engine's fastest available single-rank
configuration, measured rather than assumed (route 2's three modes are tabulated
above; route 3's alternative OOMs), but they are not the *same* strategy. The
comparison is "each engine at its best on one rank", which is the question a
user has, rather than "the same algorithm through two code paths".

## The incidental finding the plan asked to note

`symmetrix` ships **both CPU (OpenMP/Serial) and GPU (CUDA) Kokkos paths** for
its pair style — `symmetrix/mace/kk/host` and `symmetrix/mace/kk/device` are
both registered, from one templated implementation. That is direct evidence
that a Kokkos ML pair style can support CPU, which reinforces Phase 12's point
that `pair_style jax/kk` being CUDA-only is a code-structure choice rather than
an inherent constraint. Worth citing if the CPU backend is raised with the
`lammps-jax` maintainer.

## Not measured, and why

- **4096 atoms.** The realistic-shape Phase 8 series stops at 1728 and this one
  matches it. Not attempted; not estimated.
- **Route 3 in f64.** The exporter hardcodes `float32`. Patching it was out of
  scope; the f32-vs-f32 comparison is the like-for-like one and it exists.
- **MACE-MP-0 small through `symmetrix`.** Refused by its extractor (above).
  It appears as a route-3-only point and **no 2-vs-3 ratio is reported for it**.
- **MACE-OFF23**, `symmetrix`'s other tested model: organic-molecule only, no
  Si, so out of scope for this structure set.
- **Any decomposition of where route 2 or route 3 spends its time.** No isolated
  stage timing appears anywhere in this file. Every ratio here is a
  whole-computation difference, per the rule Phase 8 arrived at the hard way.
- **Multi-rank.** One rank throughout. `symmetrix`'s `no_domain_decomposition`
  is single-rank by construction, so a multi-rank comparison would be a
  different experiment with different modes on both sides.
- **Run-to-run scatter.** Each point is a single timed run of 50 steps after a
  3-step warm-up. The route-1 rows agreeing with Phase 8's independent
  measurements to 0.1–1.2% in f64 is the only repeatability evidence offered,
  and it is evidence for the ACE rows only.

## Reproducing

```bash
# on moriarty (NOT lestrade -- the build is AVX-512)
cd /storage/eng/essswb/phase13
./build_sym.sh                                        # LAMMPS + pair_symmetrix, ~35 min

# route 2: torch checkpoint -> symmetrix json, Si only
venv-p13/bin/python extract_sym.py checkpoints/mace-mp-0b3-medium.model \
                                   checkpoints/mace-mp-0b3-medium-Si.json

# route 3: torch checkpoint -> mace-jax bundle -> lammps-jax bundles
PYTHONPATH=~/si-ace/lammps-jax/python JAX_PLATFORMS=cpu venv-p13/bin/python \
  ~/si-ace/lammps-jax/examples/export_mace.py convert \
  checkpoints/mace-mp-0b3-medium.model mace-mp-0b3-medium-jax
PYTHONPATH=~/si-ace/lammps-jax/python JAX_PLATFORMS=cpu venv-p13/bin/python \
  scripts/export_all.py m0b3med mace-mp-0b3-medium-jax comm 2 3 4 5 6

# the gate, then the sweep
STEPS=5 WARM=1 ./scripts/runone.sh 3 sym64 $PWD/checkpoints/mace-mp-0b3-medium-Si.json \
      -var dodump 1 -var dumpfile $PWD/verify/m0b3med_s64.dump
venv-p13/bin/python scripts/mace_reference.py verify/m0b3med_s64.dump \
      checkpoints/mace-mp-0b3-medium.model --pe <PotEng>
STEPS=50 WARM=3 ./scripts/sweep.sh                    # ~50 min; refuses a busy GPU
```

`lammps-jax` was checked against upstream before building: the local checkout
`a4304a2` ("fp64 support") **is** `origin/main`, so the MACE exporter used here
is current.

Scripts are committed under `phase13/`; the raw sweep is
`phase13/results/phase13_sweep.txt`.
