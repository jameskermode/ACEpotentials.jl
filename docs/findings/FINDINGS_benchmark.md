# Phase 8: ML-PACE rebuild, and a throughput benchmark that is partly blocked

Host: **moriarty** — Xeon Silver 4216 (AVX-512), RTX A4500 (compute 8.6).
The reference chart is an RTX 4070 Laptop at 70 W, so absolute numbers do not
transfer; shape and same-host ratios are what carry.

Numbers in `results.md`.

## Part A — the ML-PACE rebuild: done, and verified

Built into **`lammps/build-SKX-AMPERE86-mlpace`**, a new directory. The working
`build-SKX-AMPERE86` is untouched, and the new build reproduces Phase 6 exactly:

```
PotEng  -35238.76853368402   (identical to the committed run)
forces  max|dF| = 8.299e-14 eV/A vs the committed dump
```

`pace` and `pace/kk` are both present. Nothing shifted.

**One trap worth recording.** `BUILD_SHARED_LIBS=ON` means `lmp` is a ~1 MB
driver and every style lives in `liblammps.so.0`. The documented run recipe puts
`$V/lib` first on `LD_LIBRARY_PATH`, so a *new* `lmp` silently loads the *old*
library: `lmp -h` reported no `pace` styles and `pair_style pace` errored with
"part of the ML-PACE package which is not enabled in this LAMMPS binary", while
CMake had plainly said `Enabled packages: KOKKOS;MANYBODY;ML-PACE;PLUGIN`.
The new build directory must precede `$V/lib`. Left unnoticed this would have
benchmarked the old binary.

## Part B — the `pace` comparator is blocked, one layer deeper than expected

The v0.6 route works up to the point of handing the file to LAMMPS:

| step | result |
|---|---|
| ACEpotentials **0.6.12** resolves (ACE1 0.12.5, JuLIP 0.16.0, Julia 1.11) | OK |
| fits `Si_tiny_dataset`, same data as our v0.10 model | OK |
| `export2lammps` -> `si_v06.yace`, 9.6 MB | OK |
| loads under `pair_style pace/kk` | **blocked** |

`export2lammps` needed `Eref` — without it the potential is a 2-component
`SumIP` (`PolyPairPot`, `PIPotential`) and the exporter demands three
(plus `OneBody`). `Eref = [:Si => 0.0]` matches our v0.10 model, whose
`Vref.E0[Si]` is also 0.

### Basis sizes side by side

| | v0.6.12 | v0.10.2 |
|---|---|---|
| `length(model.basis)` / `n_B` | **110** | **110** |
| pair basis | included in the 110 | 10 (separate) |
| correlation order | 3 | 3 |
| rcut | 6.0 | 6.0 |
| elements | 1 (Si) | 1 (Si) |

At identical hyperparameters the two versions give the **same many-body basis
size**, which is the reassuring answer. The exported yace reports 210 ctilde
functions and `lmax: 2`; those are pacemaker's own counting over ms-combinations
and its radial-angular block, not comparable term-for-term with `n_B`.

### Why it will not load

The v0.6 exporter emits a radial basis as **tabulated spline nodal values**:

```yaml
bonds:
  [0, 0]:
    radbasename: "ACE.jl"
    nradial: 17
    nbins: 9999
    splinenodalvals: ...
```

Upstream ICAMS `lammps-user-pace` v.2023.11.25 — what LAMMPS's CMake fetches by
default — expects `ChebPow`/`ChebExpCos` with `radcoefficients`, and also
requires `deltaSplineBins` and `nradbasemax`, neither of which the v0.6 exporter
writes. It fails with `Exception: bad conversion`.

The **`wcwitt/lammps-user-pace` fork does carry `acejl_radial.cpp`**, which
handles `splinenodalvals`, so I rebuilt again against it
(`build-SKX-AMPERE86-acejl`, `PACELIB_URL` pointed at the fork tarball). That
gets further and then fails differently:

```
Exception: map::at
```

i.e. the fork parses the file but a required key is missing on lookup — most
likely a version skew between this ACEpotentials vintage's output and the fork's
`main`. Resolving it means finding the fork branch/tag contemporary with
ACEpotentials 0.6.12, which is past the point where this was worth pressing on.

**No contract was loosened and no field was invented to force a pass.**

## The MD abort, and why the method changed

An earlier version of this file reported `jax/kk` aborting with
`cudaErrorIllegalAddress` beyond ~50 MD steps and suspected the reneighbour
repack path. That was wrong about the cause. `compute-sanitizer` put the fault
in LAMMPS's own `NBinKokkos::bin_atoms()` with no `pair_jax_kokkos` frame, and
the root cause is that the `Si_tiny` test potential has **no repulsive core**:
its dimer curve turns over around 1.5 A and diverges attractively, so atoms
collapse into each other. It reproduces in pure ASE NVE with no LAMMPS at all.
See `FINDINGS_lammps.md`.

The benchmark therefore uses `timestep 0.0`: the configuration is frozen and
`run N` becomes N repeated single-point evaluations. That measures the pair
style and the model without touching the instability, at the cost of excluding
reneighbouring, which is stated in `bench/results.md` and in the deck.

## What the numbers do support

- **acejax on GPU, f64: ~2.5-2.8e5 atom-steps/s; f32: ~6.4-9.6e5**, plateauing
  by ~500 atoms.
- **f64 costs only ~2.5x f32**, corroborating Phase 0's 3-5.4x on different
  hardware. For a memory-bound descriptor the 1/32 FP64 arithmetic rate is not
  what governs.
- Plugin overhead looks like ~25% at 1728 atoms (1.91e5 vs 2.55e5), but with
  compilation still in the loop that is indicative only.

## Sizing capacities per point mattered

Phase 6's bundle has `max_edges` 163840 against 9880 used. Sized per point, the
216-atom bundle needs 13184 — a 12x reduction. Benchmarking at the Phase 6
capacity would have measured padding.

## Reproducing

```bash
# on moriarty; LMP must point at a build whose dir precedes $V/lib on the path
cd stage1/bench
python make_bundles.py --npz ../si_fitted.npz --reps 2 3 4 5 6
STEPS=20 REPS="2 3 4 5 6" ./run_bench.sh jax
python bench_acejax.py --npz ../si_fitted.npz --reps 2 3 4 5 6 [--f32]
# v0.6 comparator (fit + yace export both work; loading does not)
cd v06 && julia +1.11 --project=. fit_v06.jl
```
