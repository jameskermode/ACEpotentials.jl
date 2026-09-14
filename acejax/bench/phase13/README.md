# Phase 13 — ACE against MACE, three routes

Scripts and raw data for `../results_phase13.md`. Everything here is the code
that actually produced those numbers on **moriarty** (Xeon Silver 4216,
RTX A4500, compute 8.6), pulled back verbatim; paths are moriarty paths.

| file | what |
|---|---|
| `build_sym.sh` | LAMMPS `patch_4Jul2026` + `pair_symmetrix`, CUDA Kokkos, into a new build dir. Flags copied exactly from the Phase 8 binary. |
| `extract_sym.py` | MACE torch checkpoint -> symmetrix `.json`, Si only (route 2) |
| `export_all.py` / `export_all.sh` | MACE torch checkpoint -> `lammps-jax` bundles, capacities sized per point (route 3) |
| `facts.py` | parameter count, cutoff, layers, channel width per MACE checkpoint |
| `in.si_phase13` | the deck, all three routes, warm run then timed run |
| `env.sh` | module loads, `LD_LIBRARY_PATH` ordering, Kokkos flags |
| `runone.sh` | one (size, engine) point -> ms/step, atom-steps/s, PotEng |
| `sweep.sh` | the full matrix, with GPU-exclusivity refusal and a contention sampler |
| `mace_reference.py` | independent torch-MACE check against a LAMMPS dump |
| `results/phase13_sweep.txt` | the raw sweep output the tables are built from |

## Reproducing

```bash
# on moriarty (NOT lestrade -- the build is AVX-512)
ROOT=/storage/eng/essswb/phase13
./build_sym.sh                                   # ~35 min
$ROOT/venv-p13/bin/python extract_sym.py $ROOT/checkpoints/<ckpt>.model <out>.json
PYTHONPATH=~/si-ace/lammps-jax/python JAX_PLATFORMS=cpu \
  $ROOT/venv-p13/bin/python export_all.py <tag> <bundle-dir> comm 2 3 4 5 6
STEPS=50 WARM=3 ./sweep.sh                       # ~50 min, refuses a busy GPU
```

The venv `venv-p13` holds torch(cpu) + mace-torch + `jax==0.11.1` + mace-jax +
jraph + flax. The jax pin matters: a bundle must be exported by a jax matching
the PJRT plugin `pair_style jax/kk` loads.

## Two things that have burned this project before

**Shared-library shadowing.** `BUILD_SHARED_LIBS=ON` puts every pair style in
`liblammps.so`, and the documented `LD_LIBRARY_PATH` ordering makes a freshly
built `lmp` load the *old* library. Three false readings so far. `env.sh` puts
`$(dirname $LMP)` first; `ldd` was checked before any timing (see
`../results_phase13.md`).

**Isolated stage timings over-count.** Nothing here is attributed from an
isolated stage timing; every ratio is a whole-computation difference.
