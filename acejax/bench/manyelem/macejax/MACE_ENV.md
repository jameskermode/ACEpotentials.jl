# MACE-JAX on the GPU — environment recipe and verification

Target host: **moriarty** (Xeon Silver 4216, RTX A4500 20 GB, compute 8.6, 62 GB RAM).
Built 2026-09-11. Everything below was run and its output recorded; nothing here
is inferred.

## Headline: the Phase-13 environment was NOT gone

The task was framed as a rebuild because `mace_jax` could not be found in
`~/si-ace/.venv`, `~/lammps-jax/.venv` or `~/mace-exp-precon/.venv`. It is not in
any of those. It is in **`/storage/eng/essswb/phase13/venv-p13`**, which is
intact — the Phase-13 README points at it (`$ROOT/venv-p13` with
`ROOT=/storage/eng/essswb/phase13`), and the whole Phase-13 tree is still there:

| path | state |
|---|---|
| `/storage/eng/essswb/phase13/venv-p13` | intact, 182 packages, `mace_jax` 0.2.0 |
| `/storage/eng/essswb/phase13/checkpoints/` | `mace-mp-0-small.model`, `mace-0b2-small.model`, `mace-mp-0b3-medium.model` |
| `/storage/eng/essswb/phase13/{bundles,mace-*-jax}/` | the exported lammps-jax bundles |
| `/storage/eng/essswb/phase13/{scripts,verify,results}/` | the Phase-13 scripts and raw output |

`pip list` reports nothing in that venv because `uv` created it without `pip` —
`$V/bin/pip` does not exist, so a `pip list | grep` returns empty and the venv
looks bare. Use `VIRTUAL_ENV=$V uv pip freeze` instead. **This is very likely
the "mace_jax is not installed anywhere" false negative**, and it is worth
knowing before anything else gets declared missing.

`~/si-ace/acejax` being a stale copy is confirmed — it has no `bench/phase13/`.
That is a stale *checkout*, not a missing environment.

## What was actually built

`venv-p13` is **CPU-only jax** — it has no `jax_plugins/`, and Phase 13 exported
bundles under `JAX_PLATFORMS=cpu`, doing its GPU work inside LAMMPS via
`pair_style jax/kk`, which loads the PJRT plugin from a *different* venv. So
there was no environment in which `mace_jax` itself ran on the GPU. That is what
this recipe adds.

**`/storage/eng/essswb/macejax-gpu/`**

| file | what |
|---|---|
| `make_env.sh` | builds the venv; `MACEJAX_WITH_ACEJAX=1` adds the acejax extras |
| `requirements.lock.txt` | 95 lines, fully pinned |
| `requirements-acejax-extra.txt` | the 3 packages acejax adds |
| `verify_gpu.py` | the verification below |
| `venv/` | the built environment (3.8 GB) |
| `bundles/mace-mp-0-small/` | bundle reconverted from the torch checkpoint by this venv |

The lock was **frozen from the working `venv-p13`** and extended with the jax
CUDA plugin pins copied verbatim from `~/si-ace/.venv`. It was not resolved from
scratch, so there is no opportunity for a silent downgrade; `uv` reported no
conflicts and nothing was pinned-until-it-resolved.

```bash
cd /storage/eng/essswb/macejax-gpu && ./make_env.sh          # ~4 min
```

Key pins (`jax` matches acejax exactly — see "like-for-like" below):

```
jax==0.11.1   jaxlib==0.11.1   jax-cuda12-plugin==0.11.1   jax-cuda12-pjrt==0.11.1
mace-jax @ git+https://github.com/ACEsuit/mace-jax.git@19cf364a405fcf9f2f82590a98d61bc870e05881
mace-torch==0.3.16   torch==2.14.0+cpu   e3nn-jax==0.21.0   numpy==2.5.3
```

`torch` is deliberately the `+cpu` build: MACE-JAX needs torch only to read the
torch checkpoint, and a CUDA torch would contend with jax for the card.

## Checkpoint: MACE-MP-0 small — the one that was asked for

`mace-mp-0-small.model` (32.5 MB) works. It was **reconverted from the torch
checkpoint by the new venv**, not copied from Phase 13, so the toolchain is
proven end to end:

```bash
JAX_PLATFORMS=cpu venv/bin/mace-jax-from-torch \
  --torch-model /storage/eng/essswb/phase13/checkpoints/mace-mp-0-small.model \
  --output bundles/mace-mp-0-small/params.msgpack
```

No fallback to a different checkpoint was needed. `~/.cache/mace/` holds 15
further checkpoints, untested here, including the ones named in the task
(`mace-mpa-0-medium.model`, `mace-omat-0-medium.model`,
`2023-12-03-mace-128-L1_epoch-199.model`) plus `mace-mh-1.model`,
`MACE-OFF23_medium.model`, `MACE-matpes-pbe-omat-ft.model` and
`MACE-matpes-r2scan-omat-ft.model`. Several appear twice under a
punctuation-stripped name (e.g. `macempa0mediummodel`), same size, which looks
like an older cache-naming scheme.

## Verification — energy against an independent reference

`verify_gpu.py` evaluates the **same checkpoint two independent ways on the same
structure** and compares. Structure: 64-atom cubic Si diamond (a = 5.43 Å,
2x2x2), positions perturbed by a seeded 0.05 Å Gaussian so the forces are
non-trivial and a sign or indexing error cannot hide behind an all-zero force
array.

* reference — `mace-torch` `MACECalculator`, **float64**, CPU
* test — `mace-jax` on the **GPU** (`CudaDevice(id=0)`, RTX A4500), float64 mode

```
jax 0.11.1  devices: [CudaDevice(id=0)]
GPU: NVIDIA RTX A4500
jax backend platform(s): {'gpu'}

  atoms                64
  E (mace-jax, GPU)    -341.694622424 eV
  E (mace-torch, CPU)  -341.694600191 eV
  dE/atom              3.474e-07 eV      (tol 1.0e-06)
  |F| scale            1.587921 eV/A
  max|dF|              2.446e-07 eV/A    (tol 1.0e-05)
  PASS
```

Raw output: `verify_gpu.out`.

**On the tolerance.** These are agreement tolerances between two independent
implementations, not precision tolerances. The residual is not arbitrary: the
serialized bundle stores **float32** parameters (45 float32 leaves, 2 float64,
11 int32 — checked directly), because `mace-jax-from-torch` runs without x64
enabled. So this compares float64 arithmetic on float32-rounded weights against
float64 throughout, and `2.446e-07 / 1.588 = 1.5e-7` relative on forces is
exactly float32 round-off. The agreement is as good as the stored weights allow.
If a tighter comparison is ever needed, the bundle would have to be reconverted
with `jax_enable_x64` on first.

The GPU was confirmed idle (`108 / 20470 MiB`, no compute processes) before the
run, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` with `MEM_FRACTION=.25` was set so
the card was not preallocated.

## Separate venv or shared process? — Both work; tested, not assumed

**They can share one process.** acejax needs exactly three packages beyond this
environment — `equinox`, `jaxtyping`, `wadler-lindig` — none of which touch
jax/jaxlib. Installed at the pins `~/si-ace/.venv` uses, then verified:

```
jax 0.11.1 [CudaDevice(id=0)]
mace_jax 0.2.0
acejax imported from /home/eng/essswb/si-ace/acejax/acejax/__init__.py
gpu matmul ok: 256.0
COEXIST OK
```

(acejax is not an installed package in either venv; it is used off `PYTHONPATH`
from a source tree.)

**The timing comparison is like-for-like either way**: this venv and
`~/si-ace/.venv` both run **jax 0.11.1 / jaxlib 0.11.1 / jax-cuda12-plugin
0.11.1 on Python 3.12.8**, so the compiler and PJRT runtime are identical. A
shared process additionally removes any doubt about GPU clock state or context
differences between runs, so it is the better choice for the head-to-head — but
a separate venv is not a methodological problem here.

## What did not work / what is not done

* **Not benchmarked.** This task delivered the environment and a correctness
  check only. No MACE timing number was measured, so none is quoted.
* **Bundle params are float32** (above). Fine for timing; a caveat for any
  high-precision energy comparison.
* **Only MACE-MP-0 small was verified.** The other two Phase-13 checkpoints are
  present and converted bundles exist, but were not re-verified in this venv.
* **`~/si-ace/acejax` is stale** and was used only for the import test. A
  head-to-head needs a fresh checkout of the `jax-eval` acejax.
* The `uv` hardlink warning during the build is benign — cache and target are on
  different filesystems, so it copies instead.

## Remaining before a head-to-head timing run

1. Refresh `~/si-ace/acejax` to the current `jax-eval` tree.
2. Decide the common structure set and the many-element ACE model to time
   against (`acejax/bench/manyelem/` has the basis-scaling work).
3. Pick the harness — in-process (both libs, shown to coexist) is simplest and
   avoids cross-process GPU state differences; Phase 13's route was through
   LAMMPS `pair_style jax/kk`, which is a different question (plumbing held
   constant) and needs the bundles + the LAMMPS build, both of which survive.
4. Warm-up/compile exclusion: Phase 13 found MACE-MP-0b3 medium compiles for
   ~30 s, which must be outside the timed region.
