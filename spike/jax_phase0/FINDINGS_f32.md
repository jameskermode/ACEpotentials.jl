# Phase 0 follow-up: the f32 GPU Jacobian anomaly

Investigated because a single 5.3e-1 naive-vs-hybrid Jacobian discrepancy was
too large to be roundoff. It was not. There are **two independent effects**.

## Effect 1 — TF32 is on by default (documented, not a bug, but a trap)

Ada is compute 8.9, so XLA uses TF32 (10-bit mantissa) for f32 matmuls by
default. Setting `jax_default_matmul_precision=highest` recovers true f32:

| quantity | TF32 (default) | precision=highest | ratio |
|---|---|---|---|
| forward descriptor `B` vs Julia | 1.1705e-03 | 2.9252e-06 | 400× |
| Jacobian naive-vs-hybrid | 5.418e-04 | 2.091e-07 | 2600× |
| forces vs f64 | ~7.7e-05 | — | — |

**A 1.2e-3 relative error in the descriptor is not acceptable for a production
potential without a conscious decision.** Stage 1 must set matmul precision
explicitly, and must verify the setting survives `jax.export` into the
lammps-jax bundle.

## Effect 2 — XLA GPU autotuning picks a wrong kernel (genuine correctness bug)

~30% of *processes* produce a structurally wrong Jacobian element: -0.023 where
the true value is -19.159. Stable within a process, varying between processes.

Not a precision failure: at that element the terms sum with
`sum|t| / |sum t| = 1.002`, i.e. no cancellation at all.

| configuration | runs | outcome |
|---|---|---|
| f32, default | 20 | 14 good (5.418e-04), **6 wrong (5.247e-01)** |
| f32, `--xla_gpu_autotune_level=0` | 10 | **10 good** |
| f32, `matmul_precision=highest` | 10 | 8 good (2.091e-07), **2 wrong (5.245e-01)** |
| f64, default | 10 | **10 good** (4.869e-16) |
| forward descriptor, f32 | 20 | **20 stable** (1.1705e-03) |
| forces via `jax.grad`, f32 | 20 | **20 stable** (spread ~1e-8 rel) |

Scope: **f32 only, Jacobian only.** It is eliminated by disabling autotuning and
is independent of TF32 (it still occurs at precision=highest). The affected
operation is the batched einsum `'ebm,emc->ebc'` — 2154 batches of
(110x43) @ (43x3), a very skinny batched GEMM with many candidate kernels.

## Consequences

- **Stage 1 is not exposed to Effect 2.** Forward and forces are both stable
  across processes. But it *is* exposed to Effect 1 and must pin matmul
  precision.
- **Stage 2 is exposed to both**, because fit assembly is exactly the Jacobian.
  f64 is clean in every test, which is now a third independent reason to
  assemble the design matrix in f64.
- Worth reporting upstream to JAX/XLA with this reproducer.

## Process notes

- `etace_jax.py` calls `jax.config.update("jax_enable_x64", True)` at **import
  time**. Any script setting precision before importing it is silently
  overridden. This cost one wasted diagnostic run and must not survive into
  Stage 1 — no module-level global config.
- `check.py` and `fwd_f32.py` differ only in that patch; `check.py` always runs
  f64 regardless of `X64=0`.

## Reproducer

```bash
cd /tmp/jax_phase0
for i in $(seq 1 20); do X64=0 JAX_PLATFORMS=cuda .venv/bin/python diag_f32.py \
  2>/dev/null | grep -oE "hybrid\| = [0-9.e-]+"; done | sort | uniq -c
```
