"""Where in the gather does the cost sit: the forward gather or its adjoint?

Splits `gather_probe.py`'s comparison three ways at each buffer length:
forward-only against value_and_grad, DUPLICATE column indices (what ACE has:
43 entries into 37 columns) against a PERMUTATION (unique indices), and both
against the one-hot matmul.  Answers whether the duplicates are the problem.
"""
import os, time
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

rng = np.random.default_rng(0)
NR, NY, NA = 37, 25, 43
dup_r = jnp.asarray(rng.integers(0, NR, NA).astype(np.int32))
dup_y = jnp.asarray(rng.integers(0, NY, NA).astype(np.int32))
perm  = jnp.asarray(rng.permutation(NR).astype(np.int32))     # unique indices, na = nr
Sr = jnp.asarray(np.eye(NR)[np.asarray(dup_r)].T)
Sy = jnp.asarray(np.eye(NY)[np.asarray(dup_y)].T)

def t(fn, args, reps=5):
    j = jax.jit(fn); jax.block_until_ready(j(*args))
    b = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(j(*args)); b = min(b, time.perf_counter()-t0)
    return b*1e3

print(f"{'E':>8}" + "".join(f"{n:>13}" for n in
      ["dup fwd","dup grad","perm fwd","perm grad","mm fwd","mm grad"]))
for E in (17057, 62027, 248112, 496224):
    R = jnp.asarray(rng.normal(size=(E, NR))); Y = jnp.asarray(rng.normal(size=(E, NY)))
    dup  = lambda R, Y: jnp.sum(R[:, dup_r] * Y[:, dup_y])
    pm   = lambda R, Y: jnp.sum(R[:, perm] * Y[:, dup_y[:NR]])
    mm   = lambda R, Y: jnp.sum((R @ Sr) * (Y @ Sy))
    vals = []
    for f in (dup, pm, mm):
        vals.append(t(f, (R, Y)) * 1e6 / E)
        vals.append(t(jax.value_and_grad(f, argnums=(0, 1)), (R, Y)) * 1e6 / E)
    # reorder to fwd,grad pairs already in that order
    print(f"{E:>8}" + "".join(f"{v:>13.1f}" for v in vals) + "   ns/edge")
