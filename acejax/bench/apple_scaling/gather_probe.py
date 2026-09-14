"""Isolate the operation `ace_stage_scan.py` fingered: the axis-1 gather-product

    A = Rnl[:, aspec_r] * Ylm[:, aspec_y]           # (E, 37),(E, 25) -> (E, 43)

whose reverse-mode adjoint is a scatter-add along axis 1 with DUPLICATE indices
(43 entries into 37 and 25 columns).  Compared against an algebraically identical
one-hot matmul form, whose adjoint is a matmul and involves no scatter.

Inputs are plain random arrays -- no ACE, no splines -- so the only thing under
test is the gather/scatter.
"""
import argparse
import os
import time

import numpy as np

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edges", type=int, nargs="+", required=True)
    p.add_argument("--nr", type=int, default=37)
    p.add_argument("--ny", type=int, default=25)
    p.add_argument("--na", type=int, default=43)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    ar = jnp.asarray(rng.integers(0, a.nr, a.na).astype(np.int32))
    ay = jnp.asarray(rng.integers(0, a.ny, a.na).astype(np.int32))
    Sr = jnp.asarray(np.eye(a.nr)[np.asarray(ar)].T)      # (nr, na)
    Sy = jnp.asarray(np.eye(a.ny)[np.asarray(ay)].T)      # (ny, na)

    def gather(R, Y):
        return jnp.sum(R[:, ar] * Y[:, ay])

    def matmul(R, Y):
        return jnp.sum((R @ Sr) * (Y @ Sy))

    print(f"jax {jax.__version__} x64={jax.config.jax_enable_x64}  "
          f"nr={a.nr} ny={a.ny} na={a.na}")
    print(f"{'E':>8} {'gather ms':>10} {'ns/edge':>9} {'matmul ms':>10} {'ns/edge':>9} "
          f"{'ratio':>7} {'max|dR diff|':>13}")
    for E in a.edges:
        R = jnp.asarray(rng.normal(size=(E, a.nr)))
        Y = jnp.asarray(rng.normal(size=(E, a.ny)))
        ts = {}
        for name, fn in (("gather", gather), ("matmul", matmul)):
            g = jax.jit(jax.value_and_grad(fn, argnums=(0, 1)))
            v, gr = g(R, Y)
            jax.block_until_ready(gr)
            best = float("inf")
            for _ in range(a.reps):
                t0 = time.perf_counter()
                jax.block_until_ready(g(R, Y))
                best = min(best, time.perf_counter() - t0)
            ts[name] = (best * 1e3, gr)
        d = float(jnp.max(jnp.abs(ts["gather"][1][0] - ts["matmul"][1][0])))
        print(f"{E:>8} {ts['gather'][0]:>10.3f} {ts['gather'][0]*1e6/E:>9.1f} "
              f"{ts['matmul'][0]:>10.3f} {ts['matmul'][0]*1e6/E:>9.1f} "
              f"{ts['gather'][0]/ts['matmul'][0]:>7.2f} {d:>13.2e}", flush=True)


if __name__ == "__main__":
    main()
