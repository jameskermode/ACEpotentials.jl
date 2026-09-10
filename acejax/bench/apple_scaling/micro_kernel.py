"""A stripped-down stand-in for the ACE edge kernel, to see whether the
buffer-size degradation is specific to ACE or generic to XLA-CPU on this host.

Same skeleton as `Model.site_energies`: an elementwise chain over an (E, K)
per-edge array, a segment_sum into (N, K) nodes, a quadratic readout, and
`value_and_grad` through all of it.  Nothing about ACE survives except the
shapes.  E (the padded edge count) is the axis under test; N and the number of
*real* edges are held fixed, exactly as in sweep B.

    python micro_kernel.py --n-nodes 216 --k 43 --edges 17057 31013 62027 ...
"""
import argparse
import json
import os
import time

import numpy as np

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
import jax.numpy as jnp
from jax import lax


def build(n_nodes, k, n_edges, n_real, dtype, seed=0):
    rng = np.random.default_rng(seed)
    # real edges point at real nodes; padded slots point at node 0, the way
    # ghost_exchange_subgraph fills the tail of its buffer
    seg = np.zeros(n_edges, np.int32)
    seg[:n_real] = rng.integers(0, n_nodes, n_real)
    x = np.zeros((n_edges, 3))
    x[:n_real] = rng.normal(size=(n_real, 3))
    w = rng.normal(size=(k,)) / np.sqrt(k)
    return (jnp.asarray(x, dtype), jnp.asarray(seg), jnp.asarray(w, dtype))


def make_fn(n_nodes, k):
    def energy(x, seg, w):
        r = jnp.sqrt(jnp.sum(x * x, axis=-1) + 1e-12)          # (E,)
        u = 1.0 / (1.0 + r)
        # a Chebyshev-like recurrence to K columns: the per-edge (E, K) array
        cols = [jnp.ones_like(u), u]
        for _ in range(k - 2):
            cols.append(2.0 * u * cols[-1] - cols[-2])
        P = jnp.stack(cols, axis=-1)                            # (E, K)
        P = P * jnp.exp(-r)[:, None]
        A = jax.ops.segment_sum(P, seg, num_segments=n_nodes)   # (N, K)
        return jnp.sum((A * A) @ w)
    return energy


def time_point(n_nodes, k, n_edges, n_real, dtype, inner, reps):
    x, seg, w = build(n_nodes, k, n_edges, n_real, dtype)
    energy = make_fn(n_nodes, k)

    @jax.jit
    def leg(x):
        def body(_, xx):
            g = jax.grad(energy)(xx, seg, w)
            return xx + 1e-6 * g
        return lax.fori_loop(0, inner, body, x)

    jax.block_until_ready(leg(x))
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(leg(x))
        best = min(best, (time.perf_counter() - t0) / inner)
    mem = {}
    try:
        m = leg.lower(x).compile().memory_analysis()
        mem = dict(temp=int(m.temp_size_in_bytes))
    except Exception as exc:
        mem = dict(error=repr(exc))
    return dict(n_nodes=n_nodes, k=k, n_edges=n_edges, n_real=n_real,
                dtype=str(x.dtype), ms=best * 1e3, ns_per_edge=best * 1e9 / n_edges,
                mem=mem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-nodes", type=int, default=216)
    p.add_argument("--k", type=int, default=43)
    p.add_argument("--n-real", type=int, default=15218)
    p.add_argument("--edges", type=int, nargs="+", required=True)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()
    jax.config.update("jax_enable_x64", not a.f32)
    dtype = jnp.float32 if a.f32 else jnp.float64
    print(f"jax {jax.__version__} x64={jax.config.jax_enable_x64} devices={jax.devices()}")
    rows = []
    for e in a.edges:
        r = time_point(a.n_nodes, a.k, e, min(a.n_real, e), dtype, a.inner, a.reps)
        rows.append(r)
        print(f"  E={e:>7d} real={r['n_real']:>7d} {r['dtype']:>8}  "
              f"{r['ms']:9.3f} ms  {r['ns_per_edge']:7.1f} ns/edge  "
              f"temp={r['mem'].get('temp', 0)/2**20:8.1f} MB", flush=True)
    if a.out:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
