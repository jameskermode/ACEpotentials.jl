"""Which part of the ACE edge kernel scales superlinearly in the padded edge count?

Times `value_and_grad` (w.r.t. the edge vectors, as MD does) of a chain of nested
PREFIXES of `Model.site_energies`, at a fixed number of *real* edges and a varying
padded buffer length -- the same variable sweep B varies.

Attribution is by DIFFERENCE between consecutive prefixes, never by timing a stage
on its own.  Each prefix ends in a scalar sum so the AD graph is comparable; the
sum itself is O(E) and cancels in the differences.

    python ace_stage_scan.py --model ../../fixtures/si_fitted.npz --edges 17057 ... 
"""
import argparse
import json
import os
import time

import numpy as np

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import acejax
from acejax.model import pool_sparse, highest_precision

STAGES = ["norm", "radial", "angular", "edge_features", "pool", "full"]


def make(model, stage, segs, n_nodes, mask, node_z, zi, zj):
    def f(rij):
        if stage == "norm":
            return jnp.sum(jnp.linalg.norm(rij, axis=-1))
        if stage == "radial":
            Rnl, Rpair = model.radial(rij, zi, zj)
            return jnp.sum(Rnl) + jnp.sum(Rpair)
        if stage == "angular":
            Rnl, Rpair = model.radial(rij, zi, zj)
            return jnp.sum(Rnl) + jnp.sum(Rpair) + jnp.sum(model.angular(rij))
        if stage == "edge_features":
            A, Rpair = model.edge_features(rij, zi, zj)
            return jnp.sum(A) + jnp.sum(Rpair)
        if stage == "pool":
            A, Rpair = model.edge_features(rij, zi, zj)
            return (jnp.sum(pool_sparse(A, segs, n_nodes, mask))
                    + jnp.sum(pool_sparse(Rpair, segs, n_nodes, mask)))
        if stage == "full":
            return jnp.sum(model.site_energies(rij, zi, zj, segs, n_nodes, node_z, mask))
        raise ValueError(stage)
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--edges", type=int, nargs="+", required=True)
    p.add_argument("--n-nodes", type=int, default=216)
    p.add_argument("--n-real", type=int, default=15218)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--stages", nargs="+", default=STAGES)
    p.add_argument("--out", default="")
    a = p.parse_args()

    calc = acejax.ACECalculator(a.model, dtype=jnp.float64)
    model = calc.model
    rcut = float(np.max(np.asarray(model.pair_envelope)[..., 0]))
    print(f"jax {jax.__version__} x64={jax.config.jax_enable_x64} rcut={rcut}")

    rows = []
    for E in a.edges:
        rng = np.random.default_rng(0)
        n_real = min(a.n_real, E)
        rij = np.zeros((E, 3))
        # real edges: random directions at physical distances; padded slots sit at
        # the cutoff on the x axis, exactly as ace_dist.ace_node_energies places them
        d = rng.uniform(1.8, 5.9, n_real)
        u = rng.normal(size=(n_real, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        rij[:n_real] = u * d[:, None]
        rij[n_real:] = [rcut, 0.0, 0.0]
        segs = np.zeros(E, np.int32); segs[:n_real] = rng.integers(0, a.n_nodes, n_real)
        mask = np.zeros(E, bool); mask[:n_real] = True
        zi = np.zeros(E, np.int32); zj = np.zeros(E, np.int32)
        node_z = np.zeros(a.n_nodes, np.int32)
        args = [jnp.asarray(x) for x in (rij, segs, mask, node_z, zi, zj)]
        rij_j, segs_j, mask_j, node_z_j, zi_j, zj_j = args

        out = {"n_edges": E, "n_real": n_real}
        for stage in a.stages:
            f = make(model, stage, segs_j, a.n_nodes, mask_j, node_z_j, zi_j, zj_j)
            g = jax.jit(jax.value_and_grad(f))
            with highest_precision():
                jax.block_until_ready(g(rij_j))
                best = float("inf")
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    jax.block_until_ready(g(rij_j))
                    best = min(best, time.perf_counter() - t0)
            out[stage] = best * 1e3
        rows.append(out)
        cells = "  ".join(f"{s}={out[s]:8.3f}" for s in a.stages)
        print(f"E={E:>7d}  {cells}", flush=True)

    print("\nns per padded edge, and the DIFFERENCE from the previous prefix:")
    hdr = f"{'E':>8}" + "".join(f"{s:>15}" for s in a.stages)
    print(hdr)
    for out in rows:
        cells = ""
        prev = 0.0
        for s in a.stages:
            ns = out[s] * 1e6 / out["n_edges"]
            dns = (out[s] - prev) * 1e6 / out["n_edges"]
            prev = out[s]
            cells += f"{ns:7.1f}/{dns:+6.1f}"
        print(f"{out['n_edges']:>8}{cells}")
    if a.out:
        with open(a.out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
