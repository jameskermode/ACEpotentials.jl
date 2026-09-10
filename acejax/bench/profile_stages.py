#!/usr/bin/env python3
"""Per-stage breakdown of site_basis, to account for the pace gap rather than
attribute it.

Times radial, angular, A pooling, AA products, A2B and readout in isolation and
compares the sum against the whole.  A shortfall means the cost is in something
not enumerated here, which is itself the finding.

Also reports the padded-edge fraction, since fixed shapes mean dead edges are
evaluated while pace processes exactly the live neighbours.
"""
import argparse, itertools, pathlib, sys, time
import jax, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def diamond(reps, a=5.43, rattle=0.05, seed=0):
    b = np.array([[0,0,0],[.25,.25,.25],[0,.5,.5],[.25,.75,.75],
                  [.5,0,.5],[.75,.25,.75],[.5,.5,0],[.75,.75,.25]])
    pos = np.array([(np.array([i,j,k])+v)*a
                    for i in range(reps) for j in range(reps) for k in range(reps) for v in b])
    return pos + rattle*np.random.default_rng(seed).normal(size=pos.shape), np.diag([a*reps]*3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--a2b-sparse", action="store_true")
    p.add_argument("--repeats", type=int, default=20)
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from acejax import load, highest_precision, sparse_graph
    from acejax.model import pool_sparse

    model, meta, _ = load(a.npz, dtype=jnp.float32 if a.f32 else jnp.float64,
                          a2b_sparse=a.a2b_sparse)
    pos, cell = diamond(a.reps)
    g = sparse_graph(pos, cell, (True,)*3, float(meta["rcut"]))
    n = g.n_nodes
    nz = jnp.zeros(n, jnp.int32)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    rij = jnp.asarray(g.rij, dtype=jnp.float32 if a.f32 else jnp.float64)
    zi, zj = nz[send], nz[recv]

    def bench(fn, *args):
        f = jax.jit(fn)
        jax.block_until_ready(f(*args))
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter(); jax.block_until_ready(f(*args))
            ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    with highest_precision():
        # intermediates, so each stage can be timed on real inputs
        Rnl, Rpair = model.radial(rij, zi, zj)
        Ylm = model.angular(rij)
        edge_A = Rnl[:, model.aspec_r] * Ylm[:, model.aspec_y]
        A = pool_sparse(edge_A, send, n, None)
        AA = jnp.concatenate([jnp.prod(A[:, gg], axis=-1) for gg in model.aa_specs], -1)
        Apair = pool_sparse(Rpair, send, n, None)
        Bdense = AA @ model.A2B.T

        stages = [
            ("radial", lambda r: model.radial(r, zi, zj), (rij,)),
            ("angular", lambda r: model.angular(r), (rij,)),
            ("edge_A product", lambda R, Y: R[:, model.aspec_r] * Y[:, model.aspec_y], (Rnl, Ylm)),
            ("A pooling", lambda e: pool_sparse(e, send, n, None), (edge_A,)),
            ("AA products", lambda AA_in: jnp.concatenate(
                [jnp.prod(AA_in[:, gg], axis=-1) for gg in model.aa_specs], -1), (A,)),
            ("A2B contraction", lambda aa: (
                jax.ops.segment_sum((aa[:, model.a2b_cols] * model.a2b_vals).T,
                                    model.a2b_rows, num_segments=model.A2B.shape[0]).T
                if model.a2b_sparse else aa @ model.A2B.T), (AA,)),
            ("readout", lambda B, Ap: model._readout(B, Ap, nz), (Bdense, Apair)),
        ]
        rows = [(name, bench(fn, *args)) for name, fn, args in stages]
        whole = bench(lambda r: model.site_basis(r, zi, zj, send, n, None), rij)
        full = bench(lambda r: model.site_energies(r, zi, zj, send, n, nz), rij)

    tag = f"n_B={meta['n_B']} {'f32' if a.f32 else 'f64'} A2B={'sparse' if a.a2b_sparse else 'dense'}"
    print(f"\n=== {tag}  {n} atoms, {len(g.senders)} edges, backend {jax.default_backend()} ===")
    tot = sum(t for _, t in rows)
    for name, t in rows:
        print(f"  {name:<18} {t:8.3f} ms   {100*t/whole:5.1f}% of site_basis")
    print(f"  {'-'*18} {'-'*8}")
    print(f"  {'sum of stages':<18} {tot:8.3f} ms   {100*tot/whole:5.1f}%")
    print(f"  {'site_basis (whole)':<18} {whole:8.3f} ms")
    print(f"  {'site_energies':<18} {full:8.3f} ms")
    print(f"  {'UNACCOUNTED':<18} {whole-tot:8.3f} ms   {100*(whole-tot)/whole:5.1f}%")


if __name__ == "__main__":
    main()
