#!/usr/bin/env python3
"""Our pure-JAX spherical harmonics against sphericart, on GPU.

The per-stage breakdown puts the angular embedding at ~52% of site_basis, and it
is the one component deliberately hand-written: sphericart lowers to an FFI
custom call, which would put a custom-call target in exported StableHLO and drag
in LAMMPS_JAX_FFI_HANDLERS.  That was right for the export path; this measures
what it costs.

Isolated AND end-to-end, because timing stages in isolation over-counts (XLA
fuses them in the real computation) -- the end-to-end delta is the one that
decides anything.
"""
import argparse, pathlib, sys, time
import jax, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def diamond(reps, a=5.43, rattle=0.05, seed=0):
    b = np.array([[0,0,0],[.25,.25,.25],[0,.5,.5],[.25,.75,.75],
                  [.5,0,.5],[.75,.25,.75],[.5,.5,0],[.75,.75,.25]])
    pos = np.array([(np.array([i,j,k])+v)*a
                    for i in range(reps) for j in range(reps) for k in range(reps) for v in b])
    return pos + rattle*np.random.default_rng(seed).normal(size=pos.shape), np.diag([a*reps]*3)


def timeit(fn, *args, repeats=20):
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--a2b-sparse", action="store_true")
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import sphericart.jax as scj
    import acejax.model as M
    from acejax import load, highest_precision, sparse_graph
    from acejax.harmonics import real_spherical_harmonics

    model, meta, _ = load(a.npz, dtype=jnp.float32 if a.f32 else jnp.float64,
                          a2b_sparse=a.a2b_sparse)
    lmax = int(meta["lmax"])
    pos, cell = diamond(a.reps)
    g = sparse_graph(pos, cell, (True,)*3, float(meta["rcut"]))
    n = g.n_nodes
    nz = jnp.zeros(n, jnp.int32)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    rij = jnp.asarray(g.rij, dtype=jnp.float32 if a.f32 else jnp.float64)
    zi, zj = nz[send], nz[recv]

    print(f"\n=== n_B={meta['n_B']} lmax={lmax} {'f32' if a.f32 else 'f64'} "
          f"{n} atoms {len(g.senders)} edges  backend {jax.default_backend()} ===")

    with highest_precision():
        t_ours = timeit(lambda r: real_spherical_harmonics(r, lmax), rij)
        t_sc = timeit(lambda r: scj.spherical_harmonics(r, lmax), rij)
        print(f"  isolated   ours {t_ours:7.3f} ms   sphericart {t_sc:7.3f} ms   "
              f"ratio {t_ours/t_sc:5.2f}x")

        # end-to-end: swap the implementation the model calls
        base = M.real_spherical_harmonics
        e_ours = timeit(lambda r: model.site_basis(r, zi, zj, send, n, None), rij)
        M.real_spherical_harmonics = lambda xyz, L: scj.spherical_harmonics(xyz, L)
        try:
            e_sc = timeit(lambda r: model.site_basis(r, zi, zj, send, n, None), rij)
        finally:
            M.real_spherical_harmonics = base
        print(f"  site_basis ours {e_ours:7.3f} ms   sphericart {e_sc:7.3f} ms   "
              f"ratio {e_ours/e_sc:5.2f}x   delta {e_ours-e_sc:+.3f} ms")
        share = 100 * (e_ours - e_sc) / e_ours if e_ours else 0
        print(f"  -> switching would cut site_basis by {share:.1f}%")


if __name__ == "__main__":
    main()
