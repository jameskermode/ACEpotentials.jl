#!/usr/bin/env python3
"""Time acejax in Python on the same GPU, to separate plugin overhead from model cost.

Reports atom-steps/s for a forces evaluation (the quantity MD actually needs),
at the same sizes the LAMMPS runs use.
"""
import argparse, itertools, pathlib, sys, time
import jax
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def diamond(reps, a=5.43, rattle=0.05, seed=0):
    b = np.array([[0,0,0],[.25,.25,.25],[0,.5,.5],[.25,.75,.75],
                  [.5,0,.5],[.75,.25,.75],[.5,.5,0],[.75,.75,.25]])
    pos = np.array([(np.array([i,j,k])+v)*a
                    for i in range(reps) for j in range(reps) for k in range(reps) for v in b])
    rng = np.random.default_rng(seed)
    return pos + rattle*rng.normal(size=pos.shape), np.diag([a*reps]*3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, nargs="+", default=[2,3,4,5,6,8])
    p.add_argument("--f32", action="store_true")
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from acejax import load, highest_precision
    # self-contained periodic edge list: matscipy-neighbours needs a build
    # toolchain that the benchmark host lacks, and stage1/tests already pins
    # matscipy's list against Julia's, so an independent route is fine here.
    def edges(pos, cell, rcut):
        n = len(pos)
        reps_ = [int(np.ceil(rcut / cell[k, k])) for k in range(3)]
        ii, jj, rr = [], [], []
        for sh in itertools.product(*[range(-r, r + 1) for r in reps_]):
            d = (pos[None, :, :] + np.array(sh) @ cell) - pos[:, None, :]
            r = np.linalg.norm(d, axis=-1)
            m = (r < rcut) & (r > 1e-10)
            a, b = np.where(m)
            ii.append(a); jj.append(b); rr.append(d[a, b])
        ii = np.concatenate(ii); jj = np.concatenate(jj); rr = np.concatenate(rr)
        o = np.argsort(ii, kind="stable")
        return ii[o].astype(np.int32), jj[o].astype(np.int32), rr[o]

    model, meta, _ = load(a.npz, dtype=jnp.float32 if a.f32 else jnp.float64)
    rcut = float(meta["rcut"])
    print(f"# acejax {'f32' if a.f32 else 'f64'} on {jax.default_backend()}  "
          f"n_B={meta['n_B']} rcut={rcut}")
    print(f"# {'atoms':>8} {'edges':>9} {'ms/step':>10} {'atom-steps/s':>14}")
    for reps in a.reps:
        pos, cell = diamond(reps)
        ii, jj, rr = edges(pos, cell, rcut)
        n = len(pos)
        nz = jnp.zeros(n, jnp.int32)
        send, recv = jnp.asarray(ii), jnp.asarray(jj)
        rij = jnp.asarray(rr, dtype=jnp.float32 if a.f32 else jnp.float64)
        with highest_precision():
            f = jax.jit(lambda r: model.energy_forces_virial(
                r, nz[send], nz[recv], send, recv, n, nz)[:2])
            jax.block_until_ready(f(rij))
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter(); jax.block_until_ready(f(rij))
                ts.append(time.perf_counter()-t0)
        dt = min(ts)
        print(f"  {n:>8} {len(ii):>9} {dt*1e3:>10.3f} {n/dt:>14.4g}")


if __name__ == "__main__":
    main()
