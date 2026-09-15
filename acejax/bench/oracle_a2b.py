#!/usr/bin/env python3
"""Ceiling for removing the A2B contraction: time E+F with A2B replaced by a
free identity (B := AA[:, :n_B]).  Attribution by difference, whole
computation both times.  THROWAWAY -- evidence for the fold, not product.

  python bench/oracle_a2b.py --npz fixtures/si_l2849.npz --reps 6 --a2b-sparse
"""
import argparse, itertools, pathlib, sys, time

import jax, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from bench_acejax import diamond          # same structures as the benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--a2b-sparse", action="store_true")
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from acejax import load, highest_precision
    from acejax.model import ACEModel, pool_sparse

    def edges(pos, cell, rcut):
        n = len(pos)
        reps_ = [int(np.ceil(rcut / cell[k, k])) for k in range(3)]
        ii, jj, rr = [], [], []
        for sh in itertools.product(*[range(-r, r + 1) for r in reps_]):
            d = (pos[None, :, :] + np.array(sh) @ cell) - pos[:, None, :]
            r = np.linalg.norm(d, axis=-1)
            m = (r < rcut) & (r > 1e-10)
            x, y = np.where(m)
            ii.append(x); jj.append(y); rr.append(d[x, y])
        ii = np.concatenate(ii); jj = np.concatenate(jj); rr = np.concatenate(rr)
        o = np.argsort(ii, kind="stable")
        return ii[o].astype(np.int32), jj[o].astype(np.int32), rr[o]

    dt = jnp.float32 if a.f32 else jnp.float64
    model, meta, _ = load(a.npz, dtype=dt, a2b_sparse=a.a2b_sparse)
    pos, cell = diamond(a.reps)
    ii, jj, rr = edges(pos, cell, float(meta["rcut"]))
    n = len(pos)
    nz = jnp.zeros(n, jnp.int32)
    send, recv, rij = jnp.asarray(ii), jnp.asarray(jj), jnp.asarray(rr, dtype=dt)

    def timeit(m):
        with highest_precision():
            f = jax.jit(lambda r: m.energy_forces_virial(r, nz[send], nz[recv],
                                                         send, recv, n, nz)[:2])
            jax.block_until_ready(f(rij))
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter(); jax.block_until_ready(f(rij))
                ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    n_B = int(meta["n_B"])
    orig = ACEModel._from_pooled

    def oracle(self, A, Apair):
        AA = jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in self.aa_specs], axis=-1)
        return AA[:, :n_B], Apair          # free identity of the right shape

    t_real = timeit(model)
    ACEModel._from_pooled = oracle
    t_orac = timeit(model)
    ACEModel._from_pooled = orig
    print(f"# {a.npz.name} n_B={n_B} atoms={n} {'f32' if a.f32 else 'f64'} "
          f"a2b={'sparse' if a.a2b_sparse else 'dense'} on {jax.default_backend()}")
    print(f"  E+F with A2B      {t_real:9.3f} ms")
    print(f"  E+F oracle (no A2B){t_orac:9.3f} ms")
    print(f"  ceiling            {t_real / t_orac:9.3f}x")


if __name__ == "__main__":
    main()
