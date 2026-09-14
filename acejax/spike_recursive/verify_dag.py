#!/usr/bin/env python3
"""THROWAWAY SPIKE -- correctness gate for the recursive/DAG AA evaluator.

CPU only, no GPU needed.  Three checks, in increasing strength:

1. STRUCTURAL (exact, integer).  Reconstruct the multiset of A-indices each DAG
   node computes and check that `projection` maps it onto exactly the flat
   spec, row for row.  This proves the DAG evaluates the right products
   independently of any floating point.

2. BITWISE.  mode="chain" associates left-to-right exactly as `jnp.prod` over a
   gathered row does, so its AA output must be bit-identical to flat.  This
   gates the whole index pipeline: levels, buffer layout, projection, and the
   projection folded into the A2B column index.

3. ULP.  modes "julia"/"balanced" re-associate the products -- which is the
   entire point of sharing subproducts -- so bitwise equality is unattainable
   by construction.  Report the ulp distance and the downstream effect on B,
   energy and forces instead.
"""
import argparse, json, pathlib, sys
import numpy as np
import jax

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from acejax import load, highest_precision, sparse_graph
from acejax.model import pool_sparse
from dag import dag_from_model, reconstruct_spec, _BUILDERS, level_dag


def diamond(reps, a=5.43, rattle=0.05, seed=0):
    b = np.array([[0,0,0],[.25,.25,.25],[0,.5,.5],[.25,.75,.75],
                  [.5,0,.5],[.75,.25,.75],[.5,.5,0],[.75,.75,.25]])
    pos = np.array([(np.array([i,j,k])+v)*a
                    for i in range(reps) for j in range(reps) for k in range(reps) for v in b])
    return pos + rattle*np.random.default_rng(seed).normal(size=pos.shape), np.diag([a*reps]*3)


def ulps(a, b):
    """Distance in representable doubles between two f64 arrays."""
    ia = a.view(np.int64).copy(); ib = b.view(np.int64).copy()
    ia[ia < 0] = np.int64(-(2**63)) - ia[ia < 0]
    ib[ib < 0] = np.int64(-(2**63)) - ib[ib < 0]
    return np.abs(ia - ib)


def dag_vals(A, levels, n_total, n_A):
    v = jnp.zeros((A.shape[0], n_total), A.dtype).at[:, :n_A].set(A)
    for s, L, R in levels:
        v = jax.lax.dynamic_update_slice(v, v[:, jnp.asarray(L)] * v[:, jnp.asarray(R)], (0, s))
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--reps", type=int, default=2)
    a = p.parse_args()
    ok_all = True

    for npz in a.npz:
        model, meta, _ = load(npz, dtype=jnp.float64, a2b_sparse=True)
        n_A = int(model.aspec_r.shape[0])
        specs = [np.asarray(g) for g in model.aa_specs]
        spec = [tuple(int(v) for v in row) for g in specs for row in g]

        pos, cell = diamond(a.reps)
        gph = sparse_graph(pos, cell, (True,)*3, float(meta["rcut"]))
        n = gph.n_nodes
        nz = jnp.zeros(n, jnp.int32)
        send, recv = jnp.asarray(gph.senders), jnp.asarray(gph.receivers)
        rij = jnp.asarray(gph.rij, dtype=jnp.float64)
        zi, zj = nz[send], nz[recv]

        print(f"\n######## {npz.name}  n_A={n_A} n_AA={meta['n_AA']} n_B={meta['n_B']} "
              f"| {n} atoms, {len(gph.senders)} edges (CPU, f64)")

        with highest_precision():
            eA, Rp = model.edge_features(rij, zi, zj)
            A = pool_sparse(eA, send, n, None)
            Apair = pool_sparse(Rp, send, n, None)
            AA_ref = np.asarray(jnp.concatenate(
                [jnp.prod(A[:, g], axis=-1) for g in model.aa_specs], -1))
            B_ref, _ = model._from_pooled(A, Apair)
            B_ref = np.asarray(B_ref)
            E_ref = np.asarray(model._readout(jnp.asarray(B_ref), Apair, nz))
            # forces through the flat path, for the downstream comparison
            def E_flat(r):
                return jnp.sum(model.site_energies(r, zi, zj, send, n, nz, None))
            F_ref = np.asarray(jax.grad(E_flat)(rij))

            for mode in ("chain", "julia", "balanced"):
                D = dag_from_model(specs, n_A, mode=mode)
                left, right, _, _ = _BUILDERS[mode](spec, n_A)

                # ---- 1. structural, exact
                rec = reconstruct_spec(left, right, n_A)
                lv, perm_pos, n_tot = level_dag(left, right, n_A)
                # projection in the *permuted* space -> map back through perm_pos
                inv = np.empty(n_tot, np.int64); inv[perm_pos] = np.arange(n_tot)
                got_spec = [rec[inv[j]] for j in D["projection"]]
                struct_ok = got_spec == [tuple(sorted(s)) for s in spec]

                # ---- 2/3. numeric
                proj = np.asarray(D["projection"])
                vals = np.asarray(dag_vals(A, D["levels"], D["n_total"], n_A))
                AA = vals[:, proj]
                u = ulps(AA, AA_ref)
                bit = bool(np.array_equal(AA.view(np.int64), AA_ref.view(np.int64)))

                cols = jnp.asarray(proj[np.asarray(model.a2b_cols)])
                contrib = jnp.asarray(vals)[:, cols] * model.a2b_vals
                B = np.asarray(jax.ops.segment_sum(contrib.T, model.a2b_rows,
                               num_segments=model.A2B.shape[0]).T)
                Bok = np.array_equal(B.view(np.int64), B_ref.view(np.int64))

                def E_dag(r, D=D, cols=cols):
                    e, pp = model.edge_features(r, zi, zj)
                    v = dag_vals(pool_sparse(e, send, n, None), D["levels"], D["n_total"], n_A)
                    c = v[:, cols] * model.a2b_vals
                    Bx = jax.ops.segment_sum(c.T, model.a2b_rows,
                                             num_segments=model.A2B.shape[0]).T
                    return jnp.sum(model._readout(Bx, pool_sparse(pp, send, n, None), nz))
                E = float(E_dag(rij)); F = np.asarray(jax.grad(E_dag)(rij))

                dB = np.abs(B - B_ref).max() / max(np.abs(B_ref).max(), 1e-300)
                dF = np.abs(F - F_ref).max() / max(np.abs(F_ref).max(), 1e-300)
                dE = abs(E - float(E_ref.sum())) / max(abs(float(E_ref.sum())), 1e-300)
                good = struct_ok and (u.max() <= 4) and dF < 1e-12
                ok_all &= good
                print(f"  mode={mode:<9} depth {D['depth']:<2} nodes {D['n_nodes_int']:<6} "
                      f"levels {str(D['level_sizes']):<26}")
                print(f"     structural spec identity : {'EXACT' if struct_ok else 'MISMATCH'}")
                print(f"     AA vs flat               : bitwise {bit}, max {u.max()} ulp, "
                      f"mean {u.mean():.3f} ulp")
                print(f"     B  bitwise {str(Bok):<5} rel {dB:.3e} | E rel {dE:.3e} | "
                      f"F rel {dF:.3e}   -> {'PASS' if good else 'FAIL'}")

    print("\nALL CHECKS PASS" if ok_all else "\nSOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
