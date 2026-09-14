#!/usr/bin/env python3
"""THROWAWAY SPIKE -- recursive/DAG AA products vs the flat evaluator.

Correctness first, then END-TO-END timing, never an isolated AA number: XLA
fuses the flat AA into its surroundings, so isolated AA over-counts (established
twice already in this project).  Two end-to-end metrics are reported:

  site_basis        -- forward descriptor only
  energy + forces   -- value_and_grad, which is what MD actually pays

A `model.site_basis` baseline is timed alongside the hand-rolled `flat` path to
confirm the comparison baseline is the shipped one.
"""
import argparse, pathlib, sys, time
import numpy as np
import jax

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))


def diamond(reps, a=5.43, rattle=0.05, seed=0):
    b = np.array([[0,0,0],[.25,.25,.25],[0,.5,.5],[.25,.75,.75],
                  [.5,0,.5],[.75,.25,.75],[.5,.5,0],[.75,.75,.25]])
    pos = np.array([(np.array([i,j,k])+v)*a
                    for i in range(reps) for j in range(reps) for k in range(reps) for v in b])
    return pos + rattle*np.random.default_rng(seed).normal(size=pos.shape), np.diag([a*reps]*3)


def run(npz, reps, f32, sparse, repeats, modes):
    import jax.numpy as jnp
    from acejax import load, highest_precision, sparse_graph
    from acejax.model import pool_sparse
    from dag import dag_from_model
    from dagjax import MAKERS

    dt = jnp.float32 if f32 else jnp.float64
    model, meta, _ = load(npz, dtype=dt, a2b_sparse=sparse)
    n_A = int(model.aspec_r.shape[0])
    specs = [np.asarray(g) for g in model.aa_specs]
    flat_m = sum(int(g.shape[0]) * (int(g.shape[1]) - 1) for g in specs)
    flat_g = sum(int(g.shape[0]) * int(g.shape[1]) for g in specs)

    pos, cell = diamond(reps)
    g = sparse_graph(pos, cell, (True,)*3, float(meta["rcut"]))
    n = g.n_nodes
    nz = jnp.zeros(n, jnp.int32)
    send = jnp.asarray(g.senders)
    rij = jnp.asarray(g.rij, dtype=dt)
    zi, zj = nz[send], nz[jnp.asarray(g.receivers)]

    def a2b(AA, cols):
        return jax.ops.segment_sum((AA[:, cols] * model.a2b_vals).T, model.a2b_rows,
                                   num_segments=model.A2B.shape[0]).T

    def flat_AA(A):
        return jnp.concatenate([jnp.prod(A[:, gg], axis=-1) for gg in model.aa_specs], -1)

    def site_basis_flat(r):
        eA, Rp = model.edge_features(r, zi, zj)
        A = pool_sparse(eA, send, n, None)
        return a2b(flat_AA(A), model.a2b_cols), pool_sparse(Rp, send, n, None)

    def bench(fn, *args):
        f = jax.jit(fn)
        try:
            jax.block_until_ready(f(*args))
        except Exception as e:
            print("    FAILED:", type(e).__name__, str(e)[:180]); return float("nan")
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter()-t0)
        return min(ts) * 1e3

    def efn(sb):
        return lambda r: jnp.sum(model._readout(*sb(r), nz))

    print(f"\n########  {npz.name}  n_A={n_A} n_AA={meta['n_AA']} n_B={meta['n_B']} "
          f"| {n} atoms {len(g.senders)} edges | {'f32' if f32 else 'f64'} "
          f"A2B={'sparse' if sparse else 'dense'} | {jax.default_backend()}")

    with highest_precision():
        eA, Rp = model.edge_features(rij, zi, zj)
        A = pool_sparse(eA, send, n, None)
        AA_ref = np.asarray(flat_AA(A))

        t_ship = bench(lambda r: model.site_basis(r, zi, zj, send, n, None), rij)
        t_flat = bench(site_basis_flat, rij)
        t_flatE = bench(jax.value_and_grad(efn(site_basis_flat)), rij)
        # ORACLE: same shapes, same A2B, but the AA products replaced by a plain
        # gather -- zero multiplies.  No AA optimisation can beat this, so
        # (flat - oracle) is the entire budget any recursive scheme competes for.
        idx_cheap = jnp.asarray(np.arange(int(meta["n_AA"])) % n_A, jnp.int32)
        def site_basis_oracle(r):
            e, p_ = model.edge_features(r, zi, zj)
            return (a2b(pool_sparse(e, send, n, None)[:, idx_cheap], model.a2b_cols),
                    pool_sparse(p_, send, n, None))
        t_orac = bench(site_basis_oracle, rij)
        t_oracE = bench(jax.value_and_grad(efn(site_basis_oracle)), rij)
        F_ref = np.asarray(jax.grad(efn(site_basis_flat))(rij))
        Fs = max(np.abs(F_ref).max(), 1e-300)
        print(f"  baseline  model.site_basis {t_ship:8.3f} ms | "
              f"spike flat site_basis {t_flat:8.3f} ms | flat E+F {t_flatE:8.3f} ms")
        print(f"  ORACLE (AA products deleted, gather only): site_basis {t_orac:8.3f} ms "
              f"| E+F {t_oracE:8.3f} ms")
        print(f"  => whole AA budget: site_basis {t_flat-t_orac:7.3f} ms "
              f"({100*(t_flat-t_orac)/t_flat:4.1f}%) | E+F {t_flatE-t_oracE:7.3f} ms "
              f"({100*(t_flatE-t_oracE)/t_flatE:4.1f}%)  <- ceiling for any AA scheme")

        for mode in modes:
            t0 = time.perf_counter()
            D = dag_from_model(specs, n_A, mode=mode)
            tb = time.perf_counter() - t0
            n_total, proj = D["n_total"], np.asarray(D["projection"])
            cols_dag = jnp.asarray(proj[np.asarray(model.a2b_cols)])

            print(f"  -- mode={mode}: build {tb:.2f}s, nodes {D['n_nodes_int']} "
                  f"(aux {D['n_extra']}), cols {n_total} vs {meta['n_AA']}, depth {D['depth']}, "
                  f"levels {D['level_sizes']}")
            print(f"     mults {D['n_nodes_int']} vs {flat_m} ({flat_m/D['n_nodes_int']:.2f}x), "
                  f"reads {2*D['n_nodes_int']} vs {flat_g} ({flat_g/(2*D['n_nodes_int']):.2f}x)")
            for name in ("dus", "concat", "cvjp"):
                vf = MAKERS[name](D["levels"], n_total, n_A)
                got = np.asarray(vf(A))[:, proj]
                rel = np.abs(got - AA_ref) / np.maximum(np.abs(AA_ref), 1e-300)
                def sb(r, vf=vf):
                    e, p_ = model.edge_features(r, zi, zj)
                    return a2b(vf(pool_sparse(e, send, n, None)), cols_dag), pool_sparse(p_, send, n, None)
                dF = np.abs(np.asarray(jax.grad(efn(sb))(rij)) - F_ref).max() / Fs
                t = bench(sb, rij)
                tE = bench(jax.value_and_grad(efn(sb)), rij)
                print(f"     {name:<7} AA max-rel {rel.max():.2e} dF {dF:.2e} | "
                      f"site_basis {t:8.3f} ms ({t_flat/t:5.3f}x) | "
                      f"E+F {tE:8.3f} ms ({t_flatE/tE:5.3f}x)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, nargs="+", required=True)
    p.add_argument("--reps", type=int, nargs="+", default=[6])
    p.add_argument("--f32", action="store_true")
    p.add_argument("--both-precisions", action="store_true")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--modes", nargs="+", default=["julia", "balanced", "chain"])
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    precs = [False, True] if a.both_precisions else [a.f32]
    for npz in a.npz:
        for reps in a.reps:
            for f32 in precs:
                run(npz, reps, f32, not a.dense, a.repeats, a.modes)


if __name__ == "__main__":
    main()
