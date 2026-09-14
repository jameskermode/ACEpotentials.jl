"""THROWAWAY. Same-process, same-host cost of the dist/parallel route.

Everything here is a whole jitted call, min-of-repeats, compilation excluded --
no isolated stage timings.  `bench/results.md` records three occasions on which
isolated timings over-counted; the rule there is to attribute by difference over
whole computations, so the only quantity claimed is a ratio of two end-to-end
force evaluations measured in the same process.

  raw    `model.energy_forces_virial` on the exact edge list -- no neighbour
         list, no communication, no integration, no padding.  Same call and
         same min-of-repeats method as `bench/bench_acejax.py`, i.e. the
         dashed-line baseline of `bench/results.md` (it times E, F and V and
         discards V, exactly as that benchmark does).
  dist   one `shard_map` step through `ghost_exchange`: exchange_apply, the
         masked energy, the reverse communication in the VJP, and the psum.
         Padded to the capacities `create_config`/`subgraph_params` pick.

The ratio dist/raw is the analogue of results.md's "plugin gap" column: the
fraction of raw model throughput that survives the deployment path.
"""

import argparse
import os
import time
from functools import partial

import numpy as np

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

import jax.numpy as jnp
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ase.build import bulk

import acejax
from acejax.model import highest_precision
from acejax.nlist import sparse_graph
from lammps_jax.dist.parallel import ghost_exchange as gx

from ace_dist import ace_node_energies, ghost_energy


def timeit(f, repeats):
    jax.block_until_ready(f())               # compile outside the measurement
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="../fixtures/si_fitted.npz")
    p.add_argument("--reps", type=int, nargs="+", default=[3, 4, 5])
    p.add_argument("--ranks", type=int, nargs="+", default=[1])
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--skin", type=float, default=0.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--a2b-sparse", action="store_true")
    p.add_argument("--cap-owned", type=float, default=1.5,
                   help="create_config capacity_mult (atom slots)")
    p.add_argument("--cap-sub", type=float, default=2.0,
                   help="subgraph_params capacity_mult (edge/cell slots)")
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    dtype = jnp.float32 if a.f32 else jnp.float64

    model, meta, _ = acejax.load(a.model, dtype=dtype, a2b_sparse=a.a2b_sparse)
    rcut = float(meta["rcut"])
    rcut_env = float(np.max(np.asarray(model.pair_envelope)[..., 0]))
    print(f"# {jax.default_backend()}  {'f32' if a.f32 else 'f64'}  n_B={meta['n_B']}  "
          f"rcut={rcut}  a2b={'sparse' if a.a2b_sparse else 'dense'}  skin={a.skin}")
    print(f"{'atoms':>6} {'edges':>7} {'rk':>3} {'slots a/e':>12} "
          f"{'raw ms':>9} {'dist ms':>9} {'ratio':>7} {'rebuild ms':>11} "
          f"{'raw at-st/s':>12} {'dist at-st/s':>13}")

    for rep in a.reps:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(rep)
        rng = np.random.default_rng(0)
        atoms.positions += 0.05 * rng.standard_normal(atoms.positions.shape)
        atoms.wrap()
        n_atoms = len(atoms)
        box_np = np.asarray(atoms.get_cell().array).T
        box = jnp.asarray(box_np, dtype)

        g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                         atoms.get_pbc(), rcut)
        n_edges = len(g.senders)
        nz = jnp.zeros(n_atoms, jnp.int32)
        send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
        rij = jnp.asarray(g.rij, dtype)
        with highest_precision():
            raw = jax.jit(lambda r: model.energy_forces_virial(
                r, nz[send], nz[recv], send, recv, n_atoms, nz)[:2])
            t_raw = timeit(lambda: raw(rij), a.repeats)

        species = np.zeros(n_atoms, np.int32)
        ids = np.arange(n_atoms, dtype=np.int32)
        R = np.asarray(atoms.get_scaled_positions() % 1.0)
        node_energy_fn = partial(ace_node_energies, model, rcut_env)

        for n_ranks in a.ranks:
            if n_ranks > jax.local_device_count():
                continue
            config = gx.create_config(n_ranks, n_atoms, box_np, rcut,
                                      capacity_mult=a.cap_owned, skin=a.skin)
            cells, cell_cap, max_nb, max_edges = gx.subgraph_params(
                config, box_np, n_atoms, capacity_mult=a.cap_sub)
            mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
            sharding = NamedSharding(mesh, P("i"))
            owned, owned_ints, counts, _ = gx.pack_tiles(R, config, ints=np.stack([species, ids], 1))
            put = lambda x: jax.device_put(jnp.asarray(x, dtype) if x.dtype.kind == "f"
                                           else jnp.asarray(x), sharding)
            owned_m, ints_m, counts_m = put(owned), put(owned_ints), put(counts)

            @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 3,
                     out_specs=(P("i"),) * 8 + (P(),), check_vma=False)
            def rebuild(owned_stack, ints_stack, counts_arr):
                pos, ints_local, n_owned, migrate = gx.redistribute(
                    owned_stack.squeeze(0), counts_arr[0], config, "i",
                    owned_ints=ints_stack.squeeze(0))
                ghost_data, n_ghost, exch, plan = gx.ghost_exchange(pos, n_owned, config, "i")
                ints_nodes = gx.exchange_apply(plan, ints_local, config, "i")
                ghost_ints = lax.dynamic_slice_in_dim(ints_nodes, n_owned, config.max_ghost, axis=0)
                _, snd, rcv, node_species, emask, overflow = gx.ghost_exchange_subgraph(
                    pos, ghost_data, ints_local[:, 0], ghost_ints[:, 0],
                    ints_local[:, 1], ghost_ints[:, 1], n_owned, n_ghost, box,
                    cells, cell_cap, max_nb, max_edges, config, "i")
                flags = lax.psum(jnp.stack([migrate[0] | exch[0],
                                            jnp.any(migrate[1:]) | exch[1] | overflow]).astype(jnp.int32), "i")
                stacked = jax.tree.map(lambda x: x[None], (pos, ints_local, n_owned, plan,
                                                           snd, rcv, node_species, emask))
                return (*stacked, flags)

            @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 8,
                     out_specs=(P(), P("i")), check_vma=False)
            def step(owned_stack, ints_stack, counts_arr, plan, snd, rcv, node_species, emask):
                pos, n_owned = owned_stack.squeeze(0), counts_arr[0]
                plan, snd, rcv, node_species, emask = jax.tree.map(
                    lambda x: x.squeeze(0), (plan, snd, rcv, node_species, emask))
                E, gr = jax.value_and_grad(lambda q: ghost_energy(
                    node_energy_fn, q, n_owned, node_species, snd, rcv, emask, plan,
                    box, config, "i"))(pos)
                return lax.psum(E, "i"), gr[None]

            rebuild_j, step_j = jax.jit(rebuild), jax.jit(step)
            with highest_precision():
                *state, flags = rebuild_j(owned_m, ints_m, counts_m)
                assert not int(flags[0]) and not int(flags[1]), f"flags {flags}"
                t_dist = timeit(lambda: step_j(*state)[1], a.repeats)
                t_rb = timeit(lambda: rebuild_j(owned_m, ints_m, counts_m)[-1], a.repeats)

            slots = f"{config.max_owned * n_ranks / n_atoms:.2f}/{max_edges * n_ranks / n_edges:.2f}"
            print(f"{n_atoms:6d} {n_edges:7d} {n_ranks:3d} {slots:>12} "
                  f"{t_raw * 1e3:9.3f} {t_dist * 1e3:9.3f} {t_raw / t_dist:7.2f} "
                  f"{t_rb * 1e3:11.3f} {n_atoms / t_raw:12.3e} {n_atoms / t_dist:13.3e}")


if __name__ == "__main__":
    main()
