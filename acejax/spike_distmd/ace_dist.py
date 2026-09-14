"""THROWAWAY. acejax through lammps_jax.dist.parallel.ghost_exchange.

Mirrors `lammps_jax/dist/scripts/nequix_bench.py::run_ghost`, with two changes:

1. The node-energy function is ACE, not nequix.  ACE is a one-hop model, so the
   per-layer `exchange_fn` that `feature_exchange.node_energies` needs is never
   called -- everything ACE needs is already in the edge list.
2. `feature_exchange.ghost_energy` is reimplemented here (12 lines) because the
   shipped one lives in the only module of the four that imports e3nn/jax_md.

Reference is `acejax.ACECalculator` on the same ASE Atoms.
"""

import argparse
import os
import time
from functools import partial

import numpy as np

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ase.build import bulk

import acejax
from acejax.model import highest_precision
from lammps_jax.dist.parallel import ghost_exchange as gx


# --------------------------------------------------------------- the adapter

def ace_node_energies(model, rcut, dR, edge_mask, node_species, senders, receivers,
                      n_rows, exchange_fn=None):
    """Per-row site energies (n_rows,) for the owned rows.

    Signature matches `feature_exchange.node_energies` bar `node_species` (the
    nequix path recovers ghost species through `exchange_fn`; `ghost_exchange_
    subgraph` already returns them, so we just carry them).

    `dR = pos[receivers] - pos[senders]` with the *receiver* the owned centre,
    which is the opposite of acejax's own convention (`rij = r[j] - r[i]`,
    sender = centre), hence the negation and the swapped species lookup.
    """
    rij = -dR
    # Padded edges go to the cutoff, where the envelope vanishes and the
    # gradient stays defined -- same trick as `energy_from_positions`.  `rcut`
    # is bound outside the trace: shard_map lifts closed-over arrays to tracers.
    pad = jnp.asarray([rcut, 0.0, 0.0], rij.dtype)
    rij = jnp.where(edge_mask[:, None], rij, pad)
    safe_s = jnp.where(edge_mask, senders, 0)
    safe_r = jnp.where(edge_mask, receivers, 0)
    zi, zj = node_species[safe_r], node_species[safe_s]
    node_z = node_species[:n_rows]
    return model.site_energies(rij, zi, zj, safe_r, n_rows, node_z, edge_mask)


def ghost_energy(node_energy_fn, owned_pos, n_owned, node_species, senders,
                 receivers, edge_mask, plan, box, config, axis_name):
    """`feature_exchange.ghost_energy` without the jax_md/e3nn imports."""
    pos_nodes = gx.exchange_apply(plan, owned_pos, config, axis_name, unwrap=True)
    wrap = jnp.asarray([1.0 if r == 1 else 0.0 for r in config.grid], owned_pos.dtype)
    dr_frac = pos_nodes[receivers] - pos_nodes[senders]
    dr_frac = dr_frac - jnp.round(dr_frac) * wrap
    dR = dr_frac @ box.T                      # box columns are lattice vectors
    edge_mask = edge_mask & (jnp.sum(dR ** 2, axis=-1) < config.cutoff ** 2)
    rows = owned_pos.shape[0]
    e = node_energy_fn(dR, edge_mask, node_species, senders, receivers, rows)
    return jnp.sum(jnp.where(jnp.arange(rows) < n_owned, e, 0.0))


# ------------------------------------------------------------------- driver

def build(rep=4, a=5.43):
    at = bulk("Si", "diamond", a=a, cubic=True).repeat(rep)
    rng = np.random.default_rng(0)
    at.positions += 0.05 * rng.standard_normal(at.positions.shape)  # break symmetry
    at.wrap()
    return at


def reference(atoms, model_path):
    calc = acejax.ACECalculator(model_path, dtype=jnp.float64)
    atoms = atoms.copy()
    atoms.calc = calc
    return float(atoms.get_potential_energy()), np.asarray(atoms.get_forces()), calc


def run(model_path, rep, n_ranks, skin, steps, warmup, verbose=True):
    atoms = build(rep)
    E_ref, F_ref, calc = reference(atoms, model_path)
    model = calc.model
    cutoff = calc.cutoff
    n_atoms = len(atoms)

    box = jnp.asarray(atoms.get_cell().array.T)          # columns = lattice vectors
    R = jnp.asarray(atoms.get_scaled_positions() % 1.0)
    species = np.asarray(calc._species_index(atoms.get_atomic_numbers()), np.int32)
    ids = np.arange(n_atoms, dtype=np.int32)

    config = gx.create_config(n_ranks, n_atoms, np.asarray(atoms.get_cell().array.T),
                              cutoff, capacity_mult=1.5, skin=skin)
    cells, cell_cap, max_nb, max_edges = gx.subgraph_params(
        config, np.asarray(atoms.get_cell().array.T), n_atoms, capacity_mult=2.0)
    if verbose:
        print(f"grid={config.grid} max_owned={config.max_owned} max_ghost={config.max_ghost} "
              f"cells={list(cells)} cell_cap={cell_cap} max_nb={max_nb} max_edges={max_edges}")

    mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
    sharding = NamedSharding(mesh, P("i"))
    owned, owned_ints, counts, _ = gx.pack_tiles(
        R, config, ints=np.stack([species, ids], 1))
    owned_m, ints_m, counts_m = (jax.device_put(jnp.asarray(x), sharding)
                                 for x in (owned, owned_ints, counts))
    rcut_env = float(np.max(np.asarray(model.pair_envelope)[..., 0]))
    node_energy_fn = partial(ace_node_energies, model, rcut_env)

    @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 3,
             out_specs=(P("i"),) * 8 + (P(),), check_vma=False)
    def rebuild(owned_stack, ints_stack, counts_arr):
        owned_pos, ints_local, n_owned, migrate = gx.redistribute(
            owned_stack.squeeze(0), counts_arr[0], config, "i",
            owned_ints=ints_stack.squeeze(0))
        ghost_data, n_ghost, exch, plan = gx.ghost_exchange(owned_pos, n_owned, config, "i")
        ints_nodes = gx.exchange_apply(plan, ints_local, config, "i")
        ghost_ints = lax.dynamic_slice_in_dim(ints_nodes, n_owned, config.max_ghost, axis=0)
        node_idx, snd, rcv, node_species, emask, overflow = gx.ghost_exchange_subgraph(
            owned_pos, ghost_data, ints_local[:, 0], ghost_ints[:, 0],
            ints_local[:, 1], ghost_ints[:, 1], n_owned, n_ghost, box,
            cells, cell_cap, max_nb, max_edges, config, "i")
        flags = lax.psum(jnp.stack([
            migrate[0] | exch[0],
            jnp.any(migrate[1:]) | exch[1] | overflow]).astype(jnp.int32), "i")
        stacked = jax.tree.map(lambda x: x[None], (owned_pos, ints_local, n_owned, plan,
                                                   snd, rcv, node_species, emask))
        return (*stacked, flags)

    @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 8,
             out_specs=(P(), P("i"), P("i")), check_vma=False)
    def step(owned_stack, ints_stack, counts_arr, plan, snd, rcv, node_species, emask):
        owned_pos, n_owned = owned_stack.squeeze(0), counts_arr[0]
        plan, snd, rcv, node_species, emask = jax.tree.map(
            lambda x: x.squeeze(0), (plan, snd, rcv, node_species, emask))
        E, g = jax.value_and_grad(lambda p: ghost_energy(
            node_energy_fn, p, n_owned, node_species, snd, rcv, emask, plan,
            box, config, "i"))(owned_pos)
        return lax.psum(E, "i"), g[None], counts_arr

    rebuild_j, step_j = jax.jit(rebuild), jax.jit(step)
    with highest_precision():
        *state, flags = rebuild_j(owned_m, ints_m, counts_m)
        if int(flags[0]):
            raise RuntimeError("an owned atom lies outside its tile after redistribute")
        if int(flags[1]):
            raise RuntimeError("ghost exchange overflow")
        E_par, g_frac, counts_out = step_j(*state)

    # dE/dr_frac -> dE/dr_real: g_frac_row = g_real_row @ box, box columns are
    # lattice vectors, so g_real_row = g_frac_row @ inv(box).
    inv_box = np.linalg.inv(np.asarray(box))
    F_owned = -np.asarray(g_frac) @ inv_box
    owned_ints_out, counts_out = np.asarray(state[1]), np.asarray(counts_out)
    F = np.zeros((n_atoms, 3))
    seen = np.zeros(n_atoms, bool)
    for d in range(n_ranks):
        live = int(counts_out[d])
        gid = owned_ints_out[d, :live, 1]
        F[gid] = F_owned[d, :live]
        seen[gid] = True
    assert seen.all(), f"{(~seen).sum()} atoms never owned"

    e_err = abs(float(E_par) - E_ref)
    f_err = float(np.max(np.abs(F - F_ref)))
    if verbose:
        print(f"n_atoms={n_atoms} ranks={n_ranks}  E_ref={E_ref:.12f}  E_par={float(E_par):.12f}")
        print(f"  |dE| = {e_err:.3e}  ({e_err / abs(E_ref):.3e} rel)")
        print(f"  max|dF| = {f_err:.3e}   max|F_ref| = {np.max(np.abs(F_ref)):.4f}")

    timing = None
    if steps:
        f = lambda: step_j(*state)[1]
        with highest_precision():
            for _ in range(warmup):
                jax.block_until_ready(f())
            t0 = time.perf_counter()
            for _ in range(steps):
                jax.block_until_ready(f())
            timing = (time.perf_counter() - t0) / steps
            g = lambda: rebuild_j(owned_m, ints_m, counts_m)[-1]
            for _ in range(warmup):
                jax.block_until_ready(g())
            t0 = time.perf_counter()
            for _ in range(steps):
                jax.block_until_ready(g())
            t_rebuild = (time.perf_counter() - t0) / steps
        if verbose:
            print(f"  step   {timing * 1e3:.3f} ms   ({1 / timing:.1f} steps/s)")
            print(f"  rebuild {t_rebuild * 1e3:.3f} ms")
    return dict(n_atoms=n_atoms, ranks=n_ranks, e_err=e_err, f_err=f_err, t_step=timing)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fixtures/si_fitted.npz")
    p.add_argument("--rep", type=int, default=4)
    p.add_argument("--ranks", type=int, nargs="+", default=[1])
    p.add_argument("--skin", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    a = p.parse_args()
    for n in a.ranks:
        print(f"=== ranks={n} ===")
        run(a.model, a.rep, n, a.skin, a.steps, a.warmup)


if __name__ == "__main__":
    main()
