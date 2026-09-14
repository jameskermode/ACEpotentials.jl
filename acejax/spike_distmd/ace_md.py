"""THROWAWAY. Distributed NVE MD for acejax on lammps_jax.dist.parallel.

`ace_dist.py` proves one force evaluation goes through the plumbing.  This runs
an actual trajectory: `--inner` velocity-Verlet steps inside a single jitted
`shard_map` call, so the inner loop never returns to the host -- the property
the LAMMPS route cannot have (the pair style pays two MPI_Allreduce and an
exec.fence every step).  Between calls the host triggers a full rebuild
(redistribute + ghost exchange + cell-list edge build), which is the reneighbour
cadence.

Velocities ride in columns 3:6 of `owned_data`, so `redistribute` migrates them
with their atoms; `ghost_exchange` only ever sees the position columns.

Validation: after the trajectory, gather positions and recompute PE and forces
with `acejax.ACECalculator`.  Energy conservation is reported as drift per atom
per picosecond.
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

from ase import units
from ase.build import bulk

import acejax
from acejax.model import highest_precision
from lammps_jax.dist.parallel import ghost_exchange as gx

from ace_dist import ace_node_energies, ghost_energy


def run(model_path, rep, n_ranks, skin, n_outer, n_inner, dt_fs, temperature,
        f32=False, quiet=False):
    dtype = jnp.float32 if f32 else jnp.float64
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(rep)
    rng = np.random.default_rng(0)
    atoms.positions += 0.02 * rng.standard_normal(atoms.positions.shape)
    atoms.wrap()
    n_atoms = len(atoms)
    mass = float(atoms.get_masses()[0])

    calc = acejax.ACECalculator(model_path, dtype=dtype)
    model, cutoff = calc.model, calc.cutoff
    box_np = np.asarray(atoms.get_cell().array).T      # columns = lattice vectors
    box = jnp.asarray(box_np, dtype)
    inv_box = np.linalg.inv(box_np)

    R = np.asarray(atoms.get_scaled_positions() % 1.0)
    species = np.asarray(calc._species_index(atoms.get_atomic_numbers()), np.int32)
    ids = np.arange(n_atoms, dtype=np.int32)

    config = gx.create_config(n_ranks, n_atoms, box_np, cutoff,
                              capacity_mult=1.5, skin=skin)
    cells, cell_cap, max_nb, max_edges = gx.subgraph_params(
        config, box_np, n_atoms, capacity_mult=2.0)
    print(f"grid={config.grid}  atom slots {config.max_owned * n_ranks}/{n_atoms} "
          f"= {config.max_owned * n_ranks / n_atoms:.2f}x  max_ghost={config.max_ghost}  "
          f"edge slots {max_edges * n_ranks}")

    mesh = Mesh(jax.devices()[:n_ranks], axis_names=("i",))
    sharding = NamedSharding(mesh, P("i"))
    owned, owned_ints, counts, _ = gx.pack_tiles(
        np.asarray(R, np.float64), config, ints=np.stack([species, ids], 1))

    v_real = rng.normal(size=(n_atoms, 3)) * np.sqrt(units.kB * temperature / mass)
    v_real -= v_real.mean(0)
    v_frac_all = v_real @ inv_box.T
    v_packed = np.zeros_like(owned)
    for d in range(n_ranks):
        live = int(counts[d])
        v_packed[d, :live] = v_frac_all[owned_ints[d, :live, 1]]
    data0 = np.concatenate([owned, v_packed], axis=-1)

    put = lambda x: jax.device_put(jnp.asarray(x, dtype) if x.dtype.kind == "f"
                                   else jnp.asarray(x), sharding)
    data_m, ints_m, counts_m = (put(data0), put(owned_ints), put(counts))

    rcut_env = float(np.max(np.asarray(model.pair_envelope)[..., 0]))
    node_energy_fn = partial(ace_node_energies, model, rcut_env)
    # F_real_row = -g_frac_row @ inv(box);  a_frac_row = F_real_row @ inv(box).T / m
    accel_map = jnp.asarray(-inv_box @ inv_box.T / mass, dtype)
    dt = dt_fs * units.fs

    @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 3,
             out_specs=(P("i"),) * 8 + (P(),), check_vma=False)
    def rebuild(data_stack, ints_stack, counts_arr):
        data, ints_local, n_owned, migrate = gx.redistribute(
            data_stack.squeeze(0), counts_arr[0], config, "i", owned_ints=ints_stack.squeeze(0))
        owned_pos = data[:, :3]
        ghost_data, n_ghost, exch, plan = gx.ghost_exchange(owned_pos, n_owned, config, "i")
        ints_nodes = gx.exchange_apply(plan, ints_local, config, "i")
        ghost_ints = lax.dynamic_slice_in_dim(ints_nodes, n_owned, config.max_ghost, axis=0)
        _, snd, rcv, node_species, emask, overflow = gx.ghost_exchange_subgraph(
            owned_pos, ghost_data, ints_local[:, 0], ghost_ints[:, 0],
            ints_local[:, 1], ghost_ints[:, 1], n_owned, n_ghost, box,
            cells, cell_cap, max_nb, max_edges, config, "i")
        flags = lax.psum(jnp.stack([migrate[0] | exch[0],
                                    jnp.any(migrate[1:]) | exch[1] | overflow]).astype(jnp.int32), "i")
        stacked = jax.tree.map(lambda x: x[None], (data, ints_local, n_owned, plan,
                                                   snd, rcv, node_species, emask))
        return (*stacked, flags)

    @partial(shard_map, mesh=mesh, in_specs=(P("i"),) * 8,
             out_specs=(P("i"), P(), P(), P("i")), check_vma=False)
    def trajectory(data_stack, ints_stack, counts_arr, plan, snd, rcv, node_species, emask):
        data, n_owned = data_stack.squeeze(0), counts_arr[0]
        plan, snd, rcv, node_species, emask = jax.tree.map(
            lambda x: x.squeeze(0), (plan, snd, rcv, node_species, emask))
        pos, v = data[:, :3], data[:, 3:6]
        live = (jnp.arange(pos.shape[0]) < n_owned)[:, None]

        def eg(p):
            return jax.value_and_grad(lambda q: ghost_energy(
                node_energy_fn, q, n_owned, node_species, snd, rcv, emask, plan,
                box, config, "i"))(p)

        def body(_, carry):
            pos, v, g = carry
            v = v + 0.5 * dt * (g @ accel_map) * live
            pos = pos + dt * v * live
            g = eg(pos)[1]
            v = v + 0.5 * dt * (g @ accel_map) * live
            return (pos, v, g)

        pos, v, _ = lax.fori_loop(0, n_inner, body, (pos, v, eg(pos)[1]))
        E, g_final = eg(pos)
        v_real_loc = v @ box.T
        ke = 0.5 * mass * jnp.sum(jnp.where(live, v_real_loc ** 2, 0.0))
        out = jnp.concatenate([pos, v], axis=-1)
        return (out[None], lax.psum(E, "i"), lax.psum(ke, "i"), g_final[None])

    rebuild_j, trajectory_j = jax.jit(rebuild), jax.jit(trajectory)

    with highest_precision():
        *state, flags = rebuild_j(data_m, ints_m, counts_m)
        assert not int(flags[0]) and not int(flags[1]), f"rebuild flags {flags}"
        e0 = None
        t_step, t_rebuild, n_timed = 0.0, 0.0, 0
        for outer in range(n_outer):
            t0 = time.perf_counter()
            data_out, E, KE, g_final = trajectory_j(state[0], *state[1:])
            jax.block_until_ready((data_out, E))
            dt_traj = time.perf_counter() - t0
            tot = float(E) + float(KE)
            e0 = tot if e0 is None else e0
            T = 2 * float(KE) / (3 * n_atoms * units.kB)
            if not quiet:
                print(f"  leg {outer:2d}  PE={float(E):.6f}  KE={float(KE):.4f}  T={T:6.1f} K  "
                      f"Etot={tot:.6f}  drift={tot - e0:+.3e} eV  ({dt_traj / n_inner * 1e3:.3f} ms/step)")
            prev = state
            t1 = time.perf_counter()
            *state, flags = rebuild_j(data_out, prev[1], prev[2])
            jax.block_until_ready(state[0])
            dt_rb = time.perf_counter() - t1
            assert not int(flags[0]) and not int(flags[1]), f"rebuild flags {flags} at leg {outer}"
            if outer:                        # leg 0 carries compilation
                t_step += dt_traj
                t_rebuild += dt_rb
                n_timed += 1

    per_step = t_step / max(n_timed, 1) / n_inner
    n_steps = n_outer * n_inner
    ps = n_steps * dt_fs / 1000.0
    print(f"\n{n_steps} steps ({n_outer} legs of {n_inner}), {ps:.3f} ps, "
          f"{n_atoms} atoms, {n_ranks} rank(s), {'f32' if f32 else 'f64'}")
    print(f"  Etot drift  {(tot - e0) / n_atoms / max(ps, 1e-12) * 1e3:+.4f} meV/atom/ps")
    print(f"  inner step  {per_step * 1e3:.4f} ms   ({n_atoms / per_step:.3e} atom-steps/s)")
    print(f"  rebuild     {t_rebuild / max(n_timed, 1) * 1e3:.4f} ms  "
          f"(amortised over {n_inner}: {t_rebuild / max(n_timed, 1) / n_inner * 1e3:.4f} ms/step)")

    # ---- validate against the ASE calculator at the final configuration
    # `state` is post-rebuild, `prev` is the ordering `g_final` and `data_out`
    # were produced in; redistribute permutes rows, so the two index maps differ
    # the moment an atom migrates.  Gather each from its own map.
    oi, cts = np.asarray(prev[1]), np.asarray(prev[2])
    pos_dev = np.asarray(data_out).reshape(n_ranks, -1, 6)[..., :3]
    final = np.zeros((n_atoms, 3))
    for d in range(n_ranks):
        final[oi[d, :int(cts[d]), 1]] = pos_dev[d, :int(cts[d])]
    at2 = atoms.copy()
    at2.set_scaled_positions(final % 1.0)
    at2.calc = calc
    E_ase, F_ase = float(at2.get_potential_energy()), np.asarray(at2.get_forces())

    F = np.zeros((n_atoms, 3))
    Fg = -np.asarray(g_final).reshape(n_ranks, -1, 3) @ inv_box
    for d in range(n_ranks):
        F[oi[d, :int(cts[d]), 1]] = Fg[d, :int(cts[d])]
    print(f"  final PE    dist {float(E):.9f}   ASE {E_ase:.9f}   |d| {abs(float(E) - E_ase):.3e}")
    print(f"  final |dF|  max {np.max(np.abs(F - F_ase)):.3e}   (max|F| = {np.max(np.abs(F_ase)):.4f})")
    return dict(per_step=per_step, n_atoms=n_atoms,
                e_err=abs(float(E) - E_ase), f_err=float(np.max(np.abs(F - F_ase))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="fixtures/si_fitted.npz")
    p.add_argument("--rep", type=int, default=4)
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--skin", type=float, default=1.0)
    p.add_argument("--outer", type=int, default=6)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    run(a.model, a.rep, a.ranks, a.skin, a.outer, a.inner, a.dt, a.temperature,
        a.f32, a.quiet)


if __name__ == "__main__":
    main()
