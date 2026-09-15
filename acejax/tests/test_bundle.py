"""The LAMMPS bundle's energy_fn, tested without LAMMPS or lammps-jax.

`lammps/export_bundle.py::build` is the only thing the plugin path adds on top
of the core; if it is right here, the LAMMPS-side check (lammps/check_vs_python.py)
only has to catch plugin-contract mismatches.
"""
import importlib.util
import pathlib
import types

import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from acejax import highest_precision, load, sparse_graph

ROOT = pathlib.Path(__file__).parent.parent


def _export_bundle():
    spec = importlib.util.spec_from_file_location(
        "export_bundle", ROOT / "lammps" / "export_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_importable_without_lammps_jax():
    mod = _export_bundle()
    assert callable(mod.build)


def _cluster(npz, n_pad_atoms=37, n_pad_edges=101):
    """Non-periodic cluster from the fixture geometry, so positions alone define
    rij (the bundle computes rij from positions; periodic images would not)."""
    model, meta, z = load(npz)
    pos = np.asarray(z["test_pos"]).T
    n = len(pos)
    cell = np.eye(3) * (np.ptp(pos, axis=0).max() + 2 * meta["rcut"] + 1)
    g = sparse_graph(pos, cell, np.zeros(3, bool), meta["rcut"])
    max_atoms = n + n_pad_atoms
    E = len(g.senders)
    senders = np.concatenate([g.senders, np.full(n_pad_edges, max_atoms)]).astype(np.int32)
    receivers = np.concatenate([g.receivers, np.full(n_pad_edges, max_atoms)]).astype(np.int32)
    edge_mask = np.concatenate([np.ones(E, bool), np.zeros(n_pad_edges, bool)])
    graph = types.SimpleNamespace(senders=jnp.asarray(senders),
                                  receivers=jnp.asarray(receivers),
                                  edge_mask=jnp.asarray(edge_mask))
    positions = jnp.asarray(np.concatenate([pos, np.zeros((n_pad_atoms, 3))]))
    species = jnp.concatenate([species_index(z), jnp.zeros(n_pad_atoms, jnp.int32)])
    # reference: the core evaluated on exactly the real edges
    node_z = species_index(z)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        e_ref = model.site_energies(jnp.asarray(g.rij), node_z[send], node_z[recv],
                                    send, n, node_z)
    return positions, species, graph, np.asarray(e_ref), n, max_atoms


def test_local_axis_matches_full_axis(npz):
    mod = _export_bundle()
    positions, species, graph, e_ref, n, max_atoms = _cluster(npz)
    full = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=None)[0]
    local = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=n + 3)[0]
    with highest_precision():
        e_full = np.asarray(full(positions, species, graph))
        e_loc = np.asarray(local(positions, species, graph))
    assert e_full.shape == e_loc.shape == (max_atoms,)
    assert np.abs(e_full[:n] - e_ref).max() < 1e-12
    assert np.abs(e_loc[:n] - e_ref).max() < 1e-12
    assert np.all(e_loc[n:] == 0.0)
    assert np.all(np.isfinite(e_loc))


def test_local_axis_forces_match(npz):
    """Forces come from autodiff of the summed energy in the plugin; the local
    axis must not change the gradient w.r.t. positions, including ghost rows."""
    mod = _export_bundle()
    positions, species, graph, e_ref, n, max_atoms = _cluster(npz)
    full = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=None)[0]
    local = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=n + 3)[0]
    with highest_precision():
        gf = jax.grad(lambda p: jnp.sum(full(p, species, graph)))(positions)
        gl = jax.grad(lambda p: jnp.sum(local(p, species, graph)))(positions)
    assert np.abs(np.asarray(gf) - np.asarray(gl)).max() < 1e-12


def test_local_axis_overflow_is_nan_not_silent(npz):
    """segment_sum silently drops ids >= num_segments; the guard must turn an
    undersized max_local into NaN, never a plausible wrong energy."""
    mod = _export_bundle()
    positions, species, graph, e_ref, n, max_atoms = _cluster(npz)
    small = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=n - 1)[0]
    with highest_precision():
        e = np.asarray(small(positions, species, graph))
    assert np.all(np.isnan(e[:n - 1]))


def test_max_local_above_max_atoms_rejected(npz):
    mod = _export_bundle()
    with pytest.raises(ValueError, match="max_local"):
        mod.build(npz, max_atoms=100, edges_per_atom=1, max_local=101)
