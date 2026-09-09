"""Phases 3-4 gate.

From an ASE Atoms object -- own neighbour list, periodic images and all --
reproduce Julia's AtomsCalculators.energy_forces_virial on the same fitted
model to 1e-10 in f64: energy, forces AND virial.

Phase 1 validated the core against a Julia-supplied edge list.  This closes the
loop: matscipy-neighbours builds the edges here, so a wrong cutoff, a missing
periodic image or a shift-convention error would show up.
"""
import json
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from acejax import (ACECalculator, dense_graph, dense_to_sparse,
                    highest_precision, load, sparse_graph)

NPZ = pathlib.Path(__file__).parent.parent / "si_fitted.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="run export_model.jl first")
TOL = 1e-10


@pytest.fixture(scope="module")
def case():
    model, meta, z = load(NPZ)
    from ase import Atoms
    atoms = Atoms(numbers=np.asarray(z["test_Z"]),
                  positions=np.asarray(z["test_pos"]).T,
                  cell=np.asarray(z["test_cell"]).T,
                  pbc=np.asarray(z["test_pbc"]).astype(bool))
    return model, meta, z, atoms


def test_nlist_matches_julia(case):
    """The neighbour list itself, before any model evaluation."""
    model, meta, z, atoms = case
    g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                     atoms.get_pbc(), meta["rcut"])
    assert len(g.senders) == len(z["test_edge_i"]), "edge count differs from Julia"
    # same (i, rij) multiset -- ordering within a centre need not match
    def key(i, r):
        return sorted(zip(i.tolist(), [tuple(np.round(v, 9)) for v in r]))
    assert key(g.senders, g.rij) == key(np.asarray(z["test_edge_i"]),
                                        np.asarray(z["test_edge_rij"]).T)
    print(f"\n  edges {len(g.senders)}, sorted by i: {bool(np.all(np.diff(g.senders) >= 0))}")


def test_energy_forces_virial(case):
    model, meta, z, atoms = case
    g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                     atoms.get_pbc(), meta["rcut"])
    node_z = jnp.zeros(g.n_nodes, jnp.int32)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        E, F, V = model.energy_forces_virial(jnp.asarray(g.rij), node_z[send],
                                            node_z[recv], send, recv, g.n_nodes, node_z)
    eE = abs(float(E) - float(z["test_E"][0]))
    eF = np.max(np.abs(np.asarray(F) - np.asarray(z["test_F"]).T))
    eV = np.max(np.abs(np.asarray(V) - np.asarray(z["test_V"])))
    print(f"\n  energy |d| = {eE:.3e} eV")
    print(f"  forces |d| = {eF:.3e} eV/A   (scale {np.max(np.abs(z['test_F'])):.4f})")
    print(f"  virial |d| = {eV:.3e} eV     (scale {np.max(np.abs(z['test_V'])):.4f})")
    assert eE < TOL and eF < TOL and eV < TOL


def test_virial_sign_is_julia_convention(case):
    """Guard the sign explicitly: a flipped virial would still be 'close' in
    magnitude, so assert against the reference rather than |V|."""
    model, meta, z, atoms = case
    g = sparse_graph(atoms.get_positions(), atoms.get_cell().array,
                     atoms.get_pbc(), meta["rcut"])
    node_z = jnp.zeros(g.n_nodes, jnp.int32)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        _, _, V = model.energy_forces_virial(jnp.asarray(g.rij), node_z[send],
                                            node_z[recv], send, recv, g.n_nodes, node_z)
    Vref = np.asarray(z["test_V"])
    assert np.max(np.abs(np.asarray(V) - Vref)) < TOL
    assert np.max(np.abs(np.asarray(V) + Vref)) > 1.0, "sign test is degenerate"


def test_dense_and_sparse_pooling_agree(case):
    """The two neighbour-list layouts must give the same answer; neither is
    hard-wired, and lammps-jax needs sparse while dense avoids a scatter."""
    model, meta, z, atoms = case
    d = dense_graph(atoms.get_positions(), atoms.get_cell().array,
                    atoms.get_pbc(), meta["rcut"], 64)
    n, K = d.idx.shape
    node_z = jnp.zeros(d.n_nodes, jnp.int32)
    zi = jnp.zeros((n, K), jnp.int32)
    mask = jnp.asarray(d.mask)
    # padded slots must not be allowed to blow up: park them at the cutoff
    rij = np.where(d.mask[..., None], d.rij, np.array([meta["rcut"], 0.0, 0.0]))
    with highest_precision():
        e_dense = model.site_energies_dense(jnp.asarray(rij), zi, zi, mask, node_z)
    s = dense_to_sparse(d)
    send = jnp.asarray(s.senders)
    with highest_precision():
        e_sparse = model.site_energies(jnp.asarray(s.rij), node_z[send],
                                       node_z[jnp.asarray(s.receivers)], send,
                                       s.n_nodes, node_z)
    err = np.max(np.abs(np.asarray(e_dense) - np.asarray(e_sparse)))
    print(f"\n  dense vs sparse pooling: max|d| = {err:.3e}")
    assert err < 1e-11


def test_ase_calculator(case):
    model, meta, z, atoms = case
    atoms = atoms.copy()
    atoms.calc = ACECalculator(model, meta)
    eE = abs(atoms.get_potential_energy() - float(z["test_E"][0]))
    eF = np.max(np.abs(atoms.get_forces() - np.asarray(z["test_F"]).T))
    s = atoms.get_stress(voigt=False)
    eV = np.max(np.abs(-s * atoms.get_volume() - np.asarray(z["test_V"])))
    print(f"\n  ASE: |dE| = {eE:.3e}  |dF| = {eF:.3e}  |dV| = {eV:.3e}")
    assert eE < TOL and eF < TOL and eV < TOL
