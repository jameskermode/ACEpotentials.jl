"""ASE calculator: the Phase 4 closing of the loop from a real structure.

Builds the neighbour list with matscipy-neighbours, evaluates the edge-vector
core, and returns energy, forces and stress.  Nothing here knows about cells
beyond handing them to the neighbour list -- the strain derivative acts on edge
vectors, so the same code path serves LAMMPS, where there is no cell at all.
"""

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .model import highest_precision
from .nlist import sparse_graph


class ACECalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model, meta, cutoff=None, dtype=None, **kw):
        super().__init__(**kw)
        self.model = model
        self.meta = meta
        self.cutoff = float(cutoff if cutoff is not None else meta["rcut"])
        self._z2i = {int(z): i for i, z in enumerate(meta["elements"])}
        self.dtype = dtype

    def _species_index(self, numbers):
        try:
            return np.array([self._z2i[int(z)] for z in numbers], np.int32)
        except KeyError as e:
            raise ValueError(
                f"element Z={e.args[0]} not in model elements {self.meta['elements']}") from e

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        import jax.numpy as jnp

        g = sparse_graph(self.atoms.get_positions(), self.atoms.get_cell().array,
                         self.atoms.get_pbc(), self.cutoff)
        node_z = jnp.asarray(self._species_index(self.atoms.get_atomic_numbers()))
        rij = jnp.asarray(g.rij, dtype=self.dtype)
        send = jnp.asarray(g.senders)
        recv = jnp.asarray(g.receivers)
        with highest_precision():
            E, F, V = self.model.energy_forces_virial(
                rij, node_z[send], node_z[recv], send, recv, g.n_nodes, node_z)
        E = float(E)
        self.results["energy"] = E
        self.results["free_energy"] = E
        self.results["forces"] = np.asarray(F)
        vol = self.atoms.get_volume()
        if vol > 0:
            # ASE stress is -virial/volume, Voigt-ordered
            s = -np.asarray(V) / vol
            self.results["stress"] = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
