"""Export the fitted Si ACE model as a lammps-jax bundle.

Contract notes (cpp/lammps_jax_model.h ModelContract):
  * LAMMPS supplies the neighbour list; positions span nlocal+nghost and there
    is NO cell -- ghosts carry periodicity.  Our core is edge-vector based, so
    this is the same code path as the ASE calculator, with shifts=None.
  * Static shapes: padded edges carry senders==receivers==max_atoms and
    edge_mask False.  Those indices are OUT OF BOUNDS for a (max_atoms, 3)
    positions array, so gather with where(mask, idx, 0) as the nequip template
    does, then overwrite the padded vectors.
  * Padded edges are parked AT THE CUTOFF, where the envelope vanishes and the
    gradient stays defined; a zero pad NaNs it (stage1 tests/test_padding.py).
  * float64 requires jax_enable_x64 set BEFORE export_model, or the traced
    program silently truncates.
  * custom_call_targets is empty: acejax uses a pure-JAX harmonic recursion, so
    no FFI handler needs registering at run time.
"""
import sys

import jax

jax.config.update("jax_enable_x64", True)   # before export_model, per its check
import jax.numpy as jnp
import numpy as np
from lammps_jax.export import export_model

sys.path.insert(0, "/home/eng/essswb/si-ace/stage1")
from acejax import load

NPZ = "/home/eng/essswb/si-ace/stage1/si_fitted.npz"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/eng/essswb/si-ace/si_ace.lammps-jax.json"
MAX_ATOMS = int(sys.argv[2]) if len(sys.argv) > 2 else 2560
EDGES_PER_ATOM = int(sys.argv[3]) if len(sys.argv) > 3 else 64
MAX_EDGES = MAX_ATOMS * EDGES_PER_ATOM

model, meta, z = load(NPZ, dtype=jnp.float64)
RCUT = float(meta["rcut"])
N_SPECIES = len(meta["elements"])
print("model: elements", meta["elements"], "rcut", RCUT,
      "n_B", meta["n_B"], "lmax", meta["lmax"])
print("capacities: max_atoms", MAX_ATOMS, "max_edges", MAX_EDGES)


def energy_fn(positions, species, graph):
    """Per-atom energies. `species` is the LAMMPS type index (0-based)."""
    mask = graph.edge_mask
    centers = jnp.where(mask, graph.senders, 0)
    neighbors = jnp.where(mask, graph.receivers, 0)
    rij = positions[neighbors] - positions[centers]
    pad = jnp.asarray([RCUT, 0.0, 0.0], positions.dtype)
    rij = jnp.where(mask[:, None], rij, pad)
    node_z = jnp.clip(species, 0, N_SPECIES - 1).astype(jnp.int32)
    return model.site_energies(rij, node_z[centers], node_z[neighbors], centers,
                               positions.shape[0], node_z, mask)


export_model(
    energy_fn=energy_fn,
    path=OUT,
    max_atoms=MAX_ATOMS,
    max_edges=MAX_EDGES,
    cutoff=RCUT,
    unit_style="metal",
    precision="float64",
    force_output="atom-force",   # forces by autodiff from the energy
    newton="on",                 # energy exports are newton on only
    n_hops=1,
    n_species=N_SPECIES,
    custom_call_targets=(),      # pure-JAX harmonics: nothing to resolve
)
print("exported", OUT)
