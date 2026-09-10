#!/usr/bin/env python3
"""Export a fitted ACE model as a lammps-jax bundle.

  ./export_bundle.py --npz ../si_fitted.npz --out si_ace.lammps-jax.json

Contract notes (cpp/lammps_jax_model.h ModelContract):
  * LAMMPS supplies the neighbour list; positions span nlocal+nghost and there
    is NO cell -- ghosts carry periodicity.  Our core is edge-vector based, so
    this is the same code path as the ASE calculator, with shifts=None.
  * Static shapes: padded edges carry senders==receivers==max_atoms and
    edge_mask False.  Those indices are OUT OF BOUNDS for a (max_atoms, 3)
    positions array, so gather with where(mask, idx, 0) as the nequip template
    does, then overwrite the padded vectors.
  * Padded edges are parked AT THE CUTOFF, where the envelope vanishes and the
    gradient stays defined; a zero pad NaNs it (tests/test_padding.py).
  * float64 requires jax_enable_x64 set BEFORE export_model, or the traced
    program silently truncates.
  * custom_call_targets is empty: acejax uses a pure-JAX harmonic recursion, so
    no FFI handler needs registering at run time.

Export with the same jax as the runtime PJRT plugin ships (0.11.1 in
/storage/eng/essswb/venvs/lammps-jax).
"""
import argparse
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)   # before export_model, per its check
import jax.numpy as jnp
from lammps_jax.export import export_model

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # so `import acejax` works from anywhere
from acejax import load


def build(npz, max_atoms=2560, edges_per_atom=64, precision="float64",
          a2b_sparse=False):
    """Return (energy_fn, model, meta, rcut, max_atoms, max_edges)."""
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    model, meta, _ = load(npz, dtype=dtype, a2b_sparse=a2b_sparse)
    rcut = float(meta["rcut"])
    n_species = len(meta["elements"])
    max_edges = max_atoms * edges_per_atom

    def energy_fn(positions, species, graph):
        """Per-atom energies. `species` is the LAMMPS type index (0-based)."""
        mask = graph.edge_mask
        centers = jnp.where(mask, graph.senders, 0)
        neighbors = jnp.where(mask, graph.receivers, 0)
        rij = positions[neighbors] - positions[centers]
        pad = jnp.asarray([rcut, 0.0, 0.0], positions.dtype)
        rij = jnp.where(mask[:, None], rij, pad)
        node_z = jnp.clip(species, 0, n_species - 1).astype(jnp.int32)
        return model.site_energies(rij, node_z[centers], node_z[neighbors],
                                   centers, positions.shape[0], node_z, mask)

    return energy_fn, model, meta, rcut, max_atoms, max_edges


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=pathlib.Path,
                   default=HERE.parent / "fixtures" / "si_fitted.npz",
                   help="exported model npz (default: fixtures/si_fitted.npz)")
    p.add_argument("--out", type=pathlib.Path,
                   default=HERE / "si_ace.lammps-jax.json")
    p.add_argument("--max-atoms", type=int, default=2560)
    p.add_argument("--edges-per-atom", type=int, default=64)
    p.add_argument("--a2b-sparse", action="store_true",
                   help="gather/segment-sum A2B contraction instead of a dense "
                        "matmul; A2B is ~0.07%% occupied at large basis")
    p.add_argument("--precision", choices=["float64", "float32"], default="float64",
                   help="bundle precision; f32 is ~2.5x faster here because the "
                        "descriptor is memory-bound, not FLOP-bound")
    a = p.parse_args()
    if not a.npz.exists():
        p.error(f"npz not found: {a.npz}")

    energy_fn, model, meta, rcut, max_atoms, max_edges = build(
        a.npz, a.max_atoms, a.edges_per_atom, a.precision, a.a2b_sparse)
    print("model:", a.npz, "| elements", meta["elements"], "rcut", rcut,
          "n_B", meta["n_B"], "lmax", meta["lmax"],
          "| radial", meta["radial_kind"], "| Y", meta["ybasis_kind"])
    print("capacities: max_atoms", max_atoms, "max_edges", max_edges,
          "| precision", a.precision)

    export_model(
        energy_fn=energy_fn,
        path=a.out,
        max_atoms=max_atoms,
        max_edges=max_edges,
        cutoff=rcut,
        unit_style="metal",
        precision=a.precision,
        force_output="atom-force",   # forces by autodiff from the energy
        newton="on",                 # energy exports are newton on only
        n_hops=1,
        n_species=len(meta["elements"]),
        custom_call_targets=(),      # pure-JAX harmonics: nothing to resolve
    )
    print("exported", a.out)


if __name__ == "__main__":
    main()
