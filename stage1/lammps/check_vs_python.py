#!/usr/bin/env python3
"""The Phase 6 gate: does the LAMMPS bundle reproduce our Python calculator?

Reads the configuration LAMMPS actually used out of its own dump, rebuilds it
as an ASE Atoms, evaluates the same fitted model through acejax, and compares
energies and forces.  Reading the geometry back from the dump means the two
sides cannot silently disagree about what configuration was evaluated.

Usage: check_vs_python.py <lammps.dump> <npz> [pe_from_lammps]
"""
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np
from ase import Atoms

sys.path.insert(0, "/home/eng/essswb/si-ace/stage1")
from acejax import ACECalculator, load


def read_dump(path):
    with open(path) as f:
        lines = f.read().splitlines()
    i = len(lines) - 1 - lines[::-1].index("ITEM: NUMBER OF ATOMS")
    n = int(lines[i + 1])
    b = len(lines) - 1 - next(k for k, l in enumerate(lines[::-1])
                              if l.startswith("ITEM: BOX BOUNDS"))
    bounds = np.array([[float(v) for v in lines[b + 1 + k].split()] for k in range(3)])
    j = len(lines) - 1 - next(k for k, l in enumerate(lines[::-1])
                              if l.startswith("ITEM: ATOMS"))
    cols = lines[j].split()[2:]
    rows = np.array([[float(v) for v in l.split()] for l in lines[j + 1:j + 1 + n]])
    idx = {c: k for k, c in enumerate(cols)}
    rows = rows[np.argsort(rows[:, idx["id"]])]
    pos = rows[:, [idx[c] for c in "xyz"]]
    frc = rows[:, [idx[f"f{c}"] for c in "xyz"]]
    cell = np.diag(bounds[:, 1] - bounds[:, 0])
    return pos, frc, cell


def main():
    dump, npz = sys.argv[1], sys.argv[2]
    pe_lammps = float(sys.argv[3]) if len(sys.argv) > 3 else None
    pos, F_lmp, cell = read_dump(dump)
    model, meta, _ = load(npz)
    atoms = Atoms(numbers=np.full(len(pos), int(meta["elements"][0])),
                  positions=pos, cell=cell, pbc=True)
    atoms.calc = ACECalculator(model, meta)
    E_py = atoms.get_potential_energy()
    F_py = atoms.get_forces()
    dF = np.max(np.abs(F_py - F_lmp))
    print(f"  atoms {len(pos)}   cell diag {np.diag(cell).round(4).tolist()}")
    print(f"  python E = {E_py:.10f} eV")
    if pe_lammps is not None:
        print(f"  lammps E = {pe_lammps:.10f} eV   |dE| = {abs(E_py-pe_lammps):.3e}"
              f"   ({abs(E_py-pe_lammps)/abs(E_py):.2e} rel)")
    print(f"  max|dF| = {dF:.3e} eV/A   (|F| scale {np.max(np.abs(F_lmp)):.4f})")
    ok = dF < 1e-9 and (pe_lammps is None or abs(E_py - pe_lammps) < 1e-6)
    print("  " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
