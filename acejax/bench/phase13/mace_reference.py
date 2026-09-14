#!/usr/bin/env python3
"""Independent reference for the MACE routes: torch MACE on LAMMPS's own geometry.

UNTESTED -- written without host access; read before running.

Reads the configuration back out of the LAMMPS dump rather than rebuilding it,
so the two sides cannot silently disagree about what was evaluated -- the same
discipline check_vs_python.py uses for the ACE route, and the reason Phase 6's
rank comparison had to be rebuilt (`displace_atoms random` is decomposition
dependent).

Prints the energy difference per atom and max|dF|, and exits non-zero if either
exceeds the tolerance.  Both MACE routes run in float32, torch here runs in
float64, so the tolerance is a PRECISION tolerance, not an agreement tolerance:
1e-4 eV/atom and 5e-3 eV/A are the thresholds lammps-jax's own exporter
preflight uses.

Usage: mace_reference.py <lammps.dump> <mace.model> [--etol E] [--ftol F]
                         [--calculator symmetrix --json model.json]
"""
import argparse

import numpy as np


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
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("model")
    p.add_argument("--pe", type=float, default=None,
                   help="LAMMPS PotEng for the same frame, if comparing energies")
    p.add_argument("--etol", type=float, default=1e-4, help="eV per atom")
    p.add_argument("--ftol", type=float, default=5e-3, help="eV/A")
    p.add_argument("--calculator", choices=["mace", "symmetrix"], default="mace")
    a = p.parse_args()

    from ase import Atoms

    pos, frc, cell = read_dump(a.dump)
    atoms = Atoms("Si" + str(len(pos)), positions=pos, cell=cell, pbc=True)

    if a.calculator == "mace":
        from mace.calculators import MACECalculator
        atoms.calc = MACECalculator(model_paths=a.model, device="cpu",
                                    default_dtype="float64")
    else:
        from symmetrix import Symmetrix
        atoms.calc = Symmetrix(a.model)

    e_ref = atoms.get_potential_energy()
    f_ref = atoms.get_forces()

    df = float(np.max(np.abs(f_ref - frc)))
    fs = float(np.max(np.abs(f_ref)))
    print(f"  atoms {len(pos)}   |F| scale {fs:.4f} eV/A")
    print(f"  max|dF| = {df:.3e} eV/A   (tol {a.ftol:.1e})")
    ok = df <= a.ftol
    if a.pe is not None:
        de = abs(e_ref - a.pe) / len(pos)
        print(f"  dE/atom = {de:.3e} eV   (ref {e_ref:.9f}, lammps {a.pe:.9f}, "
              f"tol {a.etol:.1e})")
        ok = ok and de <= a.etol
    print("  PASS" if ok else "  FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
