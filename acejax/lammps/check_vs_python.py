#!/usr/bin/env python3
"""Does the LAMMPS bundle reproduce the acejax Python calculator?

Reads the configuration LAMMPS actually used out of its own dump, rebuilds the
periodic neighbour list, evaluates the same fitted model through acejax, and
compares energies and forces.  Reading the geometry back from the dump means
the two sides cannot silently disagree about what was evaluated.

The neighbour list is built here by explicit image enumeration rather than via
matscipy-neighbours, to keep the remote dependency set to jax + equinox + numpy
(matscipy-neighbours needs a build toolchain).  tests/test_efv.py
already pins matscipy's list against Julia's, so this is a cross-check by an
independent route, not a weakening.

Usage: check_vs_python.py <lammps.dump> <npz> [pe_from_lammps]
"""
import itertools
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from acejax import highest_precision, load


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


def periodic_edges(pos, cell, rcut):
    """Full pairing over periodic images; cell assumed diagonal-dominant."""
    n = len(pos)
    reps = [int(np.ceil(rcut / cell[k, k])) for k in range(3)]
    ii, jj, rr = [], [], []
    for s in itertools.product(*[range(-r, r + 1) for r in reps]):
        shift = np.array(s) @ cell
        d = (pos[None, :, :] + shift) - pos[:, None, :]
        r = np.linalg.norm(d, axis=-1)
        keep = (r < rcut) & (r > 1e-10)
        a, b = np.where(keep)
        ii.append(a); jj.append(b); rr.append(d[a, b])
    ii = np.concatenate(ii); jj = np.concatenate(jj); rr = np.concatenate(rr)
    o = np.argsort(ii, kind="stable")
    return ii[o].astype(np.int32), jj[o].astype(np.int32), rr[o]


def main():
    dump, npz = sys.argv[1], sys.argv[2]
    pe_lammps = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    pos, F_lmp, cell = read_dump(dump)
    model, meta, _ = load(npz)
    rcut = float(meta["rcut"])
    ii, jj, rij = periodic_edges(pos, cell, rcut)
    n = len(pos)
    nz = jnp.zeros(n, jnp.int32)
    send, recv = jnp.asarray(ii), jnp.asarray(jj)
    with highest_precision():
        E, F, V = model.energy_forces_virial(jnp.asarray(rij), nz[send], nz[recv],
                                             send, recv, n, nz)
    E_py = float(E); F_py = np.asarray(F)
    dF = np.max(np.abs(F_py - F_lmp))
    fs = np.max(np.abs(F_lmp))
    print(f"  atoms {n}  edges {len(ii)}  cell diag {np.diag(cell).round(4).tolist()}")
    print(f"  python E = {E_py:.10f} eV")
    ok_E = True
    if pe_lammps is not None:
        dE = abs(E_py - pe_lammps)
        print(f"  lammps E = {pe_lammps:.10f} eV   |dE| = {dE:.3e}  ({dE/abs(E_py):.2e} rel)")
        ok_E = dE < 1e-6
    print(f"  max|dF| = {dF:.3e} eV/A   (|F| scale {fs:.4f}, rel {dF/fs:.2e})")
    ok = ok_E and dF < 1e-9
    print("  " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
