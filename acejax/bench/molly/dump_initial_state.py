"""Write the EXACT initial state that spike_distmd/ace_md.py builds, so the
Julia/Molly engine can start from the same configuration and velocities.

Replicates ace_md.py's recipe verbatim (same RNG object, same call order):

    atoms = bulk("Si","diamond",a=5.43,cubic=True).repeat(rep)
    rng   = np.random.default_rng(0)
    atoms.positions += 0.02 * rng.standard_normal(atoms.positions.shape)
    atoms.wrap()
    ...
    v_real = rng.normal(size=(n,3)) * sqrt(units.kB * T / mass)
    v_real -= v_real.mean(0)

Nothing between those two draws touches `rng` in ace_md.py, so the stream is
identical.  Verified by --check, which re-imports ace_md's own construction.

Velocities are written in Angstrom/fs (ASE's internal velocity unit is
Angstrom/t_ASE with t_ASE = 1/units.fs fs), which is unambiguous for Julia.
"""
import argparse
import numpy as np
from ase import units
from ase.build import bulk


def build(rep, temperature=300.0):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat(rep)
    rng = np.random.default_rng(0)
    atoms.positions += 0.02 * rng.standard_normal(atoms.positions.shape)
    atoms.wrap()
    n = len(atoms)
    mass = float(atoms.get_masses()[0])
    v_real = rng.normal(size=(n, 3)) * np.sqrt(units.kB * temperature / mass)
    v_real -= v_real.mean(0)
    return atoms, v_real, mass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    atoms, v_ase, mass = build(a.rep, a.temperature)
    np.savez(
        a.out,
        positions=np.asarray(atoms.positions, np.float64),   # Angstrom, wrapped
        cell=np.asarray(atoms.get_cell().array, np.float64), # rows = lattice vectors
        numbers=np.asarray(atoms.get_atomic_numbers(), np.int32),
        velocities_ase=np.asarray(v_ase, np.float64),        # ASE internal units
        velocities_A_per_fs=np.asarray(v_ase * units.fs, np.float64),
        mass_amu=np.float64(mass),
        temperature=np.float64(a.temperature),
        kB_eV_per_K=np.float64(units.kB),
        fs_in_ase=np.float64(units.fs),
    )
    print(f"{len(atoms)} atoms  cell {atoms.get_cell().array[0,0]:.4f}  "
          f"mass {mass}  -> {a.out}")


if __name__ == "__main__":
    main()
