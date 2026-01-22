#!/usr/bin/env python3
"""
Compare Julia reference forces with LAMMPS results.

Note: Julia uses a FULL neighbor list (both i->j and j->i),
while LAMMPS uses a HALF neighbor list with newton on.
This means:
- Julia energy = 2 × LAMMPS energy
- Julia forces = 2 × LAMMPS forces (due to double counting)
"""

import numpy as np
import sys

def parse_lammps_dump(filename):
    """Parse LAMMPS dump file to extract positions and forces."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find ATOMS section
    for i, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            header = line.strip().split()[2:]  # Skip "ITEM: ATOMS"
            data_start = i + 1
            break

    # Parse atom data
    n_atoms = int(lines[3].strip())
    positions = np.zeros((n_atoms, 3))
    forces = np.zeros((n_atoms, 3))

    for i in range(n_atoms):
        parts = lines[data_start + i].strip().split()
        atom_id = int(parts[0]) - 1  # Convert to 0-indexed
        positions[atom_id] = [float(parts[2]), float(parts[3]), float(parts[4])]
        forces[atom_id] = [float(parts[5]), float(parts[6]), float(parts[7])]

    return positions, forces

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("Force Comparison: Julia Reference vs LAMMPS")
    print("=" * 70)

    # Load Julia reference
    ref = np.load(os.path.join(script_dir, "julia_reference.npz"))
    julia_energy_full = ref["energy"][0]
    julia_forces_full = ref["atomic_forces"]  # (n_atoms, 3)
    n_atoms = int(ref["n_atoms"][0])
    n_pairs = int(ref["n_pairs"][0])

    # Convert from full to half list values for comparison
    # (Julia uses full list, LAMMPS uses half list)
    julia_energy = julia_energy_full / 2
    julia_forces = julia_forces_full / 2

    print(f"\nJulia Reference (FULL neighbor list):")
    print(f"  Atoms: {n_atoms}")
    print(f"  Pairs: {n_pairs} (full list)")
    print(f"  Energy (full): {julia_energy_full:.10f} eV")
    print(f"  Energy (half-equivalent): {julia_energy:.10f} eV")
    print(f"  Max force (half-equiv): {np.max(np.abs(julia_forces)):.6e} eV/Å")
    print(f"  Force sum: {np.sum(julia_forces, axis=0)}")

    # Load LAMMPS results
    lammps_pos, lammps_forces = parse_lammps_dump(os.path.join(script_dir, "lammps_forces.dump"))

    # Extract LAMMPS energy from the printed output (hardcoded from run)
    lammps_energy = -0.0236770440330495

    print(f"\nLAMMPS Results (HALF neighbor list, newton on):")
    print(f"  Energy: {lammps_energy:.10f} eV")
    print(f"  Max force: {np.max(np.abs(lammps_forces)):.6e} eV/Å")
    print(f"  Force sum: {np.sum(lammps_forces, axis=0)}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    energy_diff = abs(julia_energy - lammps_energy)
    energy_rel = energy_diff / abs(julia_energy) * 100
    print(f"\nEnergy:")
    print(f"  Julia:  {julia_energy:.10f} eV")
    print(f"  LAMMPS: {lammps_energy:.10f} eV")
    print(f"  Diff:   {energy_diff:.6e} eV ({energy_rel:.4f}%)")

    print(f"\nForces per atom (eV/Å):")
    print(f"  {'Atom':<6} {'Julia Fx':<12} {'LAMMPS Fx':<12} {'Diff Fx':<12}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12}")

    force_diffs = []
    for i in range(n_atoms):
        jf = julia_forces[i]
        lf = lammps_forces[i]
        diff = np.linalg.norm(jf - lf)
        force_diffs.append(diff)
        print(f"  {i+1:<6} {jf[0]:>12.6e} {lf[0]:>12.6e} {(jf[0]-lf[0]):>12.6e}")

    print(f"\nForce comparison:")
    print(f"  Max |F_julia - F_lammps|: {np.max(force_diffs):.6e} eV/Å")
    print(f"  Mean |F_julia - F_lammps|: {np.mean(force_diffs):.6e} eV/Å")

    # Check Newton's 3rd law
    julia_sum = np.linalg.norm(np.sum(julia_forces, axis=0))
    lammps_sum = np.linalg.norm(np.sum(lammps_forces, axis=0))
    print(f"\nNewton's 3rd law check (|sum F| should be ~0):")
    print(f"  Julia:  {julia_sum:.6e}")
    print(f"  LAMMPS: {lammps_sum:.6e}")

    # Success criteria
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    energy_pass = energy_rel < 1.0  # < 1% energy difference
    force_pass = np.max(force_diffs) < 1e-3  # < 1 meV/Å force difference
    newton_pass = max(julia_sum, lammps_sum) < 1e-6  # Near-zero force sum

    print(f"\n  Energy match (<1% diff):     {'PASS ✓' if energy_pass else 'FAIL ✗'} ({energy_rel:.4f}%)")
    print(f"  Force match (<1 meV/Å diff): {'PASS ✓' if force_pass else 'FAIL ✗'} ({np.max(force_diffs):.2e} eV/Å)")
    print(f"  Newton's 3rd law:            {'PASS ✓' if newton_pass else 'FAIL ✗'}")

    all_pass = energy_pass and force_pass and newton_pass
    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")

    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
