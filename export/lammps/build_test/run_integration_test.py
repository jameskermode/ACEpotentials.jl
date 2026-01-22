#!/usr/bin/env python3
"""
LAMMPS-Julia Integration Test
==============================

This script:
1. Runs Julia to generate reference energy and forces
2. Runs LAMMPS with the same atomic configuration
3. Compares results and reports pass/fail

Usage:
    python run_integration_test.py
"""

import subprocess
import sys
import os
import re
import numpy as np
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
JULIA_SCRIPT = SCRIPT_DIR / "generate_julia_reference.jl"
LAMMPS_INPUT = SCRIPT_DIR / "test_integration.in"
JULIA_REF_FILE = SCRIPT_DIR / "julia_reference.npz"
LAMMPS_FORCES_FILE = SCRIPT_DIR / "lammps_forces.dump"

# Tolerances
ENERGY_RTOL = 1e-6  # Relative tolerance for energy
FORCE_ATOL = 1e-6   # Absolute tolerance for forces (eV/Å)


def run_julia_reference():
    """Run Julia script to generate reference data."""
    print("=" * 70)
    print("Step 1: Generate Julia Reference Data")
    print("=" * 70)

    cmd = [
        "julia", "+1.11", "--project=/home/eng/essswb/ace-potentials-julia-1.2/ACEpotentials.jl-main/export",
        str(JULIA_SCRIPT)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)

    if result.returncode != 0:
        print("FAILED: Julia script failed")
        print(result.stderr)
        return False

    print(result.stdout)

    if not JULIA_REF_FILE.exists():
        print(f"FAILED: Reference file not created: {JULIA_REF_FILE}")
        return False

    return True


def run_lammps():
    """Run LAMMPS with the test configuration."""
    print("\n" + "=" * 70)
    print("Step 2: Run LAMMPS with IREE VMFB")
    print("=" * 70)

    # Load required modules and run LAMMPS
    cmd = f"""
    source /etc/profile
    module load GCC/13.3.0 OpenMPI/5.0.3 CUDA/12.5.0 Python/3.12.3
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/eng/essswb/iree/lib
    cd {SCRIPT_DIR}
    /home/eng/essswb/lammps/lammps-22Jul2025/build/lmp -k on g 1 -sf kk -in {LAMMPS_INPUT}
    """

    print("Running LAMMPS...")
    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        print("FAILED: LAMMPS run failed")
        return None

    # Extract energy from output
    energy_match = re.search(r"LAMMPS_ENERGY:\s+([-\d.e+]+)", result.stdout)
    if energy_match:
        lammps_energy = float(energy_match.group(1))
        print(f"LAMMPS energy: {lammps_energy}")
        return lammps_energy
    else:
        print("FAILED: Could not extract energy from LAMMPS output")
        return None


def parse_lammps_dump(filename):
    """Parse LAMMPS dump file to extract forces."""
    forces = []
    positions = []

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find ATOMS section
    in_atoms = False
    for line in lines:
        if line.startswith("ITEM: ATOMS"):
            in_atoms = True
            continue
        if in_atoms and not line.startswith("ITEM:"):
            parts = line.split()
            if len(parts) >= 8:
                # id type x y z fx fy fz
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                fx, fy, fz = float(parts[5]), float(parts[6]), float(parts[7])
                positions.append([x, y, z])
                forces.append([fx, fy, fz])

    return np.array(positions), np.array(forces)


def compare_results(lammps_energy):
    """Compare Julia reference with LAMMPS results."""
    print("\n" + "=" * 70)
    print("Step 3: Compare Results")
    print("=" * 70)

    # Load Julia reference
    ref = np.load(JULIA_REF_FILE)
    julia_energy = ref['energy'][0]  # Padded energy (matches LAMMPS)
    julia_energy_actual = ref['energy_actual'][0]  # Actual pairs only
    julia_forces = ref['atomic_forces']
    julia_positions = ref['positions']
    n_atoms = int(ref['n_atoms'][0])
    n_pairs = int(ref['n_pairs'][0])
    vmfb_size = int(ref['vmfb_size'][0])

    print(f"\nJulia Reference:")
    print(f"  Energy (padded to {vmfb_size}): {julia_energy:.10f} eV")
    print(f"  Energy (actual {n_pairs} pairs): {julia_energy_actual:.10f} eV")
    print(f"  Atoms: {n_atoms}")
    print(f"  Max force: {np.max(np.abs(julia_forces)):.10f} eV/Å")

    # Parse LAMMPS forces
    if not LAMMPS_FORCES_FILE.exists():
        print(f"FAILED: LAMMPS forces file not found: {LAMMPS_FORCES_FILE}")
        return False

    lammps_positions, lammps_forces = parse_lammps_dump(LAMMPS_FORCES_FILE)

    print(f"\nLAMMPS Forces:")
    print(f"  Atoms: {len(lammps_forces)}")
    print(f"  Max force: {np.max(np.abs(lammps_forces)):.10f} eV/Å")

    # Compare forces
    print(f"\n--- Force Comparison ---")

    # Sort by position to match atoms
    # (LAMMPS may have different ordering)
    force_diffs = []
    for i in range(n_atoms):
        # Find matching atom in LAMMPS output
        julia_pos = julia_positions[i]
        distances = np.linalg.norm(lammps_positions - julia_pos, axis=1)
        j = np.argmin(distances)

        if distances[j] > 0.01:
            print(f"  WARNING: Atom {i} position mismatch: {distances[j]:.6f}")

        force_diff = lammps_forces[j] - julia_forces[i]
        force_diffs.append(force_diff)

        if np.max(np.abs(force_diff)) > FORCE_ATOL:
            print(f"  Atom {i}: Julia {julia_forces[i]} vs LAMMPS {lammps_forces[j]}")
            print(f"           Diff: {force_diff}")

    force_diffs = np.array(force_diffs)
    max_force_diff = np.max(np.abs(force_diffs))
    rms_force_diff = np.sqrt(np.mean(force_diffs**2))

    print(f"\n  Max force difference: {max_force_diff:.2e} eV/Å")
    print(f"  RMS force difference: {rms_force_diff:.2e} eV/Å")

    # Check force sum (Newton's 3rd law)
    julia_force_sum = np.sum(julia_forces, axis=0)
    lammps_force_sum = np.sum(lammps_forces, axis=0)

    print(f"\n--- Newton's 3rd Law Check ---")
    print(f"  Julia force sum:  {julia_force_sum}")
    print(f"  LAMMPS force sum: {lammps_force_sum}")

    # Energy comparison
    print(f"\n--- Energy Comparison ---")
    energy_diff = abs(lammps_energy - julia_energy)
    energy_rel_diff = energy_diff / abs(julia_energy) if julia_energy != 0 else energy_diff
    print(f"  Julia (padded):  {julia_energy:.10f} eV")
    print(f"  LAMMPS:          {lammps_energy:.10f} eV")
    print(f"  Absolute diff:   {energy_diff:.2e} eV")
    print(f"  Relative diff:   {energy_rel_diff:.2e}")

    # Final verdict
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    passed = True

    # Energy check
    if energy_rel_diff < ENERGY_RTOL:
        print(f"✓ PASS: Energy matches within tolerance ({ENERGY_RTOL} relative)")
    else:
        print(f"✗ FAIL: Energy differs by {energy_rel_diff:.2e} (tolerance: {ENERGY_RTOL})")
        passed = False

    # Force check
    if max_force_diff < FORCE_ATOL:
        print(f"✓ PASS: Forces match within tolerance ({FORCE_ATOL} eV/Å)")
    else:
        print(f"✗ FAIL: Forces differ by {max_force_diff:.2e} eV/Å (tolerance: {FORCE_ATOL})")
        passed = False

    # Newton's 3rd law check
    newton_error = np.max(np.abs(lammps_force_sum))
    if newton_error < 1e-10:
        print(f"✓ PASS: Newton's 3rd law satisfied (sum = {newton_error:.2e})")
    else:
        print(f"✗ FAIL: Newton's 3rd law violated (sum = {newton_error:.2e})")
        passed = False

    print("=" * 70)

    if passed:
        print("\n✓✓✓ INTEGRATION TEST PASSED ✓✓✓\n")
    else:
        print("\n✗✗✗ INTEGRATION TEST FAILED ✗✗✗\n")

    return passed


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("LAMMPS-Julia Integration Test")
    print("=" * 70)
    print(f"\nWorking directory: {SCRIPT_DIR}")

    # Step 1: Generate Julia reference
    if not run_julia_reference():
        print("\nFailed to generate Julia reference")
        sys.exit(1)

    # Step 2: Run LAMMPS
    lammps_energy = run_lammps()
    if lammps_energy is None:
        print("\nFailed to run LAMMPS")
        sys.exit(1)

    # Step 3: Compare results
    if compare_results(lammps_energy):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
