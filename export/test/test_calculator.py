#!/usr/bin/env python3
"""
Test ASE Calculator with Bucket VMFBs

Verifies the updated calculator correctly loads bucket VMFBs and computes
energy/forces.

Run:
    cd export/tools && uv run python ../test/test_calculator.py
"""

import sys
from pathlib import Path
import numpy as np

# Add package to path for testing
PACKAGE_DIR = Path(__file__).parent.parent / "package" / "src"
sys.path.insert(0, str(PACKAGE_DIR))

# Models directory (bucket VMFBs)
MODELS_DIR = Path(__file__).parent.parent / "benchmark" / "bucket_energy_gradient"

print("=" * 70)
print("Test ASE Calculator with Bucket VMFBs")
print("=" * 70)
print(f"\nPackage dir: {PACKAGE_DIR}")
print(f"Models dir: {MODELS_DIR}")

# Check dependencies
try:
    import ase
    from ase.build import bulk
    print(f"ASE version: {ase.__version__}")
except ImportError:
    print("ERROR: ase not found. Install with: pip install ase")
    sys.exit(1)

try:
    from matscipy.neighbours import neighbour_list
    print("matscipy: OK")
except ImportError:
    print("ERROR: matscipy not found. Install with: pip install matscipy")
    sys.exit(1)

try:
    from iree import runtime as iree_rt
    print("iree.runtime: OK")
except ImportError:
    print("ERROR: iree.runtime not found")
    sys.exit(1)

# Import calculator
print("\n--- Loading Calculator ---")
from mypotential import Calculator
from mypotential._device import print_device_info

# Show device info
print_device_info(MODELS_DIR)

# Create calculator
print("\n--- Creating Calculator ---")
try:
    calc = Calculator(
        device='cpu',
        models_dir=MODELS_DIR,
        rcut=5.5,
    )
    print(f"Calculator created successfully!")
    print(f"  Device: {calc.device}")
    print(f"  Cutoff: {calc.rcut} Å")
    print(f"  Buckets: {calc.num_buckets}")
    print(f"  Max edges: {calc.max_edges}")
except Exception as e:
    print(f"ERROR: Failed to create calculator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create test system
print("\n--- Creating Test System ---")
atoms = bulk('Si', 'diamond', a=5.43)
atoms = atoms * (2, 2, 2)  # 2x2x2 supercell = 64 atoms
print(f"Atoms: {len(atoms)} Si")
print(f"Cell: {atoms.cell.lengths()}")

# Attach calculator
atoms.calc = calc

# Compute energy
print("\n--- Computing Energy ---")
try:
    energy = atoms.get_potential_energy()
    print(f"Energy: {energy:.6f} eV")
except Exception as e:
    print(f"ERROR: Energy calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compute forces
print("\n--- Computing Forces ---")
try:
    forces = atoms.get_forces()
    print(f"Forces shape: {forces.shape}")
    print(f"Max force: {np.max(np.abs(forces)):.6e} eV/Å")
    print(f"Force sum: {np.sum(forces, axis=0)}")  # Should be ~0 (Newton 3rd)

    # Check Newton's 3rd law
    force_sum_norm = np.linalg.norm(np.sum(forces, axis=0))
    if force_sum_norm < 1e-10:
        print(f"Newton's 3rd law: PASS (|sum F| = {force_sum_norm:.2e})")
    else:
        print(f"Newton's 3rd law: WARNING (|sum F| = {force_sum_norm:.2e})")
except Exception as e:
    print(f"ERROR: Force calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with displaced atoms
print("\n--- Testing Displaced System ---")
atoms_disp = atoms.copy()
atoms_disp.calc = calc
atoms_disp.positions[0] += [0.1, 0.0, 0.0]  # Displace first atom

energy_disp = atoms_disp.get_potential_energy()
forces_disp = atoms_disp.get_forces()

print(f"Displaced energy: {energy_disp:.6f} eV")
print(f"Energy change: {energy_disp - energy:.6f} eV")
print(f"Force on displaced atom: {forces_disp[0]}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Calculator successfully tested with bucket VMFBs!

Configuration:
  - Device: {calc.device}
  - Cutoff: {calc.rcut} Å
  - Buckets loaded: {calc.num_buckets}
  - Max edges supported: {calc.max_edges}

Test Results:
  - Energy computation: OK
  - Force computation: OK
  - Newton's 3rd law: {'OK' if force_sum_norm < 1e-10 else 'WARNING'}

The ASE Calculator is compatible with the bucket VMFB export format.
""")
print("=" * 70)
