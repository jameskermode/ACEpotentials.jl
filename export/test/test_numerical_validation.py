#!/usr/bin/env python3
"""
Numerical Validation Tests
==========================

Comprehensive numerical validation of the Python ASE calculator against
Julia reference values and finite difference checks.

Run:
    cd export/tools && uv run pytest ../test/test_numerical_validation.py -v
"""

import pytest
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "benchmark" / "bucket_energy_gradient"

# Tolerances
ENERGY_RTOL = 1e-10  # Relative tolerance for energy (float64)
GRADIENT_ATOL = 1e-10  # Absolute tolerance for gradients
FORCE_FD_RTOL = 1e-4  # Relative tolerance for finite difference forces
NEWTON_ATOL = 1e-10  # Tolerance for Newton's 3rd law


@pytest.fixture
def calculator():
    """Create a Calculator instance."""
    from mypotential import Calculator
    return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)


@pytest.fixture
def bucket_manager():
    """Create a BucketManager for direct VMFB access."""
    from mypotential._iree_wrapper import BucketManager
    return BucketManager(MODELS_DIR, 'local-task')


class TestVMFBMatchesJulia:
    """Test 1: VMFB output matches Julia reference values."""

    def test_all_buckets_match_julia(self):
        """All bucket VMFBs should match their Julia reference data."""
        from iree import runtime as iree_rt

        results = []

        for bucket_dir in sorted(MODELS_DIR.glob("bucket_*")):
            bucket_name = bucket_dir.name
            vmfb_path = bucket_dir / "energy_gradient_f64_cpu.vmfb"
            test_data_path = bucket_dir / "test_data.npz"

            if not vmfb_path.exists() or not test_data_path.exists():
                continue

            # Load VMFB
            module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), driver="local-task")

            # Load reference
            data = np.load(test_data_path)
            rij = data["rij"].astype(np.float64)
            energy_ref = float(data["energy"][0])
            grad_ref = data["gradient"].astype(np.float64)

            # Run VMFB
            rij_T = np.ascontiguousarray(rij.T)
            result = module.main(rij_T)
            energy_vmfb = float(np.asarray(result[0]))
            grad_vmfb = np.asarray(result[1]).T

            # Compare
            energy_diff = abs(energy_vmfb - energy_ref)
            energy_rdiff = energy_diff / abs(energy_ref) if energy_ref != 0 else energy_diff
            grad_max_diff = np.max(np.abs(grad_vmfb - grad_ref))

            energy_ok = energy_rdiff < ENERGY_RTOL
            grad_ok = grad_max_diff < GRADIENT_ATOL

            results.append({
                "bucket": bucket_name,
                "energy_ok": energy_ok,
                "grad_ok": grad_ok,
                "energy_rdiff": energy_rdiff,
                "grad_max_diff": grad_max_diff,
            })

        # All buckets should pass
        assert len(results) > 0, "No buckets found with test data"
        for r in results:
            assert r["energy_ok"], f"{r['bucket']}: energy rel_diff={r['energy_rdiff']:.2e}"
            assert r["grad_ok"], f"{r['bucket']}: gradient max_diff={r['grad_max_diff']:.2e}"


class TestCalculatorUsesVMFB:
    """Test 2: ASE Calculator correctly calls VMFB and processes results."""

    def test_calculator_matches_direct_vmfb(self, calculator, bucket_manager):
        """Calculator output should match direct VMFB call."""
        from ase.build import bulk
        from matscipy.neighbours import neighbour_list

        # Create test system with perturbation
        atoms = bulk('Si', 'diamond', a=5.43)
        atoms = atoms * (2, 2, 2)  # 64 atoms
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05

        # Get energy/forces via Calculator
        atoms.calc = calculator
        E_calc = atoms.get_potential_energy()
        F_calc = atoms.get_forces()

        # Compute manually using BucketManager
        pair_i, pair_j, rij = neighbour_list('ijD', atoms, calculator.rcut)
        n_atoms = len(atoms)

        E_vmfb, grad_vmfb = bucket_manager(rij.astype(np.float64), calculator.rcut)

        # Compute forces from gradient
        F_manual = np.zeros((n_atoms, 3), dtype=np.float64)
        np.add.at(F_manual, pair_i, grad_vmfb)
        np.add.at(F_manual, pair_j, -grad_vmfb)

        # Compare
        E_diff = abs(E_calc - E_vmfb)
        F_diff = np.max(np.abs(F_calc - F_manual))

        assert E_diff < 1e-10, f"Energy mismatch: {E_diff:.2e}"
        assert F_diff < 1e-10, f"Force mismatch: {F_diff:.2e}"


class TestForceFiniteDifference:
    """Test 3: Forces match finite difference of energy."""

    def test_forces_match_finite_difference(self, calculator):
        """Analytical forces should match numerical gradient."""
        from ase.build import bulk

        # Small system for speed
        atoms = bulk('Si', 'diamond', a=5.43)
        np.random.seed(123)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        atoms.calc = calculator

        # Analytical forces
        F_analytical = atoms.get_forces()

        # Finite difference
        delta = 1e-5
        F_numerical = np.zeros_like(F_analytical)

        for i in range(len(atoms)):
            for d in range(3):
                atoms_plus = atoms.copy()
                atoms_plus.positions[i, d] += delta
                atoms_plus.calc = calculator
                E_plus = atoms_plus.get_potential_energy()

                atoms_minus = atoms.copy()
                atoms_minus.positions[i, d] -= delta
                atoms_minus.calc = calculator
                E_minus = atoms_minus.get_potential_energy()

                F_numerical[i, d] = -(E_plus - E_minus) / (2 * delta)

        # Compare
        max_force = np.max(np.abs(F_numerical))
        max_diff = np.max(np.abs(F_analytical - F_numerical))
        rel_diff = max_diff / max_force if max_force > 1e-10 else max_diff

        assert rel_diff < FORCE_FD_RTOL, f"Force FD rel_diff={rel_diff:.2e}"


class TestNewtonsThirdLaw:
    """Test 4: Newton's 3rd law - total force should be zero."""

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_force_sum_is_zero(self, calculator, seed):
        """Sum of all forces should be approximately zero."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
        np.random.seed(seed)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        atoms.calc = calculator

        forces = atoms.get_forces()
        total_force = np.sum(forces, axis=0)
        force_norm = np.linalg.norm(total_force)

        assert force_norm < NEWTON_ATOL, f"|sum F| = {force_norm:.2e}"


class TestForceSignConvention:
    """Test 5: Force sign convention - displaced atom feels force."""

    def test_displaced_atom_has_force(self, calculator):
        """Displaced atom should have non-zero force."""
        from ase.build import bulk

        # Displace first atom in +x direction
        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.positions[0, 0] += 0.2
        atoms.calc = calculator

        forces = atoms.get_forces()
        F_x = forces[0, 0]

        # Force should be finite and non-zero
        assert np.isfinite(F_x), "Force should be finite"
        assert abs(F_x) > 1e-10, "Displaced atom should have non-zero force"
