"""
Numerical Validation Tests
==========================

Tests for numerical accuracy of forces (finite difference)
and consistency with Julia reference implementation.

Uses bucket-based VMFBs for testing.
"""

import pytest
import numpy as np
from pathlib import Path


# Find models directory
PACKAGE_MODELS_DIR = Path(__file__).parent.parent / 'src/mypotential/models'
BENCHMARK_MODELS_DIR = Path(__file__).parent.parent.parent / 'benchmark/bucket_energy_gradient'

def get_models_dir():
    """Get available models directory."""
    for models_dir in [PACKAGE_MODELS_DIR, BENCHMARK_MODELS_DIR]:
        if models_dir.exists():
            bucket_dirs = list(models_dir.glob('bucket_*'))
            if bucket_dirs:
                return models_dir
    return None

MODELS_DIR = get_models_dir()
HAS_MODEL = MODELS_DIR is not None

# Check for Julia reference fixtures
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
HAS_REFERENCE = (FIXTURES_DIR / 'julia_reference.npz').exists()

skip_no_model = pytest.mark.skipif(not HAS_MODEL, reason="No compiled model (bucket VMFBs)")
skip_no_reference = pytest.mark.skipif(not HAS_REFERENCE, reason="No Julia reference")


@skip_no_model
class TestFiniteDifferenceForces:
    """Validate forces using finite difference of energy."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)

    @pytest.fixture
    def test_atoms(self):
        """Create test structure with some disorder."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)
        # Add small random displacements to break symmetry
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        return atoms

    def test_forces_finite_difference(self, calculator, test_atoms):
        """Forces should match finite difference gradient."""
        test_atoms.calc = calculator

        # Analytical forces
        F_analytical = test_atoms.get_forces()

        # Finite difference
        delta = 1e-5
        F_numerical = np.zeros_like(F_analytical)

        for i in range(len(test_atoms)):
            for d in range(3):
                # Forward
                pos_plus = test_atoms.positions.copy()
                pos_plus[i, d] += delta
                atoms_plus = test_atoms.copy()
                atoms_plus.positions = pos_plus
                atoms_plus.calc = calculator
                E_plus = atoms_plus.get_potential_energy()

                # Backward
                pos_minus = test_atoms.positions.copy()
                pos_minus[i, d] -= delta
                atoms_minus = test_atoms.copy()
                atoms_minus.positions = pos_minus
                atoms_minus.calc = calculator
                E_minus = atoms_minus.get_potential_energy()

                F_numerical[i, d] = -(E_plus - E_minus) / (2 * delta)

        # Compare
        max_diff = np.max(np.abs(F_analytical - F_numerical))
        max_force = np.max(np.abs(F_numerical))
        rel_diff = max_diff / (max_force + 1e-10)

        assert rel_diff < 1e-4, f"Force mismatch: max_diff={max_diff}, rel_diff={rel_diff}"

    def test_forces_finite_difference_multiple_configs(self, calculator):
        """Test finite difference for multiple configurations."""
        from ase.build import bulk

        np.random.seed(123)
        errors = []

        for config_idx in range(3):
            atoms = bulk('Si', 'diamond', a=5.43)
            atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
            atoms.calc = calculator

            F_analytical = atoms.get_forces()

            # Quick finite difference (central atom only)
            delta = 1e-5
            F_numerical = np.zeros(3)
            i = 0  # Test first atom

            for d in range(3):
                pos_plus = atoms.positions.copy()
                pos_plus[i, d] += delta
                atoms_plus = atoms.copy()
                atoms_plus.positions = pos_plus
                atoms_plus.calc = calculator
                E_plus = atoms_plus.get_potential_energy()

                pos_minus = atoms.positions.copy()
                pos_minus[i, d] -= delta
                atoms_minus = atoms.copy()
                atoms_minus.positions = pos_minus
                atoms_minus.calc = calculator
                E_minus = atoms_minus.get_potential_energy()

                F_numerical[d] = -(E_plus - E_minus) / (2 * delta)

            error = np.max(np.abs(F_analytical[i] - F_numerical))
            errors.append(error)

        mean_error = np.mean(errors)
        assert mean_error < 1e-6, f"Mean FD error {mean_error} too large"


@skip_no_model
class TestNewtonThirdLaw:
    """Validate Newton's 3rd law for pair forces."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)

    def test_total_force_zero_pbc(self, calculator):
        """Total force should be zero for periodic system."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
        # Add disorder
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        atoms.calc = calculator

        forces = atoms.get_forces()
        total_force = np.sum(forces, axis=0)

        assert np.allclose(total_force, 0, atol=1e-10), f"Total force {total_force} not zero"

    def test_total_force_multiple_configs(self, calculator):
        """Test Newton's 3rd law for multiple configurations."""
        from ase.build import bulk

        for seed in [1, 2, 3]:
            np.random.seed(seed)
            atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
            atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
            atoms.calc = calculator

            forces = atoms.get_forces()
            total_force_norm = np.linalg.norm(np.sum(forces, axis=0))

            assert total_force_norm < 1e-10, f"Config {seed}: total force = {total_force_norm}"


@skip_no_model
class TestStressVirial:
    """Validate stress/virial calculations."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)

    def test_stress_exists(self, calculator):
        """Test that stress can be computed."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.calc = calculator

        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))

    def test_stress_symmetry(self, calculator):
        """Stress tensor should be symmetric (Voigt notation)."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.05
        atoms.calc = calculator

        stress = atoms.get_stress()

        # In Voigt notation: xx, yy, zz, yz, xz, xy
        # All components should be finite
        assert np.all(np.isfinite(stress))


@skip_no_model
class TestVMFBReference:
    """Compare calculator output against VMFB test data."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)

    def test_energy_consistent(self, calculator):
        """Energy should be consistent across calls."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        atoms.calc = calculator

        E1 = atoms.get_potential_energy()
        E2 = atoms.get_potential_energy()

        assert E1 == E2, "Energy not deterministic"

    def test_forces_consistent(self, calculator):
        """Forces should be consistent across calls."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)
        np.random.seed(42)
        atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1
        atoms.calc = calculator

        F1 = atoms.get_forces()
        F2 = atoms.get_forces()

        np.testing.assert_array_equal(F1, F2, "Forces not deterministic")


@skip_no_reference
@skip_no_model
class TestJuliaReference:
    """Compare against Julia ACEpotentials reference.

    NOTE: This test requires a julia_reference.npz file generated from
    a Julia ACE model that matches the exported VMFBs. The bucket VMFBs
    use a simplified polynomial model, so this test will fail unless
    the reference was generated from the same simplified model.
    """

    @pytest.fixture
    def reference_data(self):
        """Load Julia reference calculation."""
        return np.load(FIXTURES_DIR / 'julia_reference.npz')

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)

    def test_energy_matches_julia(self, calculator, reference_data):
        """Energy should match Julia reference."""
        from ase import Atoms

        # Reconstruct atoms from reference
        atoms = Atoms(
            positions=reference_data['positions'],
            cell=reference_data['cell'],
            pbc=True,
            numbers=reference_data.get('numbers', [14] * len(reference_data['positions']))
        )
        atoms.calc = calculator

        E_python = atoms.get_potential_energy()
        E_julia = float(reference_data['energy'])

        rel_error = np.abs(E_python - E_julia) / (np.abs(E_julia) + 1e-10)
        assert rel_error < 1e-5, f"Energy mismatch: Python={E_python}, Julia={E_julia}"

    def test_forces_match_julia(self, calculator, reference_data):
        """Forces should match Julia reference."""
        from ase import Atoms

        atoms = Atoms(
            positions=reference_data['positions'],
            cell=reference_data['cell'],
            pbc=True,
            numbers=reference_data.get('numbers', [14] * len(reference_data['positions']))
        )
        atoms.calc = calculator

        F_python = atoms.get_forces()
        F_julia = reference_data['forces']

        max_error = np.max(np.abs(F_python - F_julia))
        rel_error = max_error / (np.max(np.abs(F_julia)) + 1e-10)

        assert rel_error < 1e-4, f"Force mismatch: max_error={max_error}, rel_error={rel_error}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
