"""
Numerical Validation Tests
==========================

Tests for numerical accuracy of forces (finite difference)
and consistency with Julia reference implementation.
"""

import pytest
import numpy as np
from pathlib import Path


MODELS_DIR = Path(__file__).parent.parent / 'src/mypotential/models'
HAS_MODEL = (MODELS_DIR / 'model_cpu.vmfb').exists() and (MODELS_DIR / 'params.npz').exists()
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
HAS_REFERENCE = (FIXTURES_DIR / 'julia_reference.npz').exists()

skip_no_model = pytest.mark.skipif(not HAS_MODEL, reason="No compiled model")
skip_no_reference = pytest.mark.skipif(not HAS_REFERENCE, reason="No Julia reference")


@skip_no_model
class TestFiniteDifferenceForces:
    """Validate forces using finite difference of energy."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu')

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
        delta = 1e-4
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
        rel_diff = max_diff / (np.max(np.abs(F_numerical)) + 1e-10)

        assert rel_diff < 1e-3, f"Force mismatch: max_diff={max_diff}, rel_diff={rel_diff}"

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
            delta = 1e-4
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
        assert mean_error < 1e-3, f"Mean FD error {mean_error} too large"


@skip_no_model
class TestNewtonThirdLaw:
    """Validate Newton's 3rd law for pair forces."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu')

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

        assert np.allclose(total_force, 0, atol=1e-6), f"Total force {total_force} not zero"

    def test_pair_forces_opposite(self, calculator):
        """For dimer, forces should be equal and opposite."""
        from ase import Atoms

        # Two atoms in vacuum
        atoms = Atoms('Si2', positions=[[0, 0, 0], [2.5, 0, 0]],
                      cell=[20, 20, 20], pbc=False)
        atoms.calc = calculator

        forces = atoms.get_forces()

        # F_1 = -F_2 (Newton's 3rd law)
        assert np.allclose(forces[0], -forces[1], atol=1e-6)

        # Forces should be along bond axis (x direction)
        assert np.abs(forces[0, 1]) < 1e-6  # y component
        assert np.abs(forces[0, 2]) < 1e-6  # z component


@skip_no_model
class TestStressVirial:
    """Validate stress/virial calculations."""

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu')

    def test_stress_finite_difference(self, calculator):
        """Stress should match finite difference of energy w.r.t. strain."""
        from ase.build import bulk

        atoms = bulk('Si', 'diamond', a=5.43)
        atoms.calc = calculator

        # Analytical stress
        stress_analytical = atoms.get_stress()

        # Finite difference via cell deformation
        delta = 1e-5
        stress_numerical = np.zeros(6)

        # Voigt indices: xx, yy, zz, yz, xz, xy -> strain indices
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for v_idx, (i, j) in enumerate(voigt_map):
            # Forward strain
            cell_plus = atoms.cell.array.copy()
            cell_plus[i, j] += delta * cell_plus[j, j]
            if i != j:
                cell_plus[j, i] += delta * cell_plus[i, i]

            atoms_plus = atoms.copy()
            atoms_plus.set_cell(cell_plus, scale_atoms=True)
            atoms_plus.calc = calculator
            E_plus = atoms_plus.get_potential_energy()

            # Backward strain
            cell_minus = atoms.cell.array.copy()
            cell_minus[i, j] -= delta * cell_minus[j, j]
            if i != j:
                cell_minus[j, i] -= delta * cell_minus[i, i]

            atoms_minus = atoms.copy()
            atoms_minus.set_cell(cell_minus, scale_atoms=True)
            atoms_minus.calc = calculator
            E_minus = atoms_minus.get_potential_energy()

            # dE/d(strain) / V = stress
            V = atoms.get_volume()
            factor = 2.0 if i != j else 1.0
            stress_numerical[v_idx] = (E_plus - E_minus) / (2 * delta * factor) / V

        # Compare (stress can have large relative errors for small values)
        for v_idx in range(6):
            if np.abs(stress_analytical[v_idx]) > 0.01:  # Only check significant stresses
                rel_error = np.abs(stress_analytical[v_idx] - stress_numerical[v_idx]) / np.abs(stress_analytical[v_idx])
                assert rel_error < 0.1, f"Stress[{v_idx}] mismatch: {stress_analytical[v_idx]} vs {stress_numerical[v_idx]}"


@skip_no_reference
@skip_no_model
class TestJuliaReference:
    """Compare against Julia ACEpotentials reference."""

    @pytest.fixture
    def reference_data(self):
        """Load Julia reference calculation."""
        return np.load(FIXTURES_DIR / 'julia_reference.npz')

    @pytest.fixture
    def calculator(self):
        from mypotential import Calculator
        return Calculator(device='cpu')

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

        rel_error = np.abs(E_python - E_julia) / np.abs(E_julia)
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
