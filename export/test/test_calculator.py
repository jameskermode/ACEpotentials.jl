#!/usr/bin/env python3
"""
Test ASE Calculator with Bucket VMFBs

Verifies the calculator correctly loads bucket VMFBs and computes energy/forces.

Run:
    cd export/tools && uv run pytest ../test/test_calculator.py -v
"""

import pytest
import numpy as np
from pathlib import Path

# Models directory (bucket VMFBs)
MODELS_DIR = Path(__file__).parent.parent / "benchmark" / "bucket_energy_gradient"


@pytest.fixture
def calculator():
    """Create a Calculator instance for testing."""
    from mypotential import Calculator
    return Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)


@pytest.fixture
def silicon_supercell():
    """Create a 2x2x2 silicon supercell."""
    from ase.build import bulk
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms = atoms * (2, 2, 2)  # 64 atoms
    return atoms


class TestCalculatorInit:
    """Test calculator initialization."""

    def test_calculator_creates_successfully(self, calculator):
        """Calculator should initialize with valid parameters."""
        assert calculator.device in ('cpu', 'local-task')  # IREE driver name
        assert calculator.rcut == 5.5
        assert calculator.num_buckets > 0
        assert calculator.max_edges > 0

    def test_calculator_loads_buckets(self, calculator):
        """Calculator should load all available bucket VMFBs."""
        # Check that buckets were loaded
        assert calculator.num_buckets >= 1


class TestEnergyCalculation:
    """Test energy computation."""

    def test_energy_returns_float(self, calculator, silicon_supercell):
        """Energy should be a finite float."""
        silicon_supercell.calc = calculator
        energy = silicon_supercell.get_potential_energy()

        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_energy_deterministic(self, calculator, silicon_supercell):
        """Energy should be deterministic for same structure."""
        silicon_supercell.calc = calculator

        e1 = silicon_supercell.get_potential_energy()
        e2 = silicon_supercell.get_potential_energy()

        assert e1 == e2

    def test_energy_changes_with_displacement(self, calculator, silicon_supercell):
        """Energy should change when atoms are displaced."""
        silicon_supercell.calc = calculator
        e_original = silicon_supercell.get_potential_energy()

        # Displace first atom
        atoms_disp = silicon_supercell.copy()
        atoms_disp.calc = calculator
        atoms_disp.positions[0] += [0.1, 0.0, 0.0]
        e_displaced = atoms_disp.get_potential_energy()

        assert e_original != e_displaced


class TestForceCalculation:
    """Test force computation."""

    def test_forces_shape(self, calculator, silicon_supercell):
        """Forces should have shape (n_atoms, 3)."""
        silicon_supercell.calc = calculator
        forces = silicon_supercell.get_forces()

        assert forces.shape == (len(silicon_supercell), 3)

    def test_forces_finite(self, calculator, silicon_supercell):
        """All forces should be finite."""
        silicon_supercell.calc = calculator
        forces = silicon_supercell.get_forces()

        assert np.all(np.isfinite(forces))

    def test_newtons_third_law(self, calculator, silicon_supercell):
        """Sum of all forces should be approximately zero."""
        # Add some perturbation to get non-trivial forces
        np.random.seed(42)
        silicon_supercell.positions += np.random.randn(*silicon_supercell.positions.shape) * 0.05
        silicon_supercell.calc = calculator

        forces = silicon_supercell.get_forces()
        force_sum = np.sum(forces, axis=0)
        force_sum_norm = np.linalg.norm(force_sum)

        assert force_sum_norm < 1e-10, f"Newton's 3rd law violated: |sum F| = {force_sum_norm}"

    def test_displaced_atom_has_nonzero_force(self, calculator, silicon_supercell):
        """A displaced atom should experience a force."""
        silicon_supercell.calc = calculator

        # Displace first atom significantly
        silicon_supercell.positions[0] += [0.2, 0.0, 0.0]
        forces = silicon_supercell.get_forces()

        # Force on displaced atom should be non-zero
        force_magnitude = np.linalg.norm(forces[0])
        assert force_magnitude > 1e-6, "Displaced atom should have non-zero force"


class TestConsistency:
    """Test energy-force consistency."""

    def test_force_finite_difference(self, calculator):
        """Forces should match finite difference of energy."""
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

        # Compare with relative tolerance
        max_force = np.max(np.abs(F_numerical))
        max_diff = np.max(np.abs(F_analytical - F_numerical))
        rel_diff = max_diff / max_force if max_force > 1e-10 else max_diff

        assert rel_diff < 1e-4, f"Force FD mismatch: rel_diff={rel_diff:.2e}"
