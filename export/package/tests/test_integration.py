"""
Integration Tests for MyPotential Calculator
============================================

These tests require a compiled model to be present.
Mark with @pytest.mark.integration to run separately.
"""

import pytest
import numpy as np
from pathlib import Path

# Check if we have a compiled model
MODELS_DIR = Path(__file__).parent.parent / 'src/mypotential/models'
HAS_CPU_MODEL = (MODELS_DIR / 'model_cpu.vmfb').exists()
HAS_PARAMS = (MODELS_DIR / 'params.npz').exists()
HAS_MODEL = HAS_CPU_MODEL and HAS_PARAMS

skip_no_model = pytest.mark.skipif(
    not HAS_MODEL,
    reason="No compiled model available"
)


@pytest.fixture
def calculator():
    """Create calculator for testing."""
    if not HAS_MODEL:
        pytest.skip("No compiled model")

    from mypotential import Calculator
    return Calculator(device='cpu')


@pytest.fixture
def silicon_bulk():
    """Create silicon bulk structure."""
    from ase.build import bulk
    return bulk('Si', 'diamond', a=5.43)


@pytest.fixture
def silicon_supercell():
    """Create larger silicon structure."""
    from ase.build import bulk
    return bulk('Si', 'diamond', a=5.43) * (2, 2, 2)


@skip_no_model
class TestBasicCalculation:
    """Test basic calculator operations."""

    def test_energy_calculation(self, calculator, silicon_bulk):
        """Calculate energy for silicon."""
        silicon_bulk.calc = calculator
        energy = silicon_bulk.get_potential_energy()

        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_forces_calculation(self, calculator, silicon_bulk):
        """Calculate forces for silicon."""
        silicon_bulk.calc = calculator
        forces = silicon_bulk.get_forces()

        assert forces.shape == (len(silicon_bulk), 3)
        assert np.all(np.isfinite(forces))

    def test_stress_calculation(self, calculator, silicon_bulk):
        """Calculate stress for silicon."""
        silicon_bulk.calc = calculator
        stress = silicon_bulk.get_stress()

        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))

    def test_forces_near_zero_for_perfect_crystal(self, calculator, silicon_bulk):
        """Perfect crystal should have near-zero forces."""
        silicon_bulk.calc = calculator
        forces = silicon_bulk.get_forces()

        # Forces should be very small for equilibrium structure
        max_force = np.max(np.abs(forces))
        assert max_force < 0.1, f"Max force {max_force} too large for perfect crystal"

    def test_energy_is_negative(self, calculator, silicon_bulk):
        """Cohesive energy should be negative (bound system)."""
        silicon_bulk.calc = calculator
        energy = silicon_bulk.get_potential_energy()
        energy_per_atom = energy / len(silicon_bulk)

        # Silicon cohesive energy is around -4.6 eV/atom
        # Allow for model differences but should be negative
        assert energy_per_atom < 0, f"Energy per atom {energy_per_atom} should be negative"


@skip_no_model
class TestNeighborEdgeCases:
    """Test edge cases in neighbor list handling."""

    def test_no_neighbors(self, calculator):
        """Single atom with no neighbors."""
        from ase import Atoms

        # Single atom in large cell - no neighbors
        atoms = Atoms('Si', positions=[[0, 0, 0]], cell=[100, 100, 100], pbc=True)
        atoms.calc = calculator

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        # Should return zero (or reference energy) without crashing
        assert np.isfinite(energy)
        assert forces.shape == (1, 3)
        assert np.all(np.isfinite(forces))

    def test_two_atoms(self, calculator):
        """Two atoms interacting."""
        from ase import Atoms

        # Two silicon atoms
        atoms = Atoms('Si2', positions=[[0, 0, 0], [2.35, 0, 0]], cell=[10, 10, 10], pbc=True)
        atoms.calc = calculator

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        assert np.isfinite(energy)
        assert forces.shape == (2, 3)

        # Forces should be equal and opposite (Newton's 3rd law)
        assert np.allclose(forces[0], -forces[1], atol=1e-6)


@skip_no_model
class TestConsistency:
    """Test consistency across repeated calculations."""

    def test_repeated_calculation(self, calculator, silicon_bulk):
        """Same structure should give same results."""
        silicon_bulk.calc = calculator

        energies = []
        for _ in range(5):
            energies.append(silicon_bulk.get_potential_energy())

        assert np.allclose(energies, energies[0]), "Energy not consistent across calls"

    def test_copy_gives_same_result(self, calculator, silicon_bulk):
        """Copied structure should give same energy."""
        silicon_bulk.calc = calculator
        energy1 = silicon_bulk.get_potential_energy()

        atoms_copy = silicon_bulk.copy()
        atoms_copy.calc = calculator
        energy2 = atoms_copy.get_potential_energy()

        assert np.isclose(energy1, energy2)


@skip_no_model
@pytest.mark.slow
class TestMDStability:
    """Test molecular dynamics stability."""

    def test_nve_energy_conservation(self, calculator, silicon_supercell):
        """NVE MD should conserve energy."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase import units

        silicon_supercell.calc = calculator
        MaxwellBoltzmannDistribution(silicon_supercell, temperature_K=300)

        dyn = VelocityVerlet(silicon_supercell, timestep=1*units.fs)

        energies = [silicon_supercell.get_total_energy()]

        for _ in range(50):
            dyn.run(1)
            energies.append(silicon_supercell.get_total_energy())

        energies = np.array(energies)
        energy_drift = np.abs(energies[-1] - energies[0])
        energy_drift_per_atom = energy_drift / len(silicon_supercell)

        # Allow 10 meV/atom drift over 50 steps
        assert energy_drift_per_atom < 0.01, f"Energy drift {energy_drift_per_atom} eV/atom too large"

    def test_md_no_explosion(self, calculator, silicon_supercell):
        """MD should not explode (forces stay bounded)."""
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.verlet import VelocityVerlet
        from ase import units

        silicon_supercell.calc = calculator
        MaxwellBoltzmannDistribution(silicon_supercell, temperature_K=500)

        dyn = VelocityVerlet(silicon_supercell, timestep=0.5*units.fs)

        for _ in range(100):
            dyn.run(1)
            forces = silicon_supercell.get_forces()
            max_force = np.max(np.abs(forces))

            # Forces should stay bounded (< 100 eV/A is reasonable)
            assert max_force < 100, f"Force explosion: max force = {max_force}"


@skip_no_model
@pytest.mark.parametrize("device", ["cpu"])
class TestDeviceConsistency:
    """Test that different devices give same results."""

    def test_energy_matches_reference(self, device, silicon_bulk):
        """Energy should match across devices."""
        from mypotential import Calculator

        try:
            calc = Calculator(device=device)
        except (RuntimeError, FileNotFoundError):
            pytest.skip(f"{device} not available")

        silicon_bulk.calc = calc
        energy = silicon_bulk.get_potential_energy()

        # Store first result as reference
        if not hasattr(self, '_reference_energy'):
            self._reference_energy = energy
        else:
            assert np.isclose(energy, self._reference_energy, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
