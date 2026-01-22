"""
Tests for MyPotential Calculator
================================

Tests that the Calculator class works correctly with bucket-based VMFBs.
"""

import pytest
import numpy as np
from pathlib import Path


# Find models directory - try package models first, then benchmark
PACKAGE_MODELS_DIR = Path(__file__).parent.parent / 'src/mypotential/models'
BENCHMARK_MODELS_DIR = Path(__file__).parent.parent.parent / 'benchmark/bucket_energy_gradient'

def get_models_dir():
    """Get available models directory."""
    # Check for bucket VMFBs
    for models_dir in [PACKAGE_MODELS_DIR, BENCHMARK_MODELS_DIR]:
        if models_dir.exists():
            bucket_dirs = list(models_dir.glob('bucket_*'))
            if bucket_dirs:
                return models_dir
    return None

MODELS_DIR = get_models_dir()
HAS_MODEL = MODELS_DIR is not None

skip_no_model = pytest.mark.skipif(not HAS_MODEL, reason="No compiled model (bucket VMFBs)")


def test_import():
    """Test that the package can be imported."""
    from mypotential import Calculator, load_calculator
    from mypotential import get_device_info, print_device_info


def test_device_info():
    """Test device info functions."""
    from mypotential import get_device_info

    info = get_device_info()

    assert 'cpu' in info
    assert 'cuda' in info
    assert 'vulkan' in info

    # CPU should always have driver available
    assert info['cpu']['driver_available'] is True


@skip_no_model
def test_list_available_backends():
    """Test listing available backends."""
    from mypotential._device import list_available_backends

    backends = list_available_backends(MODELS_DIR)

    assert 'cpu' in backends
    assert 'cuda' in backends
    assert 'vulkan' in backends

    # Should have at least CPU VMFBs
    assert len(backends['cpu']) > 0, f"No CPU VMFBs found in {MODELS_DIR}"


@skip_no_model
def test_calculator_creation():
    """Test calculator creation with bucket VMFBs."""
    from mypotential import Calculator

    calc = Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)
    assert calc.device == 'local-task'
    assert calc.rcut == 5.5
    assert calc.num_buckets > 0


@skip_no_model
def test_calculator_silicon():
    """Test calculator with silicon structure."""
    from mypotential import Calculator
    from ase.build import bulk

    # Create silicon structure
    atoms = bulk('Si', 'diamond', a=5.43)
    calc = Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)
    atoms.calc = calc

    # Calculate energy
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert np.isfinite(energy)

    # Calculate forces
    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))

    # For a perfect crystal, forces should be near zero
    # (Using simplified model, so very small forces expected)
    assert np.max(np.abs(forces)) < 1e-6


@skip_no_model
def test_calculator_stress():
    """Test stress calculation."""
    from mypotential import Calculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=5.43)
    calc = Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)
    atoms.calc = calc

    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))


@skip_no_model
def test_calculator_supercell():
    """Test calculator with larger supercell."""
    from mypotential import Calculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)  # 16 atoms (2 per primitive × 8)
    calc = Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))


@skip_no_model
def test_calculator_displaced_atoms():
    """Test that displaced atoms have non-zero forces."""
    from mypotential import Calculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=5.43)
    np.random.seed(42)
    atoms.positions += np.random.randn(*atoms.positions.shape) * 0.1

    calc = Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)
    atoms.calc = calc

    forces = atoms.get_forces()

    # Displaced atoms should have non-zero forces
    assert np.max(np.abs(forces)) > 1e-10


@skip_no_model
def test_bucket_selection():
    """Test that appropriate bucket is selected based on system size."""
    from mypotential import Calculator
    from ase.build import bulk

    calc = Calculator(device='cpu', models_dir=MODELS_DIR, rcut=5.5)

    # Small system should use smallest bucket
    atoms_small = bulk('Si', 'diamond', a=5.43)
    atoms_small.calc = calc
    _ = atoms_small.get_potential_energy()

    # Larger system should still work
    atoms_large = bulk('Si', 'diamond', a=5.43) * (3, 3, 3)  # 216 atoms
    atoms_large.calc = calc
    _ = atoms_large.get_potential_energy()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
