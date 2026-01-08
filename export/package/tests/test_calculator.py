"""
Tests for MyPotential Calculator
================================
"""

import pytest
import numpy as np
from pathlib import Path


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


def test_list_models():
    """Test listing available models."""
    from mypotential import list_available_models

    models = list_available_models()

    assert 'cpu' in models
    assert 'cuda' in models
    assert 'vulkan' in models

    # Each entry is (path, exists)
    for backend, (path, exists) in models.items():
        assert isinstance(path, Path)
        assert isinstance(exists, bool)


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath('src/mypotential/models/model_cpu.vmfb').exists(),
    reason="No compiled model available"
)
def test_calculator_creation():
    """Test calculator creation."""
    from mypotential import Calculator

    calc = Calculator(device='cpu')
    assert calc.device == 'local-task'
    assert calc.rcut > 0


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath('src/mypotential/models/model_cpu.vmfb').exists(),
    reason="No compiled model available"
)
def test_calculator_silicon():
    """Test calculator with silicon structure."""
    from mypotential import Calculator
    from ase.build import bulk

    # Create silicon structure
    atoms = bulk('Si', 'diamond', a=5.43)
    calc = Calculator(device='cpu')
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
    assert np.max(np.abs(forces)) < 1e-3


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath('src/mypotential/models/model_cpu.vmfb').exists(),
    reason="No compiled model available"
)
def test_calculator_stress():
    """Test stress calculation."""
    from mypotential import Calculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=5.43)
    calc = Calculator(device='cpu')
    atoms.calc = calc

    stress = atoms.get_stress()
    assert stress.shape == (6,)
    assert np.all(np.isfinite(stress))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
