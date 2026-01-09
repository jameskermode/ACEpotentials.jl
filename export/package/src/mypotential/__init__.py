"""
MyPotential - Pre-compiled ACE Interatomic Potential
====================================================

A portable, pre-compiled ACE potential that works across CPU and GPU
without requiring Julia, PyTorch, or JAX at runtime.

Quick Start:
    from mypotential import Calculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.calc = Calculator()  # Auto-selects best device

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

Device Selection:
    The calculator automatically selects the best available device
    in order of preference: CUDA > Vulkan > CPU

    To force a specific device:
        calc = Calculator(device='cpu')
        calc = Calculator(device='cuda')
        calc = Calculator(device='vulkan')

API:
    Calculator      - Main ASE Calculator class
    load_calculator - Convenience function to create calculator
    get_device_info - Check available devices
    print_device_info - Print device availability summary
"""

__version__ = "1.0.0"

from .calculator import Calculator, load_calculator
from ._device import (
    select_device,
    get_device_info,
    print_device_info,
    list_available_backends,
)

__all__ = [
    'Calculator',
    'load_calculator',
    'select_device',
    'get_device_info',
    'print_device_info',
    'list_available_backends',
]
