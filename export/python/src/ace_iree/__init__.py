"""
ACE-IREE: ASE Calculator for IREE-compiled ACE potentials

This package provides an ASE Calculator that uses IREE-compiled ACE models
exported from Julia/ACEpotentials.jl. The models are compiled via Reactant.jl
to StableHLO and then to IREE VMFB format.

Example:
    >>> from ace_iree import ACECalculator
    >>> calc = ACECalculator("model.vmfb", "constants.npz")
    >>> atoms.calc = calc
    >>> energy = atoms.get_potential_energy()
    >>> forces = atoms.get_forces()
    >>> stress = atoms.get_stress()
"""

from .calculator import ACECalculator
from .iree_runtime import IREEModel, SubprocessRunner

__version__ = "0.1.0"
__all__ = ["ACECalculator", "IREEModel", "SubprocessRunner"]
