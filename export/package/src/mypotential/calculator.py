"""
ASE Calculator Interface
========================

ASE Calculator implementation for IREE-compiled ACE potentials.
Provides drop-in replacement for other ASE calculators.

Example:
    from mypotential import Calculator

    calc = Calculator()  # Auto-selects best device
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import logging

from ase.calculators.calculator import Calculator as ASECalculator, all_changes
from matscipy.neighbours import neighbour_list

from ._device import select_device, get_models_dir
from ._iree_wrapper import IREEModel, ModelParams

logger = logging.getLogger(__name__)


class Calculator(ASECalculator):
    """
    ASE Calculator for IREE-compiled ACE potentials.

    Automatically selects the best available compute device (CUDA > Vulkan > CPU)
    and loads the corresponding pre-compiled model.

    Architecture:
        - IREE computes per-pair forces (GPU-accelerated, no scatter ops)
        - Host accumulates pair forces to atomic forces (CPU scatter)
        - This split avoids IREE scatter compilation issues while maintaining
          GPU acceleration for the expensive model evaluation

    Attributes:
        implemented_properties: List of computable properties
        device: Active IREE device string
        rcut: Model cutoff radius in Angstroms
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(
        self,
        device: str = 'auto',
        models_dir: Optional[Union[str, Path]] = None,
        params_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        Initialize the calculator.

        Args:
            device: Compute device to use. Options:
                - 'auto': Auto-select best available (default)
                - 'cuda': Use NVIDIA GPU
                - 'vulkan': Use Vulkan GPU
                - 'cpu': Use CPU
            models_dir: Custom directory containing model files.
                If not specified, uses package's built-in models.
            params_path: Path to .npz file with model parameters.
                If not specified, looks for params.npz in models_dir.
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        # Select device and model
        if models_dir is not None:
            models_dir = Path(models_dir)
        else:
            models_dir = get_models_dir()

        self.device, vmfb_path = select_device(device, models_dir)

        # Load model parameters
        if params_path is not None:
            self._params = ModelParams(str(params_path))
        else:
            params_file = models_dir / 'params.npz'
            if params_file.exists():
                self._params = ModelParams(str(params_file))
            else:
                raise FileNotFoundError(
                    f"Model parameters not found: {params_file}. "
                    "Provide params_path argument."
                )

        # Load IREE model
        self._iree = IREEModel(
            vmfb_path=str(vmfb_path),
            device=self.device,
        )

        # Get cutoff from model
        self.rcut = self._params.rcut

        logger.info(f"Calculator initialized: device={self.device}, rcut={self.rcut}")

    def calculate(
        self,
        atoms=None,
        properties=['energy'],
        system_changes=all_changes,
    ):
        """
        Calculate properties for the given atoms.

        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: List of changes since last calculation
        """
        super().calculate(atoms, properties, system_changes)

        # Build neighbor list
        pair_i, pair_j, rij = neighbour_list('ijD', atoms, self.rcut)
        n_atoms = len(atoms)
        n_pairs = len(pair_i)

        if n_pairs == 0:
            # No neighbors - isolated atoms
            self.results['energy'] = 0.0
            if 'forces' in properties:
                self.results['forces'] = np.zeros((n_atoms, 3), dtype=np.float64)
            if 'stress' in properties:
                self.results['stress'] = np.zeros(6, dtype=np.float64)
            return

        # Build pool matrix (for accumulating pair contributions to atoms)
        pool_matrix = np.zeros((n_atoms, n_pairs), dtype=np.float32)
        for e in range(n_pairs):
            pool_matrix[pair_i[e], e] = 1.0

        # Convert rij to float32
        rij = rij.astype(np.float32)

        # Compute energy and forces
        energy, pair_forces = self._iree.compute_energy_and_forces(
            rij, pool_matrix, self._params.to_dict()
        )

        self.results['energy'] = float(energy)

        if 'forces' in properties or 'stress' in properties:
            # Accumulate pair forces to atomic forces
            forces = np.zeros((n_atoms, 3), dtype=np.float64)
            np.add.at(forces, pair_i, -pair_forces)
            np.add.at(forces, pair_j, pair_forces)
            self.results['forces'] = forces

        if 'stress' in properties:
            # Compute virial from pair forces and displacements
            # V_ab = -sum_ij (f_ij)_a * (r_ij)_b
            virial = -np.einsum('ea,eb->ab', pair_forces, rij)

            # Convert to Voigt notation and normalize by volume
            volume = atoms.get_volume()
            stress = np.array([
                virial[0, 0],  # xx
                virial[1, 1],  # yy
                virial[2, 2],  # zz
                virial[1, 2],  # yz
                virial[0, 2],  # xz
                virial[0, 1],  # xy
            ]) / volume

            self.results['stress'] = stress

    @property
    def cutoff(self) -> float:
        """Model cutoff radius in Angstroms."""
        return self.rcut

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Get potential energy."""
        return super().get_potential_energy(atoms, force_consistent)

    def get_forces(self, atoms=None):
        """Get forces on atoms."""
        return super().get_forces(atoms)

    def get_stress(self, atoms=None):
        """Get stress tensor."""
        return super().get_stress(atoms)


# Convenience function for quick setup
def load_calculator(
    device: str = 'auto',
    models_dir: Optional[str] = None,
) -> Calculator:
    """
    Load calculator with automatic device selection.

    This is the recommended way to create a calculator:

        from mypotential import load_calculator
        calc = load_calculator()
        atoms.calc = calc

    Args:
        device: 'auto', 'cuda', 'vulkan', or 'cpu'
        models_dir: Optional custom models directory

    Returns:
        Configured Calculator instance
    """
    return Calculator(device=device, models_dir=models_dir)
