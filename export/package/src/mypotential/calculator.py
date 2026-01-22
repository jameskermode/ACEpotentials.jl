"""
ASE Calculator Interface
========================

ASE Calculator implementation for IREE-compiled ACE potentials using
bucket-based energy+gradient VMFBs.

Architecture:
- Loads bucket VMFBs from models directory (bucket_2000/, bucket_10000/, etc.)
- Auto-selects smallest bucket that fits the system at runtime
- Each VMFB computes energy and dE/drij gradient in one call
- Forces = -gradient, scattered to atoms via np.add.at

Example:
    from mypotential import Calculator

    calc = Calculator(models_dir="/path/to/bucket_energy_gradient")
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import logging

from ase.calculators.calculator import Calculator as ASECalculator, all_changes

try:
    from matscipy.neighbours import neighbour_list
    HAS_MATSCIPY = True
except ImportError:
    HAS_MATSCIPY = False

from ._device import select_device, get_models_dir
from ._iree_wrapper import BucketManager, ModelParams

logger = logging.getLogger(__name__)


class Calculator(ASECalculator):
    """
    ASE Calculator for IREE-compiled ACE potentials.

    Uses bucket-based VMFBs that compute energy and gradients directly.
    Automatically selects the smallest bucket that fits the system.

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
        rcut: float = 5.5,
        E0: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Initialize the calculator.

        Args:
            device: Compute device ('auto', 'cuda', 'cpu')
            models_dir: Directory containing bucket_*/ subdirectories with VMFBs
            params_path: Path to .npz file with E0 and rcut (optional)
            rcut: Cutoff radius in Angstroms (default 5.5, overridden by params)
            E0: One-body energies per species (optional, default zeros)
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        if not HAS_MATSCIPY:
            raise ImportError(
                "matscipy is required for neighbor list computation. "
                "Install with: pip install matscipy"
            )

        # Find models directory
        if models_dir is not None:
            models_dir = Path(models_dir)
        else:
            models_dir = get_models_dir()

        # Select device
        self.device, _ = select_device(device, models_dir)

        # Map device selection to IREE driver
        iree_device = {
            'cuda': 'cuda',
            'cpu': 'local-task',
            'auto': 'local-task',  # Will be overridden by select_device
        }.get(self.device, 'local-task')

        # Load model parameters if provided
        if params_path is not None:
            params = ModelParams(str(params_path))
            self.rcut = params.rcut
            self.E0 = params.E0
        else:
            # Check for params.npz in models_dir
            params_file = models_dir / 'params.npz'
            if params_file.exists():
                params = ModelParams(str(params_file))
                self.rcut = params.rcut
                self.E0 = params.E0
            else:
                # Use provided values or defaults
                self.rcut = rcut
                self.E0 = E0 if E0 is not None else np.array([0.0])

        # Load bucket VMFBs
        self._bucket_manager = BucketManager(models_dir, iree_device)

        logger.info(
            f"Calculator initialized: device={self.device}, rcut={self.rcut}, "
            f"buckets={[b.max_edges for b in self._bucket_manager.buckets]}"
        )

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
        # Returns: i (source), j (neighbor), D (displacement j - i)
        pair_i, pair_j, rij = neighbour_list('ijD', atoms, self.rcut)
        n_atoms = len(atoms)
        n_pairs = len(pair_i)

        if n_pairs == 0:
            # No neighbors - only E0 contributes
            self.results['energy'] = float(np.sum(self.E0[0] * n_atoms))
            if 'forces' in properties:
                self.results['forces'] = np.zeros((n_atoms, 3), dtype=np.float64)
            if 'stress' in properties:
                self.results['stress'] = np.zeros(6, dtype=np.float64)
            return

        # Convert to float64 for VMFB
        rij_f64 = rij.astype(np.float64)

        # Call VMFB to get energy and gradient
        energy, gradient = self._bucket_manager(rij_f64, self.rcut)

        # Add E0 (one-body) contribution
        # For single-species: E0[0] * n_atoms
        E0_total = float(self.E0[0]) * n_atoms
        total_energy = energy + E0_total

        self.results['energy'] = float(total_energy)

        if 'forces' in properties or 'stress' in properties:
            # Forces = -gradient (negative of dE/drij)
            # Then scatter to atoms:
            #   F_i += -(-gradient) = +gradient  (force on source atom i)
            #   F_j += -gradient                  (force on neighbor atom j)
            #
            # Convention: rij = pos[j] - pos[i], so gradient is dE/d(pos[j]-pos[i])
            # Force on i: -dE/dpos[i] = +dE/drij = +gradient
            # Force on j: -dE/dpos[j] = -dE/drij = -gradient
            forces = np.zeros((n_atoms, 3), dtype=np.float64)
            np.add.at(forces, pair_i, gradient)    # F_i += gradient
            np.add.at(forces, pair_j, -gradient)   # F_j -= gradient
            self.results['forces'] = forces

        if 'stress' in properties:
            # Compute virial from pair forces and displacements
            # virial_ab = -sum_ij (F_ij)_a * (r_ij)_b
            # where F_ij = -gradient (force contribution from edge ij)
            pair_forces = -gradient  # F_ij = -dE/drij
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

    @property
    def num_buckets(self) -> int:
        """Number of loaded bucket VMFBs."""
        return len(self._bucket_manager.buckets)

    @property
    def max_edges(self) -> int:
        """Maximum edges supported by largest bucket."""
        if self._bucket_manager.buckets:
            return self._bucket_manager.buckets[-1].max_edges
        return 0


# Convenience function for quick setup
def load_calculator(
    device: str = 'auto',
    models_dir: Optional[str] = None,
) -> Calculator:
    """
    Load calculator with automatic device selection.

    Args:
        device: 'auto', 'cuda', or 'cpu'
        models_dir: Directory containing bucket_*/ subdirectories

    Returns:
        Configured Calculator instance
    """
    return Calculator(device=device, models_dir=models_dir)


# Alias for consistent naming
ACECalculator = Calculator
