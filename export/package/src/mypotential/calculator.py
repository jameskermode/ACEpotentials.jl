"""
ASE Calculator Interface
========================

ASE Calculator implementation for IREE-compiled ACE potentials.

Architecture:
- Loads all available VMFB contributions (ace, pair, etc.)
- Each contribution returns (energy, pair_forces)
- Calculator sums all contributions
- E0 (one-body) added separately in Python
- Forces accumulated from pair_forces to atomic forces

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
from ._iree_wrapper import load_contribution, ModelParams, IREEContribution

logger = logging.getLogger(__name__)


class Calculator(ASECalculator):
    """
    ASE Calculator for IREE-compiled ACE potentials.

    Automatically selects the best available compute device (CUDA > Vulkan > CPU)
    and loads all available contribution modules.

    Architecture:
        - Each contribution VMFB computes (energy, pair_forces)
        - Forces computed via Enzyme autodiff inside IREE
        - Python sums contributions and accumulates pair_forces to atomic forces

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
            device: Compute device to use ('auto', 'cuda', 'vulkan', 'cpu')
            models_dir: Custom directory containing model files
            params_path: Path to .npz file with model parameters
            **kwargs: Additional arguments passed to ASE Calculator
        """
        super().__init__(**kwargs)

        # Select device and find models directory
        if models_dir is not None:
            models_dir = Path(models_dir)
        else:
            models_dir = get_models_dir()

        self.device, _ = select_device(device, models_dir)

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

        # Get cutoff and shapes from parameters
        self.rcut = self._params.rcut

        # Load metadata for shapes
        metadata_file = models_dir / 'metadata.json'
        if metadata_file.exists():
            import json
            with open(metadata_file) as f:
                metadata = json.load(f)
            model_meta = metadata.get('model', {})
            self.max_atoms = model_meta.get('max_atoms', 256)
            self.max_pairs = model_meta.get('max_pairs', 10000)
        else:
            self.max_atoms = 256
            self.max_pairs = 10000

        # Load all available contributions for this device
        self._contributions = self._load_contributions(models_dir)

        if not self._contributions:
            raise RuntimeError(
                f"No VMFB files found for device '{self.device}' in {models_dir}"
            )

        logger.info(
            f"Calculator initialized: device={self.device}, rcut={self.rcut}, "
            f"contributions={len(self._contributions)}"
        )

    def _load_contributions(self, models_dir: Path) -> List[IREEContribution]:
        """Load all available contribution VMFB files for the selected device."""
        contributions = []

        # Map device to backend suffix
        backend_suffix = {
            'cuda': '_cuda.vmfb',
            'vulkan': '_vulkan.vmfb',
            'local-task': '_cpu.vmfb',
        }.get(self.device, '_cpu.vmfb')

        # Find all VMFB files for this backend
        for vmfb_file in models_dir.glob(f'*{backend_suffix}'):
            try:
                contrib = load_contribution(
                    str(vmfb_file),
                    self.device,
                    self.max_pairs,
                    self.max_atoms,
                )
                contributions.append(contrib)
                logger.info(f"Loaded contribution: {contrib.name}")
            except Exception as e:
                logger.warning(f"Failed to load {vmfb_file}: {e}")

        return contributions

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
            # No neighbors - only E0 contributes
            self.results['energy'] = float(np.sum(self._params.E0))
            if 'forces' in properties:
                self.results['forces'] = np.zeros((n_atoms, 3), dtype=np.float64)
            if 'stress' in properties:
                self.results['stress'] = np.zeros(6, dtype=np.float64)
            return

        # Convert to float32 for IREE
        rij_f32 = rij.astype(np.float32)

        # Initialize totals
        total_energy = 0.0
        total_pair_forces = np.zeros((n_pairs, 3), dtype=np.float32)

        # Sum contributions from all IREE modules
        for contrib in self._contributions:
            energy, pair_forces = contrib(rij_f32, pair_i, n_atoms, self._params)
            total_energy += energy
            total_pair_forces += pair_forces

        # Add E0 (one-body) contribution
        # For single-species: E0[0] * n_atoms
        # For multi-species: sum E0[species_index] for each atom
        E0_total = self._params.E0[0] * n_atoms  # TODO: multi-species
        total_energy += E0_total

        self.results['energy'] = float(total_energy)

        if 'forces' in properties or 'stress' in properties:
            # Accumulate pair forces to atomic forces
            # For edge (i -> j) with displacement rij = pos[j] - pos[i]:
            #   F_i += -pair_forces
            #   F_j += +pair_forces
            forces = np.zeros((n_atoms, 3), dtype=np.float64)
            np.add.at(forces, pair_i, -total_pair_forces)
            np.add.at(forces, pair_j, total_pair_forces)
            self.results['forces'] = forces

        if 'stress' in properties:
            # Compute virial from pair forces and displacements
            virial = -np.einsum('ea,eb->ab', total_pair_forces, rij)

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
    def num_contributions(self) -> int:
        """Number of loaded IREE contributions."""
        return len(self._contributions)

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

    Args:
        device: 'auto', 'cuda', 'vulkan', or 'cpu'
        models_dir: Optional custom models directory

    Returns:
        Configured Calculator instance
    """
    return Calculator(device=device, models_dir=models_dir)


# Alias for consistent naming
ACECalculator = Calculator
