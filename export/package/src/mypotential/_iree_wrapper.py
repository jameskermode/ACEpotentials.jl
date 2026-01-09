"""
IREE Runtime Wrapper
====================

Low-level interface to IREE-compiled energy contributions.

Architecture:
- Each contribution is a separate VMFB file: <name>_<backend>.vmfb
- Each returns (energy, pair_forces) when called
- Calculator loads all available contributions and sums results
- E0 (one-body) is handled separately in Python (constant per atom)

This design is extensible - new contributions can be added without
modifying existing code, as long as they follow the interface.
"""

from pathlib import Path
from typing import Tuple, List, Optional, Callable
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class IREEContribution:
    """
    Generic IREE-compiled energy contribution.

    Each contribution takes rij and returns (energy, pair_forces).
    The specific parameters passed depend on the contribution type,
    determined by the VMFB's expected inputs.
    """

    def __init__(
        self,
        vmfb_path: str,
        device: str = 'local-task',
        max_pairs: int = 10000,
        max_atoms: int = 256,
    ):
        """
        Load an IREE-compiled contribution.

        Args:
            vmfb_path: Path to compiled .vmfb file
            device: IREE device string ('local-task', 'cuda', 'vulkan')
            max_pairs: Maximum number of pairs (for padding)
            max_atoms: Maximum number of atoms (for padding)
        """
        import iree.runtime as iree_rt

        self.vmfb_path = Path(vmfb_path)
        self.name = self.vmfb_path.stem.rsplit('_', 1)[0]  # e.g., "ace" from "ace_cpu.vmfb"
        self.device_str = device
        self.max_pairs = max_pairs
        self.max_atoms = max_atoms

        logger.info(f"Loading contribution '{self.name}' from {vmfb_path}")

        self.config = iree_rt.Config(device)

        with open(vmfb_path, 'rb') as f:
            vmfb_data = f.read()

        self.context = iree_rt.SystemContext(config=self.config)
        self.vm_module = iree_rt.VmModule.copy_buffer(
            self.context.instance, vmfb_data
        )
        self.context.add_vm_module(self.vm_module)

        self._bound_module = self.context.modules[self.vm_module.name]
        self._main_fn = self._find_function()

        logger.info(f"Contribution '{self.name}' loaded successfully")

    def _find_function(self):
        """Find the main entry point function."""
        # Try common function names
        candidates = [
            f'{self.name}_energy_and_forces',
            'main',
            'energy_and_forces',
        ]

        for fn_name in candidates:
            try:
                fn = getattr(self._bound_module, fn_name, None)
                if fn is not None:
                    logger.debug(f"Found function: {fn_name}")
                    return fn
            except Exception:
                pass

        available = [x for x in dir(self._bound_module) if not x.startswith('_')]
        raise ValueError(f"No entry point found for '{self.name}'. Available: {available}")

    def _pad_array(
        self,
        arr: np.ndarray,
        target_shape: Tuple[int, ...],
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Pad array to target shape."""
        result = np.full(target_shape, fill_value, dtype=arr.dtype)
        slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
        result[slices] = arr[slices]
        return result

    def _pad_rij(self, rij: np.ndarray, max_pairs: int, rcut: float) -> np.ndarray:
        """
        Pad rij array, using large displacements for padding (beyond cutoff).

        This is critical: zero-displacement padding causes numerical explosions
        in the outer envelope computation. Padding with r > rcut ensures
        padded pairs contribute zero energy due to the cutoff mask.
        """
        n_pairs = rij.shape[0]
        result = np.zeros((max_pairs, 3), dtype=rij.dtype)
        result[:n_pairs] = rij

        # Pad remaining with large displacement (2x cutoff) to ensure zero contribution
        if n_pairs < max_pairs:
            result[n_pairs:, 0] = rcut * 2.0  # Large x-displacement

        return result

    def __call__(
        self,
        rij: np.ndarray,
        pair_i: np.ndarray,
        n_atoms: int,
        params: 'ModelParams',
    ) -> Tuple[float, np.ndarray]:
        """
        Compute energy and pair forces for this contribution.

        Args:
            rij: Edge displacement vectors [n_pairs, 3]
            pair_i: Source atom indices [n_pairs]
            n_atoms: Number of atoms
            params: Model parameters

        Returns:
            (energy, pair_forces) where pair_forces is [n_pairs, 3]
        """
        n_pairs = rij.shape[0]

        # Prepare common inputs - use special padding for rij that avoids zero-distance artifacts
        rij_padded = self._pad_rij(rij.astype(np.float32), self.max_pairs, params.rcut)
        rij_T = rij_padded.T.astype(np.float32)

        # Get contribution-specific inputs
        inputs = self._prepare_inputs(rij_T, pair_i, n_atoms, n_pairs, params)

        # Call IREE
        result = self._main_fn(*inputs)

        return self._unpack_result(result, n_pairs)

    def _prepare_inputs(
        self,
        rij_T: np.ndarray,
        pair_i: np.ndarray,
        n_atoms: int,
        n_pairs: int,
        params: 'ModelParams',
    ) -> list:
        """
        Prepare inputs for this contribution type.

        Override in subclasses or use contribution-specific logic.
        """
        # Default: assume ACE-like inputs (pool matrix + selection matrices)
        # Build pool matrix
        pool_matrix = np.zeros((n_atoms, n_pairs), dtype=np.float32)
        for e in range(n_pairs):
            pool_matrix[pair_i[e], e] = 1.0

        pool_padded = self._pad_array(pool_matrix, (self.max_atoms, self.max_pairs))
        pool_T = pool_padded.T.astype(np.float32)

        return [
            rij_T,
            pool_T,
            params.selector_R,
            params.selector_Y,
            params.symm_sel1,
            params.symm_sel2_1,
            params.symm_sel2_2,
            params.A2Bmap,
            params.readout_params,
            params.W_radial,
        ]

    def _unpack_result(self, result, n_pairs: int) -> Tuple[float, np.ndarray]:
        """Unpack IREE result into (energy, pair_forces)."""
        if isinstance(result, (list, tuple)):
            energy_raw = result[0]
            pair_forces_raw = result[1] if len(result) > 1 else None
        else:
            energy_raw = result
            pair_forces_raw = None

        if hasattr(energy_raw, 'to_host'):
            energy_raw = energy_raw.to_host()
        energy = float(np.asarray(energy_raw))

        if pair_forces_raw is not None:
            if hasattr(pair_forces_raw, 'to_host'):
                pair_forces_raw = pair_forces_raw.to_host()
            pair_forces_padded = np.asarray(pair_forces_raw).T
            pair_forces = pair_forces_padded[:n_pairs].astype(np.float32)
        else:
            pair_forces = np.zeros((n_pairs, 3), dtype=np.float32)

        return energy, pair_forces


class PairContribution(IREEContribution):
    """
    Pair contribution - simpler inputs (no pool matrix needed).
    """

    def _prepare_inputs(
        self,
        rij_T: np.ndarray,
        pair_i: np.ndarray,
        n_atoms: int,
        n_pairs: int,
        params: 'ModelParams',
    ) -> list:
        """Pair only needs rij and its own weights."""
        return [
            rij_T,
            params.pair_W_radial_T,
            params.pair_W_readout,
        ]


def load_contribution(vmfb_path: str, device: str, max_pairs: int, max_atoms: int) -> IREEContribution:
    """
    Load a contribution, using the appropriate class based on the name.
    """
    name = Path(vmfb_path).stem.rsplit('_', 1)[0]

    # Use specialized class if available
    if name == 'pair':
        return PairContribution(vmfb_path, device, max_pairs, max_atoms)
    else:
        return IREEContribution(vmfb_path, device, max_pairs, max_atoms)


class ModelParams:
    """
    Container for model parameters.

    Loads from NPZ file exported by create_package.jl.
    Parameters are organized by contribution but accessed uniformly.
    """

    def __init__(self, npz_path: str):
        self.path = Path(npz_path)

        with np.load(npz_path) as data:
            self._params = {key: data[key] for key in data.files}

        # Core parameters
        self.rcut = float(self._params.get('rcut', [6.0])[0])
        self.n_polys = int(self._params.get('n_polys', [4])[0])
        self.n_rnl = int(self._params.get('n_rnl', self._params.get('n_polys', [4]))[0])
        self.maxl = int(self._params.get('maxl', [2])[0])
        self.n_species = int(self._params.get('n_species', [1])[0])
        self.species_Z = self._params.get('species_Z', np.array([14]))

        # One-body E0 (handled in Python, not IREE)
        self.E0 = self._params.get('E0', np.zeros(self.n_species)).astype(np.float64)

        # ACE parameters (transpose for column-major convention)
        self.selector_R = self._params['selector_R'].T.astype(np.float32)
        self.selector_Y = self._params['selector_Y'].T.astype(np.float32)
        self.symm_sel1 = self._params['symm_sel1'].T.astype(np.float32)
        self.symm_sel2_1 = self._params['symm_sel2_1'].T.astype(np.float32)
        self.symm_sel2_2 = self._params['symm_sel2_2'].T.astype(np.float32)
        self.A2Bmap = self._params['A2Bmap'].T.astype(np.float32)
        self.readout_params = self._params['params'].astype(np.float32)
        W_radial_default = np.eye(self.n_rnl, self.n_polys, dtype=np.float32)
        self.W_radial = self._params.get('W_radial', W_radial_default).T.astype(np.float32)

        # Pair parameters (if present)
        self.has_pair = int(self._params.get('has_pair', [0])[0]) > 0
        if self.has_pair:
            # pair_W_radial has shape (n_basis, n_polys, n_species_pairs)
            # For IREE we need (n_polys, n_basis), so squeeze species dim and transpose
            pair_W = self._params['pair_W_radial']
            if pair_W.ndim == 3:
                pair_W = pair_W[:, :, 0]  # Take first species pair for now
            self.pair_W_radial_T = pair_W.T.astype(np.float32)  # (n_polys, n_basis)
            self.pair_W_readout = self._params['pair_W_readout'][:, 0].astype(np.float32)

    def __repr__(self) -> str:
        return f"ModelParams(rcut={self.rcut}, n_species={self.n_species}, has_pair={self.has_pair})"


# Legacy alias
IREEModel = IREEContribution
