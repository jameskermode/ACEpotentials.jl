"""
IREE Runtime Wrapper
====================

Low-level interface to IREE-compiled ACE models.
Handles model loading, input padding, and result extraction.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


class IREEModel:
    """
    Wrapper for IREE-compiled ACE potential model.

    Handles:
    - Loading compiled VMFB modules
    - Input padding to fixed compiled shapes
    - Calling compute_energy and compute_pair_forces functions
    - Output unpacking
    """

    def __init__(
        self,
        vmfb_path: str,
        device: str = 'local-task',
        metadata_path: Optional[str] = None,
    ):
        """
        Load an IREE-compiled model.

        Args:
            vmfb_path: Path to compiled .vmfb file
            device: IREE device string ('local-task', 'cuda', 'vulkan')
            metadata_path: Optional path to metadata.json with model info
        """
        import iree.runtime as iree_rt

        self.vmfb_path = Path(vmfb_path)
        self.device_str = device

        # Load metadata
        if metadata_path:
            self.metadata = self._load_metadata(metadata_path)
        else:
            # Try to find metadata next to vmfb
            meta_path = self.vmfb_path.parent / 'metadata.json'
            if meta_path.exists():
                self.metadata = self._load_metadata(str(meta_path))
            else:
                self.metadata = self._default_metadata()

        # Extract shapes from metadata
        self.max_atoms = self.metadata.get('max_atoms', 4096)
        self.max_pairs = self.metadata.get('max_pairs', 200000)
        self.rcut = self.metadata.get('cutoff', 6.0)

        # Load IREE module
        logger.info(f"Loading VMFB: {vmfb_path}")
        logger.info(f"Device: {device}")

        self.config = iree_rt.Config(device)

        with open(vmfb_path, 'rb') as f:
            vmfb_data = f.read()

        self.context = iree_rt.SystemContext(config=self.config)
        self.vm_module = iree_rt.VmModule.copy_buffer(
            self.context.instance, vmfb_data
        )
        self.context.add_vm_module(self.vm_module)

        # Get function handles
        self._energy_fn = None
        self._pair_forces_fn = None
        self._main_fn = None

        # Try to find exported functions
        try:
            self._energy_fn = self.vm_module.lookup_function('compute_energy')
        except ValueError:
            logger.debug("compute_energy function not found")

        try:
            self._pair_forces_fn = self.vm_module.lookup_function('compute_pair_forces')
        except ValueError:
            logger.debug("compute_pair_forces function not found")

        try:
            self._main_fn = self.vm_module.lookup_function('main')
        except ValueError:
            logger.debug("main function not found")

        if self._energy_fn is None and self._pair_forces_fn is None and self._main_fn is None:
            raise ValueError("No recognized functions found in VMFB module")

        logger.info("IREE model loaded successfully")

    def _load_metadata(self, path: str) -> dict:
        """Load metadata from JSON file."""
        with open(path) as f:
            return json.load(f)

    def _default_metadata(self) -> dict:
        """Default metadata when none provided."""
        return {
            'max_atoms': 4096,
            'max_pairs': 200000,
            'cutoff': 6.0,
            'elements': [],
        }

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

    def compute_energy(
        self,
        rij: np.ndarray,
        pool_matrix: np.ndarray,
        model_params: dict,
    ) -> float:
        """
        Compute total energy.

        Args:
            rij: Pair displacement vectors [n_pairs, 3]
            pool_matrix: Pooling matrix [n_atoms, n_pairs]
            model_params: Dictionary of model parameters

        Returns:
            Total energy (scalar)
        """
        if self._energy_fn is None and self._main_fn is None:
            raise RuntimeError("No energy computation function available")

        # Pad inputs
        n_pairs = rij.shape[0]
        n_atoms = pool_matrix.shape[0]

        rij_padded = self._pad_array(
            rij.astype(np.float32),
            (self.max_pairs, 3)
        )

        pool_padded = self._pad_array(
            pool_matrix.astype(np.float32),
            (self.max_atoms, self.max_pairs)
        )

        # Call IREE
        # Note: Actual signature depends on compiled model
        # This is a placeholder - real implementation needs to match export

        if self._main_fn:
            result = self._main_fn(rij_padded, pool_padded, *self._unpack_params(model_params))
            if isinstance(result, tuple):
                return float(result[0])
            return float(result)

        return 0.0

    def compute_pair_forces(
        self,
        rij: np.ndarray,
        pool_matrix: np.ndarray,
        model_params: dict,
    ) -> np.ndarray:
        """
        Compute per-pair force contributions.

        Args:
            rij: Pair displacement vectors [n_pairs, 3]
            pool_matrix: Pooling matrix [n_atoms, n_pairs]
            model_params: Dictionary of model parameters

        Returns:
            Pair forces [n_pairs, 3]
        """
        n_pairs = rij.shape[0]

        # Pad inputs
        rij_padded = self._pad_array(
            rij.astype(np.float32),
            (self.max_pairs, 3)
        )

        pool_padded = self._pad_array(
            pool_matrix.astype(np.float32),
            (self.max_atoms, self.max_pairs)
        )

        # Call IREE
        # The model returns (energy, pair_forces) from the main function
        # which includes automatic differentiation w.r.t. rij

        if self._main_fn:
            result = self._main_fn(rij_padded, pool_padded, *self._unpack_params(model_params))

            if isinstance(result, tuple) and len(result) >= 2:
                # Result is (energy, pair_forces, ...)
                pair_forces_padded = np.asarray(result[1])
                return pair_forces_padded[:n_pairs]

        return np.zeros((n_pairs, 3), dtype=np.float32)

    def compute_energy_and_forces(
        self,
        rij: np.ndarray,
        pool_matrix: np.ndarray,
        model_params: dict,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute energy and pair forces in one call.

        More efficient than calling compute_energy and compute_pair_forces
        separately when both are needed.

        Args:
            rij: Pair displacement vectors [n_pairs, 3]
            pool_matrix: Pooling matrix [n_atoms, n_pairs]
            model_params: Dictionary of model parameters

        Returns:
            Tuple of (energy, pair_forces)
        """
        n_pairs = rij.shape[0]

        rij_padded = self._pad_array(
            rij.astype(np.float32),
            (self.max_pairs, 3)
        )

        pool_padded = self._pad_array(
            pool_matrix.astype(np.float32),
            (self.max_atoms, self.max_pairs)
        )

        if self._main_fn:
            result = self._main_fn(rij_padded, pool_padded, *self._unpack_params(model_params))

            if isinstance(result, tuple):
                energy = float(result[0])
                pair_forces = np.asarray(result[1])[:n_pairs] if len(result) > 1 else np.zeros((n_pairs, 3), dtype=np.float32)
                return energy, pair_forces

            return float(result), np.zeros((n_pairs, 3), dtype=np.float32)

        return 0.0, np.zeros((n_pairs, 3), dtype=np.float32)

    def _unpack_params(self, params: dict) -> list:
        """Unpack parameter dictionary into list for IREE call."""
        # Order must match model compilation signature
        required_keys = [
            'selector_R', 'selector_Y',
            'symm_sel1', 'symm_sel2_1', 'symm_sel2_2',
            'A2Bmap', 'params'
        ]

        result = []
        for key in required_keys:
            if key in params:
                arr = params[key]
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr, dtype=np.float32)
                result.append(arr.astype(np.float32))

        return result


class ModelParams:
    """
    Container for pre-compiled model parameters.

    Loaded from NPZ file exported during model compilation.
    """

    def __init__(self, npz_path: str):
        """
        Load model parameters from NPZ file.

        Args:
            npz_path: Path to .npz file with model parameters
        """
        self.path = Path(npz_path)

        with np.load(npz_path) as data:
            self.params = {key: data[key] for key in data.files}

        # Extract key parameters
        self.rcut = float(self.params.get('rcut', [6.0])[0])

    def to_dict(self) -> dict:
        """Return parameters as dictionary."""
        return self.params

    def __repr__(self) -> str:
        return f"ModelParams(path={self.path}, keys={list(self.params.keys())})"
