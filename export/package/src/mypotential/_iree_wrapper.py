"""
IREE Runtime Wrapper
====================

Low-level interface to IREE-compiled energy+gradient VMFBs.

Architecture:
- Uses bucket-based VMFBs: each bucket handles a maximum number of edges
- VMFBs compute energy and dE/drij gradient in one call
- Calculator selects smallest bucket that fits the system
- Forces = -gradient, scattered to atoms

VMFB Interface (bucket_energy_gradient):
- Input:  rij.T (3, n_edges) float64 - edge displacement vectors
- Output: (energy, gradient, _) where:
  - energy: scalar float64
  - gradient: (3, n_edges) float64 - dE/drij
"""

from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BucketVMFB:
    """
    Single bucket VMFB for energy+gradient computation.

    Each bucket handles up to max_edges edges. At runtime, the calculator
    selects the smallest bucket that can fit the system.
    """

    def __init__(
        self,
        vmfb_path: str,
        device: str = 'local-task',
        max_edges: int = 2000,
    ):
        """
        Load a bucket VMFB.

        Args:
            vmfb_path: Path to compiled .vmfb file
            device: IREE device string ('local-task', 'cuda', 'vulkan')
            max_edges: Maximum edges this bucket can handle
        """
        import iree.runtime as iree_rt

        self.vmfb_path = Path(vmfb_path)
        self.device_str = device
        self.max_edges = max_edges

        logger.info(f"Loading bucket VMFB: {vmfb_path} (max_edges={max_edges})")

        # Load VMFB
        self.module = iree_rt.load_vm_flatbuffer_file(
            str(vmfb_path),
            driver=device
        )

        logger.info(f"Bucket VMFB loaded: max_edges={max_edges}")

    def __call__(
        self,
        rij: np.ndarray,
        rcut: float = 5.5,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute energy and gradient for given edge displacements.

        Args:
            rij: Edge displacement vectors [n_edges, 3] float64
            rcut: Cutoff radius (for padding beyond cutoff)

        Returns:
            (energy, gradient) where:
            - energy: scalar float64
            - gradient: [n_edges, 3] float64 - dE/drij
        """
        n_edges = rij.shape[0]

        if n_edges > self.max_edges:
            raise ValueError(
                f"System has {n_edges} edges but bucket only supports {self.max_edges}. "
                "Use a larger bucket."
            )

        # Pad rij to bucket size
        # Use large displacement (2x cutoff) for padding to ensure zero contribution
        rij_padded = np.zeros((self.max_edges, 3), dtype=np.float64)
        rij_padded[:n_edges] = rij
        rij_padded[n_edges:, 0] = rcut * 2.0  # Beyond cutoff

        # VMFB expects (3, n_edges) due to column-major conversion
        rij_T = np.ascontiguousarray(rij_padded.T)

        # Call VMFB
        result = self.module.main(rij_T)

        # Parse output: (energy_scalar, gradient (3, max_edges), duplicate)
        energy = float(np.asarray(result[0]))
        gradient_T = np.asarray(result[1])  # (3, max_edges)
        gradient = gradient_T.T[:n_edges]   # (n_edges, 3), unpad

        return energy, gradient


class BucketManager:
    """
    Manages multiple bucket VMFBs and selects appropriate one at runtime.
    """

    def __init__(
        self,
        models_dir: Path,
        device: str = 'local-task',
    ):
        """
        Load all available bucket VMFBs from models directory.

        Args:
            models_dir: Directory containing bucket_*/energy_gradient_*.vmfb
            device: IREE device string
        """
        self.device = device
        self.buckets: List[BucketVMFB] = []

        # Map device to VMFB suffix
        vmfb_suffix = {
            'cuda': 'f64_cuda.vmfb',
            'local-task': 'f64_cpu.vmfb',
        }.get(device, 'f64_cpu.vmfb')

        # Find bucket directories
        bucket_dirs = sorted(
            [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('bucket_')],
            key=lambda d: int(d.name.split('_')[1])  # Sort by max_edges
        )

        if not bucket_dirs:
            raise FileNotFoundError(
                f"No bucket directories found in {models_dir}. "
                "Expected bucket_*/ directories with energy_gradient_*.vmfb files."
            )

        # Load each bucket
        for bucket_dir in bucket_dirs:
            max_edges = int(bucket_dir.name.split('_')[1])
            vmfb_path = bucket_dir / f"energy_gradient_{vmfb_suffix}"

            if not vmfb_path.exists():
                logger.warning(f"VMFB not found: {vmfb_path}")
                continue

            try:
                bucket = BucketVMFB(str(vmfb_path), device, max_edges)
                self.buckets.append(bucket)
            except Exception as e:
                logger.warning(f"Failed to load bucket {bucket_dir.name}: {e}")

        if not self.buckets:
            raise RuntimeError(f"No bucket VMFBs loaded from {models_dir}")

        logger.info(f"Loaded {len(self.buckets)} buckets: {[b.max_edges for b in self.buckets]}")

    @property
    def num_buckets(self) -> int:
        """Number of loaded buckets."""
        return len(self.buckets)

    @property
    def max_edges(self) -> int:
        """Maximum edges supported (from largest bucket)."""
        return self.buckets[-1].max_edges if self.buckets else 0

    def select_bucket(self, n_edges: int) -> BucketVMFB:
        """Select smallest bucket that can handle n_edges."""
        for bucket in self.buckets:
            if n_edges <= bucket.max_edges:
                return bucket

        # No bucket large enough
        max_supported = self.buckets[-1].max_edges if self.buckets else 0
        raise ValueError(
            f"System has {n_edges} edges but largest bucket only supports {max_supported}. "
            "Generate a larger bucket VMFB."
        )

    def __call__(
        self,
        rij: np.ndarray,
        rcut: float = 5.5,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute energy and gradient, auto-selecting appropriate bucket.

        Args:
            rij: Edge displacement vectors [n_edges, 3] float64
            rcut: Cutoff radius

        Returns:
            (energy, gradient) - see BucketVMFB.__call__
        """
        bucket = self.select_bucket(rij.shape[0])
        return bucket(rij, rcut)


# ============================================================================
# Legacy support for old-style ModelParams (used by create_package.jl)
# ============================================================================

class ModelParams:
    """
    Container for model parameters.

    For bucket-based VMFBs, only E0 and rcut are needed.
    The VMFB has all other parameters baked in.
    """

    def __init__(self, npz_path: str):
        self.path = Path(npz_path)

        with np.load(npz_path) as data:
            self._params = {key: data[key] for key in data.files}

        # Core parameters
        self.rcut = float(self._params.get('rcut', [5.5])[0])
        self.n_species = int(self._params.get('n_species', [1])[0])

        # One-body E0 (handled in Python, not IREE)
        self.E0 = self._params.get('E0', np.zeros(self.n_species)).astype(np.float64)

    def __repr__(self) -> str:
        return f"ModelParams(rcut={self.rcut}, n_species={self.n_species})"


# ============================================================================
# Legacy aliases for backward compatibility
# ============================================================================

# Old-style contribution interface (not used with bucket VMFBs)
class IREEContribution:
    """Legacy: Old-style contribution interface. Use BucketVMFB for new code."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "IREEContribution is deprecated. Use BucketManager for bucket-based VMFBs."
        )


def load_contribution(*args, **kwargs):
    """Legacy: Use BucketManager instead."""
    raise NotImplementedError(
        "load_contribution is deprecated. Use BucketManager for bucket-based VMFBs."
    )
