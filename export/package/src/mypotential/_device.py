"""
Device Selection for IREE Runtime
=================================

Automatically selects the best available compute device (CUDA > Vulkan > CPU)
and finds the corresponding pre-compiled model files.

Supports two directory layouts:
1. Bucket-based: bucket_2000/energy_gradient_f64_cpu.vmfb, etc.
2. Flat: model_cpu.vmfb, model_cuda.vmfb (legacy)
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    """Check if CUDA driver is available."""
    try:
        import ctypes
        ctypes.CDLL('libcuda.so.1')
        return True
    except OSError:
        return False


def _vulkan_available() -> bool:
    """Check if Vulkan driver is available."""
    try:
        import ctypes
        ctypes.CDLL('libvulkan.so.1')
        return True
    except OSError:
        return False


def _check_iree_device(device: str) -> bool:
    """Check if IREE can create the specified device."""
    try:
        import iree.runtime as iree_rt
        config = iree_rt.Config(device)
        return True
    except Exception:
        return False


def get_models_dir() -> Path:
    """Get the models directory for this package."""
    return Path(__file__).parent / "models"


def _find_bucket_vmfbs(models_dir: Path, backend: str) -> List[Path]:
    """
    Find bucket VMFBs for a given backend.

    Searches for bucket_*/ directories containing energy_gradient_f64_*.vmfb files.
    """
    suffix_map = {
        'cpu': 'f64_cpu.vmfb',
        'cuda': 'f64_cuda.vmfb',
        'vulkan': 'f64_vulkan.vmfb',
    }
    suffix = suffix_map.get(backend, 'f64_cpu.vmfb')

    vmfb_files = []
    for bucket_dir in models_dir.glob('bucket_*'):
        if bucket_dir.is_dir():
            vmfb = bucket_dir / f'energy_gradient_{suffix}'
            if vmfb.exists():
                vmfb_files.append(vmfb)

    return sorted(vmfb_files)


def _find_flat_vmfbs(models_dir: Path, backend: str) -> List[Path]:
    """
    Find flat-layout VMFBs for a given backend (legacy support).

    Searches for *_<backend>.vmfb files directly in models_dir.
    """
    suffix_map = {
        'cpu': '_cpu.vmfb',
        'cuda': '_cuda.vmfb',
        'vulkan': '_vulkan.vmfb',
    }
    suffix = suffix_map.get(backend, '_cpu.vmfb')
    return list(models_dir.glob(f'*{suffix}'))


def list_available_backends(models_dir: Optional[Path] = None) -> Dict[str, List[Path]]:
    """
    List available pre-compiled model backends.

    Checks both bucket-based and flat layouts.

    Returns:
        Dictionary mapping backend names to list of available VMFB paths
    """
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir)

    available = {}
    for backend in ['cpu', 'cuda', 'vulkan']:
        # Try bucket layout first
        bucket_files = _find_bucket_vmfbs(models_dir, backend)
        if bucket_files:
            available[backend] = bucket_files
        else:
            # Fall back to flat layout
            available[backend] = _find_flat_vmfbs(models_dir, backend)

    return available


def select_device(
    requested: str = 'auto',
    models_dir: Optional[Path] = None
) -> Tuple[str, Path]:
    """
    Select the best available device and models directory.

    Supports bucket-based VMFB layout (bucket_*/energy_gradient_*.vmfb)
    and flat layout (*_cpu.vmfb, *_cuda.vmfb).

    Args:
        requested: Device to use. Options:
            - 'auto': Auto-select best available (CUDA > Vulkan > CPU)
            - 'cuda': Use CUDA GPU
            - 'vulkan': Use Vulkan GPU
            - 'cpu': Use CPU
        models_dir: Optional custom directory containing model files.
            If not specified, uses the package's built-in models directory.

    Returns:
        Tuple of (device_string, models_dir) where:
            - device_string: IREE device string (e.g., 'cuda', 'vulkan', 'local-task')
            - models_dir: Path to the directory containing VMFB files

    Raises:
        RuntimeError: If no suitable device/model combination is available
    """
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir)

    # Device configurations: (backend_name, iree_device_string)
    device_configs = [
        ('cuda', 'cuda'),
        ('vulkan', 'vulkan'),
        ('cpu', 'local-task'),
    ]

    if requested == 'auto':
        # Try devices in priority order
        for backend, iree_device in device_configs:
            # Check for VMFBs (bucket or flat layout)
            bucket_files = _find_bucket_vmfbs(models_dir, backend)
            flat_files = _find_flat_vmfbs(models_dir, backend)
            vmfb_files = bucket_files or flat_files

            if not vmfb_files:
                logger.debug(f"No VMFB files found for {backend}")
                continue

            # Check device availability
            if backend == 'cuda' and not _cuda_available():
                logger.debug("CUDA not available")
                continue
            elif backend == 'vulkan' and not _vulkan_available():
                logger.debug("Vulkan not available")
                continue

            # Verify IREE can use this device
            if not _check_iree_device(iree_device):
                logger.debug(f"IREE cannot use device: {iree_device}")
                continue

            layout = 'bucket' if bucket_files else 'flat'
            logger.info(f"Selected device: {backend} ({iree_device}), {len(vmfb_files)} VMFBs ({layout})")
            return iree_device, models_dir

        raise RuntimeError(
            f"No suitable device found. Models directory: {models_dir}, "
            f"Available: {list(models_dir.glob('*.vmfb'))} and {list(models_dir.glob('bucket_*'))}"
        )

    else:
        # User requested specific device
        requested_lower = requested.lower()

        device_map = {
            'cuda': ('cuda', 'cuda'),
            'gpu': ('cuda', 'cuda'),  # Alias
            'vulkan': ('vulkan', 'vulkan'),
            'cpu': ('cpu', 'local-task'),
            'local-task': ('cpu', 'local-task'),  # Alias
        }

        if requested_lower not in device_map:
            raise ValueError(
                f"Unknown device: {requested}. "
                f"Options: {list(device_map.keys())}"
            )

        backend, iree_device = device_map[requested_lower]

        # Check for VMFBs
        bucket_files = _find_bucket_vmfbs(models_dir, backend)
        flat_files = _find_flat_vmfbs(models_dir, backend)
        vmfb_files = bucket_files or flat_files

        if not vmfb_files:
            raise FileNotFoundError(
                f"No VMFB files found for {requested} in {models_dir}. "
                f"Expected bucket_*/energy_gradient_f64_{backend}.vmfb or *_{backend}.vmfb"
            )

        return iree_device, models_dir


def get_device_info(models_dir: Optional[Path] = None) -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device availability information
    """
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir)

    backends = list_available_backends(models_dir)

    info = {}
    for backend in ['cuda', 'vulkan', 'cpu']:
        vmfb_files = backends.get(backend, [])
        iree_device = 'local-task' if backend == 'cpu' else backend

        driver_available = True
        if backend == 'cuda':
            driver_available = _cuda_available()
        elif backend == 'vulkan':
            driver_available = _vulkan_available()

        info[backend] = {
            'driver_available': driver_available,
            'iree_available': _check_iree_device(iree_device) if driver_available else False,
            'vmfb_count': len(vmfb_files),
            'vmfb_files': [str(p) for p in vmfb_files],
        }

    return info


def print_device_info(models_dir: Optional[Path] = None):
    """Print device availability information."""
    info = get_device_info(models_dir)

    print("Device Availability")
    print("=" * 50)

    for device, details in info.items():
        status = []
        n_vmfbs = details['vmfb_count']
        if details['driver_available'] and details['iree_available'] and n_vmfbs > 0:
            status.append(f"READY ({n_vmfbs} VMFBs)")
        else:
            if not details['driver_available']:
                status.append("no driver")
            if not details['iree_available']:
                status.append("IREE error")
            if n_vmfbs == 0:
                status.append("no models")

        status_str = ", ".join(status) if status else "unknown"
        print(f"  {device:8s}: {status_str}")

    print()

    # Show which device would be selected
    try:
        selected_device, selected_dir = select_device('auto', models_dir)
        print(f"Auto-selected: {selected_device}")
        print(f"Models directory: {selected_dir}")
    except RuntimeError as e:
        print(f"Auto-selection failed: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print_device_info()
