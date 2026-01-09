"""
Device Selection for IREE Runtime
=================================

Automatically selects the best available compute device (CUDA > Vulkan > CPU)
and finds the corresponding pre-compiled model file.
"""

from pathlib import Path
from typing import Tuple, Optional
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


def list_available_backends(models_dir: Optional[Path] = None) -> dict:
    """
    List available pre-compiled model backends.

    With contribution-based architecture, each backend has multiple VMFB files
    named <contribution>_<backend>.vmfb (e.g., ace_cpu.vmfb, pair_cpu.vmfb).

    Returns:
        Dictionary mapping backend names to list of available VMFB paths
    """
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir)

    backend_suffixes = {
        'cpu': '_cpu.vmfb',
        'cuda': '_cuda.vmfb',
        'vulkan': '_vulkan.vmfb',
    }

    available = {}
    for backend, suffix in backend_suffixes.items():
        vmfb_files = list(models_dir.glob(f'*{suffix}'))
        available[backend] = vmfb_files

    return available


def select_device(
    requested: str = 'auto',
    models_dir: Optional[Path] = None
) -> Tuple[str, Path]:
    """
    Select the best available device and models directory.

    With contribution-based architecture, each backend has multiple VMFB files
    named <contribution>_<backend>.vmfb. This function selects the device
    and returns the models directory (not a specific model file).

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

    # Backend suffixes for contribution-based naming
    backend_suffixes = {
        'cuda': '_cuda.vmfb',
        'vulkan': '_vulkan.vmfb',
        'cpu': '_cpu.vmfb',
    }

    # Device configurations: (backend_name, iree_device_string)
    device_configs = [
        ('cuda', 'cuda'),
        ('vulkan', 'vulkan'),
        ('cpu', 'local-task'),
    ]

    if requested == 'auto':
        # Try devices in priority order
        for backend, iree_device in device_configs:
            suffix = backend_suffixes[backend]
            vmfb_files = list(models_dir.glob(f'*{suffix}'))

            if not vmfb_files:
                logger.debug(f"No VMFB files found for {backend}: *{suffix}")
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

            logger.info(f"Selected device: {backend} ({iree_device}), {len(vmfb_files)} contributions")
            return iree_device, models_dir

        raise RuntimeError(
            "No suitable device found. Available models: "
            f"{list(models_dir.glob('*.vmfb'))}"
        )

    else:
        # User requested specific device
        requested_lower = requested.lower()

        device_map = {
            'cuda': 'cuda',
            'gpu': 'cuda',  # Alias
            'vulkan': 'vulkan',
            'cpu': 'local-task',
            'local-task': 'local-task',  # Alias
        }

        if requested_lower not in device_map:
            raise ValueError(
                f"Unknown device: {requested}. "
                f"Options: {list(device_map.keys())}"
            )

        iree_device = device_map[requested_lower]

        # Get backend name for suffix lookup
        backend_name = 'cpu' if iree_device == 'local-task' else iree_device
        suffix = backend_suffixes.get(backend_name, '_cpu.vmfb')
        vmfb_files = list(models_dir.glob(f'*{suffix}'))

        if not vmfb_files:
            raise FileNotFoundError(
                f"No VMFB files found for {requested}: {models_dir}/*{suffix}"
            )

        return iree_device, models_dir


def get_device_info(models_dir: Optional[Path] = None) -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device availability information
    """
    backends = list_available_backends(models_dir)

    info = {
        'cuda': {
            'driver_available': _cuda_available(),
            'iree_available': _check_iree_device('cuda') if _cuda_available() else False,
            'contributions': len(backends['cuda']),
            'vmfb_files': [str(p) for p in backends['cuda']],
        },
        'vulkan': {
            'driver_available': _vulkan_available(),
            'iree_available': _check_iree_device('vulkan') if _vulkan_available() else False,
            'contributions': len(backends['vulkan']),
            'vmfb_files': [str(p) for p in backends['vulkan']],
        },
        'cpu': {
            'driver_available': True,  # Always available
            'iree_available': _check_iree_device('local-task'),
            'contributions': len(backends['cpu']),
            'vmfb_files': [str(p) for p in backends['cpu']],
        },
    }

    return info


def print_device_info(models_dir: Optional[Path] = None):
    """Print device availability information."""
    info = get_device_info(models_dir)

    print("Device Availability")
    print("=" * 50)

    for device, details in info.items():
        status = []
        n_contrib = details['contributions']
        if details['driver_available'] and details['iree_available'] and n_contrib > 0:
            status.append(f"READY ({n_contrib} contributions)")
        else:
            if not details['driver_available']:
                status.append("no driver")
            if not details['iree_available']:
                status.append("IREE error")
            if n_contrib == 0:
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
