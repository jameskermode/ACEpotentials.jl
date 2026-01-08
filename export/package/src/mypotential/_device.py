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


def list_available_models() -> dict:
    """
    List available pre-compiled model files.

    Returns:
        Dictionary mapping backend names to (path, available) tuples
    """
    models_dir = get_models_dir()

    backends = {
        'cpu': 'model_cpu.vmfb',
        'cuda': 'model_cuda.vmfb',
        'vulkan': 'model_vulkan.vmfb',
    }

    available = {}
    for backend, filename in backends.items():
        path = models_dir / filename
        available[backend] = (path, path.exists())

    return available


def select_device(
    requested: str = 'auto',
    models_dir: Optional[Path] = None
) -> Tuple[str, Path]:
    """
    Select the best available device and corresponding model file.

    Args:
        requested: Device to use. Options:
            - 'auto': Auto-select best available (CUDA > Vulkan > CPU)
            - 'cuda': Use CUDA GPU
            - 'vulkan': Use Vulkan GPU
            - 'cpu': Use CPU
        models_dir: Optional custom directory containing model files.
            If not specified, uses the package's built-in models directory.

    Returns:
        Tuple of (device_string, model_path) where:
            - device_string: IREE device string (e.g., 'cuda', 'vulkan', 'local-task')
            - model_path: Path to the corresponding .vmfb file

    Raises:
        RuntimeError: If no suitable device/model combination is available
    """
    if models_dir is None:
        models_dir = get_models_dir()
    else:
        models_dir = Path(models_dir)

    # Device configurations: (backend_name, iree_device_string, model_filename)
    device_configs = [
        ('cuda', 'cuda', 'model_cuda.vmfb'),
        ('vulkan', 'vulkan', 'model_vulkan.vmfb'),
        ('cpu', 'local-task', 'model_cpu.vmfb'),
    ]

    if requested == 'auto':
        # Try devices in priority order
        for backend, iree_device, model_file in device_configs:
            model_path = models_dir / model_file

            if not model_path.exists():
                logger.debug(f"Model file not found: {model_path}")
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

            logger.info(f"Selected device: {backend} ({iree_device})")
            return iree_device, model_path

        raise RuntimeError(
            "No suitable device found. Available models: "
            f"{list(models_dir.glob('*.vmfb'))}"
        )

    else:
        # User requested specific device
        requested_lower = requested.lower()

        device_map = {
            'cuda': ('cuda', 'model_cuda.vmfb'),
            'gpu': ('cuda', 'model_cuda.vmfb'),  # Alias
            'vulkan': ('vulkan', 'model_vulkan.vmfb'),
            'cpu': ('local-task', 'model_cpu.vmfb'),
            'local-task': ('local-task', 'model_cpu.vmfb'),  # Alias
        }

        if requested_lower not in device_map:
            raise ValueError(
                f"Unknown device: {requested}. "
                f"Options: {list(device_map.keys())}"
            )

        iree_device, model_file = device_map[requested_lower]
        model_path = models_dir / model_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found for {requested}: {model_path}"
            )

        return iree_device, model_path


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns:
        Dictionary with device availability information
    """
    models = list_available_models()

    info = {
        'cuda': {
            'driver_available': _cuda_available(),
            'iree_available': _check_iree_device('cuda') if _cuda_available() else False,
            'model_available': models['cuda'][1],
            'model_path': str(models['cuda'][0]),
        },
        'vulkan': {
            'driver_available': _vulkan_available(),
            'iree_available': _check_iree_device('vulkan') if _vulkan_available() else False,
            'model_available': models['vulkan'][1],
            'model_path': str(models['vulkan'][0]),
        },
        'cpu': {
            'driver_available': True,  # Always available
            'iree_available': _check_iree_device('local-task'),
            'model_available': models['cpu'][1],
            'model_path': str(models['cpu'][0]),
        },
    }

    return info


def print_device_info():
    """Print device availability information."""
    info = get_device_info()

    print("Device Availability")
    print("=" * 50)

    for device, details in info.items():
        status = []
        if details['driver_available'] and details['iree_available'] and details['model_available']:
            status.append("READY")
        else:
            if not details['driver_available']:
                status.append("no driver")
            if not details['iree_available']:
                status.append("IREE error")
            if not details['model_available']:
                status.append("no model")

        status_str = ", ".join(status) if status else "unknown"
        print(f"  {device:8s}: {status_str}")

    print()

    # Show which device would be selected
    try:
        selected_device, selected_path = select_device('auto')
        print(f"Auto-selected: {selected_device}")
        print(f"Model file: {selected_path}")
    except RuntimeError as e:
        print(f"Auto-selection failed: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print_device_info()
