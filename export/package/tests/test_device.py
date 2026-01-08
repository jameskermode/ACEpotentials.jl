"""
Tests for Device Selection Module
=================================
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os


class TestDeviceDetection:
    """Test device driver detection functions."""

    def test_cuda_detection_returns_bool(self):
        """_cuda_available should return boolean."""
        from mypotential._device import _cuda_available
        result = _cuda_available()
        assert isinstance(result, bool)

    def test_vulkan_detection_returns_bool(self):
        """_vulkan_available should return boolean."""
        from mypotential._device import _vulkan_available
        result = _vulkan_available()
        assert isinstance(result, bool)

    def test_cuda_detection_with_mock_failure(self):
        """Test CUDA detection when driver not found."""
        from mypotential._device import _cuda_available

        with patch('ctypes.CDLL', side_effect=OSError("not found")):
            assert _cuda_available() is False

    def test_vulkan_detection_with_mock_failure(self):
        """Test Vulkan detection when driver not found."""
        from mypotential._device import _vulkan_available

        with patch('ctypes.CDLL', side_effect=OSError("not found")):
            assert _vulkan_available() is False


class TestModelsDir:
    """Test models directory functions."""

    def test_get_models_dir_returns_path(self):
        """get_models_dir should return Path object."""
        from mypotential._device import get_models_dir
        result = get_models_dir()
        assert isinstance(result, Path)

    def test_models_dir_is_inside_package(self):
        """Models dir should be inside the mypotential package."""
        from mypotential._device import get_models_dir
        result = get_models_dir()
        assert 'mypotential' in str(result)
        assert result.name == 'models'


class TestListAvailableModels:
    """Test model listing function."""

    def test_list_models_structure(self):
        """list_available_models returns correct structure."""
        from mypotential._device import list_available_models
        result = list_available_models()

        assert 'cpu' in result
        assert 'cuda' in result
        assert 'vulkan' in result

        for backend, (path, exists) in result.items():
            assert isinstance(path, Path)
            assert isinstance(exists, bool)

    def test_list_models_filenames(self):
        """Model filenames follow expected pattern."""
        from mypotential._device import list_available_models
        result = list_available_models()

        # result is a dict: {backend: (path, exists)}
        assert result['cpu'][0].name == 'model_cpu.vmfb'
        assert result['cuda'][0].name == 'model_cuda.vmfb'
        assert result['vulkan'][0].name == 'model_vulkan.vmfb'


class TestSelectDevice:
    """Test device selection logic."""

    def test_select_device_cpu_explicit(self):
        """Explicitly requesting CPU should work."""
        from mypotential._device import select_device

        # Create temp dir with CPU model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy model file
            model_path = Path(tmpdir) / 'model_cpu.vmfb'
            model_path.write_bytes(b'dummy')

            device, path = select_device('cpu', models_dir=Path(tmpdir))
            assert device == 'local-task'
            assert 'cpu' in str(path)

    def test_select_device_aliases(self):
        """Test device name aliases."""
        from mypotential._device import select_device

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'model_cpu.vmfb'
            model_path.write_bytes(b'dummy')

            # 'local-task' should map to CPU
            device, _ = select_device('local-task', models_dir=Path(tmpdir))
            assert device == 'local-task'

    def test_select_device_missing_model(self):
        """Should raise FileNotFoundError for missing model."""
        from mypotential._device import select_device

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                select_device('cuda', models_dir=Path(tmpdir))

    def test_select_device_unknown_device(self):
        """Should raise ValueError for unknown device."""
        from mypotential._device import select_device

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown device"):
                select_device('unknown_device', models_dir=Path(tmpdir))

    def test_select_device_auto_fallback_to_cpu(self):
        """Auto should fall back to CPU when GPU unavailable."""
        from mypotential._device import select_device

        with tempfile.TemporaryDirectory() as tmpdir:
            # Only create CPU model
            (Path(tmpdir) / 'model_cpu.vmfb').write_bytes(b'dummy')

            with patch('mypotential._device._cuda_available', return_value=False):
                with patch('mypotential._device._vulkan_available', return_value=False):
                    with patch('mypotential._device._check_iree_device', return_value=True):
                        device, path = select_device('auto', models_dir=Path(tmpdir))
                        assert device == 'local-task'


class TestGetDeviceInfo:
    """Test device info function."""

    def test_device_info_structure(self):
        """get_device_info returns correct structure."""
        from mypotential._device import get_device_info
        info = get_device_info()

        assert 'cpu' in info
        assert 'cuda' in info
        assert 'vulkan' in info

        for device_info in info.values():
            assert 'driver_available' in device_info
            assert 'iree_available' in device_info
            assert 'model_available' in device_info
            assert 'model_path' in device_info

    def test_cpu_always_has_driver(self):
        """CPU driver should always be available."""
        from mypotential._device import get_device_info
        info = get_device_info()
        assert info['cpu']['driver_available'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
