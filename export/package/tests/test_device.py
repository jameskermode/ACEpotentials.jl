"""
Tests for Device Selection Module
=================================

Tests for bucket-based VMFB detection and device selection.
"""

import pytest
from pathlib import Path
from unittest.mock import patch
import tempfile


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


class TestListAvailableBackends:
    """Test backend listing function."""

    def test_list_backends_structure(self):
        """list_available_backends returns correct structure."""
        from mypotential._device import list_available_backends

        # Use a temp dir to get predictable results
        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_available_backends(Path(tmpdir))

            assert 'cpu' in result
            assert 'cuda' in result
            assert 'vulkan' in result

            for backend, files in result.items():
                assert isinstance(files, list)

    def test_list_backends_finds_bucket_vmfbs(self):
        """list_available_backends finds bucket-based VMFBs."""
        from mypotential._device import list_available_backends

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create bucket directory structure
            bucket_dir = tmpdir / 'bucket_2000'
            bucket_dir.mkdir()
            (bucket_dir / 'energy_gradient_f64_cpu.vmfb').write_bytes(b'dummy')
            (bucket_dir / 'energy_gradient_f64_cuda.vmfb').write_bytes(b'dummy')

            result = list_available_backends(tmpdir)

            assert len(result['cpu']) == 1
            assert len(result['cuda']) == 1
            assert len(result['vulkan']) == 0
            assert 'bucket_2000' in str(result['cpu'][0])


class TestSelectDevice:
    """Test device selection logic."""

    def test_select_device_cpu_explicit(self):
        """Explicitly requesting CPU should work."""
        from mypotential._device import select_device

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create bucket VMFB
            bucket_dir = tmpdir / 'bucket_2000'
            bucket_dir.mkdir()
            (bucket_dir / 'energy_gradient_f64_cpu.vmfb').write_bytes(b'dummy')

            device, path = select_device('cpu', models_dir=tmpdir)
            assert device == 'local-task'
            assert path == tmpdir

    def test_select_device_aliases(self):
        """Test device name aliases."""
        from mypotential._device import select_device

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create bucket VMFB
            bucket_dir = tmpdir / 'bucket_2000'
            bucket_dir.mkdir()
            (bucket_dir / 'energy_gradient_f64_cpu.vmfb').write_bytes(b'dummy')

            # 'local-task' should map to CPU
            device, _ = select_device('local-task', models_dir=tmpdir)
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
            tmpdir = Path(tmpdir)

            # Create bucket VMFB for CPU
            bucket_dir = tmpdir / 'bucket_2000'
            bucket_dir.mkdir()
            (bucket_dir / 'energy_gradient_f64_cpu.vmfb').write_bytes(b'dummy')

            with patch('mypotential._device._cuda_available', return_value=False):
                with patch('mypotential._device._vulkan_available', return_value=False):
                    with patch('mypotential._device._check_iree_device', return_value=True):
                        device, path = select_device('auto', models_dir=tmpdir)
                        assert device == 'local-task'


class TestGetDeviceInfo:
    """Test device info function."""

    def test_device_info_structure(self):
        """get_device_info returns correct structure."""
        from mypotential._device import get_device_info

        with tempfile.TemporaryDirectory() as tmpdir:
            info = get_device_info(Path(tmpdir))

            assert 'cpu' in info
            assert 'cuda' in info
            assert 'vulkan' in info

            for device_info in info.values():
                assert 'driver_available' in device_info
                assert 'iree_available' in device_info
                assert 'vmfb_count' in device_info
                assert 'vmfb_files' in device_info

    def test_cpu_always_has_driver(self):
        """CPU driver should always be available."""
        from mypotential._device import get_device_info

        with tempfile.TemporaryDirectory() as tmpdir:
            info = get_device_info(Path(tmpdir))
            assert info['cpu']['driver_available'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
