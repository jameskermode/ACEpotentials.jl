"""
Tests for IREE Wrapper Module
=============================
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json


class TestPadArray:
    """Test array padding utility."""

    def test_pad_array_2d_expand(self):
        """Padding should expand array to target shape."""
        from mypotential._iree_wrapper import IREEModel

        # Create instance without loading (just to access method)
        model = object.__new__(IREEModel)

        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        result = model._pad_array(arr, (4, 5), fill_value=0.0)

        assert result.shape == (4, 5)
        assert result[0, 0] == 1.0
        assert result[1, 2] == 6.0
        assert result[2, 0] == 0.0  # Padded region
        assert result[0, 3] == 0.0  # Padded region

    def test_pad_array_1d(self):
        """1D array padding."""
        from mypotential._iree_wrapper import IREEModel

        model = object.__new__(IREEModel)

        arr = np.array([1, 2, 3], dtype=np.float32)
        result = model._pad_array(arr, (10,), fill_value=-1.0)

        assert result.shape == (10,)
        assert result[0] == 1.0
        assert result[2] == 3.0
        assert result[3] == -1.0  # Fill value

    def test_pad_array_preserves_dtype(self):
        """Padding should preserve data type."""
        from mypotential._iree_wrapper import IREEModel

        model = object.__new__(IREEModel)

        arr = np.array([[1, 2]], dtype=np.float64)
        result = model._pad_array(arr, (2, 4), fill_value=0.0)

        assert result.dtype == np.float64

    def test_pad_array_truncates_if_smaller(self):
        """If target is smaller, should truncate."""
        from mypotential._iree_wrapper import IREEModel

        model = object.__new__(IREEModel)

        arr = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)
        result = model._pad_array(arr, (1, 3), fill_value=0.0)

        assert result.shape == (1, 3)
        assert result[0, 0] == 1.0
        assert result[0, 2] == 3.0


class TestModelParams:
    """Test ModelParams class."""

    def test_load_npz(self):
        """Should load parameters from NPZ file."""
        from mypotential._iree_wrapper import ModelParams

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name,
                     rcut=np.array([5.0]),
                     selector_R=np.eye(4, dtype=np.float32),
                     params=np.random.randn(10).astype(np.float32))
            f.flush()

            params = ModelParams(f.name)

            assert params.rcut == 5.0
            assert 'selector_R' in params.params
            assert 'params' in params.params

    def test_to_dict(self):
        """to_dict should return parameter dictionary."""
        from mypotential._iree_wrapper import ModelParams

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            np.savez(f.name,
                     rcut=np.array([6.0]),
                     test_param=np.array([1, 2, 3]))
            f.flush()

            params = ModelParams(f.name)
            d = params.to_dict()

            assert isinstance(d, dict)
            assert 'rcut' in d
            assert 'test_param' in d


class TestIREEModelMetadata:
    """Test metadata loading in IREEModel."""

    def test_default_metadata(self):
        """Should have sensible defaults without metadata file."""
        from mypotential._iree_wrapper import IREEModel

        model = object.__new__(IREEModel)
        metadata = model._default_metadata()

        assert 'max_atoms' in metadata
        assert 'max_pairs' in metadata
        assert 'cutoff' in metadata
        assert metadata['max_atoms'] > 0
        assert metadata['cutoff'] > 0

    def test_load_metadata_from_json(self):
        """Should load metadata from JSON file."""
        from mypotential._iree_wrapper import IREEModel

        model = object.__new__(IREEModel)

        with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
            json.dump({
                'max_atoms': 1000,
                'max_pairs': 50000,
                'cutoff': 4.5,
                'elements': ['Si', 'O'],
            }, f)
            f.flush()

            metadata = model._load_metadata(f.name)

            assert metadata['max_atoms'] == 1000
            assert metadata['cutoff'] == 4.5


class TestIREEModelShapes:
    """Test shape handling in compute functions."""

    def test_compute_energy_returns_scalar(self):
        """Energy should be a scalar float."""
        # This test requires a real VMFB - mark as integration
        pytest.skip("Requires compiled VMFB model")

    def test_compute_pair_forces_shape(self):
        """Pair forces should be [n_pairs, 3]."""
        pytest.skip("Requires compiled VMFB model")


@pytest.mark.integration
class TestIREEModelIntegration:
    """Integration tests requiring actual IREE runtime."""

    @pytest.fixture
    def vmfb_path(self):
        """Get path to test VMFB if available."""
        # Look for test model in various locations
        paths = [
            Path(__file__).parent.parent / 'src/mypotential/models/model_cpu.vmfb',
            Path(__file__).parent / 'fixtures/test_model.vmfb',
        ]
        for p in paths:
            if p.exists():
                return str(p)
        pytest.skip("No test VMFB available")

    def test_model_load(self, vmfb_path):
        """Should load VMFB without error."""
        from mypotential._iree_wrapper import IREEModel

        model = IREEModel(vmfb_path, device='local-task')
        assert model.max_atoms > 0
        assert model.rcut > 0

    def test_compute_energy_and_forces(self, vmfb_path):
        """Should compute energy and forces."""
        from mypotential._iree_wrapper import IREEModel

        model = IREEModel(vmfb_path, device='local-task')

        # Dummy inputs
        n_pairs = 100
        n_atoms = 10
        rij = np.random.randn(n_pairs, 3).astype(np.float32)
        pool_matrix = np.zeros((n_atoms, n_pairs), dtype=np.float32)

        # This would need actual model params
        # energy, forces = model.compute_energy_and_forces(rij, pool_matrix, {})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
