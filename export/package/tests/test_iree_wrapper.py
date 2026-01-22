"""
Tests for IREE Wrapper Module
=============================

Tests for BucketVMFB and BucketManager classes.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile


# Find models directory for integration tests
BENCHMARK_MODELS_DIR = Path(__file__).parent.parent.parent / 'benchmark/bucket_energy_gradient'

def get_models_dir():
    """Get available models directory."""
    if BENCHMARK_MODELS_DIR.exists():
        bucket_dirs = list(BENCHMARK_MODELS_DIR.glob('bucket_*'))
        if bucket_dirs:
            return BENCHMARK_MODELS_DIR
    return None

MODELS_DIR = get_models_dir()
HAS_MODEL = MODELS_DIR is not None

skip_no_model = pytest.mark.skipif(not HAS_MODEL, reason="No compiled model (bucket VMFBs)")


class TestBucketVMFBClass:
    """Test BucketVMFB class structure."""

    def test_import(self):
        """Should be able to import BucketVMFB."""
        from mypotential._iree_wrapper import BucketVMFB
        assert BucketVMFB is not None

    def test_bucket_manager_import(self):
        """Should be able to import BucketManager."""
        from mypotential._iree_wrapper import BucketManager
        assert BucketManager is not None


@skip_no_model
class TestBucketVMFBIntegration:
    """Integration tests for BucketVMFB."""

    def test_bucket_vmfb_load(self):
        """Should load bucket VMFB successfully."""
        from mypotential._iree_wrapper import BucketVMFB

        # Find a bucket VMFB - sort by numeric value
        bucket_dirs = sorted(
            MODELS_DIR.glob('bucket_*'),
            key=lambda d: int(d.name.split('_')[1])
        )
        assert len(bucket_dirs) > 0

        # Use smallest bucket
        bucket_dir = bucket_dirs[0]
        max_edges = int(bucket_dir.name.split('_')[1])
        vmfb_path = bucket_dir / 'energy_gradient_f64_cpu.vmfb'
        assert vmfb_path.exists()

        # Load it
        bucket_vmfb = BucketVMFB(str(vmfb_path), 'local-task', max_edges)
        assert bucket_vmfb.max_edges > 0

    def test_bucket_vmfb_compute(self):
        """BucketVMFB should compute energy and gradient."""
        from mypotential._iree_wrapper import BucketVMFB

        # Find smallest bucket - sort by numeric value
        bucket_dirs = sorted(
            MODELS_DIR.glob('bucket_*'),
            key=lambda d: int(d.name.split('_')[1])
        )
        bucket_dir = bucket_dirs[0]
        max_edges = int(bucket_dir.name.split('_')[1])
        vmfb_path = bucket_dir / 'energy_gradient_f64_cpu.vmfb'

        bucket_vmfb = BucketVMFB(str(vmfb_path), 'local-task', max_edges)

        # Create test input (smaller than bucket size)
        n_edges = min(100, bucket_vmfb.max_edges // 2)
        np.random.seed(42)
        rij = np.random.randn(n_edges, 3).astype(np.float64) * 0.5

        # Compute
        energy, gradient = bucket_vmfb(rij, rcut=5.5)

        # Check outputs
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert gradient.shape == (n_edges, 3)
        assert np.all(np.isfinite(gradient))


@skip_no_model
class TestBucketManagerIntegration:
    """Integration tests for BucketManager."""

    def test_bucket_manager_creation(self):
        """Should create BucketManager successfully."""
        from mypotential._iree_wrapper import BucketManager

        mgr = BucketManager(MODELS_DIR, 'local-task')
        assert mgr.num_buckets > 0
        assert mgr.max_edges > 0

    def test_bucket_manager_compute(self):
        """BucketManager should compute energy and gradient."""
        from mypotential._iree_wrapper import BucketManager

        mgr = BucketManager(MODELS_DIR, 'local-task')

        # Create test input
        n_edges = 100
        np.random.seed(42)
        rij = np.random.randn(n_edges, 3).astype(np.float64) * 0.5

        # Compute
        energy, gradient = mgr(rij, rcut=5.5)

        # Check outputs
        assert isinstance(energy, float)
        assert np.isfinite(energy)
        assert gradient.shape == (n_edges, 3)
        assert np.all(np.isfinite(gradient))

    def test_bucket_manager_selects_appropriate_bucket(self):
        """BucketManager should auto-select bucket based on input size."""
        from mypotential._iree_wrapper import BucketManager

        mgr = BucketManager(MODELS_DIR, 'local-task')

        # Test with different sizes
        for n_edges in [50, 500, 1500]:
            if n_edges > mgr.max_edges:
                continue

            np.random.seed(42)
            rij = np.random.randn(n_edges, 3).astype(np.float64) * 0.5

            energy, gradient = mgr(rij, rcut=5.5)

            assert np.isfinite(energy)
            assert gradient.shape == (n_edges, 3)

    def test_bucket_manager_deterministic(self):
        """Same input should give same output."""
        from mypotential._iree_wrapper import BucketManager

        mgr = BucketManager(MODELS_DIR, 'local-task')

        np.random.seed(42)
        rij = np.random.randn(100, 3).astype(np.float64) * 0.5

        energy1, grad1 = mgr(rij, rcut=5.5)
        energy2, grad2 = mgr(rij, rcut=5.5)

        assert energy1 == energy2
        np.testing.assert_array_equal(grad1, grad2)


@skip_no_model
class TestBucketManagerWithTestData:
    """Test BucketManager against reference test data."""

    def test_matches_test_data(self):
        """BucketManager should match test_data.npz reference."""
        from mypotential._iree_wrapper import BucketManager

        mgr = BucketManager(MODELS_DIR, 'local-task')

        # Load test data from first bucket
        bucket_dirs = sorted(MODELS_DIR.glob('bucket_*'))
        test_data_path = bucket_dirs[0] / 'test_data.npz'

        if not test_data_path.exists():
            pytest.skip("No test_data.npz available")

        data = np.load(test_data_path)
        rij = data['rij'].astype(np.float64)
        energy_ref = float(data['energy'][0])
        gradient_ref = data['gradient'].astype(np.float64)

        # Compute via BucketManager
        energy, gradient = mgr(rij, rcut=5.5)

        # Check (should match to numerical precision)
        energy_diff = abs(energy - energy_ref)
        assert energy_diff < 1e-10, f"Energy diff: {energy_diff}"

        grad_diff = np.max(np.abs(gradient - gradient_ref))
        assert grad_diff < 1e-10, f"Gradient diff: {grad_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
