#!/usr/bin/env python3
"""
Test VMFB Equivalence - Verify exported VMFBs match Julia reference values.

This test validates that:
1. VMFB energy matches Julia reference energy
2. VMFB gradients match Julia reference gradients

Run:
    cd export/tools && uv run pytest ../test/test_vmfb_equivalence.py -v

Expected: All tests pass with numerical precision < 1e-10 (float64)
"""

import pytest
import numpy as np
from pathlib import Path

from iree import runtime as iree_rt

SCRIPT_DIR = Path(__file__).parent
BENCHMARK_DIR = SCRIPT_DIR.parent / "benchmark"
BUCKET_DIR = BENCHMARK_DIR / "bucket_energy_gradient"

# Tolerances for float64
ENERGY_TOL = 1e-10
GRADIENT_TOL = 1e-10


def get_available_buckets():
    """Get list of bucket directories that have both VMFB and test data."""
    buckets = []
    if BUCKET_DIR.exists():
        for d in sorted(BUCKET_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("bucket_"):
                vmfb = d / "energy_gradient_f64_cpu.vmfb"
                data = d / "test_data.npz"
                if vmfb.exists() and data.exists():
                    buckets.append(d.name)
    return buckets


AVAILABLE_BUCKETS = get_available_buckets()


@pytest.fixture(params=AVAILABLE_BUCKETS)
def bucket_name(request):
    """Parametrized fixture for each available bucket."""
    return request.param


@pytest.fixture
def bucket_data(bucket_name):
    """Load VMFB module and reference data for a bucket."""
    bucket_path = BUCKET_DIR / bucket_name
    vmfb_path = bucket_path / "energy_gradient_f64_cpu.vmfb"
    test_data_path = bucket_path / "test_data.npz"

    # Load VMFB
    module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), driver="local-task")

    # Load reference data
    data = np.load(test_data_path)
    rij = data["rij"].astype(np.float64)
    energy_ref = float(data["energy"][0])
    gradient_ref = data["gradient"].astype(np.float64)

    return {
        "module": module,
        "rij": rij,
        "energy_ref": energy_ref,
        "gradient_ref": gradient_ref,
        "n_edges": rij.shape[0],
    }


class TestVMFBEquivalence:
    """Test that VMFB output matches Julia reference values."""

    def test_energy_matches_reference(self, bucket_name, bucket_data):
        """VMFB energy should match Julia reference to numerical precision."""
        module = bucket_data["module"]
        rij = bucket_data["rij"]
        energy_ref = bucket_data["energy_ref"]

        # Run VMFB - expects (3, n_edges) due to column-major conversion
        rij_T = np.ascontiguousarray(rij.T)
        result = module.main(rij_T)

        energy_vmfb = float(np.asarray(result[0]))
        energy_diff = abs(energy_vmfb - energy_ref)

        assert energy_diff < ENERGY_TOL, (
            f"Energy mismatch in {bucket_name}: "
            f"ref={energy_ref:.10f}, vmfb={energy_vmfb:.10f}, diff={energy_diff:.2e}"
        )

    def test_gradient_matches_reference(self, bucket_name, bucket_data):
        """VMFB gradient should match Julia reference to numerical precision."""
        module = bucket_data["module"]
        rij = bucket_data["rij"]
        gradient_ref = bucket_data["gradient_ref"]

        # Run VMFB
        rij_T = np.ascontiguousarray(rij.T)
        result = module.main(rij_T)

        gradient_vmfb_T = np.asarray(result[1])  # (3, n_edges)
        gradient_vmfb = gradient_vmfb_T.T  # (n_edges, 3)

        gradient_diff = np.max(np.abs(gradient_vmfb - gradient_ref))

        assert gradient_diff < GRADIENT_TOL, (
            f"Gradient mismatch in {bucket_name}: max_diff={gradient_diff:.2e}"
        )


class TestVMFBOutput:
    """Test VMFB output format and properties."""

    def test_output_is_tuple(self, bucket_data):
        """VMFB should return a tuple."""
        module = bucket_data["module"]
        rij = bucket_data["rij"]

        rij_T = np.ascontiguousarray(rij.T)
        result = module.main(rij_T)

        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) >= 2, f"Expected at least 2 outputs, got {len(result)}"

    def test_energy_is_scalar(self, bucket_data):
        """Energy output should be a scalar."""
        module = bucket_data["module"]
        rij = bucket_data["rij"]

        rij_T = np.ascontiguousarray(rij.T)
        result = module.main(rij_T)

        energy = np.asarray(result[0])
        assert energy.size == 1, f"Energy should be scalar, got shape {energy.shape}"

    def test_gradient_shape(self, bucket_data):
        """Gradient should have shape (3, n_edges)."""
        module = bucket_data["module"]
        rij = bucket_data["rij"]
        n_edges = bucket_data["n_edges"]

        rij_T = np.ascontiguousarray(rij.T)
        result = module.main(rij_T)

        gradient = np.asarray(result[1])
        assert gradient.shape == (3, n_edges), (
            f"Expected gradient shape (3, {n_edges}), got {gradient.shape}"
        )
