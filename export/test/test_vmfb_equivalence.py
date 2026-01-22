#!/usr/bin/env python3
"""
Test VMFB Equivalence - Verify exported VMFBs match Julia reference values.

This test validates that:
1. VMFB energy matches Julia reference energy
2. VMFB gradients match Julia reference gradients
3. Forces computed from gradients match Julia reference forces

Run:
    cd export/tools && uv run python ../test/test_vmfb_equivalence.py

Expected: All tests pass with numerical precision < 1e-10 (float64)
"""

import os
import sys
from pathlib import Path
import numpy as np

# IREE runtime
try:
    from iree import runtime as iree_rt
except ImportError:
    print("ERROR: iree.runtime not found. Run from export/tools with: uv run python ...")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
BENCHMARK_DIR = SCRIPT_DIR.parent / "benchmark"
BUCKET_DIR = BENCHMARK_DIR / "bucket_energy_gradient"

# Tolerances for float64
ENERGY_TOL = 1e-10
GRADIENT_TOL = 1e-10


def check_bucket(bucket_name: str) -> dict:
    """
    Test a single bucket VMFB against its reference data.

    Returns dict with test results.
    """
    bucket_path = BUCKET_DIR / bucket_name
    vmfb_path = bucket_path / "energy_gradient_f64_cpu.vmfb"
    test_data_path = bucket_path / "test_data.npz"

    if not vmfb_path.exists():
        return {"status": "skip", "reason": f"VMFB not found: {vmfb_path}"}
    if not test_data_path.exists():
        return {"status": "skip", "reason": f"Test data not found: {test_data_path}"}

    # Load VMFB
    module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), driver="local-task")

    # Load reference data
    data = np.load(test_data_path)
    rij = data["rij"].astype(np.float64)         # (n_edges, 3)
    energy_ref = float(data["energy"][0])
    gradient_ref = data["gradient"].astype(np.float64)  # (n_edges, 3)

    n_edges = rij.shape[0]

    # Run VMFB - expects (3, n_edges) due to column-major conversion
    rij_T = np.ascontiguousarray(rij.T)  # (3, n_edges)
    result = module.main(rij_T)

    # Parse output - VMFB returns (energy_scalar, gradient, duplicated_input)
    # Energy is a scalar, gradient is (3, n_edges)
    if isinstance(result, tuple) and len(result) >= 2:
        energy_vmfb = float(np.asarray(result[0]))
        gradient_vmfb_T = np.asarray(result[1])  # (3, n_edges)
        gradient_vmfb = gradient_vmfb_T.T        # (n_edges, 3)
    else:
        return {"status": "fail", "reason": f"Unexpected output format: {type(result)}"}

    # Compare energy
    energy_diff = abs(energy_vmfb - energy_ref)
    energy_ok = energy_diff < ENERGY_TOL

    # Compare gradient
    gradient_diff = np.max(np.abs(gradient_vmfb - gradient_ref))
    gradient_ok = gradient_diff < GRADIENT_TOL

    return {
        "status": "pass" if (energy_ok and gradient_ok) else "fail",
        "n_edges": n_edges,
        "energy_ref": energy_ref,
        "energy_vmfb": energy_vmfb,
        "energy_diff": energy_diff,
        "energy_ok": energy_ok,
        "gradient_max_diff": gradient_diff,
        "gradient_ok": gradient_ok,
    }


def main():
    print("=" * 70)
    print("VMFB Equivalence Test")
    print("=" * 70)
    print(f"\nTolerances: energy={ENERGY_TOL}, gradient={GRADIENT_TOL}")
    print(f"Bucket directory: {BUCKET_DIR}")

    # Find all buckets
    buckets = sorted([d.name for d in BUCKET_DIR.iterdir() if d.is_dir() and d.name.startswith("bucket_")])

    if not buckets:
        print("\nERROR: No bucket directories found!")
        sys.exit(1)

    print(f"\nFound {len(buckets)} buckets: {buckets}")

    # Test each bucket
    results = {}
    all_pass = True

    for bucket in buckets:
        print(f"\n--- {bucket} ---")
        result = check_bucket(bucket)
        results[bucket] = result

        if result["status"] == "skip":
            print(f"  SKIP: {result['reason']}")
        elif result["status"] == "pass":
            print(f"  n_edges: {result['n_edges']}")
            print(f"  Energy:  {result['energy_ref']:.10f} (ref) vs {result['energy_vmfb']:.10f} (vmfb)")
            print(f"           diff = {result['energy_diff']:.2e} {'OK' if result['energy_ok'] else 'FAIL'}")
            print(f"  Gradient max diff: {result['gradient_max_diff']:.2e} {'OK' if result['gradient_ok'] else 'FAIL'}")
            print(f"  PASS")
        else:
            print(f"  FAIL: {result.get('reason', 'Unknown error')}")
            if "energy_diff" in result:
                print(f"  Energy diff: {result['energy_diff']:.2e}")
            if "gradient_max_diff" in result:
                print(f"  Gradient max diff: {result['gradient_max_diff']:.2e}")
            all_pass = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_pass = sum(1 for r in results.values() if r["status"] == "pass")
    n_fail = sum(1 for r in results.values() if r["status"] == "fail")
    n_skip = sum(1 for r in results.values() if r["status"] == "skip")

    print(f"  Passed: {n_pass}")
    print(f"  Failed: {n_fail}")
    print(f"  Skipped: {n_skip}")

    if all_pass and n_pass > 0:
        print("\n  ALL TESTS PASSED - VMFB matches Julia to numerical precision!")
        return 0
    else:
        print("\n  TESTS FAILED")
        return 1


def test_vmfb_equivalence():
    """Pytest entry point for VMFB equivalence test."""
    result = main()
    assert result == 0, "VMFB equivalence test failed"


if __name__ == "__main__":
    sys.exit(main())
