#!/usr/bin/env python3
"""
Test Bucket Energy+Gradient VMFBs

Quick validation that the production bucket VMFBs work correctly.
For comprehensive tests, see: export/test/test_vmfb_equivalence.py

Usage:
    cd export/tools && uv run python ../benchmark/test_enzyme_gradient_vmfb.py
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).parent

# IREE runtime
try:
    from iree import runtime as iree_rt
except ImportError:
    print("ERROR: iree.runtime not found. Run from export/tools with: uv run python ...")
    sys.exit(1)

print("=" * 70)
print("Test Bucket Energy+Gradient VMFBs")
print("=" * 70)

# Test the smallest bucket
bucket_dir = script_dir / "bucket_energy_gradient" / "bucket_2000"
vmfb_path = bucket_dir / "energy_gradient_f64_cpu.vmfb"

if not vmfb_path.exists():
    print(f"ERROR: VMFB not found: {vmfb_path}")
    print("Run: julia +1.11 --project=export export/benchmark/export_bucket_energy_gradient.jl")
    sys.exit(1)

# Load VMFB
print(f"\nLoading VMFB: {vmfb_path}")
module = iree_rt.load_vm_flatbuffer_file(str(vmfb_path), driver="local-task")

# Load test data
test_data = np.load(bucket_dir / "test_data.npz")
rij = test_data["rij"].astype(np.float64)      # (n_edges, 3)
energy_ref = float(test_data["energy"][0])
grad_ref = test_data["gradient"].astype(np.float64)  # (n_edges, 3)

n_edges = rij.shape[0]
print(f"Input: {n_edges} edges")
print(f"Reference energy: {energy_ref:.10f}")
print(f"Reference max gradient: {np.max(np.abs(grad_ref)):.6f}")

# Run VMFB
rij_T = np.ascontiguousarray(rij.T)  # (3, n_edges) for IREE
print("\n--- Running VMFB ---")
result = module.main(rij_T)

# Parse output - VMFB returns (energy_scalar, gradient, duplicated_input)
energy_vmfb = float(np.asarray(result[0]))
grad_vmfb_T = np.asarray(result[1])  # (3, n_edges)
grad_vmfb = grad_vmfb_T.T            # (n_edges, 3)

print(f"VMFB energy: {energy_vmfb:.10f}")
print(f"VMFB max gradient: {np.max(np.abs(grad_vmfb)):.6f}")

# Compare
energy_diff = abs(energy_vmfb - energy_ref)
grad_diff = np.max(np.abs(grad_vmfb - grad_ref))

print(f"\n--- Comparison ---")
print(f"Energy diff: {energy_diff:.2e}")
print(f"Gradient max diff: {grad_diff:.2e}")

TOLERANCE = 1e-10
energy_ok = energy_diff < TOLERANCE
grad_ok = grad_diff < TOLERANCE

if energy_ok and grad_ok:
    print(f"\nPASS - VMFB matches Julia reference (tol={TOLERANCE})")
else:
    print(f"\nFAIL - Mismatch exceeds tolerance {TOLERANCE}")
    sys.exit(1)

# Quick benchmark
print("\n--- Benchmark ---")
n_iter = 100

# Warmup
for _ in range(5):
    _ = module.main(rij_T)

# Time
t0 = time.perf_counter()
for _ in range(n_iter):
    _ = module.main(rij_T)
t_ms = (time.perf_counter() - t0) / n_iter * 1000

print(f"  {n_edges} edges: {t_ms:.3f} ms ({t_ms*1000/n_edges:.2f} us/edge)")

print("\n" + "=" * 70)
print("TEST PASSED")
print("=" * 70)
