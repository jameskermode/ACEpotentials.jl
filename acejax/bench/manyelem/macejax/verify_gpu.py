#!/usr/bin/env python3
"""Verify that MACE-JAX evaluates CORRECTLY on the GPU, not merely that it runs.

The same MACE torch checkpoint is evaluated two independent ways on the same
structure:

  reference : mace-torch  MACECalculator, float64, CPU
  test      : mace-jax    bundle converted from that checkpoint, float64, GPU

and the total energy and forces are compared.  Both sides are float64, so the
tolerances below are AGREEMENT tolerances between two implementations, not
precision tolerances -- they are deliberately tight.

Usage:
  verify_gpu.py --checkpoint <torch.model> --bundle <dir-or-params.msgpack>
                [--etol 1e-6] [--ftol 1e-5]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile

# Do not grab the whole card: a co-tenant has been OOM-d by that before.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def build_structure():
    from ase.build import bulk
    at = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))  # 64 atoms
    # Break the perfect symmetry so forces are non-trivial and a sign or
    # index error cannot hide behind an all-zero force array.
    rng = np.random.default_rng(0)
    at.positions += 0.05 * rng.standard_normal(at.positions.shape)
    return at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", required=True)
    p.add_argument("--etol", type=float, default=1e-6, help="eV per atom")
    p.add_argument("--ftol", type=float, default=1e-5, help="eV/A")
    a = p.parse_args()

    atoms = build_structure()
    n = len(atoms)

    # ---------------- JAX / GPU side ----------------
    import jax
    devices = jax.devices()
    print(f"jax {jax.__version__}  devices: {devices}", flush=True)
    if not any(d.platform == "gpu" for d in devices):
        print("FAIL: no GPU device visible to jax -- refusing to report a GPU result")
        return 2
    print(f"GPU: {devices[0].device_kind}", flush=True)

    from ase.io import write as ase_write
    from mace_jax.cli.mace_jax_predict import main as predict_main

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="macejax-verify-"))
    xyz = tmp / "structure.xyz"
    out = tmp / "pred.npz"
    ase_write(xyz, atoms, format="extxyz")

    predict_main([a.bundle, str(xyz), "--output", str(out),
                  "--dtype", "float64", "--compute-forces", "--no-progress"])

    pred = np.load(out, allow_pickle=True)
    e_jax = float(np.asarray(pred["energy"]).reshape(-1)[0])
    f_jax = np.asarray(pred["forces"], dtype=float).reshape(-1, 3)[:n]

    # Confirm the work really landed on the GPU rather than silently on CPU.
    plats = {d.platform for d in jax.devices()}
    print(f"jax backend platform(s): {plats}", flush=True)

    # ---------------- torch reference ----------------
    from mace.calculators import MACECalculator
    ref = atoms.copy()
    ref.calc = MACECalculator(model_paths=a.checkpoint, device="cpu",
                              default_dtype="float64")
    e_ref = float(ref.get_potential_energy())
    f_ref = np.asarray(ref.get_forces(), dtype=float)

    # ---------------- compare ----------------
    de = abs(e_jax - e_ref) / n
    df = float(np.max(np.abs(f_jax - f_ref)))
    fscale = float(np.max(np.abs(f_ref)))

    print()
    print(f"  atoms                {n}")
    print(f"  E (mace-jax, GPU)    {e_jax:.9f} eV")
    print(f"  E (mace-torch, CPU)  {e_ref:.9f} eV")
    print(f"  dE/atom              {de:.3e} eV      (tol {a.etol:.1e})")
    print(f"  |F| scale            {fscale:.6f} eV/A")
    print(f"  max|dF|              {df:.3e} eV/A    (tol {a.ftol:.1e})")
    ok = de <= a.etol and df <= a.ftol
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
