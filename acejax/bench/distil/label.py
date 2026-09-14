"""Label an xyz set with a mace-jax bundle: energies, forces, virials.

Runs `mace_jax_predict` (GPU if available), then writes an extxyz with the keys
`fit_distilled.jl` reads (`mace_energy`, `mace_force`, `mace_virial`).

Virial convention: ACEpotentials defines virial = -sum_i dV_i (x) R_i, which is
-V * sigma with sigma the ASE stress MACE reports.  FD-verified in RESULTS.md.
The virial is written as a flat 9-vector: ExtXYZ.jl parses a nested [[..]]
list as characters.

`--fmax` drops structures whose largest force exceeds the threshold AFTER
labelling: a foundation model is not a reliable teacher where it is
extrapolating, and a few such cells dominated the test error of the first
round (RESULTS.md, "per-structure breakdown").

    python label.py bundles/mace-mh-1.msgpack in.xyz out.xyz --fmax 10
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np
from ase.io import read, write

ap = argparse.ArgumentParser()
ap.add_argument("bundle")
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--fmax", type=float, default=None, help="drop structures with max|F| above this (eV/A)")
ap.add_argument("--python", default=sys.executable)
a = ap.parse_args()

ats = read(a.inp, ":")
with tempfile.TemporaryDirectory() as td:
    npz = os.path.join(td, "lab.npz")
    cmd = [a.python, "-m", "mace_jax.cli.mace_jax_predict", a.bundle, a.inp,
           "--output", npz, "--dtype", "float64",
           "--compute-forces", "--compute-stress", "--no-progress"]
    subprocess.run(cmd, check=True)
    d = np.load(npz, allow_pickle=True)
    assert len(d["energy"]) == len(ats), (len(d["energy"]), len(ats))
    order = np.argsort(d["graph_id"])
    E, F, S = d["energy"][order], d["forces"][order], d["stress"][order]

kept, dropped = [], []
for at, e, f, s in zip(ats, E, F, S):
    f = np.asarray(f, dtype=float)
    assert f.shape == (len(at), 3)
    if a.fmax is not None and np.abs(f).max() > a.fmax:
        dropped.append((np.abs(f).max(), at.get_volume() / len(at)))
        continue
    at.info["mace_energy"] = float(e)
    at.arrays["mace_force"] = f
    at.info["mace_virial"] = (-at.get_volume() * np.asarray(s, dtype=float)).reshape(-1)
    kept.append(at)

write(a.out, kept)
fm = np.array([np.abs(x.arrays["mace_force"]).max() for x in kept])
print(f"wrote {len(kept)} structures to {a.out}; dropped {len(dropped)} with max|F| > {a.fmax}")
print(f"kept: max|F| median {np.median(fm):.2f}, max {fm.max():.2f} eV/A")
for m, v in sorted(dropped, reverse=True)[:10]:
    print(f"  dropped: max|F| {m:6.2f}  vol/atom {v:5.2f}")
