#!/usr/bin/env python3
"""Export a ladder of bundles that vary max_atoms ALONE, at fixed max_edges.

Why this shape.  `pair jax/kk` retains 63/60/34% of raw acejax-in-Python
throughput at n_B = 69/710/2849 (results.md, "The plugin gap across sizes").  A
*fixed* overhead -- call boundary, neighbour build, transfers -- would amortise
away as the model grows; this one does not, which points at a cost that scales
with per-atom work.  The suspect is the atom axis: `export_bundle.py` passes
`positions.shape[0]`, i.e. max_atoms, as the node count to `site_energies`, and
`lammps_jax/export.py:wrap_energy_fn` masks rows only AFTER they are computed.
So site energies are evaluated for every ghost and pad row and then discarded.

That is an inference from three points.  This sweeps the axis directly.

max_edges is held EXACTLY fixed because the edge capacity has its own U-shape
(results.md, "Bundle capacity is a tuning parameter"), which would confound the
reading.  export_bundle.py takes --edges-per-atom rather than --max-edges, so
the ladder is max_atoms = E // k for integer k: every rung multiplies back to
the same E.

Floor: at 1728 atoms LAMMPS reports Nlocal 1728 + Nghost 3645 = nall 5373, and
the pair style aborts when nall > max_atoms.  The k=22 rung (5040) sits below
that deliberately, to pin the floor by observation rather than by arithmetic.

CPU-only: `jax.export.export(..., platforms=("cuda",))` lowers for CUDA
regardless of the local backend, so this runs under JAX_PLATFORMS=cpu and never
touches the device.
"""
import argparse
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

# 110880 = 2^5 * 3^2 * 5 * 7 * 11, chosen for its divisors: every rung below is
# an exact integer edges-per-atom.  110880 / 76920 actual edges = 1.44x, which
# is the margin the plotted series already runs at (1.40x), so the edge axis
# stays in the same regime it was measured in.
MAX_EDGES = 110880
DIVISORS = [22, 20, 18, 16, 14, 12, 10, 9, 8, 7, 6, 5]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--outdir", type=pathlib.Path, required=True)
    p.add_argument("--precision", choices=["float64", "float32"], default="float64")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--divisors", type=int, nargs="+", default=DIVISORS)
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    for k in a.divisors:
        ma = MAX_EDGES // k
        assert ma * k == MAX_EDGES, (ma, k)
        out = a.outdir / f"cap_a{ma}.lammps-jax.json"
        print(f"max_atoms={ma:>6}  edges_per_atom={k:>3}  max_edges={ma*k}  -> {out.name}",
              flush=True)
        subprocess.run([a.python, str(HERE.parent / "lammps" / "export_bundle.py"),
                        "--npz", str(a.npz), "--out", str(out),
                        "--max-atoms", str(ma), "--edges-per-atom", str(k),
                        "--precision", a.precision, "--a2b-sparse"],
                       check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
