#!/usr/bin/env python3
"""Export one bundle per benchmark point, with capacities sized to that point.

This matters more than it looks.  Phase 6's bundle has max_edges 163840 against
9880 actually used; benchmarking at ~16x over-capacity measures padding, not the
model, because the exported program has static shapes and evaluates every padded
edge.  Capacities here are the analytic estimate for a diamond lattice plus a
margin, so each point is measured near its real working set.
"""
import argparse, json, math, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
A_SI = 5.43


def capacities(reps, rcut, skin=1.0, atom_margin=1.25, edge_margin=2.0):
    """Analytic ghost/edge counts for an 8-atom-cell diamond supercell."""
    L = A_SI * reps
    n = 8 * reps**3
    rho = n / L**3
    rc = rcut + skin
    ghosts = rho * ((L + 2*rc)**3 - L**3)
    max_atoms = int(math.ceil((n + ghosts) * atom_margin))
    # full pairing within rcut (the pair style filters the skin away)
    edges = n * (4/3) * math.pi * rcut**3 * rho
    max_edges = int(math.ceil(edges * edge_margin))
    return n, max_atoms, max_edges


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    p.add_argument("--outdir", type=pathlib.Path, default=HERE / "bundles")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--precision", choices=["float64", "float32"], default="float64")
    p.add_argument("--edge-margin", type=float, default=2.0)
    p.add_argument("--a2b-sparse", action="store_true")
    a = p.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(bytes(__import__("numpy").load(a.npz)["meta_json"]).decode())
    rcut = float(meta["rcut"])
    for reps in a.reps:
        n, ma, me = capacities(reps, rcut, edge_margin=a.edge_margin)
        out = a.outdir / f"si_r{reps}_n{n}.lammps-jax.json"
        print(f"reps={reps} atoms={n} max_atoms={ma} max_edges={me} -> {out.name}", flush=True)
        subprocess.run([a.python, str(HERE.parent / "lammps" / "export_bundle.py"),
                        "--npz", str(a.npz), "--out", str(out),
                        "--max-atoms", str(ma),
                        "--edges-per-atom", str(max(1, -(-me // ma))),
                        "--precision", a.precision]
                       + (["--a2b-sparse"] if a.a2b_sparse else []),
                       check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
