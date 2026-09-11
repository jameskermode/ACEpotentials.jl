#!/usr/bin/env python3
"""Export one route-3 bundle per (model, size), capacities sized to that point."""
import json, math, os, pathlib, subprocess, sys
A_SI = 5.43
LJ = os.path.expanduser("~/si-ace/lammps-jax/examples/export_mace.py")
ROOT = pathlib.Path("/storage/eng/essswb/phase13")

def capacities(reps, rcut, n_hops, skin=1.0, atom_margin=1.25, edge_margin=1.4):
    L = A_SI*reps; n = 8*reps**3; rho = n/L**3
    rc = n_hops*rcut + skin
    nall = n + rho*((L+2*rc)**3 - L**3)
    ma = int(math.ceil(nall*atom_margin))
    rows = nall if n_hops > 1 else n
    me = int(math.ceil(rows*(4/3)*math.pi*rcut**3*rho*edge_margin))
    return n, int(nall), ma, me

def main():
    tag, bdir, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    reps_list = [int(r) for r in sys.argv[4:]]
    cfg = json.loads((ROOT/bdir/"config.json").read_text())
    rcut = float(cfg["r_max"]); nh = 1 if mode == "comm" else int(cfg["num_interactions"])
    for reps in reps_list:
        n, nall, ma, me = capacities(reps, rcut, nh)
        out = ROOT/"bundles"/f"{tag}_{mode}_r{reps}_n{n}.lammps-jax.json"
        if out.exists():
            print("skip", out.name, flush=True); continue
        cmd = [sys.executable, LJ, "export", "plain", str(out), "--mode", mode,
               "--type-z", "14", "--bundle-dir", str(ROOT/bdir),
               "--max-atoms", str(ma), "--max-edges", str(me), "--skip-check"]
        if mode == "comm": cmd += ["--owned-rows", str(n)]
        print(f"{tag} {mode} reps={reps} n={n} nall~{nall} ma={ma} me={me}", flush=True)
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
