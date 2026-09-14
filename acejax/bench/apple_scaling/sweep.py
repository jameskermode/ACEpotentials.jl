"""Driver for the Apple-Silicon scaling investigation: run ace_md_cap.py in a
fresh subprocess per point and append its RESULT json line to a jsonl file.

    python sweep.py <name> <jsonl-out> -- <points file>

Each point is one line of `key=value` pairs passed through to ace_md_cap.py.
A fresh process per point is deliberate: XLA caches compiled executables per
shape, and reusing a process would let one point's compilation warm another's.
"""
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ACEJAX = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = os.path.join(ACEJAX, ".venv", "bin", "python")
SP = os.environ.get("LJAX") or os.path.expanduser("~/gits/lammps-jax/python")
# `subgraph_params` exists only on lammps-jax `main`; set LJAX to a checkout of it.
MODEL = os.path.join(ACEJAX, "fixtures", "si_fitted.npz")


def run_point(args, extra_env=None, script="ace_md_cap.py"):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ACEJAX, "spike_distmd") + ":" + SP
    env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    if extra_env:
        env.update(extra_env)
    cmd = [PY, os.path.join(HERE, script), "--model", MODEL, "--quiet"] + args
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    if p.returncode != 0:
        return dict(failed=True, cmd=" ".join(shlex.quote(c) for c in cmd),
                    stderr=p.stderr[-2000:], wall=wall)
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
    if not line:
        return dict(failed=True, cmd=" ".join(cmd), stdout=p.stdout[-2000:], wall=wall)
    r = json.loads(line[-1][7:])
    r["wall"] = wall
    r["failed"] = False
    return r


def main():
    name, out, pts_file = sys.argv[1], sys.argv[2], sys.argv[3]
    points = [l.strip() for l in open(pts_file) if l.strip() and not l.startswith("#")]
    with open(out, "a") as fh:
        for i, line in enumerate(points):
            args = shlex.split(line)
            print(f"[{i+1}/{len(points)}] {name}: {line}", flush=True)
            r = run_point(args)
            r["sweep"] = name
            r["point"] = line
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if r["failed"]:
                print("   FAILED: " + (r.get("stderr") or r.get("stdout", ""))[-400:], flush=True)
            else:
                print(f"   {r['n_atoms']:5d} atoms  step {r['ms_step']:8.3f} ms  "
                      f"edges {r['real_edges']}/{r['max_edges']} "
                      f"({r['edge_fill']:.3f})  |dE| {r['e_err']:.2e}  "
                      f"|dF| {r['f_err']:.2e}", flush=True)


if __name__ == "__main__":
    main()
