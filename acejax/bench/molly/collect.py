"""Parse the run_julia.sh / run_jax.sh logs into one table.

    python collect.py <results-dir> [more dirs...]

Julia points come from the `RESULT` line (min over 3 in-process repeats of the
same 60-step window).  JAX points come from ace_md.py's `inner step` and
`rebuild` lines; the reported per-step cost is inner + rebuild/inner, i.e.
END-TO-END including the reneighbour the Julia side also pays, and the min over
the three invocations of each point.
"""
import re
import sys
import glob
import os
import json
from collections import defaultdict

jl = re.compile(r"^RESULT tag=(\S+) mode=(\S+) atoms=(\d+) julia_threads=(\d+) "
                r"steps=(\d+) reps=(\d+) ms_per_step_min=([\d.eE+-]+) "
                r"atom_steps_per_s=([\d.eE+-]+) spread_pct=([\d.eE+-]+) "
                r"final_PE=([-\d.eE+]+) final_KE=([-\d.eE+]+)", re.M)
jx_step = re.compile(r"inner step\s+([\d.]+) ms")
jx_reb  = re.compile(r"rebuild\s+([\d.]+) ms\s+\(amortised over (\d+): ([\d.]+) ms/step\)")
jx_pe   = re.compile(r"final PE\s+dist\s+([-\d.]+)")
jx_drift= re.compile(r"Etot drift\s+([-+\d.]+) meV/atom/ps")

rows = defaultdict(list)   # (engine, config, atoms) -> [ms/step, ...]
meta = {}

for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        base = os.path.basename(f)[:-4]
        txt = open(f, errors="replace").read()
        m = jl.search(txt)
        if m:
            tag, mode, atoms, thr, steps, reps, ms, aps, spread, pe, ke = m.groups()
            parts = tag.split("_")
            pas = parts[0]
            # the config NAME comes from the tag, not from `mode`: `tuned` is
            # mode=cached with a different skin, and must not be averaged in
            # with the skin-1.0 `cached` points.  Anything that is not a series
            # point (the skin sweeps) is skipped here and read separately.
            if len(parts) < 4 or not parts[1].startswith("rep"):
                continue
            cfg = parts[2]
            key = ("julia", f"{cfg}/t{thr}", int(atoms))
            rows[key].append(float(ms))
            meta[key + (pas,)] = dict(spread_pct=float(spread), final_PE=float(pe),
                                      steps=int(steps), reps=int(reps))
            continue
        s, r = jx_step.search(txt), jx_reb.search(txt)
        parts = base.split("_")
        # only <pass>_rep<N>_jax_<cfg>_run<i> logs are series points; the extras
        # (verify_*, skin_*, backends) land in the same directory and are skipped
        if s and r and len(parts) >= 5 and parts[2] == "jax" and parts[1].startswith("rep"):
            cfg = parts[3]
            natoms = 8 * int(parts[1][3:]) ** 3
            key = ("jax", cfg, natoms)
            rows[key].append(float(s.group(1)) + float(r.group(3)))
            pe, dr = jx_pe.search(txt), jx_drift.search(txt)
            meta[key + (parts[0],)] = dict(inner_ms=float(s.group(1)),
                                           rebuild_ms=float(r.group(1)),
                                           final_PE=float(pe.group(1)) if pe else None,
                                           drift=float(dr.group(1)) if dr else None)

print(f"{'engine':7s} {'config':14s} {'atoms':>6s} {'n':>3s} {'ms/step(min)':>13s} "
      f"{'atom-steps/s':>13s} {'spread%':>8s}")
out = {}
for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
    v = rows[key]
    mn, mx = min(v), max(v)
    spread = 100 * (mx - mn) / mn
    print(f"{key[0]:7s} {key[1]:14s} {key[2]:6d} {len(v):3d} {mn:13.4f} "
          f"{key[2]/(mn/1000):13.4g} {spread:8.2f}")
    out["|".join(map(str, key))] = dict(ms_per_step=mn, atom_steps_per_s=key[2]/(mn/1000),
                                        n=len(v), spread_pct=spread, all_ms=v)
json.dump(out, open("collected.json", "w"), indent=1)
