#!/usr/bin/env python3
"""THROWAWAY SPIKE -- parse two passes of run_gpu.sh and report agreement.

The effect being looked for may be small; a flat-vs-DAG difference smaller than
the run-to-run spread is "no measurable difference", not a speedup.  This prints
both passes side by side and the percentage disagreement for every number.
"""
import re, sys

CFG = re.compile(r"^########\s+(\S+).*?\|\s+(\d+) atoms.*?\|\s+(f\d+)")
BASE = re.compile(r"baseline\s+model\.site_basis\s+([\d.]+) ms.*?flat site_basis\s+([\d.]+) ms.*?flat E\+F\s+([\d.]+) ms")
ORAC = re.compile(r"ORACLE.*?site_basis\s+([\d.]+) ms\s+\|\s+E\+F\s+([\d.]+) ms")
MODE = re.compile(r"--\s+mode=(\w+):")
ROW = re.compile(r"^\s+(dus|concat|cvjp)\s+.*?site_basis\s+([\d.]+) ms.*?E\+F\s+([\d.]+) ms")


def parse(path):
    out, cfg, mode = {}, None, None
    for line in open(path, errors="ignore"):
        m = CFG.match(line)
        if m:
            cfg = f"{m.group(1)} {m.group(2)}at {m.group(3)}"; out[cfg] = {}; continue
        if cfg is None:
            continue
        m = BASE.search(line)
        if m:
            out[cfg]["flat"] = (float(m.group(2)), float(m.group(3)))
            out[cfg]["shipped"] = (float(m.group(1)), float("nan")); continue
        m = ORAC.search(line)
        if m:
            out[cfg]["oracle"] = (float(m.group(1)), float(m.group(2))); continue
        m = MODE.search(line)
        if m:
            mode = m.group(1); continue
        m = ROW.match(line)
        if m:
            out[cfg][f"{mode}/{m.group(1)}"] = (float(m.group(2)), float(m.group(3)))
    return out


def main(p1, p2):
    a, b = parse(p1), parse(p2)
    worst = 0.0
    for cfg in a:
        if cfg not in b:
            print(f"\n### {cfg}: MISSING from pass 2"); continue
        print(f"\n### {cfg}")
        print(f"{'variant':<18} {'site_basis p1/p2 (ms)':<26} {'disagree':<9} "
              f"{'E+F p1/p2 (ms)':<24} {'disagree':<9} {'E+F vs flat':<12}")
        f1, f2 = a[cfg]["flat"], b[cfg]["flat"]
        for k in a[cfg]:
            if k not in b[cfg] or k == "shipped":
                continue
            (s1, e1), (s2, e2) = a[cfg][k], b[cfg][k]
            ds = 100 * abs(s1 - s2) / min(s1, s2)
            de = 100 * abs(e1 - e2) / min(e1, e2)
            worst = max(worst, ds, de)
            r1, r2 = f1[1] / e1, f2[1] / e2
            print(f"{k:<18} {s1:8.3f} /{s2:8.3f}{'':<9} {ds:6.1f}%   "
                  f"{e1:8.3f} /{e2:8.3f}{'':<6} {de:6.1f}%   {r1:5.3f}x /{r2:5.3f}x")
    print(f"\nworst pass-to-pass disagreement over all numbers: {worst:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
