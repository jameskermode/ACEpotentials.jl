"""Tables for the Apple-Silicon scaling investigation from the sweep jsonl files.

    python analyse.py <jsonl> [<jsonl> ...]
"""
import json
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if not r.get("failed"):
                rows.append(r)
    return rows


def main():
    rows = load(sys.argv[1:])
    # collapse repeats of the identical configuration to their min (and record spread)
    grp = defaultdict(list)
    for r in rows:
        grp[(r["n_atoms"], r["max_edges"], r["max_local"], r["dtype"])].append(r)

    print(f"{'atoms':>6} {'edge slots':>11} {'slots/at':>9} {'fill':>6} "
          f"{'node slots':>11} {'ms/step':>9} {'ns/edge':>8} {'us/atom':>8} "
          f"{'n':>3} {'spread%':>8} {'|dF|':>9} {'temp MB':>8}")
    for key in sorted(grp):
        rs = grp[key]
        ms = [r["ms_step"] for r in rs]
        best = min(rs, key=lambda r: r["ms_step"])
        spread = (max(ms) - min(ms)) / min(ms) * 100
        temp = ""
        for r in rs:
            if "mem_trajectory" in r and "temp" in r["mem_trajectory"]:
                temp = f"{r['mem_trajectory']['temp']/2**20:8.1f}"
                break
        print(f"{key[0]:>6} {key[1]:>11} {key[1]/key[0]:>9.1f} "
              f"{best['edge_fill']:>6.3f} {key[2]:>11} {min(ms):>9.3f} "
              f"{min(ms)*1e6/key[1]:>8.1f} {min(ms)*1e3/key[0]:>8.2f} "
              f"{len(rs):>3} {spread:>8.1f} {best['f_err']:>9.1e} {temp:>8}")


if __name__ == "__main__":
    main()
