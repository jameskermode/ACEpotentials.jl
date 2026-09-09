#!/usr/bin/env python3
"""Compare two LAMMPS dumps by atom id (rewritten; the original lived in a
scratchpad that has since been cleared).

Usage: cmpdump.py A.dump B.dump [tol]
Reports max |dx| and max |dF|, and exits non-zero if either exceeds tol.
"""
import sys

import numpy as np


def read(path):
    """Return (ids-sorted array of columns, dict col->index) for the last frame."""
    with open(path) as f:
        lines = f.read().splitlines()
    i = len(lines) - 1 - lines[::-1].index("ITEM: NUMBER OF ATOMS")
    n = int(lines[i + 1])
    j = len(lines) - 1 - next(k for k, l in enumerate(lines[::-1])
                              if l.startswith("ITEM: ATOMS"))
    cols = lines[j].split()[2:]
    rows = np.array([[float(v) for v in l.split()] for l in lines[j + 1:j + 1 + n]])
    idx = {c: k for k, c in enumerate(cols)}
    return rows[np.argsort(rows[:, idx["id"]])], idx


def main(a_path, b_path, tol=1e-8):
    a, ia = read(a_path)
    b, ib = read(b_path)
    if a.shape[0] != b.shape[0]:
        print(f"MISMATCH: {a.shape[0]} vs {b.shape[0]} atoms")
        return 1
    if not np.array_equal(a[:, ia["id"]], b[:, ib["id"]]):
        print("MISMATCH: differing atom ids")
        return 1
    dx = max(np.max(np.abs(a[:, ia[c]] - b[:, ib[c]])) for c in "xyz")
    df = max(np.max(np.abs(a[:, ia[f"f{c}"]] - b[:, ib[f"f{c}"]])) for c in "xyz")
    fs = np.max(np.abs(b[:, [ib[f"f{c}"] for c in "xyz"]]))
    print(f"  atoms {a.shape[0]}   max|dx| = {dx:.3e} A   max|dF| = {df:.3e} eV/A"
          f"   (|F| scale {fs:.4f})")
    ok = dx <= tol and df <= tol
    print("  " + ("OK" if ok else f"FAIL (tol {tol:g})"))
    return 0 if ok else 1


if __name__ == "__main__":
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
    sys.exit(main(sys.argv[1], sys.argv[2], tol))
