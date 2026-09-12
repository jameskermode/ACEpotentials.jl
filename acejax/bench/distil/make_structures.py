"""Structures for distilling MACE into an ACE model of the Cantor alloy.

CrMnFeCoNi, fcc, equiatomic, with random site occupancy.

SHORT-RANGE ORDER IS NOT IMPOSED. Random occupancy is a first-test convenience,
not physics: a real Cantor alloy has chemical SRO, and a student fitted only to
randomly-occupied cells will be sampled off the distribution it would see in
use. That is a known and deliberate limitation of this first pass -- any
scientific claim needs SRO-correct configurations (e.g. Monte Carlo occupancy at
a target temperature, or SRO parameters from experiment). Do not report fit
quality here as a statement about the alloy.

The structures are otherwise deliberately varied -- lattice parameter, cell
shape, rattle amplitude -- so the fit sees compression, tension and shear rather
than one basin.
"""
import argparse
import numpy as np
from ase.build import bulk
from ase.io import write

ELEMENTS = [24, 25, 26, 27, 28]          # Cr Mn Fe Co Ni
A0 = 3.59                                 # fcc Cantor alloy lattice constant, A
MIN_SEP = 1.9                             # A; below this MACE is extrapolating


def one(rng, reps, a, rattle, strain):
    at = bulk("Ni", "fcc", a=a, cubic=True).repeat(reps)
    n = len(at)
    # equiatomic where divisible, random assignment otherwise
    z = np.repeat(ELEMENTS, n // len(ELEMENTS))
    if len(z) < n:
        z = np.concatenate([z, rng.choice(ELEMENTS, n - len(z))])
    rng.shuffle(z)
    at.set_atomic_numbers(z)
    # strain: isotropic + a little shear, so the fit sees more than one volume
    F = np.eye(3) + strain * rng.standard_normal((3, 3)) * 0.5
    F = (F + F.T) / 2
    at.set_cell(at.get_cell() @ F, scale_atoms=True)
    at.positions += rattle * rng.standard_normal(at.positions.shape)
    return at


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=250)
    p.add_argument("-o", default="cantor.xyz")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out, tries, rejected = [], 0, 0
    while len(out) < a.n and tries < 50 * a.n:
        tries += 1
        at = one(rng, rng.choice([(2, 2, 2), (2, 2, 3)]),
                 A0 * (1 + 0.06 * rng.standard_normal()),
                 float(rng.uniform(0.02, 0.16)),
                 float(abs(rng.standard_normal()) * 0.03))
        d = at.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        if d.min() < MIN_SEP:          # keep the teacher inside its comfort zone
            rejected += 1
            continue
        out.append(at)
    write(a.o, out, format="extxyz")
    sizes = sorted({len(x) for x in out})
    mind = min(float(np.min(np.where(np.eye(len(x), dtype=bool), np.inf,
                                     x.get_all_distances(mic=True)))) for x in out)
    print(f"wrote {len(out)} structures to {a.o} "
          f"(sizes {sizes}, rejected {rejected} for min-separation)")
    print(f"  min interatomic distance over the set: {mind:.3f} A (filter {MIN_SEP})")


if __name__ == "__main__":
    main()
