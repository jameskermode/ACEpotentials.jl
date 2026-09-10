"""Build a pace basis matched to our ACE model's SIZE, and write it as .yace.

For a throughput benchmark the coefficients are irrelevant: pace/kk evaluates
every basis function regardless of their values.  With `timestep 0.0` the atoms
never move, so a physically meaningless potential cannot destabilise anything.
This is a COST comparison at matched basis size, not a comparison of two fits.

Target (ours, from acejax/fixtures/si_fitted.npz):
    110 basis functions, correlation order 3, lmax 4, rcut 6.0 A, 1 element (Si)

Needs python-ace, which ships cp39 wheels only, plus setuptools (it imports
pkg_resources):

    uv venv --python 3.9 pyacenv
    uv pip install --python pyacenv/bin/python python-ace "setuptools<81"

Write the output as `.ace`, not `.yace`: to_ACECTildeBasisSet().save() emits the
text format, and a `.yace` extension makes yaml-cpp reject it.

    python make_pace_basis.py 110 3 4 6.0 si_pace110.ace
"""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from pyace import create_multispecies_basis_config, BBasisConfiguration

target = int(sys.argv[1]) if len(sys.argv) > 1 else 110
order = int(sys.argv[2]) if len(sys.argv) > 2 else 3
lmax = int(sys.argv[3]) if len(sys.argv) > 3 else 4
rcut = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
out = sys.argv[5] if len(sys.argv) > 5 else "/tmp/si_pace.ace"

def build(nradmax, nradbase, ndens=1):
    cfg = {
        "deltaSplineBins": 0.001,
        "elements": ["Si"],
        "embeddings": {"ALL": {"npot": "FinnisSinclairShiftedScaled",
                               "fs_parameters": [1, 1], "ndensity": ndens,
                               "rho_core_cut": 100000, "drho_core_cut": 250}},
        "bonds": {"ALL": {"radbase": "ChebExpCos", "radparameters": [5.25],
                          "rcut": rcut, "dcut": 0.01, "NameOfCutoffFunction": "cos"}},
        "functions": {"ALL": {"nradmax_by_orders": [nradmax] * order,
                              "lmax_by_orders": [0] + [lmax] * (order - 1)}},
    }
    bc = create_multispecies_basis_config(cfg)
    n = sum(len(b.funcspecs) for b in bc.funcspecs_blocks)
    return bc, n

# scan nradmax for the closest function count to the target
best = None
for nradmax in range(1, 20):
    try:
        bc, n = build(nradmax, max(nradmax, 8))
    except Exception:
        continue
    if best is None or abs(n - target) < abs(best[2] - target):
        best = (bc, nradmax, n)
    if n > target * 4:
        break
bc, nradmax, n = best
print(f"target {target} functions -> achieved {n} (nradmax={nradmax}, "
      f"order={order}, lmax={lmax}, rcut={rcut})")
# random coefficients: irrelevant to cost, and atoms never move
rng = np.random.default_rng(0)
for b in bc.funcspecs_blocks:
    b.set_all_coeffs(rng.normal(scale=0.01, size=len(b.get_all_coeffs())).tolist())
# pair_style pace reads the C-TILDE basis, not the B-basis config that
# BBasisConfiguration.save() writes -- converting is the whole point of this step.
from pyace import ACEBBasisSet
ctilde = ACEBBasisSet(bc).to_ACECTildeBasisSet()
ctilde.save(out)
print("wrote", out)
