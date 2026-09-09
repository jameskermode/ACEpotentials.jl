"""Both model families are exercised by every test, so neither branch can rot.

  si_fitted.npz     ace1_model : splined rbasis, splined pair, SPHERICAL Ylm,
                                 ACE1_PolyEnvelope1sR
  si_ace_model.npz  ace_model  : ANALYTIC rbasis (live Wnlq + 3-term recursion),
                                 splined pair, SOLID Ylm, PolyEnvelope1sR

Note the pair basis is splined in both: `ace_model` splinifies it
(ace_heuristics.jl:213), so the radial branches are per-basis, not per-model.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parent.parent
MODELS = {
    "ace1_spline_spherical": ROOT / "si_fitted.npz",
    "ace_analytic_solid": ROOT / "si_ace_model.npz",
}


def pytest_generate_tests(metafunc):
    if "npz" in metafunc.fixturenames:
        ids, paths = [], []
        for name, p in MODELS.items():
            ids.append(name)
            paths.append(pytest.param(p, marks=pytest.mark.skipif(
                not p.exists(), reason=f"run export_model.jl to make {p.name}")))
        metafunc.parametrize("npz", paths, ids=ids)
