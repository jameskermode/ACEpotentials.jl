"""Pin the npz orientation and the per-stage probe values.

The probe block exists so a mismatch localises to a stage (transform, envelope,
spline, Ylm) instead of only surfacing at the end in the site energy.
"""
import json
import pathlib

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)   # test-local, not library-level
import jax.numpy as jnp

from acejax import load
from acejax.radial import (agnesi_normalized, env_ace1_poly1sr, env_poly2sx,
                           spline_eval)

NPZ = pathlib.Path(__file__).parent.parent / "si_fitted.npz"
pytestmark = pytest.mark.skipif(not NPZ.exists(), reason="run export_model.jl first")


@pytest.fixture(scope="module")
def loaded():
    return load(NPZ)


def test_orientation(loaded):
    model, meta, z = loaded
    NZ = len(meta["elements"])
    assert model.WB.shape == (meta["n_B"], NZ)
    assert model.Wpair.shape == (meta["n_pair"], NZ)
    assert model.A2B.shape == (meta["n_B"], meta["n_AA"])
    assert model.rnl_coefs.shape == (NZ, NZ, meta["rnl_spline"]["ncoef"], meta["n_rnl"])
    assert z["probe_Rnl"].shape == (len(z["probe_r"]), meta["n_rnl"])
    assert z["test_pos"].shape[0] == 3 and z["test_edge_rij"].shape[0] == 3


def test_probe_transform(loaded):
    model, meta, z = loaded
    r = jnp.asarray(z["probe_r"])
    x = agnesi_normalized(r, model.rnl_transform[0, 0])
    assert np.max(np.abs(np.asarray(x) - z["probe_x"])) < 1e-14


def test_probe_envelope(loaded):
    model, meta, z = loaded
    env = env_poly2sx(jnp.asarray(z["probe_x"]), model.rnl_envelope[0, 0])
    assert np.max(np.abs(np.asarray(env) - z["probe_env"])) < 1e-14


def test_probe_rnl(loaded):
    model, meta, z = loaded
    n = len(z["probe_r"])
    zi = jnp.zeros(n, jnp.int32)
    rij = jnp.asarray(z["probe_rij"].T)
    Rnl, Rpair = model.radial(rij, zi, zi)
    e_rnl = np.max(np.abs(np.asarray(Rnl) - z["probe_Rnl"]))
    e_pair = np.max(np.abs(np.asarray(Rpair) - z["probe_Rpair"]))
    assert e_rnl < 1e-11, f"Rnl {e_rnl:.2e}"
    assert e_pair < 1e-11, f"Rpair {e_pair:.2e}"


def test_probe_ylm(loaded):
    model, meta, z = loaded
    Y = model.angular(jnp.asarray(z["probe_rij"].T))
    assert np.max(np.abs(np.asarray(Y) - z["probe_Ylm"])) < 1e-11
