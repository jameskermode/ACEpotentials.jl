"""Load an npz exported by stage1/export_model.jl.

npz rather than JSON: Julia writes matrices column-major, so 2-D arrays
round-trip through JSON transposed.  NPZ.jl records Fortran order and numpy
restores the declared shape, so array orientation here is exactly as written on
the Julia side -- `tests/test_roundtrip.py` pins that rather than trusting it.
"""

import json

import jax.numpy as jnp
import numpy as np

from .model import ACEModel


def load(path, dtype=jnp.float64):
    """Load a model.  Caller controls dtype; nothing here touches jax.config, so
    f64 requires the caller to have enabled x64 first."""
    z = np.load(path)
    meta = json.loads(bytes(z["meta_json"]).decode())
    if meta["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version {meta['schema_version']}")
    if meta["radial_kind"] != "spline":
        raise NotImplementedError(
            f"radial_kind={meta['radial_kind']!r}; only the splined branch is "
            "implemented in Stage 1 (the analytic branch is reserved for Stage 2)")

    A = lambda k: jnp.asarray(z[k], dtype=dtype)
    rs, ps_ = meta["rnl_spline"], meta["pair_spline"]
    n_orders = len(meta["aa_lens"])
    model = ACEModel(
        rnl_coefs=A("rnl_spline_coefs"),
        pair_coefs=A("pair_spline_coefs"),
        rnl_transform=A("rnl_transform"),
        pair_transform=A("pair_transform"),
        rnl_envelope=A("rnl_envelope"),
        pair_envelope=A("pair_envelope"),
        A2B=A("A2B"),
        WB=A("WB"),
        Wpair=A("Wpair"),
        E0=A("E0"),
        aspec_r=jnp.asarray(z["aspec_r"], jnp.int32),
        aspec_y=jnp.asarray(z["aspec_y"], jnp.int32),
        aa_specs=tuple(jnp.asarray(z[f"aa_spec_{k+1}"], jnp.int32) for k in range(n_orders)),
        lmax=int(meta["lmax"]),
        ysolid=(meta["ybasis_kind"] == "real_solidharmonics"),
        rnl_grid=(float(rs["x0"]), float(rs["h"]), int(rs["n"])),
        pair_grid=(float(ps_["x0"]), float(ps_["h"]), int(ps_["n"])),
        elements=tuple(int(e) for e in meta["elements"]),
    )
    return model, meta, z
