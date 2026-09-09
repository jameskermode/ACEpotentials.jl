"""Load an npz exported by stage1/export_model.jl.

npz rather than JSON: Julia writes matrices column-major, so 2-D arrays
round-trip through JSON transposed.  NPZ.jl records Fortran order and numpy
restores the declared shape, so array orientation here is exactly as written on
the Julia side -- `tests/test_roundtrip.py` pins that rather than trusting it.

Two model families are supported, and they differ in more than one place:

              ace1_model              ace_model
  rbasis      spline                  analytic (Wnlq + 3-term recursion)
  pairbasis   spline                  spline          <- both splined
  Ylm         spherical               solid
  pair env    ACE1_PolyEnvelope1sR    PolyEnvelope1sR <- different formulas

so the branches are per-basis, not per-model.
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

    def A(k, default=None):
        if k not in z.files:
            if default is None:
                raise KeyError(f"{k} missing from {path}")
            return jnp.zeros(default, dtype)
        return jnp.asarray(z[k], dtype=dtype)

    kind = meta["radial_kind"]
    pkind = meta.get("pair_radial_kind", "spline")
    for k in (kind, pkind):
        if k not in ("spline", "analytic"):
            raise NotImplementedError(f"unknown radial_kind {k!r}")

    rs = meta.get("rnl_spline") or {"x0": 0.0, "h": 1.0, "n": 2}
    ps_ = meta.get("pair_spline") or {"x0": 0.0, "h": 1.0, "n": 2}
    n_orders = len(meta["aa_lens"])
    z0 = (1, 1, 1, 1)          # placeholder shape for the unused branch
    model = ACEModel(
        rnl_coefs=A("rnl_spline_coefs", z0),
        pair_coefs=A("pair_spline_coefs", z0),
        rnl_Wnlq=A("rnl_Wnlq", z0),
        pair_Wnlq=A("pair_Wnlq", z0),
        polys_A=A("polys_A", (2,)), polys_B=A("polys_B", (2,)), polys_C=A("polys_C", (2,)),
        pair_polys_A=A("pair_polys_A", (2,)), pair_polys_B=A("pair_polys_B", (2,)),
        pair_polys_C=A("pair_polys_C", (2,)),
        rnl_transform=A("rnl_transform"),
        pair_transform=A("pair_transform"),
        rnl_envelope=A("rnl_envelope"),
        pair_envelope=A("pair_envelope"),
        A2B=A("A2B"), WB=A("WB"), Wpair=A("Wpair"), E0=A("E0"),
        aspec_r=jnp.asarray(z["aspec_r"], jnp.int32),
        aspec_y=jnp.asarray(z["aspec_y"], jnp.int32),
        aa_specs=tuple(jnp.asarray(z[f"aa_spec_{k+1}"], jnp.int32) for k in range(n_orders)),
        lmax=int(meta["lmax"]),
        ysolid=(meta["ybasis_kind"] == "real_solidharmonics"),
        radial_kind=kind,
        pair_radial_kind=pkind,
        pair_envelope_kind=meta["pair_envelope_kind"],
        rnl_grid=(float(rs["x0"]), float(rs["h"]), int(rs["n"])),
        pair_grid=(float(ps_["x0"]), float(ps_["h"]), int(ps_["n"])),
        elements=tuple(int(e) for e in meta["elements"]),
    )
    return model, meta, z
