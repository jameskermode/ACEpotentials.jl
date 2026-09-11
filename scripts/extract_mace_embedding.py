#!/usr/bin/env python3
"""
Extract the frozen element-embedding table from a MACE foundation-model
checkpoint into a plain `.npz` that Julia can read.

This script is deliberately NOT part of ACEpotentials' load path.  It is run
once, offline, and its output (`mace_element_embedding.npz`) is the artefact
that ACEpotentials consumes.  Nothing in ACEpotentials should import torch.

Only `torch` and `numpy` are needed -- *not* `mace` or `e3nn`.  MACE
checkpoints are full pickled `nn.Module` objects, so unpickling them normally
requires the mace/e3nn class definitions to be importable.  We avoid that by
substituting a generic stub class for every class we cannot import; the tensor
payloads are rebuilt by torch as usual, and we only ever read
`_parameters` / `_buffers` / `_modules` dictionaries.

What is extracted
-----------------
`node_embedding.linear.weight` -- the weight of the `LinearNodeEmbeddingBlock`,
an e3nn `Linear` mapping `num_elements x 0e -> num_channels x 0e`.  It is
stored flat, of length `num_elements * num_channels`, row-major in
(element, channel).  e3nn applies a path normalisation of
`1/sqrt(num_elements)` at evaluation time; we apply it here so that the table
is what the network actually multiplies the one-hot species vector by.

The row order follows the checkpoint's own `atomic_numbers` buffer, which is
saved alongside the table as `Z`.

Usage
-----
    python scripts/extract_mace_embedding.py CHECKPOINT.model -o OUT.npz

Provenance for the table committed with this spike:
    MACE-MP-0 "small"
    https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model
    layer: node_embedding.linear.weight   (LinearNodeEmbeddingBlock)
"""

import argparse
import hashlib
import json
import pickle

import numpy as np
import torch


class _Stub:
    """Stands in for any class we cannot import (mace.*, e3nn.*, ...)."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def __call__(self, *args, **kwargs):
        return _Stub()


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return type("Stub_" + name, (_Stub,), {"_cls": module + "." + name})


class _StubPickleModule:
    Unpickler = _StubUnpickler

    @staticmethod
    def load(f, **kwargs):
        return _StubUnpickler(f, **kwargs).load()


def _get(obj, path):
    """Walk a dotted path through _modules / _parameters / _buffers."""
    cur = obj
    parts = path.split(".")
    for i, p in enumerate(parts):
        d = getattr(cur, "__dict__", {})
        if p in d.get("_modules", {}):
            cur = d["_modules"][p]
        elif p in d.get("_parameters", {}):
            cur = d["_parameters"][p]
        elif p in d.get("_buffers", {}):
            cur = d["_buffers"][p]
        else:
            raise KeyError(f"cannot resolve {'.'.join(parts[:i+1])!r} in checkpoint")
    return cur


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("-o", "--out", default="mace_element_embedding.npz")
    ap.add_argument("--layer", default="node_embedding.linear.weight")
    ap.add_argument("--no-e3nn-norm", action="store_true",
                    help="store the raw weight without the 1/sqrt(S) e3nn path normalisation")
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    model = torch.load(args.checkpoint, map_location="cpu",
                       pickle_module=_StubPickleModule, weights_only=False)

    W = _get(model, args.layer).detach().cpu().numpy().astype(np.float64)
    Z = _get(model, "atomic_numbers").detach().cpu().numpy().astype(np.int64)

    S = len(Z)
    if W.ndim != 1 or W.size % S != 0:
        raise RuntimeError(f"unexpected weight shape {W.shape} for {S} elements")
    d = W.size // S
    emb = W.reshape(S, d)
    norm = 1.0 if args.no_e3nn_norm else 1.0 / np.sqrt(S)
    emb = emb * norm

    # The .npz is kept strictly numeric so that NPZ.jl (and any other minimal
    # reader) can load it without needing numpy object/unicode dtype support.
    # Provenance goes in a JSON sidecar next to it.
    np.savez(
        args.out,
        emb=emb,                                  # (S, d) float64
        Z=Z,                                      # (S,) atomic numbers, row order of emb
        e3nn_path_normalisation=np.array([norm]), # scalar, as a length-1 array
    )
    meta = {
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha,
        "layer": args.layer,
        "shape": list(emb.shape),
        "e3nn_path_normalisation": float(norm),
        "torch_version": torch.__version__,
        "note": "emb[i, :] is the embedding vector of element Z[i]; "
                "rows follow the checkpoint's own atomic_numbers buffer.",
    }
    side = str(args.out) + ".json" if not str(args.out).endswith(".npz") \
           else str(args.out)[:-4] + ".json"
    # The table goes into the JSON as well as the npz: ACEpotentials reads the
    # JSON (it already depends on JSON.jl), and a few hundred kB of frozen table
    # does not justify adding a binary-format dependency to the whole package.
    # The npz stays for Python consumers.
    meta["Z"] = [int(z) for z in Z]
    meta["emb"] = [[float(x) for x in row] for row in emb]
    with open(side, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {args.out} (+ {side}): emb {emb.shape}, "
          f"Z {Z.min()}..{Z.max()}, norm={norm:.6g}, sha256={sha[:16]}...")


if __name__ == "__main__":
    main()
