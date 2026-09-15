"""The LAMMPS bundle's energy_fn, tested without LAMMPS or lammps-jax.

`lammps/export_bundle.py::build` is the only thing the plugin path adds on top
of the core; if it is right here, the LAMMPS-side check (lammps/check_vs_python.py)
only has to catch plugin-contract mismatches.
"""
import importlib.util
import pathlib
import types

import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from acejax import highest_precision, load, sparse_graph

ROOT = pathlib.Path(__file__).parent.parent


def _export_bundle():
    spec = importlib.util.spec_from_file_location(
        "export_bundle", ROOT / "lammps" / "export_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_importable_without_lammps_jax():
    mod = _export_bundle()
    assert callable(mod.build)
