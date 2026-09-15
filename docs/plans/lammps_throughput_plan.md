# LAMMPS Throughput (Package 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the like-for-like `pace/kk` ÷ `jax/kk` throughput ratio from 2.6x to ≤1.5x at n_B = 2849 (≤2x at n_B = 69), f64, 1728 Si atoms, by removing wasted work in the `acejax` plugin path.

**Architecture:** Three independent, exactness-preserving changes to `acejax/`, landed and measured by difference in order: (A) fold the linear readout through `A2B` into a C-tilde vector so the `A2B` contraction and its adjoint vanish; (B) evaluate the per-atom stages of the LAMMPS bundle over a local-atom capacity `max_local` instead of the full `max_atoms` (local + ghost + pad) axis, with a NaN overflow guard; (C) size bundle capacities from a structure instead of guessing. `lammps-jax` is not modified.

**Tech Stack:** Python 3.11+, JAX, Equinox, pytest (`uv run pytest tests/ -q` inside `acejax/`); Julia 1.11 for fixture export (`julia --project=acejax/julia acejax/julia/export_model.jl`); LAMMPS + `pair_style jax/kk` on **moriarty** for Tasks 6 and 8 (see `acejax/lammps/test_si_bundle.sh` and `acejax/bench/run_bench.sh` for the host recipe — `LD_LIBRARY_PATH` order matters and has bitten three times).

**Spec:** `docs/plans/lammps_throughput_design.md`

## Global Constraints

- Every change is verified against the current path to ≤1e-10 on energies, forces and virial (1e-12 where both sides are the same code) *before* it is timed.
- Attribute by difference only: change one thing, time the whole computation twice. No isolated stage timings in any reported number.
- Nothing in `acejax/acejax/` calls `jax.config.update`; tests set `jax_enable_x64` themselves (see `tests/test_efv.py`).
- All paths below are relative to `acejax/` unless they start with `docs/` or `.github/`.
- Branch: `pr/lammps-throughput` (stacked on `pr/element-embeddings`). Commit after every task; commit messages end with the attribution trailer given in the session.
- Repeatability floor for E+F timings on moriarty is ≤1%; report differences under 3% as "no change".

---

### Task 1: A2B deletion oracle (decision gate for Task 2's ordering)

Bounds the gain from lever A before it is built. Throwaway script; the number it produces goes into the commit message and later into `bench/results.md`.

**Files:**
- Create: `bench/oracle_a2b.py`

**Interfaces:**
- Consumes: `acejax.load(npz, dtype, a2b_sparse)`, `ACEModel.energy_forces_virial`, `ACEModel._from_pooled` (monkeypatched).
- Produces: a printed table only.

- [ ] **Step 1: Write the oracle script**

```python
#!/usr/bin/env python3
"""Ceiling for removing the A2B contraction: time E+F with A2B replaced by a
free identity (B := AA[:, :n_B]).  Attribution by difference, whole
computation both times.  THROWAWAY -- evidence for the fold, not product.

  python bench/oracle_a2b.py --npz fixtures/si_l2849.npz --reps 6 --a2b-sparse
"""
import argparse, itertools, pathlib, sys, time

import jax, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from bench_acejax import diamond          # same structures as the benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", type=pathlib.Path, required=True)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--f32", action="store_true")
    p.add_argument("--a2b-sparse", action="store_true")
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    if not a.f32:
        jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from acejax import load, highest_precision
    from acejax.model import ACEModel, pool_sparse

    def edges(pos, cell, rcut):
        n = len(pos)
        reps_ = [int(np.ceil(rcut / cell[k, k])) for k in range(3)]
        ii, jj, rr = [], [], []
        for sh in itertools.product(*[range(-r, r + 1) for r in reps_]):
            d = (pos[None, :, :] + np.array(sh) @ cell) - pos[:, None, :]
            r = np.linalg.norm(d, axis=-1)
            m = (r < rcut) & (r > 1e-10)
            x, y = np.where(m)
            ii.append(x); jj.append(y); rr.append(d[x, y])
        ii = np.concatenate(ii); jj = np.concatenate(jj); rr = np.concatenate(rr)
        o = np.argsort(ii, kind="stable")
        return ii[o].astype(np.int32), jj[o].astype(np.int32), rr[o]

    dt = jnp.float32 if a.f32 else jnp.float64
    model, meta, _ = load(a.npz, dtype=dt, a2b_sparse=a.a2b_sparse, fold=False)
    pos, cell = diamond(a.reps)
    ii, jj, rr = edges(pos, cell, float(meta["rcut"]))
    n = len(pos)
    nz = jnp.zeros(n, jnp.int32)
    send, recv, rij = jnp.asarray(ii), jnp.asarray(jj), jnp.asarray(rr, dtype=dt)

    def timeit(m):
        with highest_precision():
            f = jax.jit(lambda r: m.energy_forces_virial(r, nz[send], nz[recv],
                                                         send, recv, n, nz)[:2])
            jax.block_until_ready(f(rij))
            ts = []
            for _ in range(a.repeats):
                t0 = time.perf_counter(); jax.block_until_ready(f(rij))
                ts.append(time.perf_counter() - t0)
        return min(ts) * 1e3

    n_B = int(meta["n_B"])
    orig = ACEModel._from_pooled

    def oracle(self, A, Apair):
        AA = jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in self.aa_specs], axis=-1)
        return AA[:, :n_B], Apair          # free identity of the right shape

    t_real = timeit(model)
    ACEModel._from_pooled = oracle
    t_orac = timeit(model)
    ACEModel._from_pooled = orig
    print(f"# {a.npz.name} n_B={n_B} atoms={n} {'f32' if a.f32 else 'f64'} "
          f"a2b={'sparse' if a.a2b_sparse else 'dense'} on {jax.default_backend()}")
    print(f"  E+F with A2B      {t_real:9.3f} ms")
    print(f"  E+F oracle (no A2B){t_orac:9.3f} ms")
    print(f"  ceiling            {t_real / t_orac:9.3f}x")


if __name__ == "__main__":
    main()
```

Note: `load(..., fold=False)` does not exist until Task 2. For this task, call `load(a.npz, dtype=dt, a2b_sparse=a.a2b_sparse)` and add the `fold=False` argument in Task 2 Step 8 when the flag lands.

- [ ] **Step 2: Run on moriarty, all three fixtures, 1728 atoms, f64**

Fixtures `si_s69.npz`, `si_m710.npz`, `si_l2849.npz` are gitignored and regenerable (`fixtures/.gitignore` has the recipe); they exist in the moriarty checkout under `acejax/fixtures/`.

```bash
cd acejax
for f in si_s69 si_m710 si_l2849; do
  python bench/oracle_a2b.py --npz fixtures/$f.npz --reps 6 --a2b-sparse
done
python bench/oracle_a2b.py --npz fixtures/si_s69.npz --reps 6          # dense is faster at small basis
```

Expected: three ceilings. Record them verbatim.

- [ ] **Step 3: Decide ordering**

If the ceiling at `si_l2849` is < 1.15x, Task 2 is still done (it deletes code and cost) but Task 5 (lever B) is executed before Task 2. Otherwise proceed in order.

- [ ] **Step 4: Commit**

```bash
git add acejax/bench/oracle_a2b.py
git commit -m "bench: A2B deletion oracle -- ceiling <X>x / <Y>x / <Z>x at n_B 69/710/2849, 1728 atoms f64"
```

---

### Task 2: `fold_readout` — C-tilde readout in `ACEModel`

**Files:**
- Modify: `acejax/model.py` (fields after `a_sel_y`; `_from_pooled`; `site_energies`; `site_energies_dense`; new `fold_readout` next to `with_edge_a_kind`)
- Modify: `acejax/io.py:53` (`load` gains `fold=True`)
- Modify: `acejax/__init__.py` (export `fold_readout`)
- Test: `tests/test_fold.py`

**Interfaces:**
- Produces: `fold_readout(model: ACEModel) -> ACEModel` (idempotent; sets `model.folded = True`, `model.ctilde: (n_AA, NZ)`); `load(path, dtype, a2b_sparse, edge_a_kind, fold=True)`. `site_energies` / `site_energies_dense` / `energy_forces_virial` / `energy_from_positions` produce identical values folded or not. `site_basis` and `site_descriptors` are unaffected by folding.

- [ ] **Step 1: Write the failing tests**

```python
"""Folding the linear readout through A2B (PACE's C-tilde) is exact.

`e_i = WB[:,z_i] . (A2B . AA_i)` == `(A2B^T WB[:,z_i]) . AA_i`.  The fold
removes the A2B contraction and its adjoint from the evaluation path; nothing
about the numbers may change, and descriptors (which need B) must be untouched.
"""
import jax
import numpy as np
import pytest

from conftest import species_index

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from acejax import fold_readout, highest_precision, load

TOL_SAME_CODE = 1e-12
TOL_JULIA = 1e-10


def _edges(z):
    send = jnp.asarray(np.asarray(z["test_edge_i"], np.int32))
    recv = jnp.asarray(np.asarray(z["test_edge_j"], np.int32))
    rij = jnp.asarray(np.asarray(z["test_edge_rij"]).T)
    n_nodes = int(z["test_pos"].shape[1])
    return rij, send, recv, n_nodes, species_index(z)


def _efv(model, z):
    rij, send, recv, n_nodes, node_z = _edges(z)
    with highest_precision():
        E, F, V = model.energy_forces_virial(rij, node_z[send], node_z[recv],
                                            send, recv, n_nodes, node_z)
    return float(E), np.asarray(F), np.asarray(V)


@pytest.mark.parametrize("kind", ["gather", "matmul"])
@pytest.mark.parametrize("sparse", [False, True])
def test_fold_matches_unfolded(npz, kind, sparse):
    m0, meta, z = load(npz, a2b_sparse=sparse, edge_a_kind=kind, fold=False)
    m1 = fold_readout(m0)
    assert not m0.folded and m1.folded
    assert m1.ctilde.shape == (m0.A2B.shape[1], m0.WB.shape[1])
    E0, F0, V0 = _efv(m0, z)
    E1, F1, V1 = _efv(m1, z)
    print(f"\n  |dE| {abs(E0-E1):.2e}  |dF| {np.abs(F0-F1).max():.2e}  |dV| {np.abs(V0-V1).max():.2e}")
    assert abs(E0 - E1) < TOL_SAME_CODE
    assert np.abs(F0 - F1).max() < TOL_SAME_CODE
    assert np.abs(V0 - V1).max() < TOL_SAME_CODE


def test_folded_matches_julia(npz):
    """The default load is folded; it must still hit the Julia reference."""
    m, meta, z = load(npz)
    assert m.folded
    E, F, V = _efv(m, z)
    assert abs(E - float(z["test_E"][0])) < TOL_JULIA
    assert np.abs(F - np.asarray(z["test_F"]).T).max() < TOL_JULIA
    assert np.abs(V - np.asarray(z["test_V"])).max() < TOL_JULIA


def test_fold_is_idempotent(npz):
    m, _, _ = load(npz, fold=False)
    m1 = fold_readout(m)
    assert fold_readout(m1) is m1


def test_site_basis_unchanged_by_fold(npz):
    m0, meta, z = load(npz, fold=False)
    m1 = fold_readout(m0)
    rij, send, recv, n_nodes, node_z = _edges(z)
    with highest_precision():
        B0, P0 = m0.site_basis(rij, node_z[send], node_z[recv], send, n_nodes)
        B1, P1 = m1.site_basis(rij, node_z[send], node_z[recv], send, n_nodes)
    assert np.array_equal(np.asarray(B0), np.asarray(B1))
    assert np.array_equal(np.asarray(P0), np.asarray(P1))


def test_dense_pooling_folded(npz):
    from acejax import dense_graph
    m0, meta, z = load(npz, fold=False)
    m1 = fold_readout(m0)
    from ase import Atoms
    atoms = Atoms(numbers=np.asarray(z["test_Z"]), positions=np.asarray(z["test_pos"]).T,
                  cell=np.asarray(z["test_cell"]).T, pbc=np.asarray(z["test_pbc"]).astype(bool))
    g = dense_graph(atoms.get_positions(), atoms.get_cell().array, atoms.get_pbc(),
                    meta["rcut"], max_neighbours=96)      # Si at rcut 6 has ~46 neighbours
    node_z = species_index(z)
    mask = jnp.asarray(g.mask)                             # DenseGraph.mask is a property
    zi = jnp.broadcast_to(node_z[:, None], mask.shape)
    zj = node_z[jnp.where(mask, jnp.asarray(g.idx), 0)]
    with highest_precision():
        e0 = m0.site_energies_dense(jnp.asarray(g.rij), zi, zj, mask, node_z)
        e1 = m1.site_energies_dense(jnp.asarray(g.rij), zi, zj, mask, node_z)
    assert np.abs(np.asarray(e0) - np.asarray(e1)).max() < TOL_SAME_CODE
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd acejax && uv run pytest tests/test_fold.py -q`
Expected: FAIL — `ImportError: cannot import name 'fold_readout'`.

- [ ] **Step 3: Add the fields and the fold to `acejax/model.py`**

After `a_sel_y: jax.Array = None` in the `ACEModel` field list:

```python
    # C-tilde readout (PACE's ctilde): ctilde = A2B^T @ WB, (n_AA, NZ).  When
    # `folded`, site energies contract AA directly against it and the A2B
    # contraction never runs.  `site_basis` keeps using A2B, since descriptors
    # and fitting need B itself.  See `fold_readout`.
    ctilde: jax.Array = None
    folded: bool = eqx.field(static=True, default=False)
```

Factor the AA product out of `_from_pooled` and add the folded readout:

```python
    def _aa(self, A):
        return jnp.concatenate([jnp.prod(A[:, g], axis=-1) for g in self.aa_specs], axis=-1)

    def _from_pooled(self, A, Apair):
        AA = self._aa(A)
        if self.a2b_sparse:
            contrib = AA[:, self.a2b_cols] * self.a2b_vals          # (n_nodes, nnz)
            B = jax.ops.segment_sum(contrib.T, self.a2b_rows,
                                    num_segments=self.A2B.shape[0]).T
        else:
            B = AA @ self.A2B.T
        return B, Apair

    def _readout_folded(self, A, Apair, node_z):
        e = jnp.einsum("ia,ai->i", self._aa(A), self.ctilde[:, node_z])
        e = e + jnp.einsum("ip,pi->i", Apair, self.Wpair[:, node_z])
        return e + self.E0[node_z]
```

Replace `site_energies` and `site_energies_dense`:

```python
    def site_energies(self, rij, zi, zj, segment_ids, n_nodes, node_z, mask=None):
        """Per-site energies (n_nodes,).  `node_z` is the centre species index per node."""
        if self.folded:
            edge_A, Rpair = self.edge_features(rij, zi, zj)
            return self._readout_folded(pool_sparse(edge_A, segment_ids, n_nodes, mask),
                                        pool_sparse(Rpair, segment_ids, n_nodes, mask),
                                        node_z)
        return self._readout(*self.site_basis(rij, zi, zj, segment_ids, n_nodes, mask), node_z)

    def site_energies_dense(self, rij, zi, zj, mask, node_z):
        if self.folded:
            n, K = mask.shape
            flat = lambda a: a.reshape(n * K, *a.shape[2:])
            edge_A, Rpair = self.edge_features(flat(rij), flat(zi), flat(zj))
            un = lambda a: a.reshape(n, K, -1)
            return self._readout_folded(pool_dense(un(edge_A), mask),
                                        pool_dense(un(Rpair), mask), node_z)
        return self._readout(*self.site_basis_dense(rij, zi, zj, mask), node_z)
```

Add next to `with_edge_a_kind`:

```python
# ------------------------------------------------------------------ readout fold
def fold_readout(model):
    """Return `model` with the linear readout folded through A2B.

    e_i = WB[:,z] . (A2B AA_i)  ==  (A2B^T WB[:,z]) . AA_i, so ctilde = A2B^T WB
    is computed once here and the A2B contraction -- the largest isolated stage at
    production basis size -- and its adjoint never run.  Exact (tests/test_fold.py
    holds it to 1e-12).  This is the same fold as PACE's ctilde basis.
    """
    import dataclasses
    if model.folded:
        return model
    with highest_precision():                 # TF32 would corrupt ctilde on Ampere+
        ctilde = model.A2B.T @ model.WB       # (n_AA, NZ)
    return dataclasses.replace(model, ctilde=ctilde, folded=True)
```

- [ ] **Step 4: Add `fold` to `load` in `acejax/io.py`**

Signature: `def load(path, dtype=jnp.float64, a2b_sparse=False, edge_a_kind="gather", fold=True):`. Add to the docstring:

```
    `fold` (default True) folds the linear readout through A2B (see
    `fold_readout`); pass False to keep the B-materialising path, e.g. to
    measure the fold by difference.  Descriptors are unaffected either way.
```

At the end, replace `return model, meta, z` with:

```python
    if fold:
        from .model import fold_readout
        model = fold_readout(model)
    return model, meta, z
```

- [ ] **Step 5: Export it from `acejax/__init__.py`**

Add `fold_readout` to the `from .model import ...` line and to `__all__` if there is one.

- [ ] **Step 6: Run the new tests**

Run: `cd acejax && uv run pytest tests/test_fold.py -q`
Expected: PASS (10 tests across the two fixtures × parameters).

- [ ] **Step 7: Run the whole suite — the default load is now folded, so every existing parity test exercises the fold**

Run: `cd acejax && uv run pytest tests/ -q`
Expected: PASS, same count as before plus the new tests.

- [ ] **Step 8: Update `bench/oracle_a2b.py` to `load(..., fold=False)`** (Task 1 Step 1 note) and re-run it once on `si_s69` locally to check it still runs.

- [ ] **Step 9: Commit**

```bash
git add acejax/acejax/model.py acejax/acejax/io.py acejax/acejax/__init__.py acejax/tests/test_fold.py acejax/bench/oracle_a2b.py
git commit -m "acejax: fold the linear readout through A2B (ctilde); exact to 1e-12, default on load"
```

---

### Task 3: Two-species fixture, so the per-species fold indexing is tested

All committed fixtures are single-species, so `ctilde[:, node_z]` is never indexed with more than one species. `julia/export_model.jl` already builds a round-robin multi-element test system when `ACE_ELEMENTS` has more than one entry.

**Files:**
- Create: `fixtures/sige_nofit.npz` (committed; check size ≤ 2 MB first)
- Modify: `tests/conftest.py:31-34` (`MODELS`)
- Modify: `fixtures/.gitignore` (recipe comment only)
- Modify: `.github/workflows/acejax.yml:189-190` (regenerate the third fixture in the divergence job)

**Interfaces:**
- Produces: a third `npz` parametrisation id `ace1_two_species` that every test in the suite runs against.

- [ ] **Step 1: Export the fixture**

```bash
cd acejax
ACE_ELEMENTS=Si,Ge ACE_ORDER=3 ACE_TOTALDEGREE=6 ACE_NOFIT=1 \
  julia --project=julia julia/export_model.jl fixtures/sige_nofit.npz ace1
ls -la fixtures/sige_nofit.npz
python -c "import numpy as np; z=np.load('fixtures/sige_nofit.npz'); print(z['elements'], z['WB'].shape, np.unique(z['test_Z']))"
```

Expected: `elements` = `[14 32]`, `WB` shape `(n_B, 2)`, `test_Z` contains both 14 and 32. `ACE_NOFIT=1` randomises `WB`/`Wpair` (the script does this explicitly because `ace1_model` initialises them to zero), which is what a parity test needs. If the file is over 2 MB, lower `ACE_TOTALDEGREE` to 5.

- [ ] **Step 2: Register it in `tests/conftest.py`**

```python
MODELS = {
    "ace1_spline_spherical": "si_fitted.npz",
    "ace_analytic_solid": "si_ace_model.npz",
    # two species, random (unfitted) weights: exercises every per-species index
    # path -- Wnlq[:,:,iz,jz], E0[z], WB[:,z] and the folded ctilde[:,z]
    "ace1_two_species": "sige_nofit.npz",
}
```

- [ ] **Step 3: Run the suite against it**

Run: `cd acejax && uv run pytest tests/ -q -k two_species`
Expected: PASS. If `test_efv.py::test_nlist_matches_julia` or the parity tests fail here, that is a real two-species bug in either the exporter or the model, not in this task — stop and report it with the numbers rather than loosening anything.

- [ ] **Step 4: Regenerate it in CI's divergence job**

In `.github/workflows/acejax.yml`, after line 190 (`... export_model.jl fresh/si_ace_model.npz $KIND2`), add:

```yaml
          ACE_ELEMENTS=Si,Ge ACE_ORDER=3 ACE_TOTALDEGREE=6 ACE_NOFIT=1 \
            julia --project=acejax/julia acejax/julia/export_model.jl fresh/sige_nofit.npz ace1
```

Match the indentation and the `KIND` convention of the surrounding lines exactly (read them first).

- [ ] **Step 5: Add the recipe to `fixtures/.gitignore` as a comment** (the file is committed, so it is *not* ignored):

```
# committed test fixture, two species, random weights:
#   ACE_ELEMENTS=Si,Ge ACE_ORDER=3 ACE_TOTALDEGREE=6 ACE_NOFIT=1 \
#     julia --project=julia julia/export_model.jl fixtures/sige_nofit.npz ace1
```

- [ ] **Step 6: Commit**

```bash
git add acejax/fixtures/sige_nofit.npz acejax/fixtures/.gitignore acejax/tests/conftest.py .github/workflows/acejax.yml
git commit -m "acejax: two-species test fixture (Si,Ge, unfitted) so per-species readout paths are exercised"
```

---

### Task 4: Make `lammps/export_bundle.py::build` importable without `lammps-jax`

Prerequisite for testing the bundle's `energy_fn` in the normal suite (Task 5). Pure refactor, no behaviour change.

**Files:**
- Modify: `lammps/export_bundle.py:24-31` (imports), `:70-115` (`main`)
- Test: `tests/test_bundle.py` (created here, extended in Task 5)

**Interfaces:**
- Produces: `build(npz, max_atoms, edges_per_atom, precision, a2b_sparse, edge_a_kind) -> (energy_fn, model, meta, rcut, max_atoms, max_edges)` importable via `importlib` from the tests; `energy_fn(positions, species, graph)` where `graph` has `.senders`, `.receivers`, `.edge_mask`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd acejax && uv run pytest tests/test_bundle.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lammps_jax'` (raised at module import), unless `lammps_jax` happens to be installed locally, in which case it passes trivially; proceed either way.

- [ ] **Step 3: Move the import into `main()`**

In `lammps/export_bundle.py` delete line `from lammps_jax.export import export_model` from the module header and add as the first line of `main()`:

```python
    from lammps_jax.export import export_model   # only the export needs the plugin package
```

`jax.config.update("jax_enable_x64", True)` stays at module level: the docstring's contract note says it must precede `export_model`, and the tests enable x64 themselves so importing it is harmless.

- [ ] **Step 4: Run the test**

Run: `cd acejax && uv run pytest tests/test_bundle.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add acejax/lammps/export_bundle.py acejax/tests/test_bundle.py
git commit -m "acejax/lammps: make export_bundle.build importable without lammps-jax"
```

---

### Task 5: Local-only node axis with NaN overflow guard (lever B)

**Files:**
- Modify: `lammps/export_bundle.py` (`build` signature and `energy_fn`; `--max-local` flag; contract notes in the docstring)
- Test: `tests/test_bundle.py` (extend)

**Interfaces:**
- Consumes: `load(..., fold=True)` default from Task 2.
- Produces: `build(npz, max_atoms=2560, edges_per_atom=64, precision="float64", a2b_sparse=False, edge_a_kind="gather", max_local=None) -> (energy_fn, model, meta, rcut, max_atoms, max_edges, max_local)`; `max_local=None` means `max_atoms` (old behaviour). `energy_fn` returns `(max_atoms,)` per-atom energies: rows `< max_local` are site energies, the rest zero; **all rows NaN if any masked-in sender `>= max_local`**.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bundle.py`)

```python
def _cluster(npz, n_pad_atoms=37, n_pad_edges=101):
    """Non-periodic cluster from the fixture geometry, so positions alone define
    rij (the bundle computes rij from positions; periodic images would not)."""
    model, meta, z = load(npz)
    pos = np.asarray(z["test_pos"]).T
    n = len(pos)
    cell = np.eye(3) * (np.ptp(pos, axis=0).max() + 2 * meta["rcut"] + 1)
    g = sparse_graph(pos, cell, np.zeros(3, bool), meta["rcut"])
    max_atoms = n + n_pad_atoms
    E = len(g.senders)
    senders = np.concatenate([g.senders, np.full(n_pad_edges, max_atoms)]).astype(np.int32)
    receivers = np.concatenate([g.receivers, np.full(n_pad_edges, max_atoms)]).astype(np.int32)
    edge_mask = np.concatenate([np.ones(E, bool), np.zeros(n_pad_edges, bool)])
    graph = types.SimpleNamespace(senders=jnp.asarray(senders),
                                  receivers=jnp.asarray(receivers),
                                  edge_mask=jnp.asarray(edge_mask))
    positions = jnp.asarray(np.concatenate([pos, np.zeros((n_pad_atoms, 3))]))
    species = jnp.concatenate([species_index(z), jnp.zeros(n_pad_atoms, jnp.int32)])
    # reference: the core evaluated on exactly the real edges
    node_z = species_index(z)
    send, recv = jnp.asarray(g.senders), jnp.asarray(g.receivers)
    with highest_precision():
        e_ref = model.site_energies(jnp.asarray(g.rij), node_z[send], node_z[recv],
                                    send, n, node_z)
    return positions, species, graph, np.asarray(e_ref), n, max_atoms


def test_local_axis_matches_full_axis(npz):
    mod = _export_bundle()
    positions, species, graph, e_ref, n, max_atoms = _cluster(npz)
    full = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=None)[0]
    local = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=n + 3)[0]
    with highest_precision():
        e_full = np.asarray(full(positions, species, graph))
        e_loc = np.asarray(local(positions, species, graph))
    assert e_full.shape == e_loc.shape == (max_atoms,)
    assert np.abs(e_full[:n] - e_ref).max() < 1e-12
    assert np.abs(e_loc[:n] - e_ref).max() < 1e-12
    assert np.all(e_loc[n:] == 0.0)
    assert np.all(np.isfinite(e_loc))


def test_local_axis_forces_match(npz):
    """Forces come from autodiff of the summed energy in the plugin; the local
    axis must not change the gradient w.r.t. positions, including ghost rows."""
    mod = _export_bundle()
    positions, species, graph, e_ref, n, max_atoms = _cluster(npz)
    full = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=None)[0]
    local = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=n + 3)[0]
    with highest_precision():
        gf = jax.grad(lambda p: jnp.sum(full(p, species, graph)))(positions)
        gl = jax.grad(lambda p: jnp.sum(local(p, species, graph)))(positions)
    assert np.abs(np.asarray(gf) - np.asarray(gl)).max() < 1e-12


def test_local_axis_overflow_is_nan_not_silent(npz):
    """segment_sum silently drops ids >= num_segments; the guard must turn an
    undersized max_local into NaN, never a plausible wrong energy."""
    mod = _export_bundle()
    positions, species, graph, e_ref, n, max_atoms = _cluster(npz)
    small = mod.build(npz, max_atoms=max_atoms, edges_per_atom=1, max_local=n - 1)[0]
    with highest_precision():
        e = np.asarray(small(positions, species, graph))
    assert np.all(np.isnan(e[:n - 1]))


def test_max_local_above_max_atoms_rejected(npz):
    mod = _export_bundle()
    with pytest.raises(ValueError, match="max_local"):
        mod.build(npz, max_atoms=100, edges_per_atom=1, max_local=101)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd acejax && uv run pytest tests/test_bundle.py -q`
Expected: FAIL — `TypeError: build() got an unexpected keyword argument 'max_local'`.

- [ ] **Step 3: Implement in `lammps/export_bundle.py`**

Replace `build`:

```python
def build(npz, max_atoms=2560, edges_per_atom=64, precision="float64",
          a2b_sparse=False, edge_a_kind="gather", max_local=None):
    """Return (energy_fn, model, meta, rcut, max_atoms, max_edges, max_local).

    `edge_a_kind` is baked into the exported program: the LAMMPS plugin has no
    calibration step and cannot run one, so the choice is made here, for a known
    target.  It does not change any value the bundle computes -- the two forms
    agree bit-identically on values and gradients -- only the reverse-pass cost.

    `max_local` is the capacity of the NODE axis the per-atom stages run over.
    The pair style packs edges from local centres only (PackNeighborFunctor
    iterates ilist over nlocal rows for n_hops=1), so ghost rows of the
    max_atoms axis are zero-neighbour rows that cost full per-atom work (AA,
    readout) and contribute nothing -- 3.1x nlocal at 1728 Si atoms.  None
    means max_atoms, the old behaviour.
    """
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    if max_local is None:
        max_local = max_atoms
    if not 0 < max_local <= max_atoms:
        raise ValueError(f"max_local must be in (0, max_atoms]; got {max_local} vs {max_atoms}")
    model, meta, _ = load(npz, dtype=dtype, a2b_sparse=a2b_sparse,
                          edge_a_kind=edge_a_kind)          # folded by default
    rcut = float(meta["rcut"])
    n_species = len(meta["elements"])
    max_edges = max_atoms * edges_per_atom

    def energy_fn(positions, species, graph):
        """Per-atom energies (max_atoms,). `species` is the LAMMPS type index (0-based)."""
        mask = graph.edge_mask
        centers = jnp.where(mask, graph.senders, 0)
        neighbors = jnp.where(mask, graph.receivers, 0)
        rij = positions[neighbors] - positions[centers]
        pad = jnp.asarray([rcut, 0.0, 0.0], positions.dtype)
        rij = jnp.where(mask[:, None], rij, pad)
        node_z = jnp.clip(species, 0, n_species - 1).astype(jnp.int32)
        e_local = model.site_energies(rij, node_z[centers], node_z[neighbors],
                                      centers, max_local, node_z[:max_local], mask)
        # segment_sum DROPS ids >= max_local silently; the plugin only checks
        # max_atoms.  Make an undersized bundle loud rather than subtly wrong.
        overflow = jnp.any(mask & (graph.senders >= max_local))
        e_local = jnp.where(overflow, jnp.nan, e_local)
        return jnp.zeros(positions.shape[0], e_local.dtype).at[:max_local].set(e_local)

    return energy_fn, model, meta, rcut, max_atoms, max_edges, max_local
```

In `main()`: add the flag

```python
    p.add_argument("--max-local", type=int, default=None,
                   help="node-axis capacity for local atoms (default: max_atoms). "
                        "Size it to nlocal with margin; ghosts do not need rows. "
                        "Exceeding it makes every energy NaN, by design.")
```

update the unpacking (`... max_edges, max_local = build(..., max_local=a.max_local)`), and print `max_local` with the other capacities. Add to the module docstring's contract notes:

```
  * The node axis is max_local, not max_atoms: senders are always local, so
    ghost rows are pure padding on the per-atom stages.  If nlocal exceeds
    max_local every energy is NaN -- segment_sum would otherwise drop the
    excess silently and the plugin does not check this capacity.
```

- [ ] **Step 4: Run the tests**

Run: `cd acejax && uv run pytest tests/test_bundle.py -q`
Expected: PASS (5 tests × 3 fixtures).

- [ ] **Step 5: Full suite**

Run: `cd acejax && uv run pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add acejax/lammps/export_bundle.py acejax/tests/test_bundle.py
git commit -m "acejax/lammps: per-atom stages over a max_local node axis; NaN on overflow"
```

---

### Task 6: LAMMPS-side validation of the local axis on moriarty

**Files:**
- Modify: `lammps/test_si_bundle.sh` (add a negative test)
- Modify: `lammps/in.mlip_si` only if it lacks a `thermo` line printing `pe` (read it first)

**Interfaces:**
- Consumes: `export_bundle.py --max-local` from Task 5; the host recipe in `test_si_bundle.sh` (PJRT path, `LAMMPS_PLUGIN_PATH`, `gpu/aware off`, `newton on neigh half`).

- [ ] **Step 1: Export two bundles for the 216-atom test system**

The 216-atom `in.si_setup` system on one rank has nlocal 216; `Nghost` is printed in the existing logs (`grep Nghost si.np1.log`). Use `max_atoms` from the current bundle recipe and `max_local` at 1.15x nlocal (= 249), and a deliberately undersized one at 200:

```bash
cd acejax/lammps
python export_bundle.py --npz ../fixtures/si_fitted.npz --out si_local.lammps-jax.json \
    --max-atoms 2560 --edges-per-atom 64 --max-local 249
python export_bundle.py --npz ../fixtures/si_fitted.npz --out si_undersized.lammps-jax.json \
    --max-atoms 2560 --edges-per-atom 64 --max-local 200
```

- [ ] **Step 2: Run the existing validation on the local-axis bundle**

```bash
./test_si_bundle.sh si_local.lammps-jax.json si_local
```

Expected: 1-rank and 2-rank runs complete; `cmpdump.py` np1 vs np2 within 1e-9; `check_vs_python.py` reports energy and force differences at the tolerance it already enforces (read its output; it prints the max |ΔF|). On 2 ranks nlocal per rank is ≤ 216, so `max_local` 249 still covers it.

- [ ] **Step 3: Add the negative test to `test_si_bundle.sh`**

Append:

```bash
# Undersized max_local must be LOUD: every energy NaN, never a plausible number.
if [ -f si_undersized.lammps-jax.json ]; then
  echo "### undersized max_local -> NaN ###"
  $V/bin/lmp $KK -var pjrt $PJRT -var bundle si_undersized.lammps-jax.json \
      -var dump_path undersized.dump -in in.mlip_si > undersized.log 2>&1 || true
  if grep -A1 "^ *Step" undersized.log | awk 'NR==2' | grep -qi nan; then
    echo "OK: potential energy is NaN"
  else
    echo "FAIL: undersized bundle produced a finite energy:"; grep -A1 "^ *Step" undersized.log | head -2; exit 1
  fi
fi
```

If LAMMPS aborts before printing thermo (e.g. "Non-numeric atom coords"), also accept that: extend the `if` to `grep -qiE 'nan|Non-numeric' undersized.log`. Either outcome is loud; a finite energy is the failure.

- [ ] **Step 4: Run it**

```bash
./test_si_bundle.sh si_local.lammps-jax.json si_local
```

Expected: the existing checks pass and the last block prints `OK: potential energy is NaN`.

- [ ] **Step 5: Commit** (logs and bundles are not committed; check `git status` shows only the script)

```bash
git add acejax/lammps/test_si_bundle.sh
git commit -m "acejax/lammps: validate max_local bundle in LAMMPS; undersized bundle must NaN"
```

---

### Task 7: Size capacities from a structure (lever C)

**Files:**
- Modify: `lammps/export_bundle.py` (new `size_capacities`; `--size-from`, `--skin`, `--margin-edges`, `--margin-atoms` flags)
- Modify: `bench/make_bundles.py` (pass `--max-local`; add `--no-fold`/`--no-max-local` pass-throughs for the by-difference rows in Task 8)
- Test: `tests/test_bundle.py` (extend)

**Interfaces:**
- Produces: `size_capacities(atoms, rcut, skin=1.0, margin_edges=1.3, margin_atoms=1.15) -> dict(n_local, n_ghost, n_edges, max_local, max_atoms, max_edges)` for an orthorhombic periodic `ase.Atoms`; raises `ValueError` for non-orthorhombic cells. `export_bundle.py --size-from structure.xyz` sets the three capacities from it; explicit `--max-*` flags override individually.
- `export_bundle.py --no-fold` (loads with `fold=False`; benchmark use only).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bundle.py`)

```python
def _diamond(reps, a=5.43):
    from ase.build import bulk
    return bulk("Si", "diamond", a=a, cubic=True) * (reps, reps, reps)


def test_size_capacities_counts_are_exact_and_ordered():
    mod = _export_bundle()
    atoms = _diamond(3)                       # 216 atoms, L = 16.29
    caps = mod.size_capacities(atoms, rcut=6.0, skin=1.0, margin_edges=1.0, margin_atoms=1.0)
    assert caps["n_local"] == 216
    # brute-force ghost count: images of every atom inside the box grown by rcut+skin
    pos = atoms.get_positions(); L = np.diag(atoms.get_cell().array); rc = 7.0
    ghosts = 0
    for s in np.array(np.meshgrid(*[[-1, 0, 1]] * 3)).reshape(3, -1).T:
        if not s.any():
            continue
        p = pos + s * L
        ghosts += int(np.all((p > -rc) & (p < L + rc), axis=1).sum())
    assert caps["n_ghost"] == ghosts
    g = sparse_graph(pos, atoms.get_cell().array, np.ones(3, bool), 6.0)
    assert caps["n_edges"] == len(g.senders)
    assert caps["max_local"] == 216 and caps["max_atoms"] == 216 + ghosts
    assert caps["max_edges"] == len(g.senders)


def test_size_capacities_applies_margins_and_rounds_up():
    mod = _export_bundle()
    caps = mod.size_capacities(_diamond(2), rcut=6.0, margin_edges=1.3, margin_atoms=1.15)
    assert caps["max_local"] == int(np.ceil(64 * 1.15))
    assert caps["max_atoms"] == int(np.ceil((64 + caps["n_ghost"]) * 1.15))
    assert caps["max_edges"] == int(np.ceil(caps["n_edges"] * 1.3))
    assert caps["max_local"] <= caps["max_atoms"]


def test_size_capacities_rejects_triclinic():
    mod = _export_bundle()
    atoms = _diamond(2)
    atoms.set_cell(atoms.get_cell().array + np.array([[0, 1.0, 0], [0, 0, 0], [0, 0, 0]]), scale_atoms=False)
    with pytest.raises(ValueError, match="orthorhombic"):
        mod.size_capacities(atoms, rcut=6.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd acejax && uv run pytest tests/test_bundle.py -q -k size_capacities`
Expected: FAIL — `AttributeError: module 'export_bundle' has no attribute 'size_capacities'`.

- [ ] **Step 3: Implement `size_capacities` in `lammps/export_bundle.py`**

```python
def size_capacities(atoms, rcut, skin=1.0, margin_edges=1.3, margin_atoms=1.15):
    """Single-rank capacities for `atoms` (ase.Atoms, periodic, orthorhombic).

    nlocal is exact; nghost counts periodic images inside the box grown by
    rcut + skin on every face, which is LAMMPS's communication cutoff; edges
    are full pairing within rcut, since the pair style filters skin pairs when
    it packs.  Domain decomposition only shrinks each rank's share, so the
    1-rank numbers bound every rank.  The margins cover atoms moving during
    a run; capacity cost is U-shaped (bench/results.md), so do not over-pad.
    """
    import itertools
    import numpy as np
    from acejax.nlist import sparse_graph
    cell = np.asarray(atoms.get_cell().array, float)
    if not np.allclose(cell, np.diag(np.diag(cell))):
        raise ValueError("--size-from needs an orthorhombic cell; pass --max-atoms, "
                         "--max-local and --edges-per-atom explicitly instead")
    if not np.all(atoms.get_pbc()):
        raise ValueError("--size-from needs a fully periodic structure")
    pos = atoms.get_positions(wrap=True)
    L = np.diag(cell)
    n = len(pos)
    rc = rcut + skin
    reps = np.ceil(rc / L).astype(int)
    n_ghost = 0
    for s in itertools.product(*[range(-r, r + 1) for r in reps]):
        if not any(s):
            continue
        p = pos + np.array(s) * L
        n_ghost += int(np.all((p > -rc) & (p < L + rc), axis=1).sum())
    n_edges = len(sparse_graph(pos, cell, np.ones(3, bool), rcut).senders)
    ceil = lambda x: int(math.ceil(x))
    return dict(n_local=n, n_ghost=n_ghost, n_edges=n_edges,
                max_local=ceil(n * margin_atoms),
                max_atoms=ceil((n + n_ghost) * margin_atoms),
                max_edges=ceil(n_edges * margin_edges))
```

Add `import math` at the top. Check the ghost predicate against how the test computes it (both use strict `>`/`<` on the grown box); the pair style adds a skin, so a boundary atom either way is covered by the margin.

In `main()` add flags and the sizing branch, and the `--no-fold` pass-through:

```python
    p.add_argument("--size-from", type=pathlib.Path, default=None,
                   help="size max_local/max_atoms/max_edges from this structure "
                        "(ASE-readable, periodic, orthorhombic); explicit --max-* override")
    p.add_argument("--skin", type=float, default=1.0, help="LAMMPS neighbor skin (A)")
    p.add_argument("--margin-edges", type=float, default=1.3)
    p.add_argument("--margin-atoms", type=float, default=1.15)
    p.add_argument("--no-fold", action="store_true",
                   help="keep the B-materialising readout (benchmark by difference only)")
```

Change the `--max-atoms` and `--edges-per-atom` defaults to `None` so "given explicitly" is detectable, then after parsing:

```python
    max_atoms, edges_per_atom, max_local = a.max_atoms, a.edges_per_atom, a.max_local
    if a.size_from is not None:
        import ase.io
        model_meta = json.loads(bytes(np.load(a.npz)["meta_json"]).decode())
        caps = size_capacities(ase.io.read(a.size_from), float(model_meta["rcut"]),
                               a.skin, a.margin_edges, a.margin_atoms)
        print("sized from", a.size_from, "| actual nlocal", caps["n_local"],
              "nghost", caps["n_ghost"], "edges", caps["n_edges"])
        max_atoms = max_atoms or caps["max_atoms"]
        max_local = max_local or caps["max_local"]
        edges_per_atom = edges_per_atom or max(1, -(-caps["max_edges"] // max_atoms))
    max_atoms = max_atoms or 2560
    edges_per_atom = edges_per_atom or 64
```

(`import json`, `import numpy as np` at the top.) `build` gains `fold=True` parameter passed to `load(..., fold=fold)`; `main` passes `fold=not a.no_fold`. Print the resulting `max_local`, `max_atoms`, `max_edges` and, when sized, the actual counts, so a quoted throughput can state its capacity.

- [ ] **Step 4: Run the tests**

Run: `cd acejax && uv run pytest tests/test_bundle.py -q`
Expected: PASS.

- [ ] **Step 5: Update `bench/make_bundles.py`**

In `capacities()` also return `max_local = int(math.ceil(n * atom_margin))`; add flags `--no-fold` and `--no-max-local`; pass `--max-local` unless `--no-max-local`, and `--no-fold` when given, through to `export_bundle.py`:

```python
    p.add_argument("--no-fold", action="store_true", help="baseline row: unfolded readout")
    p.add_argument("--no-max-local", action="store_true", help="baseline row: node axis = max_atoms")
    ...
        n, ma, me, ml = capacities(reps, rcut, edge_margin=a.edge_margin)
        cmd = [a.python, str(HERE.parent / "lammps" / "export_bundle.py"),
               "--npz", str(a.npz), "--out", str(out),
               "--max-atoms", str(ma), "--edges-per-atom", str(max(1, -(-me // ma))),
               "--precision", a.precision]
        if not a.no_max_local:
            cmd += ["--max-local", str(ml)]
        if a.no_fold:
            cmd += ["--no-fold"]
        if a.a2b_sparse:
            cmd += ["--a2b-sparse"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
```

Run: `cd acejax && python bench/make_bundles.py --npz fixtures/si_fitted.npz --reps 2 --outdir /tmp/mb_check` locally only if `lammps_jax` is installed; otherwise this is exercised in Task 8.

- [ ] **Step 6: Commit**

```bash
git add acejax/lammps/export_bundle.py acejax/bench/make_bundles.py acejax/tests/test_bundle.py
git commit -m "acejax/lammps: size bundle capacities from a structure (--size-from); benchmark pass-throughs"
```

---

### Task 8: Measure by difference on moriarty and record the results

**Files:**
- Modify: `bench/results.md` (new section at the end; do not edit historical sections)
- Create: `bench/rows_lever/` outputs are NOT committed (add `bench/rows_lever*/` to `.gitignore`)

**Interfaces:**
- Consumes: `bench/make_bundles.py --no-fold --no-max-local`, `bench/run_bench.sh jax pace`, `bench/bench_acejax.py`, the pace basis files from `bench/make_pace_basis.py` already on the host (read `results.md` "Comparison with pair_style pace" for the file names and the `YACE=` variable).

- [ ] **Step 1: Confirm the host and checkout**

```bash
ssh moriarty   # plain ssh; never BatchMode (see ~/.claude/CLAUDE.md)
cd <checkout of ACEpotentials on /home/eng/essswb>; git fetch jk && git switch pr/lammps-throughput
cd acejax; ls fixtures/si_s69.npz fixtures/si_m710.npz fixtures/si_l2849.npz
nvidia-smi --query-gpu=name,utilization.gpu --format=csv   # must be idle
```

- [ ] **Step 2: Export the four bundle sets per fixture**

```bash
for f in si_s69 si_m710 si_l2849; do
  S=$([ $f = si_s69 ] || echo --a2b-sparse)     # sparse A2B only at the two larger bases, as in results.md
  python bench/make_bundles.py --npz fixtures/$f.npz --reps 3 6 8 --outdir bench/rows_lever/${f}_base --no-fold --no-max-local $S
  python bench/make_bundles.py --npz fixtures/$f.npz --reps 3 6 8 --outdir bench/rows_lever/${f}_A     --no-max-local $S
  python bench/make_bundles.py --npz fixtures/$f.npz --reps 3 6 8 --outdir bench/rows_lever/${f}_AB    $S
done
```

`reps 3 6 8` = 216 / 1728 / 4096 atoms. Note that `make_bundles.py` already sizes capacities per point, so lever C changes nothing in these rows by construction; C is measured separately in Step 5.

- [ ] **Step 3: Run LAMMPS rows**

```bash
for f in si_s69 si_m710 si_l2849; do
  for row in base A AB; do
    echo "## $f $row"; BUNDLEDIR=rows_lever/${f}_$row STEPS=100 REPS="3 6 8" ./bench/run_bench.sh jax
  done
done
# pace/kk once, same day, same three matched bases.  The .ace files are built by
# bench/make_pace_basis.py <n_B> <order> <lmax> <rcut> <out> in the cp39 pyace venv
# (see its docstring); they are .ace TEXT, not .yace, and live on the host from the
# CORRECTION series (78 / 693 / 2874 functions, order 4, lmax 4/5/5).  Rebuild if missing:
#   python bench/make_pace_basis.py 78 4 4 6.0 bench/si_pace78.ace   (etc.)
for y in bench/si_pace78.ace bench/si_pace693.ace bench/si_pace2874.ace; do
  YACE=$y STEPS=100 REPS="3 6 8" ./bench/run_bench.sh pace
done
```

Save every table to `bench/rows_lever/<name>.txt`. Run each row twice; if the two disagree by more than 3% at any point, run a third and take the median.

- [ ] **Step 4: Python rows (retention denominator)**

```bash
for f in si_s69 si_m710 si_l2849; do
  S=$([ $f = si_s69 ] || echo --a2b-sparse)
  python bench/bench_acejax.py --npz fixtures/$f.npz --reps 3 6 8 $S                # folded (default)
done
```

For the unfolded Python row, temporarily run with `fold=False`: add `--no-fold` to `bench_acejax.py` (one flag, passed to `load`) in this task and commit it with the results.

- [ ] **Step 5: Lever C row**

One point, n_B = 2849, 1728 atoms: export with `--size-from` on the exact structure `in.si_bench` builds (write it out once with `write_data` in a copy of the input, or rebuild it with `bench_acejax.diamond(6)` and `ase.io.write`), default margins, then time it with `run_bench.sh` against the `AB` bundle at the same point. Record both capacities and both throughputs.

- [ ] **Step 6: Write the results section**

Append to `bench/results.md`:

```markdown
## Lever rows: fold (A), local node axis (B), sized capacities (C)

Same method as above (`timestep 0.0`, 100 steps, f64, one rank, moriarty),
measured by difference: one change per row, whole computation timed. Bundles
sized per point by `make_bundles.py`. pace/kk re-run the same day.

| n_B | atoms | base (ms/step) | +A | +A+B | pace/kk | pace ÷ jax base | pace ÷ jax A+B | retention base | retention A+B |
|---|---|---|---|---|---|---|---|---|---|
| 69 | 216 | | | | | | | | |
| ... |

A2B deletion oracle (Task 1): ceilings <X>/<Y>/<Z>x at 69/710/2849.

Lever C at n_B=2849, 1728 atoms: `--size-from` capacities (max_local/max_atoms/max_edges) = ... vs
make_bundles ...; ms/step ... vs ....

**Target (spec): pace ÷ jax ≤ 1.5x at 2849, ≤ 2x at 69.** Met / not met: ...
```

Fill every cell from the saved tables; leave nothing as a placeholder. If a target is not met, say so and which lever is next (f32 tier, edge-force export) per the spec's out-of-scope list — do not start it here.

- [ ] **Step 7: Commit**

```bash
git add acejax/bench/results.md acejax/bench/bench_acejax.py acejax/.gitignore
git commit -m "bench: lever rows A/B/C measured by difference on moriarty; pace/kk ratio <before> -> <after>"
```

---

### Task 9: Documentation and plan bookkeeping

**Files:**
- Modify: `README.md` (acejax; the LAMMPS export section)
- Modify: `docs/plans/jax_ace_port_plan.md` ("Open as of 2026-09-15" list)
- Modify: `docs/plans/lammps_throughput_design.md` (status line)

- [ ] **Step 1: README**

In the LAMMPS export section of `acejax/README.md`, replace the `--max-atoms/--edges-per-atom` example with `--size-from` and add two sentences: capacities are sized from a structure and printed; exceeding `max_local` makes every energy NaN by design, so an undersized bundle fails loudly. Mention that the readout is folded (C-tilde) by default and `fold=False` exists for measurement.

- [ ] **Step 2: Plan open items**

In `docs/plans/jax_ace_port_plan.md`, under the "Open as of 2026-09-15" list, add one line per lever with the measured before/after ratio and a pointer to `bench/results.md` "Lever rows". Change the design doc's status line to "implemented; measured — see bench/results.md".

- [ ] **Step 3: Commit**

```bash
git add acejax/README.md docs/plans/jax_ace_port_plan.md docs/plans/lammps_throughput_design.md
git commit -m "docs: LAMMPS throughput levers landed and measured"
```

---

### Task 10: Package 2 spike brief — exact yace export (timeboxed, 2 days)

Investigative; its output is a finding, not code kept. The spec (`docs/plans/lammps_throughput_design.md`, "Package 2") holds the steps; this task fixes the starting commands and the exit criteria so it can run in parallel with Tasks 5–8 on moriarty.

**Files:**
- Create: `docs/findings/FINDINGS_yace.md`
- Create (throwaway unless the spike passes): `acejax/spike_yace/`

- [ ] **Step 1: Pin what upstream parses**

```bash
cd /storage/eng/essswb/lammps-jax-build/lammps   # moriarty
grep -rn "radbasename\|ACE.jl\|splinenodalvals\|deltaSplineBins\|nradbasemax" \
   $(find . -path '*lammps-user-pace*' -name 'ace_radial.cpp' -o -path '*lammps-user-pace*' -name '*yaml*reader*.cpp' | head) | head -40
```

Record the tag (`cmake` fetch URL in `cmake/Modules/Packages/ML-PACE.cmake`) and every accepted `radbasename` in the finding.

- [ ] **Step 2: Diff a known-good pyace yace against the v0.6 export field by field**

The pyace files from `bench/make_pace_basis.py` are on the host (`bench/` — see `results.md` for names); `bench/v06/si_v06.yace` is the v0.6 export. Diff the top-level keys and the `bonds` block; write the table into the finding.

- [ ] **Step 3: Decide the radial route and, if exact, write `export_yace`**

Exact routes, in order of preference: (a) upstream accepts `ACE.jl`/`splinenodalvals` → emit our spline nodal values from the v0.10 `SplineRnlrzzBasis` (`splinify` has folded `Wnlq`); (b) a pinned fork accepts them; (c) `ChebPow`/`ChebExpCos` only if PACE's transform and cutoff reproduce `rnl_transform`/`rnl_envelope` exactly — otherwise report the fit residual and stop. Implementation goes in `src/export_yace.jl` (Julia, from the v0.10 model directly) only if (a) or (b) holds.

- [ ] **Step 4: Exactness test**

Load under `pair_style pace` and `pace/kk` on the `Si_tiny` test configuration used by `julia/export_model.jl`; compare to the Julia calculator. Pass: max |ΔF| ≤ 1e-10 eV/Å, |ΔE| ≤ 1e-10 eV.

- [ ] **Step 5: Finding and recommendation**

`docs/findings/FINDINGS_yace.md`: route taken, tag pinned, numbers, and one of: "ship as exporter + pinned-libpace CI (new plan)" or "not exact; CPU users deploy via <route>". Commit the finding; commit `spike_yace/` only if the recommendation is to ship.

```bash
git add docs/findings/FINDINGS_yace.md
git commit -m "Finding: exact yace export spike -- <pass/fail>, <route>"
```
