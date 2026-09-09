# Stage 1 — fit in Julia, evaluate in JAX

Phase 1 (+ enough of Phase 2 to validate it) of
`docs/plans/jax_ace_port_plan.md`. Exports a **fitted** `ace1_model` from Julia
and reproduces its site energies and forces in JAX.

## Status

Phases 1, 3 and 4 complete. Model fitted with `acefit!` on `Si_tiny` (BLR),
64-atom rattled Si, 2842 edges, f64.

**Phase 1 gate** — core against a Julia-supplied edge list:

| quantity | max abs error | relative |
|---|---|---|
| site energies | 2.56e-13 | 1.56e-15 |
| total energy | 0.00e+00 | — |
| forces | 1.14e-13 eV/Å | 4.15e-14 |

**Phase 3–4 gate** — from an ASE `Atoms` object, own neighbour list, periodic
images included, against `AtomsCalculators.energy_forces_virial`:

| quantity | max abs error | scale |
|---|---|---|
| energy | 1.82e-12 eV | 10441.5 |
| forces | 1.02e-13 eV/Å | 2.73 |
| virial | 6.55e-12 eV | 79.4 |
| dense vs sparse pooling | 8.53e-14 | — |

Target was 1e-10 throughout; ~2–3 orders of margin. The floor is the cubic
B-spline coefficient reproduction, measured at 1.14e-13 against Julia's
`Interpolations.jl` evaluation over 20000 random points.

The neighbour list reproduces Julia's edge set exactly (2842 edges, same
`(i, rij)` multiset). **The virial sign matches Julia's convention with no
flip**: `V = -dE/dε`, matching `site_virial = -sum(dv_i * r_i')`
(AtomsCalculatorsUtilities `sitepotentials/assembly.jl:6`).

## Run

```bash
julia --project=stage1 stage1/export_model.jl     # fit + export -> si_fitted.npz
cd stage1 && uv run pytest tests/ -q -s
```

`si_fitted.npz` is committed (0.3 MB) so the Python tests run without Julia.

## Layout

| path | purpose |
|---|---|
| `export_model.jl` | fit an `ace1_model`, export to npz |
| `acejax/radial.py` | transform, envelopes, cubic B-spline |
| `acejax/model.py` | `ACEModel` (Equinox): A → AA → B → site energy |
| `acejax/io.py` | npz loader |
| `tests/test_roundtrip.py` | array orientation + per-stage probe values |
| `tests/test_gate.py` | the gate: site energies, total energy, forces |
| `acejax/nlist.py` | matscipy-neighbours adapters, sparse and dense |
| `acejax/calculator.py` | ASE calculator |
| `tests/test_padding.py` | padded edges do not perturb or NaN the gradient |
| `tests/test_efv.py` | the Phase 3–4 gate: nlist, E/F/V, pooling layouts, ASE |

## Schema (npz, `schema_version` 1)

npz rather than JSON because Julia writes matrices column-major and 2-D arrays
round-trip through JSON transposed. `meta_json` is a UInt8 array holding JSON
(NPZ writes numeric arrays only).

```
meta_json           JSON: elements, counts, lmax, radial_kind, ybasis_kind,
                    spline grids, nnll spec, provenance
rnl_spline_coefs    (NZ,NZ,ncoef,n_rnl)   Julia's own B-spline coefficients
pair_spline_coefs   (NZ,NZ,ncoef,n_pair)
rnl_transform       (NZ,NZ,7)   p q a rin r0 yin ycut
pair_transform      (NZ,NZ,7)
rnl_envelope        (NZ,NZ,5)   x1 x2 p1 p2 s        (PolyEnvelope2sX)
pair_envelope       (NZ,NZ,3)   rcut r0 p            (ACE1_PolyEnvelope1sR)
aspec_r, aspec_y    (n_A,)      0-based Rnl / Ylm indices
aa_spec_{k}         (n_v,k)     0-based, grouped by correlation order
A2B                 (n_B,n_AA)  dense
WB (n_B,NZ), Wpair (n_pair,NZ), E0 (NZ,)
probe_*             per-stage reference values (transform, envelope, Rnl, Rpair, Ylm)
test_*              structure, edges, per-site energies, total energy, forces
```

`radial_kind` is `"spline"`. The analytic branch (`Wnlq` + poly recursion) is
reserved for Stage 2, where a trainable `Wnlq` is required; the loader raises
`NotImplementedError` rather than silently mishandling it.

## Decisions worth not re-litigating

* **Splined radials, exported as Julia's own coefficients.** `ace1_model`
  splinifies (`src/ace1_compat.jl:283`), so a fitted production model *is* the
  splined case. Exporting the coefficients means JAX and Julia agree bit-for-bit
  rather than differing by a re-tabulation error.
* **No module-level `jax.config.update`.** The Phase 0 spike set `x64` at import
  time, which silently overrode callers. Precision is the caller's; tests set it
  themselves, and `highest_precision()` is an explicit context manager.
* **Padding at the cutoff, not at zero.** Convention taken from lammps-jax's
  nequip template. `tests/test_padding.py` shows a zero pad produces 1500
  non-finite gradient entries while a cutoff pad produces none.
* **Radial coefficients are a live array leaf, not static.** For the splined
  branch Julia's `splinify` has already folded `Wnlq` into the coefficients, so
  they occupy `Wnlq`'s place in the parameter tree; keeping them a leaf is what
  keeps Stage 2 reachable.
* **Index arrays are int32 leaves, not static fields.** Marking a JAX array
  static warns and is a mistake; integer leaves are simply not differentiated.
* **Pooling is swappable, neither layout hard-wired.** `pool_sparse` (edge list,
  what lammps-jax exports) and `pool_dense` ((n, K) + count, what
  `neighbour_matrix` gives and what ET's own `(maxneigs, nnodes, nfeat)` layout
  looks like) sit behind `edge_features` / `_from_pooled`; everything downstream
  is per-node. `test_efv.py` pins that they agree to 8.5e-14.
* **Strain acts on edge vectors.** Because `rij = r[j] - r[i] + S@cell` and both
  halves transform under ε, the symmetric-displacement trick collapses to
  `rij -> rij + rij @ ε`. No cell bookkeeping, so the same code path serves
  LAMMPS, where there is no cell.
* **`dense_graph` parks padded slots at the cutoff.** `neighbour_matrix` leaves
  them as zero vectors, which NaNs the gradient the same way the sparse zero pad
  does. Done in the adapter so a caller cannot forget.
