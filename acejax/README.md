# acejax

JAX evaluator for ACE interatomic potentials fitted with
[ACEpotentials.jl](https://github.com/ACEsuit/ACEpotentials.jl).

Fit in Julia, evaluate in JAX: energies, forces, virials and site descriptors,
on CPU or GPU, with an ASE calculator and an export path to LAMMPS.

## Install

```bash
pip install acejax          # core evaluator + descriptors
pip install acejax[ase]     # + the ASE calculator
pip install acejax[cuda]    # + CUDA jaxlib
```

### Neighbour lists

acejax picks the best available backend automatically:

| | | |
|---|---|---|
| `matscipy-neighbours` | optional | GPU, DLPack, native dense layout |
| `matscipy` | **installed as a dependency** | C-accelerated, on PyPI |
| numpy fallback | built in | correct but O(N²); last resort |

`matscipy-neighbours` is not on PyPI (and PyPI rejects direct URL dependencies),
so it cannot be declared. Install it separately if you want it:

```bash
pip install git+https://github.com/libAtoms/matscipy-neighbours
```

All three produce **identical** edge sets, asserted in
`tests/test_efv.py::test_all_neighbour_backends_agree` — your results do not
depend on which is installed.

## Usage

Fit and export in Julia:

```julia
using ACEpotentials
model = ace1_model(elements = [:Si], order = 3, totaldegree = 10)
acefit!(data, model; energy_key = "dft_energy", force_key = "dft_force")
# then: julia --project=julia julia/export_model.jl si_fitted.npz ace1
```

Evaluate in Python:

```python
import jax
jax.config.update("jax_enable_x64", True)   # f64 is the caller's choice

from ase.build import bulk
from acejax import ACECalculator

atoms = bulk("Si", cubic=True) * 2
atoms.calc = ACECalculator("si_fitted.npz")

print(atoms.get_potential_energy())      # eV
print(atoms.get_forces().shape)          # (64, 3)
print(atoms.get_stress().shape)          # (6,) Voigt
```

### Site descriptors

The site basis is what the readout contracts against, so exposing it is nearly
free — and it comes from the *same single forward pass* as the energy:

```python
from acejax import site_descriptors

d = atoms.calc.get_site_descriptors(atoms)        # (n_atoms, n_basis)

# or without an ASE round-trip, which is what you want across a dataset
d = site_descriptors("si_fitted.npz", positions, numbers, cell, pbc)
d = site_descriptors("si_fitted.npz", positions, numbers, cell, pbc,
                     domain=[0, 5, 9])            # only these sites
```

The layout is species-blocked: the centre species selects which block is
populated and the rest are zero, matching `ACEpotentials.site_descriptors`.

This is one place the port is **faster than the original**, not merely
equivalent. `ACEpotentials.site_descriptors` is marked in the Julia source as
*"RETIRING THIS FOR NOW BECAUSE IT IS HIGHLY INEFFICIENT"* because it recomputes
per site; acejax takes the whole batch from one pass.

## What is covered

Both ACEpotentials model families, validated against Julia on a fitted model:

| | `ace1_model` | `ace_model` |
|---|---|---|
| radial basis | splined | analytic (`Wnlq` + 3-term recursion) |
| spherical harmonics | spherical | solid |
| energy | 1.8e-12 eV | 1.8e-12 eV |
| forces | 1.2e-13 eV/Å | 5.5e-13 eV/Å |
| virial | 9.8e-13 eV | 2.3e-12 eV |
| site descriptors | 5.7e-13 | 8.0e-14 |

(64-atom periodic Si, f64, against `AtomsCalculators.energy_forces_virial` and
`ACEpotentials.site_descriptors`.)

**Fitting still happens in Julia.** acejax evaluates; it does not fit. The
analytic branch keeps `Wnlq` as a live parameter, so trainable radials are
reachable, but no training loop is implemented.

## LAMMPS

`lammps/` exports a model as a [lammps-jax](https://github.com/abhijeetgangan/lammps-jax)
bundle for `pair_style jax/kk`:

```bash
python lammps/export_bundle.py --npz fixtures/si_fitted.npz --out si.lammps-jax.json
```

Verified on an RTX A4500: energies match the Python calculator exactly and
forces to 1.4e-13 eV/Å, on 1 and 2 MPI ranks, with 1314 and 1024/1025 ghost
atoms respectively (`lammps/run_artifacts/`).

**Known limitation.** The LAMMPS path is verified for single-point evaluation
and short runs only. Beyond roughly 50 MD steps it aborts with
`cudaErrorIllegalAddress`. This is not a capacity problem — tripling the bundle
capacities does not help — and the reneighbour repack path is the suspect. It is
unfixed. Do not use `pair_style jax/kk` for production MD yet.

`gpu/aware off` is required for multi-rank runs; with GPU-aware MPI the ghost
exchange aborts inside `CommKokkos::borders_device`, for stock bundles too.

## Precision

Nothing in acejax calls `jax.config.update` — precision is the caller's. But
matmul precision **is** pinned internally via `highest_precision()`: on Ampere
and later, XLA's TF32 default costs about 400× accuracy on this descriptor
(1.2e-3 vs 2.9e-6 against Julia). Use f64 for anything fitted; f32 costs only
~2.5× less time, because the descriptor is memory-bound rather than
FLOP-bound.

## Development

```bash
uv sync --group dev
uv run pytest tests/ -q
```

`fixtures/` holds two exported models so the suite runs without Julia. `julia/`
regenerates them. `bench/` is a throughput harness. Longer-form investigation
notes live in `../docs/findings/`.

## License

MIT.
