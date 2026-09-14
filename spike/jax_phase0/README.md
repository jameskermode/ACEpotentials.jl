# Phase 0 spike — JAX ETACE descriptor

**SPIKE CODE.** Throwaway quality, kept because the exporter and the JAX
descriptor are the two artifacts Phase 1 would build on. See
`docs/plans/jax_ace_port_plan.md`.

## Result

The JAX descriptor reproduces Julia's `ETModels.site_basis` to **5.1e-15
relative** (f64), and is **4–9× faster than Julia on CPU** for the forward pass.
Gate 3 says the naive VJP is not viable and the hand-written Jacobian is
mandatory for Stage 2.

## Files

| File | Purpose |
|---|---|
| `export_model.jl` | Dumps a Si ETACE model + reference values to `si_model.json` |
| `bench_julia.jl` | Julia baseline timings; dumps edge lists to `bench_data.json` |
| `etace_jax.py` | The JAX descriptor: A → AA → 𝔹 |
| `check.py` | Numerical agreement vs the Julia reference |
| `bench_jax.py` | CPU timings + gate 3 (Jacobian strategies) |

## Run

```bash
# from the repo root (Julia 1.12 via juliaup; ~/.juliaup/bin has no `julia` symlink)
~/.juliaup/bin/julialauncher --project=. spike/jax_phase0/export_model.jl
JULIA_NUM_THREADS=1 ~/.juliaup/bin/julialauncher --project=. spike/jax_phase0/bench_julia.jl

cd spike/jax_phase0
uv run python check.py
uv run python bench_jax.py
```

## Numbers (Apple Silicon, macOS arm64, f64, Julia single-threaded)

Model: `ace_model(elements=(:Si,), order=3, max_level=10, maxl=6, rcut=5.5)`
→ 37 Rnl, 25 Ylm, 43 A, 230 AA, 110 𝔹.

Forward descriptor:

| atoms | edges | JAX | Julia `site_basis` | ratio |
|---|---|---|---|---|
| 64 | 2154 | 0.241 ms | 0.970 ms | **4.0× faster** |
| 512 | 17206 | 0.863 ms | 7.933 ms | **9.2× faster** |

For context, Julia `site_descriptors` (classic path, 512 atoms) is 14.7 ms.
The ET path is 26 MB / 173 allocs — large buffers, not allocation churn.

Gate 3 — design-matrix Jacobian ∂𝔹/∂r (n_B = 110):

| atoms | naive vmap-VJP | hybrid (analytic dA/dr) | Julia `site_basis_jacobian` |
|---|---|---|---|
| 64 | 115.4 ms (479× fwd) | 8.2 ms (34× fwd) | 9.1 ms |
| 512 | 1480.8 ms (1715× fwd) | 129.7 ms (150× fwd) | 71.2 ms |

The two JAX strategies agree to 6.2e-16, so the hybrid is correct, not a
shortcut. Naive is 11–19× worse.

## Caveats

- **Gate 2 (GPU f32/f64) is UNMEASURED.** No CUDA device on this machine.
  `bench_jax.py` runs unmodified on a CUDA box and reports its backend.
- Uses `ace_model`, **not** `ace1_model`: `ace1_model` splinifies the radial
  basis and `convert2et` rejects `SplineRnlrzzBasis`.
- Julia 1.12.6 (the Manifest is resolved for 1.12; 1.11 fails to precompile).
  `CLAUDE.md` documents a 1.12 Dict-ordering issue affecting basis construction —
  harmless here because specs and reference 𝔹 come from the same session, so any
  ordering is self-consistent.
- Julia timed single-threaded; JAX CPU uses whatever XLA picks.
- Descriptor only. No neighbour list, no fitting, no LAMMPS export.
- JSON transposes matrices (Julia writes column-major as a list of columns).
  A real exporter should use npz.
