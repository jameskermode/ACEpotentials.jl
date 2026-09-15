# Finding: does the `lammps-export` branch reproduce a physical five-element v0.10 model exactly in LAMMPS? (2026-09-15)

**Question.** The ACEsuit `lammps-export` branch (juliac `--trim` compiled
library + `pair_style ace` plugin; tip 075d3859, Alexander Spears, 2026-08-21)
is the CPU deployment route that does not depend on ML-PACE. Its own tests
assert forces only to 1e-8, and its benchmark record puts it 2-4.6x slower
than ML-PACE per basis function. Does it (i) still build against current
dependencies and (ii) reproduce a *physical* CrMnFeCoNi v0.10 model exactly --
at 1e-10, 1e-9, 1e-8 eV/A -- through the C API and through LAMMPS, on the same
ten held-out configurations the yace spike used (`FINDINGS_yace.md` Part 2)?

**Answers.**

1. **It builds and its tests pass as-is; no rebase was needed.** `export/`
   instantiates against current packages (ACEpotentials 0.10.1 from the
   branch, EquivariantTensors 0.4.3, Polynomials4ML 0.5.8 pinned, JuliaC
   0.3.10, Julia 1.12.2); the branch's own `etace hermite multispecies`
   suite is 43/43 in 1m36s; `juliac --trim=safe` compiles the branch's Si
   test model in 102 s (first JuliaC load included) to a 2.35 MiB library, and
   the five-element libraries below in 30-37 s each.

2. **The `:polynomial` export is exact.** Through the Python C API and through
   `pair_style ace` (1 rank), energies and forces of the E0 + many-body part
   agree with the ETACE calculator *and* with the fitted classic `ACEModel` to
   **max |dF| = 6.7e-14 eV/A, |dE|/atom = 1.5e-13 eV** on forces of 3-6.7 eV/A.
   1e-10, 1e-9 and 1e-8 all pass with >= 1500x margin. The ET conversion is
   itself exact (2e-14 against the classic model, pair term included).

3. **The `:hermite_spline` export -- the README's recommended mode -- is broken
   for any multi-element model at the branch tip**: the generated code writes
   one spline table per ordered species pair but dispatches on a *symmetric*
   pair index, so for NZ >= 2 most neighbour species read the wrong table.
   As-is: **13.5 eV/A, 7.2 eV/atom** errors in LAMMPS. It is a one-line fix
   (`export/src/codegen.jl`, two dispatch lines; local commit 39d853dc on the
   host branch `lammps-export-verify`, not pushed). The branch's multi-species
   test only exercises `:polynomial`, so CI cannot see this.

4. **Even fixed, `:hermite_spline` is not exact against the fitted model**,
   because the splinification it requires (`ETModels.splinify`, cubic Hermite
   in the transformed coordinate on `Nspl` knots) is the inexact step:
   **2.7e-4 eV/A at the default `Nspl = 50`, 3.0e-6 at `Nspl = 200`**
   (|dE|/atom 5e-6 and 1.4e-8). Against the *splinified* model the fixed
   Hermite library is exact (6.8e-14). So the README's "machine precision"
   claim is true only relative to the splined model; against the fitted
   polynomial model all three tolerances **fail** at both node counts.

5. **The pair potential is not carried.** `export_ace_model(::StackedCalculator)`
   picks out `ETOneBody` (E0s) and `ETACE` and silently ignores an
   `ETPairModel`; the generated file says "ETACE has no pair potential". The
   export of the (E0, pair, ACE) stack is byte-identical to that of the
   (E0, ACE) stack. On this model the pair term carries the repulsive core:
   the E0 + many-body part alone is *attractive* on every dimer (-1.5 eV at
   3.0 A, Cr-Cr) and changes held-out forces by up to 6.9 eV/A. A deployed
   library therefore needs a separate `pair_style table` (or the pair basis
   exported), exactly as the yace route does.

**Bottom line.** With `:polynomial`, `lammps-export` is the only route measured
so far that reproduces the many-body component at float64 resolution (6.7e-14
vs 1.9e-10 for the fork-pinned yace at 1e4 nodes) and it loads in
milliseconds instead of 48 s. But at the branch tip the recommended Hermite
mode is wrong for multi-element models, the fixed Hermite mode is 1e-4-1e-6
inexact by construction, and neither mode carries the pair potential.
Throughput was not measured (box loaded, see below).

## Build status and rebase

Host `moriarty`, checkout `~/ace-potentials-julia-1.2/ACEpotentials.jl`. It was
on `lammps-export` at b66f4e12 (Jan 2026) with a dirty `export/Manifest.toml`
(2 lines); that edit is stashed as
`stash@{0}: verify-spike 2026-09-15: pre-existing export/Manifest.toml edit`.
New local branch **`lammps-export-verify`** from `acesuit/lammps-export`
(075d3859); the branch had removed `export/Manifest.toml` from git
(278d2abe, "resilient against new releases"), so the resolve is fresh.

| step | result |
|---|---|
| `Pkg.instantiate()` + precompile in `export/` | OK (193 deps precompiled, 136 s). ACEpotentials 0.10.1 (path `..`), ACEfit 0.3.1, EquivariantTensors 0.4.3, Polynomials4ML 0.5.8 (git tag pin), Lux 1.31.4, JuliaC 0.3.10, StaticArrays 1.9.20 |
| `export/test/runtests.jl etace hermite multispecies` | **43/43 pass**, 1m35.5s (ETACE 23, Hermite 7, multi-species 13). Python/LAMMPS/MPI subsets not run (MPI absent; the LAMMPS subset needs the branch's own Si build and compares to Python at 1e-6 only) |
| `juliac --trim=safe` of `test/build/test_etace_model.jl` (CI recipe: `ImageRecipe`/`LinkRecipe`, `add_ccallables`) | OK, **102 s** wall including JuliaC load, **2.35 MiB** `libace_test.so` |
| LAMMPS plugin rebuild against `~/lammps/lammps-22Jul2025/src` (`cmake ../cmake -DLAMMPS_HEADER_DIR=... -DBUILD_OMP=OFF`) | OK, 71 KB `aceplugin.so` (`verify_cantor/plugin_build/`) |
| rebase onto `origin/main` | **not needed**; not done. (`lammps-export` differs from the merge-base 3adf3a8b outside `export/` only by adding `PackageCompiler` to `Project.toml`; `main` has since touched `src/atoms_data.jl`, `src/models/ace.jl` (the 13x `evaluate_ed` fix, #326) and `Project.toml`. A rebase would be trivial.) |

Only local change on the host branch: commit 39d853dc (the Hermite dispatch
fix, below). Nothing pushed.

The host LAMMPS (`~/lammps/lammps-22Jul2025/build/lmp`, packages EXTRA-FIX
KOKKOS ML-IAP ML-SNAP MOLECULE PLUGIN PYTHON) needs
`LD_LIBRARY_PATH=/software/easybuild/software/GCCcore/14.3.0/lib64:
~/miniconda3/envs/noteable_base_chemistry/lib:/software/easybuild/software/CUDA/12.9.1/lib64`
to start at all (GLIBCXX_3.4.32, libpython3.12, libcudart.so.12), plus the
Julia `lib` and `lib/julia` directories for the ACE library.

## Model and physicality

v0.10 twin of the yace spike's v0.6 fit, built with `Models.ace_model` (the
branch's ETACE path requires learnable radials, so `ace1_model` cannot be
used):

```julia
level = TotalDegree(5.0, 1/1.5)            # ace1_model's weighting: n/NZ + 1.5 l
ace_model(elements = (:Cr,:Mn,:Fe,:Co,:Ni), order = 3, Ytype = :solid,
          level = level, max_level = 6, pair_maxn = 30,
          rin0cuts = (rin = 0, r0 = 2.54, rcut = 6.25) for all pairs,
          init_WB = :zeros, init_Wpair = :onehot, init_Wradial = :onehot,
          pair_learnable = true,            # keeps the pair basis convertible to ET
          E0s = per-element least squares of the 250 training energies)
```

Basis: 74 (n,l) radials (maxn 30, maxl 3), **1348 many-body + 30 pair
functions per species, 6890 linear parameters**. The `:onehot` radial init is
ACE1-style, R_{n'z'} = P_{n'}(r) delta_{z', Z_neighbour}, so the radial
embedding genuinely depends on the neighbour species (this is what exposes the
Hermite bug). Fit: configs 1-250 of `cantor1k_b_mh1.xyz` (9920 atoms), keys
`mace_energy/mace_force/mace_virial`, default weights (E 30, F 1, V 1),
`repulsion_restraint = true`, smoothness prior p = 4, `ACEfit.BLR()`. The
`acefit!` call was replicated inline so the assembled system could be cached
(`lsq_cantor.jld2`, 1.1 GB): assembly (31525 x 6890) 1184 s on 8 Distributed
workers, BLR 1383 s with 8 BLAS threads (47 L-BFGS iterations; a first
attempt with single-threaded BLAS ran at ~100 s/iteration and was killed).

| set | E RMSE [meV/atom] | F RMSE [eV/A] | V RMSE [meV/atom] |
|---|---|---|---|
| train (250) | 0.18 | 0.108 | 18.6 |
| held-out (10, configs 991-1000) | 8.5 | **0.146** | 67.7 |

(v0.6 twin, same data: 0.23 / 0.101 / 19.4 train, 6.0 / 0.124 / 55.8 held-out.)

Held-out max |F| per config, full potential: 3.76 1.56 1.83 1.30 3.18 2.38
3.58 3.67 2.78 3.32 (MACE reference 4.29 1.44 1.83 1.39 3.71 2.29 3.75 3.81
3.09 3.38). Dimer curves `E(r) - E0a - E0b` [eV], full potential:
Cr-Cr 0.54 (3.0 A), 2.02 (2.4), 3.29 (1.8), 3.99 (1.4), 5.25 (1.0);
Fe-Ni 0.60, 1.49, 2.73, 3.98, 6.11; Mn-Co 0.44, 1.48, 2.97, 4.25, 6.23;
Ni-Ni 0.31, 1.04, 2.41, 3.77, 5.91 -- monotonically repulsive, physical.

**The exported component is not.** E0 + many-body only (`Wpair = 0`): held-out
max |F| 6.42 6.43 3.03 4.18 4.58 5.98 5.76 6.66 4.25 3.47 (up to 6.9 eV/A of
pair force removed), and the same dimers are **attractive**: Cr-Cr -1.52
(3.0 A), -0.91 (2.4), -0.35 (1.8), -0.12 (1.4), -0.02 (1.0); Fe-Ni -1.43,
-0.90, -0.39, -0.17, -0.05; Mn-Co -2.08, -1.56, -0.84, -0.42, -0.14; Ni-Ni
-1.25, -0.92, -0.47, -0.23, -0.07. All exactness numbers below are for this
E0 + many-body component, which is what the library computes.

Max neighbours within 6.25 A over the held-out configs: 80-92 (the generated
code hard-codes `MAX_NEIGHBORS = 256`).

## Julia-side chain (ten held-out configs, the yace spike's rotated geometries read back from `cantor_k.data`)

(a) fitted classic `ACEModel`; (a_mb) the same with `Wpair = 0`;
(b) `ETModels.convert2et_full` = `StackedCalculator(ETOneBody, ETPairModel, ETACE)`;
(b_mb) `(ETOneBody, ETACE)`; (c50)/(c200) `(ETOneBody, splinify(ETACE; Nspl))`.
`dE` is per atom [eV], `dF` is max over atoms of |dF| [eV/A].

| cfg | N | max\|F\| (a) | max\|F_pair\| | dE (a)-(b) | dF (a)-(b) | dE (a_mb)-(b_mb) | dF (a_mb)-(b_mb) | dE (b_mb)-(c50) | dF (b_mb)-(c50) | dE (b_mb)-(c200) | dF (b_mb)-(c200) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 3.76 | 5.77 | 0.0 | 2.1e-14 | 9.5e-15 | 1.8e-14 | 4.0e-06 | 2.31e-04 | 1.2e-08 | 2.95e-06 |
| 2 | 32 | 1.56 | 6.88 | 2.0e-14 | 1.8e-14 | 1.4e-14 | 2.0e-14 | 1.8e-06 | 2.36e-04 | 1.2e-08 | 2.68e-06 |
| 3 | 32 | 1.83 | 3.40 | 3.6e-15 | 1.5e-14 | 7.1e-15 | 1.3e-14 | 2.5e-06 | 1.11e-04 | 1.0e-08 | 2.07e-06 |
| 4 | 32 | 1.30 | 4.60 | 5.3e-15 | 1.7e-14 | 7.1e-15 | 1.7e-14 | 2.9e-06 | 1.42e-04 | 1.2e-08 | 2.39e-06 |
| 5 | 48 | 3.18 | 5.23 | 4.7e-15 | 1.4e-14 | 9.5e-15 | 1.2e-14 | 5.2e-07 | 1.84e-04 | 8.9e-09 | 1.81e-06 |
| 6 | 48 | 2.38 | 5.06 | 1.4e-14 | 1.7e-14 | 9.5e-15 | 1.3e-14 | 4.6e-06 | 1.76e-04 | 1.2e-08 | 2.59e-06 |
| 7 | 32 | 3.58 | 4.35 | 7.1e-15 | 1.7e-14 | 0.0 | 1.2e-14 | 3.2e-06 | 1.82e-04 | 8.7e-09 | 1.80e-06 |
| 8 | 48 | 3.67 | 6.00 | 2.4e-15 | 2.0e-14 | 0.0 | 1.7e-14 | 4.0e-06 | 2.68e-04 | 1.4e-08 | 1.81e-06 |
| 9 | 32 | 2.78 | 3.50 | 7.1e-15 | 1.5e-14 | 0.0 | 1.5e-14 | 5.0e-06 | 2.29e-04 | 7.2e-09 | 1.59e-06 |
| 10 | 32 | 3.32 | 3.07 | 2.3e-14 | 1.7e-14 | 1.4e-14 | 1.0e-14 | 9.7e-07 | 2.00e-04 | 1.1e-08 | 1.78e-06 |
| **max** | | | | **2.3e-14** | **2.1e-14** | **1.4e-14** | **2.0e-14** | **5.0e-06** | **2.68e-04** | **1.4e-08** | **2.95e-06** |

So: conversion to ET is exact for all three components; **splinification is
the inexact step** -- 2.7e-4 eV/A at the branch's default `Nspl = 50` (rel.
6e-5 of max |F|), 3.0e-6 at `Nspl = 200`. The exporter's own file-size cost of
`Nspl` is the generated source: 4.7 MB (50) vs 17.6 MB (200) vs 0.6 MB
(polynomial).

## Exported code evaluated in Julia (isolates codegen from compilation)

The generated `.jl` included into a fresh module and driven site by site
(`site_energy_forces`, `NeighbourLists.PairList` at 6.25 A):

| export | vs | max \|dE\|/atom | max \|dF\| |
|---|---|---|---|
| `:polynomial` | (b_mb) | 1.4e-14 | 2.1e-14 |
| `:hermite_spline`, Nspl 50, **as-is (075d3859)** | (c50) | **7.2** | **13.5** |
| `:hermite_spline`, Nspl 200, as-is | (c200) | 7.2 | 13.5 |
| `:hermite_spline`, Nspl 50, dispatch fixed | (c50) | 1.9e-14 | 2.5e-14 |
| `:hermite_spline`, Nspl 200, dispatch fixed | (c200) | 2.1e-14 | 2.1e-14 |

### The Hermite bug

`export/src/codegen.jl`, `generate_hermite_spline_code`: the tables
`PAIR_k_F/G` are emitted for `k = 1:NZ^2` in the category order the
splinified layer uses, `k = (iz-1)*NZ + jz` (`splinify.jl`,
`extract_hermite_spline_data`, lines 157-159), but both dispatch functions
select the table with

```julia
pair_idx = zz2pair_sym(iz, jz)      # (i,j) -> (i-1)*NZ - (i-1)(i-2)/2 + (j-i+1), symmetric, 1..NZ(NZ+1)/2
```

For NZ = 5 that maps (2,2) to table 6 = category (2,1), (3,4) to 11 = (3,1),
(5,5) to 15 = (3,5), and so on; only (1,j) and, by coincidence, some
others land on a table with the right neighbour species. Probe on the
exported radials: `evaluate_Rnl(r, 2, 2)` differs from the polynomial export
by 0.70 at r = 2.5 A (|R| = 0.70), i.e. the wrong function entirely. The
`:polynomial` path (`write_radial.jl`) uses `(iz - 1) * NZ + jz` with the
comment "Asymmetric indexing for weights". For NZ = 2 (the TiAl of the
branch's benchmark) the same bug maps (2,2) to table 3 = (2,1); the
multi-species test (`test_multispecies.jl`) exports `:polynomial` only and the
Hermite test is single-species Si, so it passes.

Fix applied on the host branch (commit 39d853dc, two lines):

```julia
pair_idx = (iz - 1) * NZ + jz
```

(A better fix would emit only the tables the model has and keep one indexing
convention in one place; not done here.)

## Outside Julia: C API and LAMMPS

Libraries compiled with the CI recipe (`ImageRecipe(trim_mode = "safe",
add_ccallables = true, cpu_target = "native")` + `LinkRecipe`), one per mode:

| library | source | juliac wall | .so size |
|---|---|---|---|
| `libace_cantor_poly.so` | 0.6 MB | 37.3 s | 4.49 MiB |
| `libace_cantor_hermite50.so` (fixed) | 4.7 MB | 34.9 s | 3.50 MiB |
| `libace_cantor_hermite200.so` (fixed) | 17.6 MB | 33.8 s | 5.20 MiB |
| `libace_cantor_hermite50_asis.so` (075d3859 dispatch) | 4.7 MB | 29.5 s | 3.51 MiB |

Runtime the library actually links (`ldd`): `libjulia`, `libjulia-internal`
(14.8 MB), `libstdc++` (21 MB), `libunwind`, `libz`, `libatomic`, `libgcc_s`
-- **~38 MiB**; the branch's `build_deployment.jl` bundles ~78 MiB (adds
OpenBLAS 36 MB, SuiteSparse, MPFR, ... that the trimmed library does not
reference). The Python side (`ACELibraryCalculator`, ctypes + matscipy
neighbour lists, `export/ase-ace/src` on `PYTHONPATH`) loads the library in
0.08 s and evaluates a 32-48-atom config in 20-84 ms. LAMMPS: one fresh
process per config, `read_data` the yace spike's `cantor_k.data`,
`plugin load aceplugin.so; pair_style ace; pair_coeff * * lib.so Cr Mn Fe Co Ni`,
`run 0`, forces dumped at `%.17g`. The plugin prints the mapping at load
(`Type 1 -> Cr (Z=24)` ... `Type 5 -> Ni (Z=28)`). **Start-up: 1.55-1.62 s per
run, indistinguishable from the same LAMMPS with no potential (1.61 s); RSS 257
MB (baseline 210 MB).** The yace route needed 48 s and 3.9 GB per start at 1e4
nodes.

### `:polynomial` -- `pair_style ace` vs (b_mb) (= vs (a_mb), the fitted classic model, to the same digits)

| cfg | N | max\|F\| | max\|dF\| | rel | mean\|dF\| | \|dE\|/atom |
|---|---|---|---|---|---|---|
| 1 | 48 | 6.42 | 6.7e-14 | 1.0e-14 | 1.8e-14 | 6.6e-14 |
| 2 | 32 | 6.43 | 3.3e-14 | 5.1e-15 | 1.3e-14 | 5.0e-14 |
| 3 | 32 | 3.03 | 2.0e-14 | 6.6e-15 | 1.1e-14 | 2.1e-14 |
| 4 | 32 | 4.18 | 4.0e-14 | 9.5e-15 | 1.4e-14 | 2.8e-14 |
| 5 | 48 | 4.58 | 3.0e-14 | 6.5e-15 | 1.1e-14 | 7.6e-14 |
| 6 | 48 | 5.98 | 3.5e-14 | 5.8e-15 | 1.6e-14 | 1.1e-13 |
| 7 | 32 | 5.76 | 2.8e-14 | 4.9e-15 | 1.2e-14 | 1.2e-13 |
| 8 | 48 | 6.66 | 3.7e-14 | 5.5e-15 | 1.3e-14 | 0.0 |
| 9 | 32 | 4.25 | 2.2e-14 | 5.1e-15 | 8.5e-15 | 1.5e-13 |
| 10 | 32 | 3.47 | 2.4e-14 | 7.0e-15 | 1.1e-14 | 3.6e-14 |
| **all** | | | **6.7e-14** | **1.0e-14** | 1.3e-14 | **1.5e-13** |

Python C API, same library: max |dF| 2.3e-14, max rel 6.0e-15, |dE|/atom
2.1e-14 (vs (a_mb): 2.6e-14). The LAMMPS numbers are 2-3x the Python ones --
neighbour-order summation noise, nothing systematic.

### `:hermite_spline` (dispatch fixed) -- `pair_style ace`

| cfg | N | max\|F\| | Nspl 50 vs (c50) dF | \|dE\|/N | **Nspl 50 vs (b_mb) dF** | rel | mean | \|dE\|/N | Nspl 200 vs (c200) dF | **Nspl 200 vs (b_mb) dF** | rel | \|dE\|/N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 6.42 | 6.8e-14 | 9.5e-14 | 2.31e-04 | 3.6e-05 | 8.9e-05 | 4.0e-06 | 5.6e-14 | 2.95e-06 | 4.6e-07 | 1.2e-08 |
| 2 | 32 | 6.43 | 2.8e-14 | 1.3e-13 | 2.36e-04 | 3.7e-05 | 8.2e-05 | 1.8e-06 | 2.5e-14 | 2.68e-06 | 4.2e-07 | 1.2e-08 |
| 3 | 32 | 3.03 | 2.1e-14 | 5.0e-14 | 1.11e-04 | 3.7e-05 | 6.7e-05 | 2.5e-06 | 2.1e-14 | 2.07e-06 | 6.8e-07 | 1.0e-08 |
| 4 | 32 | 4.18 | 4.2e-14 | 4.3e-14 | 1.42e-04 | 3.4e-05 | 6.8e-05 | 2.9e-06 | 3.9e-14 | 2.39e-06 | 5.7e-07 | 1.2e-08 |
| 5 | 48 | 4.58 | 2.5e-14 | 9.5e-15 | 1.84e-04 | 4.0e-05 | 1.0e-04 | 5.2e-07 | 2.6e-14 | 1.81e-06 | 4.0e-07 | 8.9e-09 |
| 6 | 48 | 5.98 | 3.4e-14 | 9.5e-15 | 1.76e-04 | 2.9e-05 | 9.3e-05 | 4.6e-06 | 3.4e-14 | 2.59e-06 | 4.3e-07 | 1.2e-08 |
| 7 | 32 | 5.76 | 2.9e-14 | 3.6e-14 | 1.82e-04 | 3.2e-05 | 8.7e-05 | 3.2e-06 | 3.0e-14 | 1.80e-06 | 3.1e-07 | 8.7e-09 |
| 8 | 48 | 6.66 | 3.8e-14 | 9.5e-14 | 2.68e-04 | 4.0e-05 | 1.0e-04 | 4.0e-06 | 4.1e-14 | 1.81e-06 | 2.7e-07 | 1.4e-08 |
| 9 | 32 | 4.25 | 1.6e-14 | 9.9e-14 | 2.29e-04 | 5.4e-05 | 1.1e-04 | 5.0e-06 | 1.5e-14 | 1.59e-06 | 3.7e-07 | 7.2e-09 |
| 10 | 32 | 3.47 | 2.4e-14 | 1.6e-13 | 2.00e-04 | 5.8e-05 | 9.1e-05 | 9.7e-07 | 2.7e-14 | 1.78e-06 | 5.1e-07 | 1.1e-08 |
| **all** | | | **6.8e-14** | 1.6e-13 | **2.68e-04** | **5.8e-05** | 8.9e-05 | **5.0e-06** | **5.6e-14** | **2.95e-06** | **6.8e-07** | **1.4e-08** |

Python C API gives the same to the digit (2.7e-14 vs (c50); 2.68e-04 vs
(b_mb); 2.95e-06 for Nspl 200).

### `:hermite_spline` as-is (branch tip 075d3859), Nspl 50 -- `pair_style ace` and C API

max |dF| = **13.5 eV/A** (rel 1.35 of max |F|), mean |dF| 4.0 eV/A,
|dE|/atom **7.2 eV**, every config (e.g. cfg 1: 13.5 / 7.2; cfg 2: 7.6 / 5.4;
cfg 3: 7.9 / 6.1). Identical from Python and from LAMMPS, identical to the
in-Julia evaluation of the generated source -- the compiled library is a
faithful copy of wrong generated code.

### Verdicts (max |dF| over the 10 held-out configs, E0 + many-body component, `pair_style ace`, 1 rank)

Reference = the fitted v0.10 model (classic `ACEModel` with `Wpair = 0`, which
equals the polynomial ETACE to 2e-14):

| export mode | max \|dF\| | **1e-10 abs** | **1e-9 abs** | **1e-8 abs** | 1e-9 rel |
|---|---|---|---|---|---|
| `:polynomial` | 6.7e-14 | **PASS** (1500x) | **PASS** (15000x) | **PASS** (150000x) | PASS |
| `:hermite_spline` Nspl 50, as-is (tip) | 13.5 | FAIL | FAIL | FAIL | FAIL |
| `:hermite_spline` Nspl 50, dispatch fixed | 2.7e-4 | FAIL | FAIL | FAIL | FAIL |
| `:hermite_spline` Nspl 200, dispatch fixed | 3.0e-6 | FAIL | FAIL | FAIL | FAIL |

Reference = the *splinified* ETACE the Hermite export was made from (what the
branch's own tests measure):

| export mode | max \|dF\| | 1e-10 | 1e-9 | 1e-8 |
|---|---|---|---|---|
| `:hermite_spline` Nspl 50, fixed | 6.8e-14 | PASS (1460x) | PASS | PASS |
| `:hermite_spline` Nspl 200, fixed | 5.6e-14 | PASS (1780x) | PASS | PASS |
| `:hermite_spline` Nspl 50, as-is | 13.5 | FAIL | FAIL | FAIL |

Nothing was loosened; the only code change is the two-line dispatch fix,
and the as-is row is reported alongside.

## Is the pair potential carried? **No.**

- `export_ace_model(calc::StackedCalculator, ...)` (`export_ace_model.jl:60-90`)
  loops over `calc.calcs` keeping the `ETOneBody` (for E0s) and the `ETACE`;
  an `ETPairModel` matches neither branch and is dropped without a warning.
- `_write_evaluation_functions(io, tensor, NZ, false)  # No pair potential in ETACE`
  -- the `has_pair` argument is unused in `write_evaluation.jl`; the generated
  header says "ETACE has no pair potential (many-body only)".
- Measured: the `:polynomial` export of `convert2et_full(...)` (E0, pair, ACE)
  is **byte-identical** to the export of (E0, ACE).
- The E0 terms *are* carried (`site_energy_forces` adds `E0_iz` per site;
  energies above match E0 + many-body to 1e-13), so the exported energy is
  one-body + many-body, not the full potential. A `pair_style table` (or
  `hybrid/overlay` with something else) is still required, as for yace; the
  `:polynomial` codegen has the pieces (`_write_weights` knows `Wpair`) but no
  path uses them for ETACE models.

## Throughput

**Not measured.** `/proc/loadavg` was 9-19 throughout (another user's ~13-core
job), never below 2, so neither the 2000-atom `pair_style ace` run nor the
matched `pair_style pace` run was attempted; the inputs are ready
(`verify_cantor/in.bench_ace`, `in.bench_pace`, 2048-atom random fcc
CrMnFeCoNi, 100 steps, `timestep 0.0`). The branch's own record
(`benchmark/fair_comparison/MLPACE_COMPARISON.md`, TiAl, 2000 atoms, 100
steps) is ML-PACE 16.0 s (1166 functions) vs ETACE spline 19.2 s (308
functions): per basis function 4.6x slower, and that is the (buggy for NZ = 2)
Hermite mode; the exact `:polynomial` mode evaluates 30 Chebyshev
polynomials per neighbour instead of a table lookup, so expect it slower
still. Sizes and start-up above are measured.

## Comparison with the yace route (`FINDINGS_yace.md` Part 2, same held-out configs)

| | fork-pinned `.yace` (v0.6 fit) | `lammps-export` `:polynomial` | `lammps-export` `:hermite_spline` (fixed) |
|---|---|---|---|
| max \|dF\| vs its own Julia model | 1.9e-10 (1e4 nodes), 9.8e-11 (1e5) | **6.7e-14** | 2.7e-4 (Nspl 50) / 3.0e-6 (200) |
| \|dE\|/atom | 8.5e-14 | 1.5e-13 | 5e-6 / 1.4e-8 |
| 1e-10 / 1e-9 / 1e-8 | FAIL / PASS / PASS at 1e4 | PASS / PASS / PASS | FAIL / FAIL / FAIL |
| pair potential | separate `pair_style table` (0.001 A grid) | not carried | not carried |
| file / RSS / start-up | 185 MB / 3.9 GB / 48 s (1e4); 1.9 GB / 37 GB / 465 s (1e5) | 4.5 MiB + ~38 MiB runtime / 257 MB / ~0 s over LAMMPS itself | 3.5-5.2 MiB, same |
| LAMMPS dependency | wcwitt fork of ML-PACE at a pinned SHA, CPU `pair_style pace` only | stock LAMMPS with PLUGIN package + 71 KB plugin, needs the Julia runtime libs on `LD_LIBRARY_PATH` | same |
| model version | v0.6 only | v0.10 (`ace_model` + learnable radials; `ace1_model` not convertible) | same |
| state of the branch | in use | tests pass, exact, unbenchmarked in this spike | recommended mode wrong for NZ >= 2 at tip; inexact by construction after the fix |

Both routes deliver the many-body component only and need the pair term
handled separately. Where they differ is the exactness contract: the yace
route floors at ~1e-10 in dR/dr by the Hermite-in-r form and cannot be made
exact; the `lammps-export` polynomial route is exact to float64 with a 4.5 MiB
artefact, but is the slower evaluator on the branch's own numbers and needs
the Julia runtime shipped next to LAMMPS. The Hermite mode was meant to
close the speed gap and instead is where both the correctness bug and the
1e-4 floor live; if speed matters, `Nspl` would have to go well beyond 200
(or the spline be done in a way that is not O(h^3) in the radial derivative)
and the file grows linearly with it (17.6 MB of source at 200).

## Artefacts

Host `moriarty:~/ace-potentials-julia-1.2/ACEpotentials.jl/` (branch
`lammps-export-verify`, one local commit 39d853dc; `stash@{0}` holds the
pre-existing Manifest edit of `lammps-export`), directory `verify_cantor/`:
`chain_cantor.jl` (fit, physicality, chain table, exports, in-Julia check),
`reexport_hermite.jl` (patched Hermite exports + check), `probe_rnl.jl`
(radial-dispatch probe), `compile_lib.jl` (JuliaC recipe), `compare_common.py`,
`compare_pylib.py`, `compare_lammps.py`, `in.cantor_ace`, `run_lammps.sh`,
`in.bench_ace`, `in.bench_pace`; `cantor_v010_params.jld2` (fitted `ps`),
`lsq_cantor.jld2` (assembled system, 1.1 GB -- delete when done),
`chain_results.jld2`, `ref_{1..10}.txt` (17-digit references for all chain
stages), `cantor_{poly,hermite50,hermite200,hermite50_asis,poly_withpair}_model.jl`,
`lib/libace_cantor_*.so`, `plugin_build/aceplugin.so`, `lammps/{log,dump}.*`,
`log.{instantiate,tests_julia,juliac_*,chain,reexport,pylib_*,lammps_*}`.
Geometries: the yace spike's `~/si-ace/spike_yace/cantor/cantor_{1..10}.data`.
Local: this file only (no scripts added to the repo).
