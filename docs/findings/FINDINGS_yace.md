# Finding: exact yace export spike (Package 2) -- FAIL, no exact route

**Question.** Can a v0.10 `ace1_model` be written to `.yace` such that ML-PACE
reproduces our forces to 1e-10 eV/A?

**Answer.** No. Every route through libpace evaluates radial functions from a
cubic Hermite table in r (or from a fixed polynomial family we cannot produce),
and the best measured residual is **max |dF| = 5.4e-9 eV/A (3.2e-10 relative)**,
50x the criterion, with no parameter that improves it further. Details, numbers
and dead ends below.

**Recommendation (one line).** Not exact; CPU users deploy via the
`lammps-export` compiled-library route (ACEsuit `lammps-export` branch:
`juliac --trim` model library + `pair_style ace` plugin, MPI+OpenMP), which
evaluates the v0.10 model itself -- its accuracy and throughput must be
verified before it is promised, since its own tests only assert 1e-8 on forces
and it records no benchmark. A fork-pinned yace export remains *possible* as an
explicitly approximate (~1e-9 eV/A) deliverable if ML-PACE throughput is wanted
regardless; that is a different plan with a different contract, not this one.

## Route taken

1. Pinned what upstream parses (source reading + two load tests).
2. Diffed a pyace-written YAML `.yace` against the v0.6 export and against
   what the upstream `ACE.jl.*` reader wants.
3. Measured the radial re-tabulation residual of the v0.10 radials in Julia,
   replicating libpace's Hermite formula, as a function of `nbins`.
4. Measured end to end under `pair_style pace` with the one route that loads
   (wcwitt fork), using the v0.6 comparator model (which already has an
   exporter for that format) at three node counts.

No converter was written for v0.10: steps 3 and 4 show that no radial route
can reach 1e-10, so the many-body ctilde conversion (real -> complex Ylm,
`(4pi)^(rank/2)` normalisation, `ms_combs` folding) would be work on a route
that fails regardless.

## Step 1: what the pinned upstream parses

Tag: **ICAMS `lammps-user-pace` v.2023.11.25.fix2**
(`cmake/Modules/Packages/ML-PACE.cmake`, SHA256 `e0885351...`), the tag LAMMPS
`patch_4Jul2026` fetches by default.

`.yace` loading (`ace-evaluator/ace_c_basis.cpp`, `load_yaml`) branches on
`bonds::[i,j]::radbasename`:

| `radbasename` | evaluator | how radials are evaluated |
|---|---|---|
| `ChebExpCos`, `ChebPow`, `ChebLinear`, `SBessel`/`TEST_SBessel`, `TEST_*` | `ACERadialFunctions` | analytic `gr(k)` sampled every `deltaSplineBins` (0.001 A) into a **cubic Hermite table in r** (`SplineInterpolator::setupSplines`), `fr(n,l) = sum_k crad(n,l,k) gr(k)` |
| `ACE.jl*` (prefix match) | `SHIPsRadialFunctions` (`ships_radial.cpp`, C. Ortner 2020) | **analytic**, no table: `x = ((1+r0)/(1+r))^p`, `fcut = (x-xl)^pl (x-xr)^pr`, `P0 = A0 fcut`, `P1 = (A1 x + B1) P0`, `Pn = (An x + Bn) P(n-1) + Cn P(n-2)`; `fr(n,l) = P(n)` for every l; optional `polypairpot` (same form) + `reppot` core |
| anything else | -- | `invalid_argument("Unknown radial basis function name")` |

Required YAML fields for `ACE.jl*`: top-level `lmax`; per bond `rcut, r0, p,
xl, xr, pl, pr, maxn, recursion_coefficients: {A, B, C}` (each of length
`maxn`). A missing `xl` etc. makes yaml-cpp throw **`bad conversion`** -- which
is exactly the Phase 8 error: the v0.6 exporter writes the *fork's* spline
fields under the `ACE.jl` name, and upstream's `ACE.jl` reader wants polynomial
fields. It was not the `.ace`/`.yace` naming confusion.

Two load tests on the upstream binary (`build-SKX-AMPERE86-mlpace/lmp`,
`pair_style pace`, 8-atom Si cell):

| file | result |
|---|---|
| `pyace_tiny.yace` (pyace `save_yaml`, ChebExpCos) | loads, E = -1.4967751 |
| `acejl_basic_min.yace` (hand-written `ACE.jl.Basic`, maxn 3, one rank-2 term) | loads, **E = 4.044441**, reproduced to all printed digits by an independent numpy evaluation of the formulas above |

The numpy reproduction pins the conventions: `ns` 1-based; harmonics are
`sqrt(4pi)` x orthonormal complex (`ace_spherical_cart.h: Y00 = 1`) with
Condon-Shortley phase; `ms_combs` are summed literally as
`ctilde * Re(prod_t A(mu_t, n_t, l_t, m_t))` with no implicit doubling (pyace
folds the `+-m` partner into the coefficient when it writes, e.g. `[0,0]: c`,
`[1,-1]: -2c` for `ls = [1,1]`).

`pace/kk` (`src/KOKKOS/pair_pace_kokkos.cpp:252`) does
`dynamic_cast<ACERadialFunctions*>` and errors with "Chosen radial basis style
not supported by pair style pace/kk" for anything else -- so **every `ACE.jl*`
route is CPU-only**, upstream or fork. Only the ChebExpCos family reaches the
GPU.

## Step 2: field-by-field diff

Top level:

| key | pyace `.yace` (upstream writer) | v0.6.12 export | upstream `ACE.jl*` reader wants |
|---|---|---|---|
| `elements` | `[Si]` | `[Si]` | same |
| `E0` | `[0]` | `[0.0]` | same |
| `deltaSplineBins` | `0.001` | absent | ignored for `ACE.jl*` (set to 0.001 internally) |
| `embeddings.0` | `{ndensity, FS_parameters, npoti, rho_core_cutoff, drho_core_cutoff}` | same keys | same |
| `bonds` | see below | see below | see below |
| `functions.0[]` | `{mu0, rank, ndensity, num_ms_combs, mus, ns, ls, ms_combs, ctildes}` | same schema (one ctilde per m-tuple, `num_ms_combs: 1`) | same |
| `lmax` | absent (derived) | `2` | **required** |
| `polypairpot`, `reppot` | absent | absent (pair potential goes to a separate `pair_style table` file) | optional |

`bonds.[0,0]`:

| key | pyace | v0.6.12 export | upstream `ACE.jl*` | wcwitt fork `ACE.jl` |
|---|---|---|---|---|
| `radbasename` | `ChebExpCos` | `ACE.jl` | `ACE.jl*` | `ACE.jl*` |
| `rcut` | 6 | 6.000000000000001 | required | required |
| `nradmax, lmax, nradbasemax` | 2, 2, 2 | -- | -- | -- |
| `radparameters, radcoefficients, prehc, lambdahc, dcut, rcut_in, dcut_in, inner_cutoff_type` | present | -- | -- | -- |
| `r0, p, xl, xr, pl, pr, maxn, recursion_coefficients{A,B,C}` | -- | -- | **required** | -- |
| `nradial` | -- | 17 | -- | required (sets nradmax = nradbase) |
| `nbins` | -- | 9999 | -- | required |
| `splinenodalvals`, `splinenodalderivs` | -- | `{0: [...], ...}` x 17, 10000 values each | -- | required; built into a cubic Hermite table in r (`acejl_radial.cpp`, same formula as `SplineInterpolator::setupSplines`) |

Fork: `wcwitt/lammps-user-pace` main (tarball `/tmp/wcwitt.tar.gz`, SHA256
`a89bc7e9...`, no tag), built as `build-SKX-AMPERE86-acejl/lmp`.
**It loads and runs the v0.6 yace under `pair_style pace`** (E = 2054.70 on
the 8-atom cell). Phase 8's `map::at` does not reproduce with `pair_style
pace`; Phase 8 used `pace/kk`, which cannot accept this basis at all.

## Step 3: why no route is exact for v0.10 radials

The v0.10 `ace1_model` radial is `Rnl(r) = spline(x) * env(r, x)` with
`x = NormalizedTransform(GeneralizedAgnesiTransform)(r)`, a cubic B-spline on
100 uniform nodes in x in [-1, 1] with `Wnlq` folded in (`splinify`), and
`PolyEnvelope2sX` (`src/models/Rnl_splines.jl`, `ace1_compat.jl:283`).

- **(a) upstream `ACE.jl.Basic`**: the only *analytic* path, but a fixed
  family: transform `((1+r0)/(1+r))^p`, cutoff `(x-xl)^pl (x-xr)^pr`, 3-term
  recurrence, `Rnl = Pn` for all l. v0.10 has no such transform (only
  `GeneralizedAgnesiTransform`/`NormalizedTransform`), and a Wnlq-folded
  spline is not a polynomial in that x. Dead for `ace1_model`. (It *would* be
  exact by construction for a new, deliberately PACE-shaped model family:
  add a `PolyTransform`, un-splined `OrthPolyBasis1D3T`, `Wnlq = I`,
  `polypairpot` for the pair part. That is a different model, CPU-only in
  LAMMPS, and out of this spike's scope -- recorded as an option.)
- **(b) fork `ACE.jl` + `splinenodalvals`**: a cubic Hermite interpolant in r
  from Float64 nodal values and derivatives. Not our spline space (ours is in
  x, C2 with a jump in R''' at each of the 100 x-knots). Two error terms:
  interpolation `O(h^3 R'''')` (`O(h^2 [R'''])` in an interval straddling a
  knot) and rounding `O(eps |R| / h)` from differencing O(1) nodal values --
  the second grows as `nbins` grows. Neither is removable.
- **(c) `ChebExpCos`/`ChebPow` + `crad`**: a least-squares fit of our
  functions onto PACE's family *and then* the same 0.001 A Hermite table.
  Strictly dominated by (b); not attempted beyond this observation. It is the
  only family `pace/kk` accepts.

### Radial-level residual of (b) for the v0.10 model (Julia only)

`acejax/spike_yace/radial_tabulation_error.jl` (throwaway; copy at
`moriarty:~/si-ace/spike_yace/`): `ace1_model(Si, order 3, totaldegree 10)`,
Hermite formula copied from `acejl_radial.cpp`, evaluated at the 2866 edge
distances of the exporter's rattled 2x2x2 cell and at 20000 random
r in [1.8, 6]. Max abs error over the 37 many-body radial functions
(max |R| = 1.5, max |dR/dr| = 7.7):

| `nbins` | h [A] | edge max\|dR\| | edge max\|dR'\| | rand max\|dR\| | rand max\|dR'\| |
|---|---|---|---|---|---|
| 5999 | 1.0e-3 | 2.9e-10 | 1.0e-6 | 9.0e-10 | 2.2e-6 |
| 9999 | 6.0e-4 | 4.5e-11 | 3.1e-7 | 2.0e-10 | 1.1e-6 |
| 30000 | 2.0e-4 | 1.7e-13 | 1.3e-8 | 3.9e-12 | 1.1e-7 |
| 100000 | 6.0e-5 | 2.9e-14 | **4.1e-10** | 6.2e-14 | 7.4e-9 |
| 300000 | 2.0e-5 | 1.4e-14 | 1.4e-9 | 2.1e-14 | 1.5e-9 |
| 1000000 | 6.0e-6 | 1.3e-14 | 3.9e-9 | 1.8e-14 | 4.0e-9 |

The pair basis (10 functions) behaves the same (best 2.0e-10 at 1e5).
Values converge; **derivatives floor at a few 1e-10 per function and then
rise**. Diagnostics: at 1e6 bins the error is identical with BigFloat
interpolation arithmetic on the same Float64 nodal data (so it is the data's
rounding amplified by 1/h, not the arithmetic), and at 1e5 bins the largest
errors sit in intervals straddling an x-knot (7.4e-9 straddling vs 5.3e-10
not). Forces sum ~45 neighbours x 37 functions of these, so 1e-10 on forces
is out of reach on this route; production models (degree 16+) have larger
R'''' and are worse.

## Step 4: end-to-end under `pair_style pace` (fork binary)

Route (b) is the only one that loads, and ACEpotentials v0.6.12 already
writes it, so the force-level residual was measured with the v0.6 comparator
model (`acejax/bench/v06/fit_v06.jl`: `acemodel(Si, order 3, totaldegree 10,
rcut 6, Eref 0)`, BLR on `Si_tiny`; refit on the host because the fit is not
bit-reproducible, references taken from the same fit). Pipeline: analytic
ACE1 radials -> `ACE1.Splines` cubic spline in r with `nnodes` nodes ->
nodal values/derivatives -> libpace Hermite (which reproduces that r-spline
exactly, so the residual is r-spline vs analytic). Many-body component only
(the v0.6 pair potential is a separate `pair_style table`, itself
interpolated). 64-atom rattled cell `si_v06_check.xyz`, max |F| = 16.8 eV/A
(wild underdetermined coefficients; also quoted relative):

| `nnodes` (=`nbins`+1) | yace size | \|dE\| [eV] (rel) | max \|dF\| [eV/A] | rel | mean \|dF\| |
|---|---|---|---|---|---|
| 10000 (v0.6 default) | 10 MB | 9.1e-11 (5.5e-15) | 1.8e-8 | 1.1e-9 | 5.9e-9 |
| 100000 | 100 MB | 1.1e-11 (6.6e-16) | **5.4e-9** | **3.2e-10** | 1.5e-9 |
| 300000 | 300 MB | 1.1e-11 (6.6e-16) | 1.2e-8 | 7.0e-10 | 3.8e-9 |

Energies are reproduced to rounding (1e-11 on 1.6e4 eV is float64
resolution). **Forces bottom out at 5.4e-9 absolute / 3.2e-10 relative** and
degrade beyond 1e5 nodes, exactly the pattern of the Julia analysis. For a
physical model with O(1) eV/A forces this is ~3e-10 absolute at best -- still
above 1e-10, and not "exact by construction" at any setting.

**Pass criterion (max |dF| <= 1e-10, |dE| <= 1e-10): not met.** No field was
invented and the criterion was not loosened.

## Dead ends, briefly

- `/tmp/si_pace110.yace` is the `.ace` text format misnamed (no pyace YAML
  existed on the host); a real one was generated with the cached pyace 0.2.8
  wheel (`PYTHONPATH=/tmp/uvcache-pyace/archive-v0/TyBQggXjOY9QPCDM`,
  `~/micromamba/envs/pyace/bin/python`).
- First numpy reproduction of the `ACE.jl.Basic` energy was off by the
  `sqrt(4pi)` harmonic normalisation (`Y00 = 1` in PACE); fixed, then exact.
- The v0.6 Julia depot had lost `Polynomials`; `Pkg.instantiate()` restored it.
- Phase 8's `bad conversion` and `map::at` are both explained above; neither
  was a naming problem.

## Related route: the `lammps-export` branch

`~/ace-potentials-julia-1.2/ACEpotentials.jl` on branch `lammps-export`
(= `acesuit/lammps-export`, last commit 2026-01-06) is **not a yace writer**.
It exports a fitted model to trim-compatible Julia code, compiles it with
`juliac --trim` (JuliaC.jl, `cpu_target='generic'`) into a shared library
with a C API (`export/src/write_c_interface.jl`), and ships a LAMMPS plugin
`pair_style ace` (`export/lammps/plugin/src/pair_ace.cpp`, MPI + OpenMP) plus
an ASE `ACELibraryCalculator`; a `build_deployment(model, name)` script
bundles the runtime. Radials are exported as `:hermite_spline`
("machine precision") or `:polynomial` ("exact"); the ETACE path splinifies
with `Nspl=50` first. Its tests assert energies to `atol=1e-10` but forces
only to `atol=1e-8, rtol=1e-6`, and there are MPI energy-drift tests. No
throughput figure is recorded anywhere in the branch. It is the natural CPU
deployment route because it runs the v0.10 model itself, but both its
exactness (the `:polynomial` mode should be checked at 1e-10) and its speed
relative to `pair_style pace` are unmeasured.

## Artefacts

- Host (`moriarty`): `~/si-ace/spike_yace/` -- `acejl_basic_min.yace`,
  `pyace_tiny.yace`, `make_pyace_yaml.py`, `check_acejl_basic.py`,
  `si8.data`, `si64_v06.data`, `in.load`, `in.load64`, `export_n.jl`
  (v0.6.12 exporter with `nnodes` parameterised), `v06_all.jl`,
  `v06_ref_forces.txt`, `si_v06_n{10000,100000}.yace`, `dump.*`, `log.*`,
  `compare.py`, `radial_tabulation_error.jl`; pyace text files preserved in
  `~/si-ace/pace/`.
- Local (uncommitted, throwaway): `acejax/spike_yace/radial_tabulation_error.jl`.
