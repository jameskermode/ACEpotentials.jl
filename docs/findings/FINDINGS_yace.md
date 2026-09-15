# Finding: exact yace export spike (Package 2) -- FAIL, no exact route

**Question.** Can a v0.10 `ace1_model` be written to `.yace` such that ML-PACE
reproduces our forces to 1e-10 eV/A?

**Answer.** No. Every route through libpace evaluates radial functions from a
cubic Hermite table in r (or from a fixed polynomial family we cannot produce),
and the best measured residual is **max |dF| = 5.4e-9 eV/A (3.2e-10 relative)**,
54x the criterion, with no parameter that improves it further. Details, numbers
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
- **(c) `ChebExpCos`/`ChebPow` + `crad`**: PACE evaluates
  `fr(n,l) = sum_k crad(n,l,k) g_k(r)`, so the best this route can do is the
  least-squares projection of our `R_nl` onto `{g_k}`, *and then* the same
  0.001 A Hermite table. Shown below to be worse than (b) by 6-8 orders of
  magnitude. It is the only family `pace/kk` accepts.

### Route (c): PACE's transform and cutoff against ours

`ChebExpCos` (`ML-PACE/ace-evaluator/ace_radial.cpp:294-326` at
v.2023.11.25.fix2):

```
x     = 1 - 2 (exp(-lam r/rc) - exp(-lam)) / (1 - exp(-lam))        # :298-300
g_1   = T_0(x) = 1;   g_k = 1/2 (1 - T_{k-1}(x)),  k = 2..nradbase   # :303-308
env   = 1/2 (1 + cos(pi r/rc))                                      # :310
taper = 1/2 (1 + cos(pi (r - (rc-dcut))/dcut))  on [rc-dcut, rc]    # :317-323
```

(`ChebPow`, `:328-362` (doc comment + body): `x = 2 (1 - (1 - r/rc)^lam) - 1`,
`g_k = 1/2 (1 - T_k(x))`, no separate cutoff -- a polynomial in r for integer
`lam`; the same argument applies.) v0.10 `ace1_model` (parameters read
from the model, `chebexpcos_lsq.jl`):

```
s    = (r - rin)/(r0 - rin),  rin = 0, r0 = 2.40                    # GeneralizedAgnesiTransform p=2 q=4 a=0.769
y    = 1 / (1 + a s^q / (1 + s^(q-p)))
x    = clamp(-1 + 2 (y - yin)/(ycut - yin), -1, 1),  yin = 1, ycut = 0.1944   # NormalizedTransform
R_nl = B-spline_nl(x) * (x - x1)^p1 (x2 - x)^p2,  x1 = -1, x2 = 1, p1 = p2 = 2  # PolyEnvelope2sX
```

No choice of `(lam, rc, dcut, crad)` makes these equal, and not because of
parameter values: every finite combination of PACE's `g_k` is a
real-analytic function of r on `(0, rc - dcut)` (exponential, Chebyshev
polynomial, cosine), whereas `R_nl` is a cubic B-spline in `x(r)` with 100
knots -- C2 with a nonzero jump in `R'''` at each knot (the jumps are the
third differences of the `Wnlq`-folded coefficients and do not vanish). An
analytic function cannot have a discontinuous third derivative, so equality
on any interval is impossible; only approximation remains. (The un-splined
v0.10 basis would also not match: its argument is a rational function of
`(r/r0)^2`, PACE's an exponential in r, and PACE's cos envelope is not a
polynomial in either.) No special case rescues it -- `lam -> 0` makes
`x -> 1 - 2 r/rc`, still analytic.

### Route (c): least-squares residual (measured)

`acejax/spike_yace/chebexpcos_lsq.jl`: the same degree-10 model's 37 `R_nl`
projected onto `g_1..g_K` (formulas above, `dcut = 0.01`) by least squares on
20000 points, `K` = ours (10 distinct n), 2x, 4x, `lam` scanned over
{1, 2, 3.5, 5.25, 8}; best `lam` per row shown. "int." = max over
`[2.0, 5.9]`, away from the fit edges and PACE's taper.

| window | K | lam | max\|dR\| | max\|dR'\| | at r (fn) | int. max\|dR\| | int. max\|dR'\| |
|---|---|---|---|---|---|---|---|
| [0, 6] (the brief's `[rin, rcut]`) | 10 | 2.0 | 1.2 | 52 | 0.001 (6) | 1.2 | 8.8 |
| | 20 | 2.0 | 0.94 | 148 | 0.001 (10) | 0.51 | 4.6 |
| | 40 | 1.0 | 6.0e-4 | 0.17 | 0.001 (10) | 4.7e-4 | 1.6e-2 |
| [1.8, 6] (edges start at 2.14) | 10 | 2.0 | 1.1 | 54 | 1.8 (9) | 0.40 | 4.6 |
| | 20 | 3.5 | 9.8e-4 | 0.12 | 1.8 (10) | 9.8e-4 | 2.7e-2 |
| | 40 | 1.0 | 3.6e-4 | 9.7e-2 | 5.997 (10) | 2.6e-4 | 1.0e-2 |

pyace's default `lam = 5.25` is markedly worse (K = 40, [1.8, 6]: 8.7e-3 /
3.3) because `x(r)` then compresses `r > 3` into `x > 0.99`. Even at 4x our
radial count and the best `lam`, the residual is **1e-4 in value and 1e-2 in
derivative** -- seven orders above the tabulation floor of route (b), before
PACE's own 0.001 A Hermite table (which alone costs 1e-6 on derivatives at
that spacing, first row of the table below). Route (c) is worse than (b) by
measurement, and it is a fit, not an export.

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
not). The rand-set column is not monotone either (7.4e-9 -> 1.5e-9 -> 4.0e-9
from 1e5 to 1e6 bins): the two error terms cross over, and where the minimum
falls depends on which distances are sampled. Forces sum ~45 neighbours x 37
functions of these, so 1e-10 on forces is out of reach on this route. A
higher-degree model is worse, as measured with the same script at
`totaldegree = 16` (91 radial functions, max |dR/dr| = 11.5): edge
max |dR'| = 3.1e-6 at 9999 bins, best 2.6e-9 at 3e5 bins, 9.0e-9 at 1e6 --
about 10x the degree-10 number at 9999 bins, ~1.9x at 3e5 and ~2.3x at 1e6:
up to 10x at coarse tables, ~2x near the floor.

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
- Local (uncommitted, throwaway): `acejax/spike_yace/radial_tabulation_error.jl`
  (`ACE_TOTALDEGREE` selects the model) and `acejax/spike_yace/chebexpcos_lsq.jl`;
  both also copied to the host directory.

## Part 2 (2026-09-15): a physical five-element model, and exact nodal derivatives

**Questions.** (i) On a *physical* multi-element model, at what force
tolerance does the fork-pinned `.yace` route pass -- 1e-10, 1e-9 or 1e-8 eV/A?
(ii) Is the tabulation floor removable by supplying exact nodal derivatives
instead of letting libpace reconstruct them?

**Answers.** (i) Max |dF| = **1.9e-10 eV/A at 1e4 nodes** and **9.8e-11 at
1e5 nodes** (as-is), on forces of 1-4 eV/A; energies to 8.5e-14 eV/atom.
**1e-9 and 1e-8 absolute pass at both node counts with >= 5x margin; 1e-9
relative passes with >= 8x margin; 1e-10 absolute fails at 1e4 and passes
at 1e5 only by 2% (as-is) / 30% (exact derivatives)** -- not a margin to
promise on. (ii) **libpace does not reconstruct derivatives**: the fork reads
`splinenodalderivs` from the file and uses them verbatim as the Hermite end
slopes; Part 1's attribution was wrong about the mechanism (the exporter, not
libpace, was supplying spline-derived slopes). Writing exact analytic nodal
values and derivatives instead needs no C++ change, is a 20-line exporter
change, and **does not remove the floor**: it leaves 1e4 nodes unchanged and
improves 1e5 nodes by 30% (9.8e-11 -> 7.0e-11). The floor is intrinsic to the
Hermite form itself (`3(f1-f0)` in the cubic coefficients, divided by h on
evaluation) acting on Float64 nodal *values*, and shows up identically in a
Julia-only replica for both v0.6 and v0.10 radials.

### Fork source: how `splinenodalvals`/`splinenodalderivs` are consumed

`build-SKX-AMPERE86-acejl/lammps-user-pace-main/ML-PACE/ace-evaluator/acejl_radial.cpp`
(wcwitt fork, `ACEjlRadialFunctions::read_yaml`):

| line | what |
|---|---|
| 53 | `splinenodalvals = bond_yaml["splinenodalvals"].as<map<int,vector<DOUBLE_TYPE>>>()` |
| 54 | `splinenodalderivs = bond_yaml["splinenodalderivs"].as<...>()` -- **read from the file, required** |
| 74-77 | `f0 = vals[n]; f1 = vals[n+1]; f0d1 = derivs[n]*d; f1d1 = derivs[n+1]*d` |
| 79-82 | `c0 = f0; c1 = f0d1; c2 = 3(f1-f0) - f1d1 - 2 f0d1; c3 = -2(f1-f0) + f1d1 + f0d1` |
| 149-171 | `evaluate`: `spline.calcSplines(r)` (upstream `SplineInterpolator`, derivative `= (c1 + 2 c2 s + 3 c3 s^2) * rscalelookup`), then `fr(n,l) = values(n)` for every `l` |

No finite differences, no spline solve: whatever slopes are in the file are the
interpolant's slopes. What the v0.6.12 exporter puts there (`export.jl`, copied
in `export_n.jl`): `ACE1.Splines.RadialSplines(J; nnodes)` builds an
**Interpolations.jl C2 cubic B-spline in r with `Flat(OnGrid())` boundary
conditions** through the analytic radials at the nodes
(`ACE1/src/polynomials/splines.jl:104-131`), and the exporter writes that
spline's values `spl.(rg)` and its gradient `Interpolations.gradient(spl, r)` at
the nodes -- so libpace reproduces the Interpolations spline exactly, and the
residual is (C2 spline in r) vs (analytic). The derivative floor Part 1 saw is
therefore not "libpace differencing the values"; it is the Hermite
coefficients `c2, c3` differencing O(1) values (lines 81-82) and `calcSplines`
multiplying by `1/h`, which is the same whichever slopes are supplied.

**Patch.** None to the fork. Exporter-side (throwaway `export_n.jl`,
`exact_derivs = true`): replace the spline nodal data by the analytic ACE1
radials, using the same `(z, z0)` indexing `RadialSplines` uses:

```julia
if exact_derivs
    J = V3.pibasis.basis1p.J
    for iz1 in 1:size(nodalvals,2), iz2 in 1:size(nodalvals,3)
        z, z0 = zlist[iz1], zlist[iz2]; rr = ranges[1,iz1,iz2]
        for (ir, r) in enumerate(rr)
            Jv[:, ir] = ACE1.evaluate(J, r, z, z0)
            Jd[:, ir] = ACE1.evaluate_d(J, r, z, z0)
        end
        for i in 1:NB; nodalvals[i,iz1,iz2] = Jv[i,:]; nodalderivs[i,iz1,iz2] = Jd[i,:]; end
    end
end
```

The binary is unchanged, so the "as-is" and "exact" rows below are the same
`lmp` (`build-SKX-AMPERE86-acejl/lmp`, `pair_style pace`, CPU, 1 rank, 1 thread).

### The model and its physicality

ACEpotentials **v0.6.12** (Julia 1.11.7, `acejax/bench/v06` environment),
`acemodel(elements = [Cr, Mn, Fe, Co, Ni], order = 3, totaldegree = 6,
rcut = 6.25, r0 = 2.54, Eref = <per-element least squares of the training
energies>)`; `r0` must be given because JuLIP has no bond length for Mn.
Basis 4115 (pair + many-body); many-body radial `nradial = 13`, exported
functions 1065 per species (60 rank 1, 310 rank 2, 695 rank 3), **`lmax = 1`**
(ACE1x's degree weighting at totaldegree 6). Fit: `acefit!(...; solver = BLR(),
repulsion_restraint = true)`, keys `mace_energy/mace_force/mace_virial`,
default weights, on configs 1-250 of `cantor1k_b_mh1.xyz` (9920 atoms, 904 s);
held-out set = configs 991-1000 (32-48 atoms each, all five species).

| set | E RMSE [meV/atom] | F RMSE [eV/A] | V RMSE [meV/atom] |
|---|---|---|---|
| train (250) | 0.23 | 0.101 | 19.4 |
| held-out (10) | 6.0 | 0.124 | 55.8 |

Held-out max |F| per config: full potential 1.3-4.2 eV/A (MACE reference
1.4-4.3); many-body part alone 1.1-3.3. Dimer curves `E(r) - 2 E0` [eV] of the
full potential rise monotonically as r shrinks: Cr-Cr 0.31 (3.0 A), 1.47
(1.8), 2.74 (1.4), 5.90 (1.0); Fe-Ni 0.44, 1.74, 3.04, 6.17; Mn-Co 0.56, 1.99,
3.31, 6.43; Ni-Ni 0.42, 1.70, 3.00, 6.14. Caveat that matters for the
contract: the many-body component `V3` is *exactly zero* on every dimer
(ACE1x's default `delete2b`), so the repulsive core lives entirely in the
`PolyPairPot`, which the v0.6 exporter writes to a separate `pair_style table`
file (0.001 A grid) and which is **not** part of the yace comparison below.
The end-to-end numbers are for the many-body component, as in Part 1.

### Julia-side floor: v0.10 radials of the same spec (five species)

`acejax/spike_yace/radial_tabulation_error_cantor.jl` (local, Julia 1.12.7,
ACEpotentials 0.10.2): `ace1_model(CrMnFeCoNi, order 3, totaldegree 6,
rcut 6.25, r0 2.54)`, random weights, fork Hermite formula with **exact**
nodal values and derivatives, max over all 25 species pairs, on the 2638 edge
distances of held-out config 10 (r in [2.24, 6.25]) and 20000 random
r in [1.8, 6.25]. `rbasis`: 74 functions, max |R| = 1.27, max |dR/dr| = 4.7.

| `nbins` | h [A] | edge max\|dR\| | edge max\|dR'\| | rand max\|dR\| | rand max\|dR'\| |
|---|---|---|---|---|---|
| 9999 | 6.3e-4 | 1.6e-11 | 8.3e-8 | 1.8e-11 | 9.7e-8 |
| 30000 | 2.1e-4 | 4.0e-13 | 3.8e-9 | 6.0e-13 | 1.1e-8 |
| 100000 | 6.3e-5 | 7.1e-15 | **2.2e-10** | 1.7e-14 | 9.7e-10 |
| 300000 | 2.1e-5 | 6.7e-15 | 5.5e-10 | 9.8e-15 | 6.5e-10 |
| 1000000 | 6.3e-6 | 7.1e-15 | 1.9e-9 | 9.7e-15 | 2.0e-9 |

Pair basis (30 functions): best 9.2e-11 at 1e5, 9.3e-10 at 1e6. Same shape
and level as Part 1's Si table: values converge, derivatives floor at 1e5 bins
and rise.

The same replica applied to the **fitted v0.6 model's analytic ACE1 radials**
(`v06_radial_floor.jl` on the host; 13 functions, 25 pairs, 32422 held-out edge
distances in [2.09, 6.25], max |R| = 2.3, max |dR/dr| = 12.7), with the nodal
data either as the exporter writes it (`asis`: Interpolations C2 spline values
and gradient) or exact:

| `nnodes` | h [A] | nodal data | edge max\|dR\| | edge max\|dR'\| | rand max\|dR\| | rand max\|dR'\| |
|---|---|---|---|---|---|---|
| 10000 | 6.3e-4 | asis | 5.8e-12 | 2.5e-8 | 6.4e-12 | 3.1e-8 |
| 10000 | 6.3e-4 | exact | 5.8e-12 | 2.5e-8 | 6.4e-12 | 3.1e-8 |
| 30000 | 2.1e-4 | asis | 6.5e-14 | 9.9e-10 | 7.9e-14 | 1.2e-9 |
| 30000 | 2.1e-4 | exact | 6.4e-14 | 9.9e-10 | 7.9e-14 | 1.2e-9 |
| 100000 | 6.3e-5 | asis | 5.5e-14 | 1.0e-9 | 6.2e-14 | 1.0e-9 |
| 100000 | 6.3e-5 | exact | 4.4e-14 | **9.1e-10** | 4.6e-14 | 9.0e-10 |
| 300000 | 2.1e-5 | asis | 4.0e-14 | 2.1e-9 | 4.2e-14 | 2.0e-9 |
| 300000 | 2.1e-5 | exact | 3.8e-14 | 2.6e-9 | 3.9e-14 | 2.2e-9 |

At 1e4 nodes the C2 spline and the exact-slope Hermite have the *same* O(h^3)
derivative error (they agree to three digits); from 3e4 nodes both sit on a
~1e-9 floor that exact slopes do not move, and both rise beyond. So the v0.10
and v0.6 radials of this model sit on the same floor, and it is not a
derivative-supply problem.

### End to end: `pair_style pace` (fork) vs the v0.6 Julia calculator

Ten held-out configs, rotated to LAMMPS triclinic form in Julia and written
with 17 significant digits (`cantor_all.jl`; the Julia reference is computed
on the rotated cell, so both codes see identical doubles); one yace parse per
run, configs cycled with `delete_atoms` + `change_box` + `read_data add append`
(validated against a fresh parse: 1e-14, neighbour-order noise). Many-body
component only; `max|F|` is the largest force *component* in the config.

`nnodes = 10000` (yace 185 MB, 3.9 GB RSS, 48 s per LAMMPS start):

| cfg | N | max\|F\| | as-is max\|dF\| | rel | mean\|dF\| | exact max\|dF\| | rel | mean\|dF\| | \|dE\|/atom |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 3.22 | 1.89e-10 | 5.9e-11 | 4.0e-11 | 1.89e-10 | 5.9e-11 | 4.0e-11 | 8.5e-14 |
| 2 | 32 | 0.98 | 7.4e-11 | 7.6e-11 | 2.6e-11 | 7.5e-11 | 7.6e-11 | 2.6e-11 | 1.4e-14 |
| 3 | 32 | 1.21 | 1.52e-10 | 1.3e-10 | 2.8e-11 | 1.49e-10 | 1.2e-10 | 2.8e-11 | 2.8e-14 |
| 4 | 32 | 1.07 | 8.1e-11 | 7.6e-11 | 2.3e-11 | 8.0e-11 | 7.5e-11 | 2.2e-11 | 7.1e-15 |
| 5 | 48 | 2.14 | 1.24e-10 | 5.8e-11 | 3.4e-11 | 1.24e-10 | 5.8e-11 | 3.4e-11 | 4.7e-15 |
| 6 | 48 | 1.43 | 1.14e-10 | 8.0e-11 | 2.6e-11 | 1.14e-10 | 8.0e-11 | 2.6e-11 | 1.4e-14 |
| 7 | 32 | 2.51 | 1.21e-10 | 4.8e-11 | 3.6e-11 | 1.19e-10 | 4.8e-11 | 3.6e-11 | 3.6e-15 |
| 8 | 48 | 2.38 | 1.68e-10 | 7.0e-11 | 3.9e-11 | 1.71e-10 | 7.2e-11 | 3.9e-11 | 5.2e-14 |
| 9 | 32 | 1.79 | 9.9e-11 | 5.5e-11 | 2.6e-11 | 9.9e-11 | 5.6e-11 | 2.6e-11 | 1.1e-14 |
| 10 | 32 | 1.79 | 1.10e-10 | 6.1e-11 | 3.1e-11 | 1.10e-10 | 6.2e-11 | 3.1e-11 | 1.1e-14 |
| **all** | | | **1.89e-10** | **1.3e-10** | | **1.89e-10** | **1.2e-10** | | **8.5e-14** |

`nnodes = 100000` (yace 1.9 GB, **37 GB RSS, 465 s** per LAMMPS start):

| cfg | N | max\|F\| | as-is max\|dF\| | rel | mean\|dF\| | exact max\|dF\| | rel | mean\|dF\| | \|dE\|/atom |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 3.22 | 7.7e-11 | 2.4e-11 | 1.8e-11 | 4.3e-11 | 1.3e-11 | 1.2e-11 | 8.5e-14 |
| 2 | 32 | 0.98 | 6.7e-11 | 6.9e-11 | 1.8e-11 | 5.4e-11 | 5.4e-11 | 1.4e-11 | 1.8e-14 |
| 3 | 32 | 1.21 | 6.4e-11 | 5.3e-11 | 1.9e-11 | 5.3e-11 | 4.4e-11 | 1.3e-11 | 3.6e-15 |
| 4 | 32 | 1.07 | 8.1e-11 | 7.6e-11 | 2.0e-11 | 4.1e-11 | 3.8e-11 | 1.2e-11 | 7.1e-15 |
| 5 | 48 | 2.14 | 5.6e-11 | 2.6e-11 | 1.7e-11 | 3.7e-11 | 1.7e-11 | 1.1e-11 | 4.7e-15 |
| 6 | 48 | 1.43 | 7.1e-11 | 5.0e-11 | 2.0e-11 | 4.2e-11 | 2.9e-11 | 1.3e-11 | 4.7e-15 |
| 7 | 32 | 2.51 | 6.8e-11 | 2.7e-11 | 1.7e-11 | 3.7e-11 | 1.5e-11 | 1.2e-11 | 3.6e-15 |
| 8 | 48 | 2.38 | 9.8e-11 | 4.1e-11 | 1.8e-11 | 7.0e-11 | 2.9e-11 | 1.4e-11 | 5.2e-14 |
| 9 | 32 | 1.79 | 4.9e-11 | 2.8e-11 | 1.6e-11 | 4.2e-11 | 2.3e-11 | 1.1e-11 | 1.8e-14 |
| 10 | 32 | 1.79 | 6.1e-11 | 3.4e-11 | 2.1e-11 | 5.2e-11 | 2.9e-11 | 1.4e-11 | 1.1e-14 |
| **all** | | | **9.8e-11** | **7.6e-11** | | **7.0e-11** | **5.4e-11** | | **8.5e-14** |

Energies agree to 8.5e-14 eV/atom in every case (float64 resolution on
-650 to -1000 eV totals). The 1e5 rows reach the same ~1e-10 level the
Julia replica predicts once the 13 radials x ~40 neighbours are weighted by
the fitted (small, smooth) coefficients; Part 1's 5.4e-9 on the Si model was
the same floor under 10x larger forces and wild coefficients.

### Verdicts (max |dF| over the 10 held-out configs, many-body component)

| tolerance | 1e4 as-is | 1e4 exact | 1e5 as-is | 1e5 exact |
|---|---|---|---|---|
| **1e-10 eV/A absolute** | **FAIL** (1.89e-10) | **FAIL** (1.89e-10) | pass by 2% (9.8e-11) | pass by 30% (7.0e-11) |
| **1e-9 eV/A absolute** | PASS (5x margin) | PASS (5x) | PASS (10x) | PASS (14x) |
| **1e-8 eV/A absolute** | PASS (53x) | PASS (53x) | PASS (100x) | PASS (140x) |
| **1e-9 relative** (to max \|F\|) | PASS (8x) | PASS (8x) | PASS (13x) | PASS (19x) |

Nothing was loosened: same fork binary, same exporter conventions, same
comparison as Part 1; the only new fields are the exact nodal data written
into the two keys the fork already requires.

### Updated recommendation

The Part 1 recommendation stands -- **not exact; CPU users deploy via
`lammps-export`** -- but the fork-pinned yace route is now characterised
rather than dismissed: on a physical model it delivers **<= 2e-10 eV/A
absolute and ~1e-10 relative at the default 1e4 nodes** and cannot be made
exact by any nodal-data choice (the Hermite-in-r form itself floors at ~1e-9
in dR/dr). A fork-pinned exporter's contract would therefore be: forces to
**1e-9 eV/A absolute / 1e-9 relative** (not 1e-10), energies to float64;
`nnodes = 1e4` default (1e5 buys ~2x at 10x the file, 37 GB of LAMMPS RSS and
an 8-minute load per start -- not worth it); analytic nodal derivatives
written (free, harmless, ~30% at 1e5); many-body component only, with the
pair potential's accuracy a *separate* question (today a 0.001 A
`pair_style table`; folding it into the yace as rank-1 functions on the same
Hermite radials is possible in the fork's format but was not tested); CPU
`pair_style pace` only (the fork's `ACE.jl` radials are rejected by
`pace/kk`); pinned to the wcwitt fork at the tested SHA. That is an
explicitly approximate deliverable with a different contract from Package 2's,
as Part 1 said.

### Dead ends and notes

- Julia 1.12 environments on the host (`pyjuliapkg`) do not load
  ACEpotentials v0.10 (missing `MbedTLS_jll`); the v0.10 radial check was run
  locally, as in Part 1.
- `ExtXYZ.load(...)[1]` returns an `AtomView` that `PairList` rejects; use
  `ExtXYZ.Atoms(ExtXYZ.read_frame(f))`.
- `JuLIP.read_extxyz(file, k)` does not select a frame; read the whole file
  and index.
- The 1e5-node five-species yace is 1.9 GB on disk and parses to 37 GB in
  yaml-cpp; the host had 45 GB free, so runs were serialised.

### Artefacts (Part 2)

- Host `~/si-ace/spike_yace/`: `cantor_all.jl` (fit, physicality, data/ref
  files, four yace exports), `export_n.jl` (with `exact_derivs`),
  `v06_radial_floor.jl`, `in.cantor`, `in.cantor_one`, `run_pace.sh`,
  `compare_cantor.py`, `cantor_v06.json` (the fitted model),
  `cantor/` (`cantor_{1..10}.{data,box}`, `cantor_ref_*.txt`, the four
  `.yace`, the pair `.table`), `dump.n1*`, `log.n1*`, `log.cantor_all`,
  `log.v06_radial_floor`.
- Local (untracked) `acejax/spike_yace/`: the same scripts plus
  `radial_tabulation_error_cantor.jl`, `cantor_held.xyz`.
