# Findings: distributed dense least-squares for ACEfit (Stage 2 phase 18)

One-day spike, 2026-09-12, on `moriarty` (RHEL 9.8, 32 cores, 62 GB, RTX A4500
20 GB). Question as originally posed: *is ScaLAPACK.jl usable for distributed
dense least-squares in ACEpotentials' fitting path?* Scope was widened mid-day
to a survey of every viable route (Elemental.jl, TSQR over MPI.jl, distributed
LSQR, Dagger.jl, JAX/lineax on GPU) with the same ill-conditioned tests and the
same agreement check against single-node `qr(A) \ y`.

Scripts and raw logs: `docs/findings/scalapack_spike/`.

## TL;DR

- **ScaLAPACK.jl is dead and cannot be revived cheaply.** Unregistered, last
  code commit 2015-10-01, Julia-0.4 syntax (`immutable`, `typealias`, `&x`
  ccall args, `Base.LinAlg`, `Array(T,n)`), hard-coded macOS `.dylib` path. It
  does not parse on Julia 1.11. Evidence in section 1.
- **ScaLAPACK the library works from Julia today** via `SCALAPACK32_jll`
  v2.2.302 + `MPI.jl` 0.20.27 + ~35 lines of hand-written `ccall` wrappers, with
  one non-obvious gotcha (it needs an LP64 BLAS forwarded through
  libblastrampoline; Julia's default OpenBLAS is ILP64-only). `pdgels` on a
  1-D process grid consumes ACEfit's contiguous row-block layout *verbatim*, no
  redistribution — but that layout is 10x slower than a 2-D block-cyclic one.
- **Elemental.jl v0.6.1 installs, loads and solves correctly** on Julia 1.11
  (agreement with LAPACK 2e-13 .. 6e-13), but `Array(::DistMatrix)` is broken
  (undefined `copyto!`), its only fill path from a row block is per-element
  `queueUpdate` (14 s to fill a 1.9 GB matrix vs 12 s to solve it), and the
  underlying C++ library has been frozen since 2017.
- **Hand-written TSQR over MPI.jl (26 lines) is the best direct method here**:
  agreement with LAPACK 4e-13 .. 2e-12 at cond up to 1e21, one pass over the
  row blocks, no redistribution, 8.1 s on 8 ranks for 120 000 x 2 000 vs 105.6 s
  for ScaLAPACK on the same 1-D layout. Its memory floor is the n x n `R`
  factor — 2 GB at degree 12 embedded, **162 GB at degree 20** — see section 7.
- **The conditioning claim splits into two cases, and the real ACE matrix is
  the hard one.** Pure column-scaling ill-conditioning (cond 1e21, exactly
  representable) is benign: every direct method *and* diagonally-preconditioned
  LSQR solve it to 1e-13. Ill-conditioning spread across a dense mix saturates
  at cond ~1e16 in Float64 (it cannot be constructed higher) and there LSQR
  fails with or without a diagonal preconditioner while all direct methods
  agree to cond·eps. A real ACE design matrix (Si_tiny, degree 10-14) has cond
  1e11-7e11 after the prior and **still 3e9-7e9 after column equilibration** —
  it is of the mixed type.
- **LSQR's `damp` IS `QR`'s `lambda`** (measured, section 6): with λ ≥ 1e-3
  the two solutions agree to 1e-9..1e-13 in 50-850 iterations, because the
  augmented system's condition number is ‖A‖/λ regardless of cond(A). They
  diverge only as λ → 0, which is ACEfit's `QR()` default.
- **JAX/lineax on the GPU** (float64, `highest`): `jnp.linalg.qr` and
  `lineax.QR` match CPU LAPACK to 1e-12 (scaled) / cond·eps (mixed) and are
  ~10x faster than 32-thread OpenBLAS at 20000x500 but only 1-2x at
  54000x4000 (f64 on an A4500 runs at ~280 GFLOP/s) and **OOM at the degree-10
  shape (101k x 8k)**; SVD-based `lstsq` silently truncates at λ=0; lineax's LSMR/CG never converged in 10 000 steps. A
  `shard_map` TSQR (20 lines) agrees to 1e-12 on 8 virtual devices. The 20 GB
  card could not hold the 6.5 GB degree-10 embedded matrix for a QR; the
  810 GB case is not a GPU problem.
- **Recommendation** (section 11): distributed TSQR over MPI.jl for degree ≤ 16
  embedded, where `R` fits a node; the same row-block operator gives
  distributed LSQR for free (32 + 7 lines) and is the only route that reaches
  degree 20 on this hardware, *provided a nonzero λ is used*. Do not build on
  ScaLAPACK.jl, Elemental.jl or Dagger.jl.

## 1. ScaLAPACK.jl — the package (Q1)

Measured on 2026-09-12:

| item | value |
|---|---|
| repository | `JuliaParallel/ScaLAPACK.jl`, 7 stars, not archived |
| in General registry | **no** (only `SCALAPACK_jll`, `SCALAPACK32_jll`, `SCALAPACK64_jll`) |
| last commit | 2023-03-14 — a dependabot config for GitHub Actions only |
| last *code* commit | 2015-10-01 "Fix deprecations" |
| package format | `REQUIRE` file: `julia 0.4-`, `Compat`, `MPI` — no `Project.toml` |
| `Pkg.add(url=...)` on 1.11.7 | fails in `handle_repo_add!` (no Project.toml) |
| parses on 1.11.7 | **no**: `immutable` (ScaLAPACK.jl:31), `typealias` (scalapackWrappers.jl:1) are syntax errors; also `Base.LinAlg`, `Void`, `&x` by-reference ccall, `Array(T, n)`, all removed in 0.7 (2018) |
| library discovery | `const libscalapack = "/usr/local/lib/libscalapack.dylib"` hard-coded |
| process model | `DistributedArrays` + `MPIManager` (`MPI.jl` 0.x API) |

It wraps `pdgesvd`, `pdstedc`, `pdgemm`, `pdgemr2d` and BLACS. It does **not**
wrap `pdgels`. Reviving it would be a rewrite, and the rewrite is the ~35-line
`ccall` layer in `scalapack_spike/mpi_scalapack.jl`, so there is nothing to
revive.

## 2. What installs and runs (Q1, continued)

All on `julia +1.11` = 1.11.7, Linux x86_64, fresh project, registry updated
today (moriarty's General registry was dated 2022-07-07 before the update).
Nothing was held back or downgraded (`Pkg.status(outdated=true)` empty for every
project).

| package | version | pulls in |
|---|---|---|
| `MPI.jl` | 0.20.27 | `MPICH_jll` 5.0.1+0 (MPICH 5.0.1, ABI 18:1:6, ch4:ofi); `MPIPreferences` 0.1.12, binary = `MPICH_jll` |
| `SCALAPACK32_jll` | 2.2.302+0 | built against the MPICH ABI, links `libblastrampoline`, `libmpi.so.12`, `libgfortran.so.5` |
| `OpenBLAS32_jll` | 0.3.34+0 | **required by hand**, see gotcha |
| `Elemental.jl` | 0.6.1 (`Elemental_jll` 0.87) | `DistributedArrays`, Elemental C++ 0.87-dev linked against MPICH ABI and `libopenblas64_` |
| `Dagger.jl` | 0.22.4 | Distributed.jl workers, `NextLA` tile kernels |
| `IterativeSolvers.jl` | 0.9.4 | — |
| system MPI | `/usr/lib64/openmpi`, `/usr/lib64/mpich` present, no `mpiexec` on PATH; **not used** — everything ran on `MPICH_jll`'s own `mpiexec` |

**Gotcha (cost ~20 min):** `pdgels` fails with `no BLAS/LAPACK library loaded
for lsame_()` and `PDLASCL parameter 4 had an illegal value` until an LP64
BLAS is forwarded: `BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path;
clear=false)`. The JLL does not do this itself. `SCALAPACK32_jll` is 32-bit
integer indexed: every local-array dimension and leading dimension must be
< 2^31, which holds for all sizes in the plan's table but rules out
`SCALAPACK64_jll`-free use if a single rank's block ever exceeds ~2.1e9
elements (17 GB). Not tested.

`SLATE_jll` 2025.5.28 (the modern ScaLAPACK successor) is registered with the
same MPI-flavoured build matrix, but there is no Julia wrapper at all; it would
be the same `ccall` exercise with a C API. Not tested. `QRMumps`/`qr_mumps_jll`
is a *sparse* multifrontal QR and does not apply to dense design matrices.
`MAGMA_jll` is single-node GPU. No PLASMA/Chameleon/DPLASMA JLL exists.

## 3. Test design (Q2, Q3)

`scalapack_spike/testmat.jl` builds tall-skinny `m x n` matrices whose row
block `i0:i1` can be generated identically on any process, so every solver —
serial reference, MPI ranks, Distributed workers, the JAX process — sees the
*same* matrix from its own row block. `A = B*M` with `B` iid Gaussian, `M` one
of two `n x n` mixers with singular values log-spaced from 1 to 1/cond:

- **`scaled`**: `M = Diagonal(s)`. Exactly representable; **measured cond(A)
  matches the request up to 1.02e21.** This is the ACE-like structure (columns
  of high polynomial degree are tiny).
- **`mixed`**: `M = V*Diagonal(s)*V'` with random orthogonal `V`. Rounding the
  dense product to Float64 lifts its smallest singular values to ~eps·‖M‖, so
  **measured cond(A) saturates at 0.8-3e16 whatever is requested.** This is not
  a bug in the generator: a dense Float64 matrix cannot carry cond ≫ 1e16 unless
  the ill-conditioning sits in representable structure. Measured cond is
  reported in every table.

RHS `y = A·x_true + 1e-3·noise` — a nonzero residual is essential, because the
least-squares optimality metric is 0/0 on a consistent system (the first draft
of this spike made exactly that mistake).

Metrics, all computed on the unaugmented `A, y`:

- `rel.resid` = ‖Ax−y‖/‖y‖
- `normality` = ‖Aᵀ(Ax−y)‖/(‖A‖‖Ax−y‖) — the LS optimality condition; a
  backward-stable solver gives ~1e-13 *whatever* cond(A) is, normal equations
  give ~eps·cond(A). This is what separates algorithms from conditioning.
- `agreement` = ‖x − x_qr‖/‖x_qr‖ against single-node LAPACK `qr(A) \ y`, and
  for the small cases `fwd.err` against a 400-bit BigFloat QR solution.
- `cond·eps` is printed alongside: agreement cannot be expected to beat it.

## 4. Single-node reference: which algorithms survive which conditioning (Q3)

`m=2000, n=100`, 8 TSQR blocks, `scalapack_spike/logs/ref.log`. `fwd.err` is
against the BigFloat solution.

**`scaled` family (cond exactly as requested):**

| solver | cond 1e8 | cond 1e12 | cond 1e16 | cond 1e21 |
|---|---|---|---|---|
| LAPACK `qr` fwd.err | 3.0e-13 | 1.9e-13 | 3.1e-13 | 6.0e-13 |
| TSQR (R and Qᵀy through tree) | 5.1e-13 | 2.3e-13 | 1.0e-12 | 3.7e-13 |
| normal eqns (Cholesky) | 5.6e-13 | 8.8e-13 | 1.5e-12 | 3.5e-13 |
| `svd(A) \ y` (rtol truncation) | 6.7e-13 | 9.5e-13 | **0.99** | **1.0** |
| LSQR, no precond, 2000 it | **1.0** | **1.0** | **1.0** | **1.0** |

Everything direct is fine, *including normal equations*: diagonal scaling is
exact in floating point and Cholesky is invariant under it. `svd \` truncates
at its default rtol and returns a regularised answer. Unpreconditioned LSQR
does not converge in 20n iterations at any cond ≥ 1e8.

**`mixed` family (cond saturates; measured value shown):**

| solver | 1.01e8 | 1.01e12 | 7.6e15 | 3.1e16 |
|---|---|---|---|---|
| LAPACK `qr` fwd.err | 5.6e-9 | 3.6e-5 | 0.40 | 0.54 |
| TSQR full | 1.3e-8 | 3.5e-5 | 0.82 | 0.73 |
| TSQR seminormal (`RᵀR x = Aᵀy`) + 1 refinement | 7.1e-9 | 7.5e-5 | **1.1e3** | **1.2e4** |
| normal eqns (Cholesky) | **0.13** | not PD | not PD | not PD |
| normal eqns (LU) | **0.09** | **1.0** | **1.0** | **1.0** |
| LSQR, no precond | **1.0** | **1.0** | **1.0** | **1.0** |
| (cond·eps) | 2e-8 | 2e-4 | 1.7 | 7 |

Here QR/TSQR track cond·eps·(a few hundred), normal equations lose everything by
cond 1e8, and at cond ≥ 1e16 *no* Float64 solver can do better than O(1)
forward error — even the BigFloat reference evaluated in Float64 has normality
1e-2. The seminormal TSQR shortcut (skip carrying Qᵀy, solve `RᵀR`) is the
normal equations in disguise and is **not acceptable**: the full TSQR with
`Qᵀy` reduced through the tree is required.

## 5. Distributed solvers: agreement with single-node LAPACK (Q2, Q3)

`m=20000, n=500`, **8 MPI ranks**, each generating only its own 2500-row block;
`scalapack_spike/logs/sweep.log`. Every distributed solution is compared on
rank 0 against `qr(A_full) \ y` built independently. LSQR is right-preconditioned
by global column norms (one Allreduce), λ = 0, tolerance 1e-14, max 20n
iterations.

**`scaled` family, agreement ‖x_dist − x_qr‖/‖x_qr‖:**

| solver | cond 1e8 | 1e12 | 1e16 | 1e21 |
|---|---|---|---|---|
| TSQR (MPI.jl, 26 lines) | 8.6e-13 | 8.3e-13 | 1.5e-12 | 2.0e-12 |
| ScaLAPACK `pdgels`, 8x1 grid, MB=2500, NB=n (ACEfit's row blocks verbatim) | 9.8e-13 | 7.9e-13 | 1.3e-12 | 9.2e-13 |
| ScaLAPACK `pdgels`, 2x4 grid, NB=64, after `pdgemr2d` | 2.8e-12 | 2.8e-12 | 3.7e-12 | 3.9e-12 |
| Elemental.jl `leastSquares` | — | 7.6e-13 | — | — |
| Dagger.jl `qr!(DArray)` (4 workers) | — | 4.1e-13 (n=200) | 1.5e-12 | 7.1e-13 (n=200) |
| LSQR + column equilibration, λ=0 | 1.0e-12 (21 it) | 8.2e-13 (21 it) | 1.6e-12 (21 it) | 1.4e-12 (21 it) |
| LSQR + column equilibration, damp=5e-3 | 2.9e-2 | 2.9e-2 | 3.0e-2 | 3.3e-2 |

The damp=5e-3 rows differ from the λ=0 reference by construction; they solve a
different (regularised) problem — see section 6.

**`mixed` family:**

| solver | 1.02e8 | 1.02e12 | 8.9e15 | 3.1e16 |
|---|---|---|---|---|
| TSQR | 5.2e-9 | 4.9e-5 | 0.37 | 0.79 |
| ScaLAPACK 8x1 | 4.5e-9 | 3.7e-5 | 0.29 | 0.59 |
| ScaLAPACK 2x4 | 5.8e-9 | 5.4e-5 | 0.39 | 0.73 |
| Elemental.jl | — | 4.4e-5 | — | — |
| Dagger.jl (n=200) | — | 2.0e-4 | 0.92 | — |
| LSQR + equilibration, λ=0 | **1.0** (10000 it, not converged) | **1.0** | **1.0** | **1.0** |
| (cond·eps) | 2e-8 | 2e-4 | 2.0 | 4.2 |

All direct methods agree with each other and with LAPACK to ~cond·eps·(200);
they are numerically interchangeable. LSQR with a diagonal preconditioner
cannot fix non-diagonal ill-conditioning and fails at every level with λ = 0.

## 6. LSQR `damp` versus `QR` `lambda` (Q5)

The plan states LSQR's damping "is not identical to QR's lambda". Both target
`min ‖Ax−y‖² + λ²‖x‖²`; `ACEfit.QR` solves it as `qr([A; λI]) \ [y; 0]`.
Measured, `m=2000, n=100`, LSQR to 1e-15 (`scalapack_spike/damp_vs_lambda.jl`):

| family / measured cond | λ | ‖x_QR − x_LSQR‖/‖x_QR‖ | LSQR iters | cond([A; λI]) |
|---|---|---|---|---|
| scaled 1e16 | 1e-6 | 1.2e-4 | 5000 (cap) | 4.6e7 |
| scaled 1e16 | 1e-3 | 2.5e-9 | 399 | 4.6e4 |
| scaled 1e16 | 1e-1 | 1.5e-13 | 84 | 4.6e2 |
| scaled 1e21 | 1e-3 | 9.9e-11 | 237 | 4.6e4 |
| mixed 1e12 | 1e-6 | **0.58** | 5000 (cap) | 4.5e7 |
| mixed 1e12 | 1e-3 | 9.7e-11 | 855 | 4.5e4 |
| mixed 7.6e15 | 1e-3 | 5.9e-10 | 404 | 4.5e4 |
| mixed 3.1e16 | 1e-3 | 1.6e-10 | 240 | 4.5e4 |
| mixed 3.1e16 | 1e-1 | 2.6e-13 | 54 | 4.5e2 |

So: **they are the same regulariser**, and once λ is large enough that
‖A‖/λ ≲ 1e5 LSQR reproduces the QR-λ answer to 1e-9 or better in a few hundred
iterations, *independently of cond(A)*. The difference the plan worries about
is real only in the λ → 0 limit — which is `ACEfit.QR()`'s default — and there
LSQR simply does not converge on mixed-type matrices. With ACEfit's `P` the
augmented system is `[A/P; λI]`, i.e. λ‖Pθ‖; the same statement holds for the
transformed variable.

## 7. Real ACE conditioning, and the memory floor of every direct method

**Real matrix.** `scalapack_spike/ace_cond.jl`, Si_tiny (1052 observations),
`ace1_model(order=3)`, `algebraic_smoothness_prior(p=4)`, default weights, on
the `ACEpotentials-jax/acejax/julia` environment (Julia 1.12.2):

| totaldegree | A | cond(A) | cond(W·A/P) — what `QR` sees | column-norm spread | cond after column equilibration |
|---|---|---|---|---|---|
| 10 | 1052 x 120 | 2.1e10 | 1.1e11 | 4.1e5 | 2.9e9 |
| 14 | 1052 x 397 | 4.8e10 | 6.8e11 | 2.4e6 | 7.3e9 |
| 18 | 1052 x 1137 | ∞ (n > m, 160 exact zeros) | ∞ | 6.5e6 | ∞ |

Two things measured, one inferred. Measured: the prior and weights together *raise* cond by 5-14x
(it is designed to regularise, not to equilibrate); equilibration recovers only
~40-100x, leaving cond 3e9-7e9 — **the ACE matrix is of the `mixed` type**,
where λ=0 LSQR failed above. The 1e21 figure in the plan was not reproduced
here; this dataset is too small for degree ≥ 16 (underdetermined), so it is
neither confirmed nor refuted. Inferred: production fits at higher degree on
larger data will have cond somewhere between these values and 1e21, and the
conclusion — λ > 0 is required for any iterative route — holds across that
range.

**Memory floor.** Every QR-type method, distributed or not, must hold the
`n x n` upper-triangular `R` (LAPACK/ScaLAPACK store it as a full square). With
the plan's *measured* parameter counts (S=5, order 3):

| model | n | `R` as full square (Float64) | design matrix (plan table) |
|---|---|---|---|
| degree 12 embedded | 15 885 | 2.0 GB | 12.8 GB |
| degree 12 categorical | 101 540 | **82 GB** | 82 GB (matrix is ~square: ~101k rows) |
| degree 20 embedded | 142 250 | **162 GB** | 810 GB |

Degree 16 embedded is not in the measured counts; if the 5x rule stated for
degree 20 is applied to the 107 GB figure, n ≈ 52 000 and `R` ≈ 21 GB —
inferred, not measured. So on 62 GB nodes:

- degree ≤ 16 embedded: TSQR's per-rank state (`R` + a row block) fits;
  streaming assembly means the 107 GB matrix never has to exist.
- degree 12 categorical and degree 20 embedded: **no row-block-only direct
  method can work on one node**; `R` alone needs a 2-D distribution across
  ≥ 3-4 nodes (ScaLAPACK/Elemental/SLATE territory), or an iterative solver
  whose state is O(n) vectors (LSQR: ~1 MB per vector at n = 142 250).

Arithmetic, for scale (analytic, not measured): Householder QR is ~2mn² flops,
i.e. 2.9e16 at degree 20 embedded — roughly 8 hours at 1 TFLOP/s — versus
LSQR at ~4mn flops per iteration = 4e11 (0.4 s) per iteration.

## 8. Timing at a modest size (measured, one configuration)

`m=120000, n=2000` (1.92 GB), `scaled`, cond 1e16, **8 MPI ranks x 2 BLAS
threads** (16 cores), `scalapack_spike/logs/timing.log`. The single-node
reference ran on rank 0 with the same 2 BLAS threads and is therefore *not* a
fair CPU baseline — see the 32-thread figure below it.

| solver | time | agreement |
|---|---|---|
| TSQR (MPI.jl) | **8.1 s** | 9.3e-13 |
| ScaLAPACK `pdgels`, 8x1 grid, NB=n=2000 (row blocks verbatim) | 105.6 s | 1.6e-12 |
| ScaLAPACK `pdgels`, 2x4 grid, NB=64 | 11.9 s + 0.4 s `pdgemr2d` | 6.8e-12 |
| Elemental.jl `leastSquares` | 11.8 s + **14.1 s** per-element fill | 6.3e-13 |
| LSQR + equilibration, λ=0 (trivially conditioned after scaling) | 2.5 s, 19 it | 1.4e-12 |
| LAPACK `qr(A) \ y`, 1 node, 2 threads | 27.6 s | — |
| LAPACK `qr(A) \ y`, 1 node, 32 threads | 44-49 s (measured while a GPU job shared the host; OpenBLAS `dgeqrf` scales poorly) | — |

The 1-D ScaLAPACK layout is 9x slower than 2-D because with `NB = n` the whole
matrix is one panel and the trailing update vanishes. So the "clean path" from
ACEfit's row blocks — no redistribution — is correct but slow, and the
`pdgemr2d` redistribution (0.4 s here) is cheap but needs a second copy of the
matrix resident during the copy. Dagger.jl's tiled QR was 10-50x slower than
LAPACK at 4000x200 .. 20000x500 (0.8-8 s vs 0.02-0.4 s) with a 134 s first-call
compile; not timed larger.

## 9. JAX / lineax on the GPU

Environment: `/storage/eng/essswb/macejax-gpu/venv` — jax 0.11.1, jaxlib 0.11.1,
jax-cuda12-plugin 0.11.1, equinox 0.13.8 (already present), **lineax 0.1.1
added** (`uv pip install lineax`; dry-run showed it the only new package).
Python 3.12.8. RTX A4500, 20 GB, 108 MiB in use by others at the start;
`XLA_PYTHON_CLIENT_PREALLOCATE=false`; `jax_enable_x64=True`;
`jax_default_matmul_precision="highest"`. The matrices are the *same* ones
(dumped from Julia with the CPU LAPACK solution alongside, `dump_cases.jl`);
the augmented system `[A; λI] \ [y; 0]` is what is solved when λ > 0, and the
agreement column is against CPU `qr([A; λI]) \ [y; 0]`. Scripts:
`scalapack_spike/jax_arm.py`, `jax_tsqr_shard.py`.

**Accuracy, `m=20000, n=500`, float64 on the GPU** (agreement vs CPU LAPACK qr;
`scalapack_spike/logs/jax_acc.log`):

| method | scaled 1e12 λ=0 | scaled 1e21 λ=0 | mixed 1.0e12 λ=0 | mixed 1.9e16 λ=0 | scaled 1e21 λ=1e-3 | mixed 1.9e16 λ=1e-3 |
|---|---|---|---|---|---|---|
| `jnp.linalg.qr` + `solve_triangular` | 2.3e-12 | 3.3e-12 | 3.2e-5 | 0.56 | 1.1e-12 | 2.8e-11 |
| `lineax.QR` | 6.9e-13 | 1.0e-12 | 3.2e-5 | 0.56 | 1.1e-12 | 2.8e-11 |
| `jnp.linalg.lstsq` (SVD, default rcond) | **0.98** | **1.0** | **0.98** | **1.0** | 1.0e-12 | 2.8e-11 |
| `lineax.SVD` | **0.99** | **1.0** | **1.0** | **1.0** | 1.0e-12 | 2.8e-11 |
| `lineax.LSMR` (rtol=atol=1e-14, 10000 steps) | **1.0**, cap hit | **1.0**, cap | **1.0**, cap | **1.0**, cap | **1.2e-3**, cap | **0.10**, cap |
| `lineax.Normal(CG)` (same) | **1.0**, cap | **1.0**, cap | **1.0**, cap | **1.0**, cap | **0.16**, cap | **0.30**, cap |
| (cond·eps) | 2e-4 | 2e5 | 2e-4 | 4 | — | — |

So GPU Householder QR (either entry point) is numerically the same solver as
CPU LAPACK and as the MPI TSQR/ScaLAPACK arms: 1e-12 on representable
conditioning, cond·eps on mixed, and 1e-11 on the regularised system. The
SVD-based routes truncate at their default `rcond` and return a *different,
regularised* answer at λ=0 — the same behaviour as Julia's `svd(A) \ y` — which
is fine only if that is the regularisation you meant. lineax's iterative
solvers did not converge within 10 000 steps on any case at these tolerances,
including λ=1e-3 where the hand-written LSQR in section 6 converged in ~400;
this was not investigated further (it may be a tolerance-semantics mismatch)
and is reported as measured.

**Speed.** GPU times are for the second call (compiled), with
`block_until_ready`; CPU is Julia `qr(A)\y` with 32 OpenBLAS threads on the
same host while the GPU job was idle.

| size (f64) | CPU LAPACK 32 thr | `jnp.linalg.qr`+solve | `lineax.QR` | GPU speed-up |
|---|---|---|---|---|
| 20 000 x 500 (80 MB) | 0.8-1.4 s | 0.13-0.14 s | 0.085-0.091 s | ~10x |
| 50 000 x 4 000 + 4000 λ-rows (1.7 GB) | 11.8 s | 11.3 s | 6.1 s | 1-2x |
| 101 000 x 8 000 + 8000 λ-rows (7.0 GB; degree-10 embedded shape) | 77.8 s | **OOM** | **OOM** | — |

The 101k x 8k case fails in XLA's autotuner trying to allocate a 6.51 GiB
transpose copy on top of the resident matrix, for both entry points
(`scalapack_spike/logs/jax_time.log`). The 20000x500 speed-up is mostly
OpenBLAS being inefficient on a small QR; at 54k x 4k the GPU runs at
~280 GFLOP/s in f64 (2mn²/t), which is close to the A4500's f64 peak (GA102:
1/64 of f32 rate), so **on this card an f64 QR is CPU-speed**. The plan's
"f64 costs 3-5x f32" figure came from a different workload and does not
transfer to dense QR. The degree-12 embedded shape (101k x 15 885, 12.8 GB)
was therefore not attempted.

**Sharded / multi-device path.** `jnp.linalg.qr` on a sharded array is *not*
a distributed QR (XLA gathers). A TSQR written with `shard_map` over row
blocks — local `qr` per device, `all_gather` of the `n x n` R factors and
`Qᵀy`, one more `qr` of the stacked R's, replicated triangular solve — is 20
lines (`jax_tsqr_shard.py`). There is one GPU on this host, so it was run on
**8 virtual CPU devices** (`--xla_force_host_platform_device_count=8`); the same
code runs on a multi-GPU mesh by changing the mesh. Agreement with CPU LAPACK
on all 12 cases: 6.4e-13 .. 3.1e-12 (scaled), cond·eps on mixed at λ=0, and
2.8e-11 .. 3.6e-11 at λ=1e-3 — identical quality to the MPI TSQR. **Not
measured: anything across real multiple GPUs.**

**The memory ceiling, stated plainly.** The card holds 20 GB. Householder QR
needs the matrix plus `Q` (same size, `mode="reduced"`) plus workspace, so
roughly 2.2x the matrix: the largest design matrix a single-GPU QR can take
here is ~8-9 GB, i.e. the degree-10 embedded case (6.5 GB) — the degree-12
embedded matrix (12.8 GB) does not fit on this card, and 810 GB is not a GPU
problem at all. A multi-GPU shard_map TSQR has the *same* `R`-factor floor as
the MPI one (section 7): 2 GB at degree 12, 162 GB at degree 20 per device
after the all-gather (it is replicated, not distributed, in this formulation).

## 10. Integration cost against ACEfit's row-block assembly (Q4)

What exists: `ACEfit.assemble` (`assemble.jl:24-45`) allocates
`SharedArray`s `A, Y` and fills row ranges with `pmap` over packets on
Distributed.jl workers; `acefit!` (`fit_model.jl:134-160`) then forms
`Diagonal(W) * (A / P)` in place and calls `ACEfit.solve(solver, Ap, Y)`.

Common to every distributed route: assembly must produce **per-process row
blocks that stay where they are** instead of a `SharedArray`, i.e. each process
runs `feature_matrix` on its own packets and applies `W` and `/P` locally
(`P` is diagonal, so `A/P` is a column scaling — no communication). That is a
new `assemble_distributed` beside the current one, ~100-150 lines including
packet partitioning by row count, and does not touch `feature_matrix`.

| route | solver code | process model | obstacles | estimate |
|---|---|---|---|---|
| **TSQR over MPI.jl** | 26 lines (spike) → ~80 with `[A; λP]` rows folded in as a local block, committee/BLR hooks, tests | `mpiexec -n N julia` (MPI ranks assemble their own packets) or `MPIClusterManagers.jl` to keep `addprocs`/`pmap` | R must fit a rank (section 7); reduction tree only helps once block rows ≫ n; needs its own `solve(::TSQR, ...)` that never materialises `A` | **3-4 days** incl. distributed assembly |
| **distributed LSQR** | 32 + 7 lines (spike) → ~60 | same as above | needs λ > 0 (section 6); iteration count ~ ‖A‖/λ; column-norm preconditioner is one Allreduce | **+1 day** on top of TSQR (shares the operator) |
| **ScaLAPACK via `SCALAPACK32_jll`** | ~35 lines wrappers + ~40 driver (spike) | MPI, as above | LP64 BLAS forwarding; 1-D layout is 9x slow, 2-D needs `pdgemr2d` and 2x transient memory; Int32 descriptors; BLACS context lifetime; no Julia-side tests exist | **4-6 days** |
| **Elemental.jl** | 12 lines (spike) | MPI, `El.Initialize` must precede `MPI.Init`, `El.Finalize` before `MPI.Finalize` | `Array(::DistMatrix)` broken in 0.6.1; per-element `queueUpdate` fill (14 s / 1.9 GB) — would need a bulk local-part API that the wrapper does not expose; upstream C++ frozen 2017 | **5+ days**, not recommended |
| **Dagger.jl** | ~10 lines | Distributed.jl — **keeps `addprocs` and `pmap` unchanged** | needs `DArray` from remote chunks (internal constructor); uniform square tiles; 10-50x slower than LAPACK at spike sizes; CAQR (`p>1`) forces `ib=1` | **2-3 days** to a working but slow solver |

Named obstacles for the recommended route: (i) `acefit!` assumes `A` is a
`Matrix` for `A / P` and for `compute_errors` on the training set — the latter
needs a distributed `A*x`; (ii) BLR (section 19 of the plan) wants `AᵀA` or an
SVD, which TSQR provides via `R` (`AᵀA = RᵀR`, `svd(R)`) only while `R` fits;
(iii) the validation-set path calls `solve(solver, Ap, Y, Avp, Yv)` and would
need the same treatment.

## 11. Recommendation for Stage 2 phase 18

1. **Do not use ScaLAPACK.jl, Elemental.jl, or Dagger.jl.** The first is
   dead; the second is correct but bit-rotted at the edges, fill-bound, and on
   a frozen upstream; the third is alive but slow and needs an internal
   constructor to avoid gathering.
2. **Build the row-block distributed operator once, and put two solvers on
   it**: TSQR (direct, QR-equivalent, for degree ≤ 16 embedded where `R` fits a
   rank) and LSQR (iterative, O(n) state, the only thing that reaches degree 20
   / categorical 12 on 62 GB nodes). The operator is ~150 lines; the two solvers
   are ~150 lines together; both were validated in this spike against LAPACK to
   1e-12 on the direct side and to the QR-λ answer on the iterative side.
3. **Make λ > 0 mandatory on the LSQR path** and validate every LSQR fit
   against TSQR at a size where both run, as the plan already says. Section 6
   gives the expected agreement (1e-9..1e-13 for ‖A‖/λ ≲ 1e5).
4. Keep ScaLAPACK-via-JLL in reserve for the case where `R` does not fit
   *and* a direct factorisation is required (BLR at degree 20). It works, the
   wrapper is in `scalapack_spike/mpi_scalapack.jl`, and a 2-D grid is the way
   to run it.
5. **On the GPU / Stage 2A question:** the JAX arm is numerically equivalent to
   the Julia direct solvers, but on this card f64 QR is CPU-speed (6-11 s vs
   11.8 s at 54k x 4k), it OOMs at the degree-10 embedded shape, and it is
   single-device unless a TSQR is written by hand (20 lines, validated on
   virtual devices only). It does not change what
   phase 18 has to build — a row-block operator with a TSQR reduction is the
   same design in either language — so it is not by itself a reason to pull
   Stage 2A forward. It *is* a reason to keep the phase-18 design language-
   agnostic: the reduction is `qr` on local blocks + one all-gather, which JAX
   and MPI.jl both express in ~20-30 lines.

## 12. Reproducing

All on moriarty under `/tmp/essswb-scalapack-spike/` (projects `proj/` with
MPI + SCALAPACK32_jll + OpenBLAS32_jll + IterativeSolvers, `elem/` with
Elemental + MPI, `dag/` with Dagger). From `scalapack_spike/`:

```
julia +1.11 --project=../proj ref.jl                    # section 4
bash run.sh 8 mpi_tsqr.jl m=20000 n=500 cond=1e16 family=mixed
bash run.sh 8 mpi_scalapack.jl ... grid=2d              # section 5, 8
bash run.sh 8 mpi_lsqr.jl ... precond=1 damp=0
bash run_elem.sh 8 mpi_elemental.jl ...
julia +1.11 --project=../dag -p 4 dagger_qr.jl
julia +1.11 --project=../proj damp_vs_lambda.jl         # section 6
julia +1.12 --project=~/ACEpotentials-jax/acejax/julia ace_cond.jl   # section 7
julia +1.11 --project=../proj dump_cases.jl cases small && bash jax_run.sh   # section 9
```

`run.sh` caps each rank at 6 GB virtual (`ulimit -v`); the 120000x2000 and
JAX big runs used 14-60 GB caps. Peak host memory observed during the day was
11 GB of 62.

## 13. Not determined

- cond of a *production* ACE design matrix (multi-element, degree ≥ 16); only
  Si_tiny degree 10-14 was measured.
- `SCALAPACK64_jll`, `SLATE_jll`; multi-node runs of anything (one host only).
- Performance of any route at n > 2000; all timings are single-configuration
  and on 16 of 32 cores.
- Elemental.jl at cond 1e16 on the small cases (two runs were lost when a
  `pkill` pattern matched the ssh session; the 120000x2000 cond-1e16 run did
  complete, agreement 6.3e-13).
- Whether the categorical degree-12 matrix (101 540 columns, ~101k rows) is
  well-posed at all at 1 observation per parameter — it is nearly square.
