# Reactant spike: why the Julia ACE export is wrong, and what it means

Investigating `lammps-jax` `dev/julia_export` `examples/julia/ace_export.jl`.
Julia 1.12.6, CPU (Apple Silicon). SPIKE CODE.

**Version note:** the miscompilation reproduces on **both Reactant 0.2.222 and
0.2.285** (the latest as of 2026-09-09), so it is not fixed by upgrading.
Re-confirmed on Reactant 0.2.285 in clean single-dependency environments across
both platforms and every Julia version Reactant tests:

| platform | Julia | in Reactant CI? | result |
|---|---|---|---|
| macOS aarch64 (Apple Silicon) | 1.11.9 | yes (version sweep) | MISMATCH |
| macOS aarch64 | 1.12.7 | yes (version sweep) | MISMATCH |
| macOS aarch64 | 1.13.0 | **no** | MISMATCH |
| Linux x86_64 (alderlake) | 1.11.7 | yes (version sweep) | MISMATCH |
| Linux x86_64 (alderlake) | 1.12.2 | yes (version sweep) | MISMATCH |

Identical wrong output in all five: `[9.0 9.0; 36.0 36.0]` for an expected
`[9.0 6.0; 36.0 30.0]`. So it is neither platform- nor architecture-specific,
and not an artefact of the Apple Silicon build.

Reactant's `Project.toml` says `julia = "1.10"`, an open upper bound, so the
resolver installs happily on 1.13 -- but its CI matrix is 1.10 (primary) plus a
1.11/1.12 sweep, with no 1.13 row. **Report this upstream against 1.11 or 1.12,
not 1.13**, or the version is an easy way to dismiss the bug. The 1.13 row is
included only to show the behaviour does not change there.

**EquivariantTensors cannot be installed alongside Reactant > 0.2.222.**
ET depends on WignerD, which pins StructArrays <= 0.6.21, while Reactant 0.2.285
requires StructArrays >= 0.7.2. So 0.2.222 is not an arbitrary choice -- it is
the newest Reactant that can coexist with ET at all.

**This was cheap to fix, and is now done:
[ACEsuit/EquivariantTensors.jl#143](https://github.com/ACEsuit/EquivariantTensors.jl/pull/143).**
WignerD is used in ET's `src/` at exactly two places, in `O3/O3_utils.jl`,
inside `D_from_angles` / `QD_from_angles` -- utilities that build a rotation Q
and matching Wigner-D matrix so equivariance can be checked as `y о Q = D * y`.
They have **no callers anywhere in `src/`**, are not exported, and are used
only by `test/O3/test_O3_transforms.jl`. WignerD was already in ET's test
target, so the PR moves both functions into `test/test_utils/utils_testO3.jl`
and drops WignerD from `[deps]`.

Note the failure mode is subtler than "cannot be installed": asking for ET and
Reactant 0.2.285 together does **not** error, it silently resolves to
**EquivariantTensors 0.1.2**, four minor versions back, from before WignerD was
a dependency. The conflict is only visible if ET is pinned:

```
Unsatisfiable requirements detected for package StructArrays [09ab397b]:
 ├─restricted by compatibility requirements with Reactant [3c362404] to versions: 0.7.2 - 0.7.3
 └─restricted by compatibility requirements with WignerD [87c4ff3e] to versions: 0.5.0 - 0.6.21 — no versions left
```

With #143 applied, a dev'd ET and Reactant 0.2.285 resolve and precompile
together, so the KA-tracing question below can finally be tested against a
current Reactant.

## Headline

The ACE export failure is a **silent Reactant miscompilation**, reduced to a
two-line reproducer. Eager Julia is correct; the compiled program is wrong and
raises nothing.

```julia
using Reactant
Reactant.set_default_backend("cpu")

f(u) = hcat(u[:, 3] .* u[:, 3], u[:, 2] .* u[:, 3])

u  = [1.0 2.0 3.0; 4.0 5.0 6.0]
ru = Reactant.to_rarray(u)

f(u)                                    # [9.0 6.0; 36.0 30.0]   correct
Array((@compile f(ru))(ru))             # [9.0 9.0; 36.0 36.0]   WRONG
```

Runnable as-is in an environment with only Reactant installed; it is also
committed as `spike/reactant_phase0/reactant_bug_repro.jl`, which is what to
run rather than retyping the snippet.

Two distinct elementwise products are deduplicated into one. See
`reactant_bug_repro.jl`.

**Filed upstream as
[EnzymeAD/Reactant.jl#3267](https://github.com/EnzymeAD/Reactant.jl/issues/3267)**,
"Correctness issue when compiling elementwise operations" (2026-09-10, open).

### Localised to an optimisation pass

The traced IR is **correct**; an optimisation pass corrupts it.
`@code_hlo optimize=false` contains two distinct `enzyme.batch` multiplies --
`z*z` from `slice [2:3]` twice, and `y*z` from `slice [1:2]` and `slice [2:3]`
-- concatenated. After the default pipeline the whole thing has collapsed to:

```mlir
%0 = stablehlo.slice %arg0 [2:3, 0:2] : (tensor<3x2xf64>) -> tensor<1x2xf64>
%1 = stablehlo.multiply %0, %0 : tensor<1x2xf64>
%2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x2xf64>) -> tensor<2x2xf64>
return %2 : tensor<2x2xf64>
```

One multiply, then a broadcast filling both columns: the concatenate of two
different columns has become a broadcast of one. Bracketing by `optimize=`
level pins which passes are responsible:

| `optimize=` | result |
|---|---|
| `:just_batch` | `[9.0 6.0; 36.0 30.0]` **correct** |
| `:before_kernel` | `[9.0 9.0; 36.0 36.0]` wrong |
| `:all` (default) | `[9.0 9.0; 36.0 36.0]` wrong |
| `:canonicalize`, `:none` | fail to compile (`enzyme.batch` never lowered) |

So the corruption is introduced **after `:just_batch` and at or before
`:before_kernel`**. `@compile optimize=:just_batch f(ru)` is a working
workaround for anyone hitting this, at the cost of the rest of the pipeline.

## How it surfaces in ACE

`ace_export.jl` cannot trace SpheriCart, so it rebuilds Ylm as monomials of the
unit vector mapped onto SpheriCart values. The bug corrupts exactly the
monomials that are products of two *distinct* components:

| l | n_mono | status | corrupted exponents |
|---|---|---|---|
| 0 | 1 | OK | - |
| 1 | 3 | OK | - |
| **2** | **6** | **CORRUPT** | (0,1,1), (1,0,1), (1,1,0) |
| 3 | 10 | OK | - |
| 4 | 15 | OK | - |
| 5 | 21 | OK | - |

Each mixed monomial becomes `u[:,k_last]^2`. Size-independent (N = 2, 4, 64,
2048 all corrupt at l=2). Any ACE model with LMAX >= 2 is silently wrong, which
is every realistic model.

## Hypotheses tested and rejected

Recorded because three plausible explanations were wrong, and the measurements
that killed them are cheap to redo.

1. **Ill-conditioned monomial -> Ylm least-squares map.** Rejected:
   `ylm_conditioning.jl` measures cond(feats) 1.0-10.1 up to l=4, held-out error
   ~1e-15, and a 1e-16 monomial perturbation amplifies to only ~3e-16.
2. **Eager and compiled run different code paths.** The parity assert calls
   `site_energies(..., em = nothing)` while `core_energy` calls it with the edge
   mask, so the masking arithmetic is never covered by the check. Plausible, but
   rejected: `mask_path_test.jl` runs both eagerly and they agree to **exactly
   0.0**, both matching ACEpotentials to 4.5e-13.
3. **The `col === nothing` accumulator loop.** Rejected: a workaround avoiding
   the accumulator entirely is *equally* corrupt, and single mixed monomials
   compiled alone are all fine.

The trigger is narrower than any of these: two 2-factor products, one a square
and one mixed, in the same traced program.

## Is the STANDARD ETACE path traceable? (the question that matters)

An earlier draft of this file claimed "Reactant does not save the rewrite",
generalising from `ace_export.jl`'s hand-written arrays. **That was wrong.**
`ace_export.jl` hand-writing arrays is its own choice, not a demonstration that
hand-writing is required. The ETACE rewrite was designed so the standard path
would be traceable, and that is testable directly (`etace_traceable.jl`,
`trace_detail.jl`, `confirm.jl`).

Measured on Reactant 0.2.222 / CPU, 64-atom Si, `ace_model(order=3,
max_level=10, maxl=6)`:

| component | traces? | agreement |
|---|---|---|
| `yembed` — SpheriCart solid harmonics via P4ML | **YES** | 1.7e-10 |
| `rembed` minus `SelectLinL` — Agnesi + polys + envelope | **YES** | 1.1e-13 |
| `rembed` complete | no | scalar indexing |
| `site_basis`, `et_model(G, ps, st)` | no | scalar indexing |
| `abasis` / `aabasis` via `ka_evaluate` | no | `ka_with_reactant` MethodError |

So the standard path is **two specific layers away from traceable**, not
architecturally incompatible.

**Blocker 1 — `ET.SelectLinL`.** Every failure in `site_basis` bottoms out at
`selectlinl.jl:56` -> `:68`:

```julia
@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
   iB, jB = @index(Global, NTuple)
   i_x = selector(X[iB])          # scalar index into an array of PStates
   ...
end
```

EquivariantTensors' own comments already mark this as a stopgap: *"Morally this
should work, but it doesn't like the views it seems?! so we need write a
kernel"* and *"there was a problem applying the selector when it was type
unstable; now that this is fixed, maybe try to go back to the above
implementation ... that way we don't have to write a custom rrule."*
Precomputing the category as an integer array per edge and doing a gather plus
batched matmul is array-expressible, traceable, and removes the custom rrule.

**Blocker 2 — KernelAbstractions, and it is broader than first thought.**
Re-tested on Reactant 0.2.285 with the kernels replicated standalone
(`ka_blockers.jl`, `ka_triggers.jl`, since ET itself cannot be installed at that
version):

| kernel | traces? |
|---|---|
| `A[i] = 2*X[i]`, 1-D ndrange, all traced | **no** — `ka_with_reactant` MethodError |
| same, 2-D ndrange | **no** |
| 2-D + inner loop + host index array (as ET has it) | **no** |
| 2-D + inner loop + traced index array | **no** |
| **the same SelectLinL operation as pure array ops, no KA** | **yes, 4.44e-16** |

So it is not about kernel complexity, ndrange rank, scalar indexing, or
host-vs-traced index arrays. **No KernelAbstractions kernel traces at all in this
environment**, including a trivial one. Everything in ET's evaluation path routes
through `ka_evaluate` or `_ka_apply_selectlinl!`, so nothing traces.

**That caveat turned out to be the whole story.** This machine has no CUDA, and
Reactant's KA bridge lives in its CUDA extension. Re-measured on a CUDA host,
every one of those kernels traces. See **KA on a CUDA host** below; the table
above is superseded for blocker 2, and the CPU-only numbers in it should be read
as "measured without CUDA loaded".

**Notable side finding:** SpheriCart traces fine. `ace_export.jl` replaces Ylm
with fitted monomials on the premise that SpheriCart is untraceable; on this
evidence that substitution was unnecessary — and it is the substitution that
exposed the miscompilation bug above.

## KA on a CUDA host: the decisive answer (2026-09-11, `moriarty`)

**Yes -- KernelAbstractions kernels trace under Reactant once CUDA is loaded,
on the CPU backend as well as the CUDA backend.** The earlier "no KA kernel
traces at all, not even `A[i] = 2*X[i]`" finding was an artefact of the test
machine having no CUDA, exactly as the caveat above suspected. Blocker 2 as
stated is gone.

The mechanism is not subtle, and the test prints it:

```
Reactant.ka_with_reactant: 1 method(s)
    ka_with_reactant(ndrange, workgroupsize, obj, args...) @ ReactantCUDAExt
        ~/.julia/packages/Reactant/ct5sa/ext/ReactantCUDAExt.jl:620
```

`ka_with_reactant` -- the function whose `MethodError` blocked every kernel in
the previous spike -- has exactly one method, and it is defined **only inside
`ReactantCUDAExt`**. With no CUDA loaded the extension never loads, the method
does not exist, and `@compile` of *any* KA kernel fails with a `MethodError`
irrespective of what the kernel does. With CUDA loaded the method exists and
the kernels trace. It is a packaging property of Reactant, not a property of
KA kernels, of kernel complexity, of ndrange rank, or of scalar indexing.

Note on how firmly this is established: the mechanism is read straight off
`methods(Reactant.ka_with_reactant)` in the *same* process that then compiles
the kernels successfully. The complementary control -- re-running the same
cases on the same box with CUDA deliberately not loaded, to watch the method
disappear and the `MethodError` return -- is scripted as `ka_nocuda.jl` but
**was not run**: the host was lost before I got to it. It is a nice-to-have, not
load-bearing, since the extension-only method definition is not ambiguous.

### Environment

| | |
|---|---|
| host | `moriarty` -- Xeon Silver 4216 (32 threads), NVIDIA RTX A4500 20 GB, 62 GB RAM |
| Julia | 1.11.7 (inside Reactant's CI matrix) |
| Reactant | **0.2.285** (current; not the 0.2.222 the old numbers used) |
| KernelAbstractions | 0.9.42 |
| CUDA.jl | 6.2.2, `CUDA.functional() = true`, driver 610.57.4, toolkit 13.1, cuDNN 9.21 |
| EquivariantTensors | 0.4.3 + the WignerD-drop patch (see below) |
| ACEpotentials | 0.10.2, branch `jax-eval` |

### What traces

Each case runs in its own process (`ka_one.jl`), so a crash is attributable and
cannot swallow the results after it. Numbers are `max|compiled - eager|`.

| # | case | Reactant CPU backend | Reactant CUDA backend |
|---|---|---|---|
| 0b | KA on a `CuArray`, no Reactant (control) | 0.0 | 0.0 |
| 1 | **trivial KA, 1-D ndrange, all args traced** | **TRACES, 0.0** | **TRACES, 0.0** |
| 2 | 2-D ndrange | TRACES, 0.0 | TRACES, 0.0 |
| 3 | 2-D + inner loop + **host** index array (as ET has it) | TRACES, 0.0 | TRACES, 8.88e-16 |
| 4 | 2-D + inner loop + **traced** index array | TRACES, 0.0 | TRACES, 8.88e-16 |
| B1 | ET `SelectLinL` kernel, `X::Vector{Int}` | TRACES, 0.0 | TRACES, 4.44e-16 |
| B1b | ET `SelectLinL` kernel, `X::Vector{Edge}` (struct array) | **OOM-killed** | **OOM-killed** |
| B2 | ET `PooledSparseProduct` kernel (the `abasis` kernel) | TRACES, 0.0 | TRACES, 1.78e-15 |
| C | `SelectLinL` as pure array ops, no KA | TRACES, 8.88e-16 | TRACES, 0.0 |
| D | the #3267 reproducer | **wrong values** | **wrong values** |

Row 1 is the deliverable: the trivial kernel that previously failed now traces
and is bit-exact. Row B2 matters most for the ETACE path -- the actual
`PooledSparseProduct` kernel shape that `abasis`/`aabasis` go through traces and
agrees to 1.8e-15.

### The one thing that does not: struct arrays into a KA kernel

Row B1b is the only genuine failure, and it is not a `MethodError`. Passing a
**host array of structs** plus a field-extracting closure into a KA kernel makes
Reactant's compilation allocate without bound: RSS climbs to **60 GB** on a
62 GB machine and the process is killed (`exitcode=137`, SIGKILL). It is not
slow-but-finite -- RSS sat flat at 60.5 GB for minutes, i.e. pinned against
physical memory, before the OOM killer fired. This happened on both the CPU and
the CUDA backend.

The only difference between B1 (exact, fast) and B1b (60 GB, killed) is the
element type of the index argument: `Vector{Int}` versus `Vector{Edge}` with
`selector = e -> e.z0`. This is **exactly the shape `ET.SelectLinL` has**, where
`X` is an array of `PState`s and the selector pulls the species out of each one:

```julia
@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
   iB, jB = @index(Global, NTuple)
   i_x = selector(X[iB])          # scalar index into an array of PStates
   ...
```

Note the failure mode: **not** "does not trace" but "traces into an
unboundedly-large program". On the first run this took the whole machine down --
`sshd` stopped answering and the box was unreachable for the best part of an
hour. `ladder.sh` and `etace_ladder.sh` have since been hardened to run every
case under an RSS watchdog that kills it at 25 GiB (the numbers above were taken
before that, which is how the outage happened). **Do not run these tests on a
shared box without that watchdog.**

### #3267 is still present here

Row D: `[9.0 9.0; 36.0 36.0]` for an expected `[9.0 6.0; 36.0 30.0]`, on both
backends, on Reactant 0.2.285 / Julia 1.11.7 / x86_64 with CUDA loaded. So the
miscompilation is not fixed and is not CPU-backend-specific. This row exists so
that a *wrong value* elsewhere in the table is not mistaken for a tracing
failure; nothing else in the table returned wrong values.

### ET + current Reactant now co-install

The WignerD -> StructArrays pin that capped Reactant at 0.2.222 is really gone:
with WignerD moved out of ET's `[deps]`, `Pkg.add(name="Reactant",
version="0.2.285")` into an environment holding a dev'd ACEpotentials
(`jax-eval`) and a dev'd ET 0.4.3 resolves cleanly and precompiles, including
`ReactantCUDAExt`. All the ET-level numbers below are therefore against
**current** Reactant, not 0.2.222.

Caveat on provenance: the `drop-wignerd-dep` branch is not on
`github.com/ACEsuit/EquivariantTensors.jl` (`git ls-remote --heads` does not
list it, though a local clone has an `origin/drop-wignerd-dep` remote-tracking
ref). It is also cut from ET **0.5.0**, which ACEpotentials' `[compat]` of
`"0.4.3"` excludes. So for this spike the same two-line change was applied to
the **v0.4.3 tag** instead: delete the `WignerD` line from `[deps]`, and delete
`D_from_angles`/`QD_from_angles` from `src/O3/O3_utils.jl` (they have no callers
in `src/`). That is the entirety of the src-side diff.

### The ETACE path re-measured on the same host

Same environment, 64-atom Si, `ace_model(order=3, max_level=10, maxl=6)`,
2148 edges, `maxneigs=34`. One process per component (`etace_one.jl`).

| component | traces? | agreement | note |
|---|---|---|---|
| `yembed` — SpheriCart solid harmonics via P4ML | **YES** | 1.16e-10 | re-confirmed on Reactant 0.2.285 |
| `rembed` minus `SelectLinL` — Agnesi + polys + envelope | **YES** | 1.14e-13 | re-confirmed on Reactant 0.2.285 |
| the #3267 reproducer inside this env | traces, **wrong** | 6.0 | still broken |
| `abasis` / `aabasis` via `ka_evaluate` | **not established** | — | see below |
| `rembed` complete, `site_basis`, `et_model(G, ps, st)` | **not measured** | — | host lost |

The two components previously found to trace still trace, on a CUDA host and
against current Reactant — so the old measurements were not environmental, and a
negative elsewhere is a real negative.

**`abasis` was not established, and the honest reason is that it took the machine
down.** The `abasis` case was the one running when `moriarty` stopped responding;
`sshd` never came back within the session and the remaining cases (`aabasis`,
`rembed`, `site_basis`, `site_basis_R`, `sitee`) were never run. That is
*consistent* with the same unbounded-allocation failure as B1b, but it is not
confirmed, and it must not be written down as if it were.

What *is* confirmed is the mechanism that would explain it, from ET's own source
(v0.4.3). Both ET kernels on the evaluation path index a **host array of
composite elements** once per work-item — the exact shape that blew up as B1b,
and not the shape that traced as B1/B2:

```julia
# ace/sparseprodpool_ka.jl — spec::Vector{NTuple{NB,Int}}
@kernel function _ka_evaluate_PooledSparseProduct_batched_v1!(A, BB, spec, nneig, ::Val{NB})
   iA, inode = @index(Global, NTuple)
   ϕ = spec[iA]                                   # tuple pulled out of a host array
   ...  b = ntuple(t -> BB[t][ineig, inode, ϕ[t]], NB)

# utils/selectlinl.jl — X is an array of PStates
@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
   iB, jB = @index(Global, NTuple)
   i_x = selector(X[iB])                          # field pulled out of a host array
```

My standalone replication of the pooled-product kernel (row B2) traces exactly
**because it flattens `spec` into two plain `Vector{Int}`s** — which is precisely
the change that would be needed. The distinction between B1 and B1b is the whole
result: same arithmetic, same ndrange, same host-vs-traced index array; only the
element type differs, and that alone is the difference between 4.44e-16 and a
dead machine.

### What actually blocks the standard ETACE path now

Not KernelAbstractions. The revised blocker list is:

1. **Composite-element host arrays passed into KA kernels.** `ET.SelectLinL`'s
   `X::Vector{<:PState}` + selector closure, and `PooledSparseProduct`'s
   `spec::Vector{NTuple{NB,Int}}`. Measured to blow up to 60 GB and be killed in
   the `SelectLinL` shape; strongly suspected but *not measured* for `spec`. The
   fix is the same in both cases and is mechanical: precompute plain integer
   arrays (a category index per edge; two or NB parallel `Vector{Int}`s for the
   spec) and index those. Both flattened forms are **measured to trace exactly**
   (4.44e-16 / 1.78e-15 / 0.0).
2. **Reactant #3267**, still live on this host, on both backends, on 0.2.285.
   Independent of anything ACE does; nothing traceable is safe until it is fixed
   or `optimize=:just_batch` is accepted.
3. **Unmeasured**: whether `rembed` complete, `site_basis` and the full
   `et_model` call trace once (1) is addressed. The previous "scalar indexing"
   failures for these were recorded *without* CUDA loaded, so they are not
   trustworthy as they stand and need re-running.

### How much work is left for single-source-of-truth?

Smaller than the pre-test worst case, larger than the best case.

- The feared outcome — "de-KA the entire ET evaluation path" — **is off the
  table**. KA kernels trace. That was the expensive branch and it is closed.
- What remains inside ET is a **data-layout change, not an architecture change**:
  stop handing kernels arrays of tuples and arrays of `PState`s, hand them
  integer arrays. For `SelectLinL` this is the rewrite already wanted for its own
  reasons (ET's own comments ask for it, and it drops a hand-written rrule). For
  `PooledSparseProduct` it is a spec-flattening at construction time. Neither
  touches the maths.
- What remains **unknown** is everything above those two layers: `site_basis`
  and the full model call have never been traced with CUDA loaded. It would be
  wrong to assume they fall out for free.
- #3267 is a hard external dependency and is not ours to fix.

So: one bounded, already-desired change inside ET; one unknown that a single
afternoon on a working GPU box would settle; one upstream bug out of our hands.

### Caveats on this session's measurements

- Every number above was taken on **`moriarty` (RTX A4500, compute 8.6)**.
  Nothing was measured on lestrade (RTX 4000 Ada) — after moriarty fell over I
  moved there to finish the ETACE cases, but `/home` is shared between the two
  and became unreadable from lestrade as well, so no further data was collected.
- The ETACE environment resolved **CUDA.jl 5.11.3**, not the 6.2.2 of the
  KA-only environment (Lux/MLDataDevices constrain it). `ReactantCUDAExt` loaded
  and `ka_with_reactant` was present in both, which is what matters here.
- Julia 1.12 was not tested. 1.11.7 only.

### Reproduction

```bash
ssh moriarty
cd ~/reactant_ka_cuda     # Project.toml: Reactant, CUDA, KernelAbstractions, StaticArrays

# the single decisive test -- trivial KA kernel, CUDA loaded
julia +1.11 --project=. ka_one.jl 1 cpu
julia +1.11 --project=. ka_one.jl 1 gpu

# the same kernel with CUDA NOT loaded -- the control that isolates the cause
julia +1.11 --project=. ka_nocuda.jl 1 cpu

# the whole ladder, one process per case, both backends
./ladder.sh                      # writes ladder.log

# the ETACE path (needs a dev'd ACEpotentials + patched ET)
cd ~/reactant_etace
./etace_ladder.sh cpu gpu        # RSS watchdog at 25 GiB -- keep it
```

Building the two environments from scratch:

```bash
# 1. KA-only env
mkdir reactant_ka_cuda && cd reactant_ka_cuda
cp <repo>/spike/reactant_phase0/Project_ka_cuda.toml Project.toml
cp <repo>/spike/reactant_phase0/{ka_one.jl,ka_cuda.jl,ka_nocuda.jl,ladder.sh} .
julia +1.11 --project=. -e 'using Pkg; Pkg.instantiate()'   # ~20 min, mostly CUDA artifacts

# 2. ETACE env: ET v0.4.3 with WignerD removed from [deps]
git clone https://github.com/ACEsuit/EquivariantTensors.jl EquivariantTensors
cd EquivariantTensors && git checkout v0.4.3
#   delete the WignerD line from [deps] in Project.toml, and delete
#   D_from_angles / QD_from_angles from src/O3/O3_utils.jl (no callers in src/)
cd .. && mkdir reactant_etace && cd reactant_etace
julia +1.11 --project=. -e 'using Pkg;
    Pkg.develop(path="../EquivariantTensors"); Pkg.develop(path="../ACEpotentials");
    Pkg.add(name="Reactant", version="0.2.285"); Pkg.add("CUDA"); Pkg.instantiate()'
cp <repo>/spike/reactant_phase0/{etace_one.jl,etace_ladder.sh} .
```

## Performance context

`ace_export.jl`'s `segment_sum` is a dense one-hot matmul:

```julia
segment_sum(rows, idx, n) = permutedims(idx .== permutedims(1:n)) * rows
```

At MAX_ATOMS 2560 x MAX_EDGES 40960 that is a 105M-entry matrix per call, which
explains the reported ~5x slowdown and memory blowup versus JAX's scatter. This
is a property of that implementation, not of Reactant, and a traceable standard
path would not have it.

## Recommendation

Proceed with the Stage 1 JAX port now: it is validated at 1.4e-15, and the
Reactant route currently miscompiles for l >= 2.

But do not write Reactant off, and the case for it is now stronger than when
this was written. The standard ETACE path is two known stopgaps away from
tracing, both inside EquivariantTensors, both array-expressible, and one already
flagged for revisiting in its own source. Crucially, the worry that the whole
ET evaluation path would have to be de-KA-ed has been **tested and refuted**:
KernelAbstractions kernels trace fine under Reactant on a CUDA host. If the
remaining stopgaps are fixed, Julia
gets an XLA/GPU path without a parallel Python implementation to maintain -- and
because lammps-jax consumes StableHLO, which Reactant also emits, potentially
the same LAMMPS pair style too.

Two things worth doing regardless of the port:
1. ~~File the miscompilation bug upstream~~ -- **done**, filed as
   [#3267](https://github.com/EnzymeAD/Reactant.jl/issues/3267). Searched
   EnzymeAD/Reactant.jl first: there is a cluster of "silent wrong result"
   reports (#3038, #3039, #2981, #3046, all closed; #2969 open) but none matched
   -- those are KA-kernel raising or complex-array gather, whereas this is plain
   broadcast plus `hcat` on Float64. #2846 ("Correctness and performance issue
   with Reactant + KA kernel code", open) is relevant to blocker 2, not to this.
   Watch #3267: if the offending pass is fixed, the Julia export route becomes
   worth re-timing.
2. ~~First check whether KA kernels trace on a CUDA host~~ -- **done**, and
   they do; see **KA on a CUDA host** above. The ET evaluation path does *not*
   need de-KA-ing. What it needs is narrower: stop passing kernels arrays of
   tuples and arrays of `PState`s. Do the `SelectLinL` rewrite (small, already
   wanted, drops a hand-written rrule, array formulation measured at 4.44e-16),
   and flatten `PooledSparseProduct`'s `spec::Vector{NTuple{NB,Int}}` into plain
   integer arrays at construction time. Then re-run `etace_ladder.sh`, which
   still has `rembed` complete, `site_basis` and the full model call unmeasured.
3. ~~Raise the ET -> WignerD -> StructArrays pin, which currently caps Reactant
   at 0.2.222 and blocks testing against current releases.~~ -- **done**, filed
   as [EquivariantTensors.jl#143](https://github.com/ACEsuit/EquivariantTensors.jl/pull/143).
   With the same change applied to the v0.4.3 tag, ACEpotentials `jax-eval`,
   ET and Reactant 0.2.285 were installed together and the ET-level components
   re-measured against current Reactant. The fix does what it was meant to do.

Performance was not measured. Benchmarking a miscompiling path is not
meaningful, and the two blockers mean the standard path cannot yet be timed at
all.

## Files

| file | purpose |
|---|---|
| `reactant_bug_repro.jl` | the two-line minimal reproducer |
| `ylm_conditioning.jl` | rejects hypothesis 1 |
| `mask_path_test.jl` | rejects hypothesis 2 |
| `reactant_bisect.jl` | stagewise compiled-vs-eager bisect |
| `pattern.jl` | the divergence pattern on a real model |
| `lsweep.jl`, `workaround.jl`, `minimal.jl` | trigger characterisation |
| `etace_traceable.jl` | does the standard ETACE path trace? |
| `trace_detail.jl` | backtraces locating the blockers |
| `confirm.jl` | confirms yembed and rembed-minus-SelectLinL trace |
| `_ace_setup.jl` | shared model/table/cluster setup |
| `ka_cuda.jl` | the whole KA ladder in one process, CUDA loaded, backend as ARGS[1] |
| `ka_one.jl` | **one** KA case per process, so an OOM-kill is attributable |
| `ka_nocuda.jl` | the control: same cases with CUDA deliberately *not* loaded |
| `ladder.sh` | drives `ka_one.jl` over every case x both backends |
| `etace_one.jl` | one ETACE component per process (yembed, abasis, site_basis, ...) |
| `etace_ladder.sh` | drives `etace_one.jl` **under a 25 GiB RSS watchdog** -- keep it |
| `Project_ka_cuda.toml` | the minimal Reactant+CUDA+KA environment for the ladder |
