# Reactant spike: why the Julia ACE export is wrong, and what it means

Investigating `lammps-jax` `dev/julia_export` `examples/julia/ace_export.jl`.
Julia 1.12.6, CPU (Apple Silicon). SPIKE CODE.

**Version note:** the miscompilation reproduces on **both Reactant 0.2.222 and
0.2.285** (the latest as of 2026-09-09), so it is not fixed by upgrading.
Re-confirmed on Reactant 0.2.285 / Reactant_jll 0.0.407 in clean
single-dependency environments across every Julia version Reactant tests:

| Julia | in Reactant CI? | result |
|---|---|---|
| 1.11.9 | yes (version sweep) | MISMATCH |
| 1.12.7 | yes (version sweep) | MISMATCH |
| 1.13.0 | **no** | MISMATCH |

Reactant's `Project.toml` says `julia = "1.10"`, an open upper bound, so the
resolver installs happily on 1.13 -- but its CI matrix is 1.10 (primary) plus a
1.11/1.12 sweep, with no 1.13 row. **Report this upstream against 1.11 or 1.12,
not 1.13**, or the version is an easy way to dismiss the bug. The 1.13 row is
included only to show the behaviour does not change there.

**EquivariantTensors cannot be installed alongside Reactant > 0.2.222.**
ET depends on WignerD, which pins StructArrays <= 0.6.21, while Reactant 0.2.285
requires StructArrays >= 0.7.2. So 0.2.222 is not an arbitrary choice -- it is
the newest Reactant that can coexist with ET at all.

**This looks cheap to fix.** WignerD is used in ET's `src/` at exactly two
places, `O3_utils.jl:173` and `:185`, inside `D_from_angles` /
`QD_from_angles` -- utilities that build a rotation Q and matching Wigner-D
matrix so equivariance can be checked as `y о Q = D * y`. They have **no callers
anywhere in `src/`**, are not exported, and are used only by
`test/O3/test_O3_transforms.jl`. WignerD is also already listed in ET's test
target. Moving those two functions into the test suite, or behind a weakdep
extension, would drop the dependency and lift the Reactant ceiling from 0.2.222
to current -- a one-PR change that unblocks testing ETACE against current
Reactant.

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
`reactant_bug_repro.jl`. Worth filing upstream.

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

**Caveat, untested:** this machine has no CUDA. Reactant's KA extension may
require CUDA to be loaded even for the CPU backend -- issue #3038's environment
lists "CUDA 6.2.0 ... CPU backend (CUDA loaded for ReactantCUDAExt)". If that is
the explanation, blocker 2 may largely evaporate on a CUDA host and only
`SelectLinL` would need changing. This is worth testing on a GPU box before
investing in a rewrite.

**Notable side finding:** SpheriCart traces fine. `ace_export.jl` replaces Ylm
with fitted monomials on the premise that SpheriCart is untraceable; on this
evidence that substitution was unnecessary — and it is the substitution that
exposed the miscompilation bug above.

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

But do not write Reactant off. The standard ETACE path is two known stopgaps
away from tracing, both inside EquivariantTensors, both array-expressible, and
one already flagged for revisiting in its own source. If those are fixed, Julia
gets an XLA/GPU path without a parallel Python implementation to maintain -- and
because lammps-jax consumes StableHLO, which Reactant also emits, potentially
the same LAMMPS pair style too.

Two things worth doing regardless of the port:
1. File the miscompilation bug upstream (`reactant_bug_repro.jl`). Searched
   EnzymeAD/Reactant.jl first: there is a cluster of "silent wrong result"
   reports (#3038, #3039, #2981, #3046, all closed; #2969 open) but none matches
   -- those are KA-kernel raising or complex-array gather, whereas this is plain
   broadcast plus `hcat` on Float64. #2846 ("Correctness and performance issue
   with Reactant + KA kernel code", open) is relevant to blocker 2, not to this.
2. Try the `SelectLinL` rewrite. It is small, already wanted for other reasons
   (it would drop a hand-written rrule), and the array-op formulation is
   *measured* to trace exactly (4.44e-16). But first check whether KA kernels
   trace on a CUDA host -- if they do not, the whole ET evaluation path needs
   de-KA-ing, not just this one layer, which is a much larger commitment.
3. Raise the ET -> WignerD -> StructArrays pin, which currently caps Reactant at
   0.2.222 and blocks testing against current releases.

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
