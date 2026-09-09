# Reactant spike: why the Julia ACE export is wrong, and what it means

Investigating `lammps-jax` `dev/julia_export` `examples/julia/ace_export.jl`.
Reactant 0.2.222, Julia 1.12.6, CPU (Apple Silicon). SPIKE CODE.

## Headline

The ACE export failure is a **silent Reactant miscompilation**, reduced to a
two-line reproducer. Eager Julia is correct; the compiled program is wrong and
raises nothing.

```julia
f(u) = hcat(u[:, 3] .* u[:, 3], u[:, 2] .* u[:, 3])
u = [1.0 2.0 3.0; 4.0 5.0 6.0]
f(u)                                    # [9.0 6.0; 36.0 30.0]   correct
Array((@compile f(ru))(ru))             # [9.0 9.0; 36.0 36.0]   WRONG
```

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

## What this says about Reactant for ACE

**Reactant does not save the rewrite.** `ace_export.jl` is a 234-line
hand-written array reimplementation of ACE. It never traces ACEpotentials'
kernels; it substitutes:

- **radials** -> the model's splines re-tabulated on a 32768-point grid, then
  cubic Hermite interpolation (its own comment records a 6e-11 force floor)
- **Ylm** -> monomials + a least-squares map, because SpheriCart is untraceable

So the array reformulation is the same intellectual work the JAX port needs;
only the host language differs.

**Its `segment_sum` is a dense one-hot matmul:**

```julia
segment_sum(rows, idx, n) = permutedims(idx .== permutedims(1:n)) * rows
```

At MAX_ATOMS 2560 x MAX_EDGES 40960 that is a 105M-entry matrix per call. This
is an implementation choice, not a Reactant limitation, but it explains the
reported ~5x slowdown and memory blowup versus JAX's scatter.

**Accuracy comparison.** Our Stage 1 exports the model's own splines and uses
sphericart directly: measured 1.4e-15 against Julia. `ace_export.jl` stacks a
Hermite re-tabulation on top of the already-splined radials (6e-11) plus a
fitted monomial Ylm. Our path is strictly better and needs no 32768-point table.

## Recommendation

Proceed with the Stage 1 JAX port. Reactant is not ready for this workload:
correctness is broken for l >= 2 and the reference implementation is not
performance-competitive. The bug is narrow and worth reporting upstream --
it may well be fixed quickly, at which point Reactant becomes worth
re-evaluating for the Julia GPU story (roadmap Phase 4), independently of
whether the JAX port proceeds.

Performance was not measured: benchmarking a miscompiling path is not
meaningful, and the dense `segment_sum` makes the reference non-competitive
regardless.

## Files

| file | purpose |
|---|---|
| `reactant_bug_repro.jl` | the two-line minimal reproducer |
| `ylm_conditioning.jl` | rejects hypothesis 1 |
| `mask_path_test.jl` | rejects hypothesis 2 |
| `reactant_bisect.jl` | stagewise compiled-vs-eager bisect |
| `pattern.jl` | the divergence pattern on a real model |
| `lsweep.jl`, `workaround.jl`, `minimal.jl` | trigger characterisation |
| `_ace_setup.jl` | shared model/table/cluster setup |
