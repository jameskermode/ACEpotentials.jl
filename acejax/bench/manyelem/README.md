# Many-element scaling: categorical species vs a frozen element embedding

**No fitting.** Basis size and throughput depend on the basis, not on the
weights, and there is no many-element dataset here to fit against. Accuracy is a
separate and still-open question — see the Stage 1E section of
`docs/plans/jax_ace_port_plan.md`, which measures the embedding as 16–20% worse
on forces at matched `n_B` at **two** elements.

## What is established (basis size, order 3, degree 6, CPU)

`EMB=<embedding.json> julia --project=acejax/julia bench/manyelem/basis_scaling.jl`

| S | categorical `n_B` | embedded `n_B` (`d<=16`) | widths |
|---|---|---|---|
| 2 | 123 | 77 | [2, 3, 4] |
| 5 | 1 348 | 323 | [5, 15, 16] |
| 10 | 9 465 | 364 | [10, 16, 16] |
| 20 | 70 755 | 400 | [16, 16, 16] |

**Categorical grows 575x from S=2 to S=20; the embedded basis saturates at ~400**
once every per-order width hits the `d_max` cap. That saturation is the whole
point: with `d_max` fixed the basis size stops depending on the number of
elements, which is the property MACE has and categorical ACE does not.

Note this requires the **capped** (lossy) regime. With lossless widths
`d_nu = C(S+nu-1, nu)` the widths — and so `n_B` — keep growing with S, so an
S-independent cost is bought by truncation, not for free.

The categorical sweep is capped at `SMAX_CAT` (default 20). Beyond that it is
not a benchmark but an OOM risk; the growth rate is established well before it.

## Element pool

Taken from `data/length_scales_VASP_auto_length_scales.yaml` intersected with the
embedding table's rows — **75 elements**. Picking elements outside that set gives
`UndefVarError: rnn not defined in DefaultHypers`, which looks like a scaling
failure and is not one. This cost a false "categorical is unbuildable at S=40"
before it was caught.

## GPU throughput (moriarty, RTX A4500) — FIXED, and now flat in S

64-atom cell, energy+forces, f64, `ACE_NOFIT=1`, `d_max = 16`:

| S | n_B | npz MB | ms/eval | atom-steps/s |
|---|---|---|---|---|
| 2 | 77 | 0.3 | 0.282 | 2.27e5 |
| 10 | 364 | 2.8 | 0.952 | 6.72e4 |
| 20 | 400 | 7.0 | 1.894 | **3.38e4** |
| 40 | 400 | 17.4 | 2.074 | **3.09e4** |
| 75 | 400 | 40.5 | 1.992 | **3.21e4** |

**Throughput is flat from S=20 to S=75** — 3.38e4 / 3.09e4 / 3.21e4 atom-steps/s
for 20, 40 and 75 elements. That is the S-independent cost the basis-size
saturation predicted, and it is what categorical ACE cannot do at any price.
The remaining rise from S=2 to S=20 tracks `n_B` (77 -> 400) while the per-order
widths climb to the `d_max` cap; once they saturate, so does the cost.

### What was wrong, and the correction to the first diagnosis

The first measurement had throughput falling **9x** from S=2 to S=20. The cause
was `rnl_spline_coefs`, shape `(NZ, NZ, ncoef, n_rnl)` — O(S^2) to store and,
worse, an `(E, ncoef, n_rnl)` gather per edge.

The first diagnosis said that table was "entirely redundant" with a frozen
embedding. **That was wrong.** `_default_rin0cuts` derives rin/r0/rcut from
per-PAIR bond lengths, so the radial *shape* genuinely differs per pair —
measured, **39 distinct transforms across 100 pairs** at S=10. The table only
factorises if the cutoffs are uniform.

So the fix is a **modelling** choice plus a storage one:

1. `ace_embedding_model(...; uniform_cutoffs = true)` — one transform for all
   pairs, as MACE does. This gives up per-pair bond-length adaptation, which is
   exactly the thing that does not scale to many elements.
2. The exporter then *detects* the factorisation numerically (residual 5.9e-16;
   it falls back to the dense table silently if it does not hold) and stores
   `(ncoef, n1) + (NZ, d)` instead of `(NZ, NZ, ncoef, n_rnl)`.
3. `acejax` evaluates `Rnl[e,i] = P[e, n'(i)] * emb[zj[e], k(i)]`, with **no
   species gather over the spline table at all**.

Effect at S=75: the radial table drops from **896 MB to 0.01 MB**, and the model
becomes evaluable at all — S=40 and S=75 could not previously be run.

Remaining file size is now dominated by `test_desc`, the exported *test fixture*
(n_B x NZ x n_atoms = 14.6 MB at S=75), not by the model.

## Superseded: earlier note that GPU throughput was not done

The basis-size result above says what the model costs to *build*. The throughput
comparison — embedded ACE at many elements against MACE through the same
`pair_style jax/kk`, reusing the Phase 13 harness in `bench/phase13/` — needs a
GPU host and has not been run. Export with:

```
ACE_ELEMENTS=<...> ACE_ORDER=3 ACE_TOTALDEGREE=6 ACE_NOFIT=1 \
ACE_EMBEDDING=<embedding.json> ACE_DMAX=16 \
julia --project=acejax/julia acejax/julia/export_model.jl out.npz embedding
```


## Head-to-head against MACE — what it actually measured

moriarty, RTX A4500, 64-atom cell with 64 distinct elements, f64. Timed **by
difference**, `t(1000 frames) - t(1 frame)`, so compile, model load and file I/O
cancel rather than being estimated. (A first attempt at 33 frames gave a
*negative* per-frame time — the ~38 s fixed cost swamped the variable part.
1000 frames puts the variable part above the noise.)

| what | per frame | atom-steps/s |
|---|---|---|
| ACE S=75, **kernel only** (edge list supplied) | **1.99 ms** | 3.21e4 |
| ACE S=20, kernel + per-frame Python neighbour list | 64.5 ms | 9.9e2 |
| ACE S=75, kernel + per-frame Python neighbour list | 75.6 ms | 8.5e2 |
| MACE-MP-0 small, `mace_jax` CLI end-to-end | 16.9 ms | 3.79e3 |

### CORRECTED

The first version of this benchmark did not pad the edge list. Edge counts vary
frame to frame, so **every frame was a new shape and jax retraced on every one**
— the numbers below measured recompilation, not evaluation. Padding to a fixed
capacity, which is the whole reason the export contract uses one, cuts the ACE
per-frame cost by ~4x:

| what | per frame | atom-steps/s |
|---|---|---|
| ACE S=20, padded, matscipy nlist | **18.1 ms** | 3.54e3 |
| ACE S=75, padded, matscipy nlist | **17.3 ms** | 3.71e3 |
| MACE-MP-0 small, CLI end-to-end | 16.9 ms | 3.79e3 |
| *(unpadded, retracing every frame)* | *64-76 ms* | *~9e2* |
| ACE S=75, kernel only, edge list supplied | 1.99 ms | 3.21e4 |

**ACE and MACE are level on this workload** — 17.3 ms against 16.9 ms for a
64-atom, 64-element cell, both end-to-end including their own neighbour lists.
ACE's kernel is 1.99 ms of its 17.3, so the remaining ~15 ms is neighbour list,
host-to-device transfer and Python overhead; MACE's split was not isolated.

A hypothesis that the numpy fallback was to blame is **refuted**:
`acejax.nlist.backend()` reports `matscipy` in that venv. The cost was
retracing, not the neighbour list. (`backend()` is now printed by the harness —
not recording it is what let the wrong explanation stand.)

Environment: `/storage/eng/essswb/macejax-gpu/` (recipe in `macejax/`), MACE
bundle from `/storage/eng/essswb/phase13/mace-mp-0-small-jax` — the bundle the
new recipe wrote lacks `config.json` and the CLI cannot load it, so Phase 13's
complete bundle was reused.
