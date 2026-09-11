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

## GPU throughput (moriarty, RTX A4500) — MEASURED, and it does not yet follow

64-atom cell, energy+forces, f64, `ACE_NOFIT=1` models:

| S | n_B | descriptor len | npz MB | ms/eval | atom-steps/s |
|---|---|---|---|---|---|
| 2 | 77 | 166 | 0.5 | 0.327 | 1.96e5 |
| 10 | 364 | 3 700 | 21.1 | 1.656 | 3.87e4 |
| 20 | 400 | 8 120 | 80.1 | 2.974 | 2.15e4 |

**Throughput falls 9x from S=2 to S=20 even though `n_B` saturates (77 -> 400,
and only 1.1x from S=10 to S=20).** So the basis-size saturation above does
**not** currently translate into S-independent cost, and the many-element
throughput claim is NOT demonstrated.

The cause is the export, not the method. `rnl_spline_coefs` has shape
`(NZ, NZ, 102, n_rnl)` — **O(S^2)** — and is 95% of every file: 896 MB of the
936 MB at S=75. Timing tracks that table (npz 21 -> 80 MB, 3.8x) far better than
it tracks `n_B` (1.1x), i.e. these evaluations are memory-bandwidth bound on it.

**And with a frozen embedding that table is entirely redundant.** Since
`R(n'k)l(r, Z1, Z2) = P_n'(r) * emb[Z2, k]`, the radial *shape* does not depend
on the species pair: all `S^2` blocks are the same `n_rnl` splines scaled by
embedding values. Storing one spline table plus the `(S, d)` embedding is O(1)
in S — about 0.2 MB instead of 896 MB at S=75, a ~4500x reduction — and should
restore near-S-independent evaluation.

**That fix is the prerequisite for a meaningful many-element throughput number,
and for the comparison against MACE.** Until it lands, these figures measure the
exporter's redundancy rather than the method, and S=40/75 were not run at all:
at 296 MB and 937 MB they would measure it even more thoroughly.

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
