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

## Not done yet: GPU throughput

The basis-size result above says what the model costs to *build*. The throughput
comparison — embedded ACE at many elements against MACE through the same
`pair_style jax/kk`, reusing the Phase 13 harness in `bench/phase13/` — needs a
GPU host and has not been run. Export with:

```
ACE_ELEMENTS=<...> ACE_ORDER=3 ACE_TOTALDEGREE=6 ACE_NOFIT=1 \
ACE_EMBEDDING=<embedding.json> ACE_DMAX=16 \
julia --project=acejax/julia acejax/julia/export_model.jl out.npz embedding
```
