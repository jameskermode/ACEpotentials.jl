# Frozen MACE element embeddings in ACEpotentials: one-day spike

**Branch** `jax-eval` · **Date** 2026-09-11 · **Machine** Apple Silicon Mac, CPU only
**Julia** 1.12.6 · **ACEpotentials** 0.10.2 (repo `a82a7a92`) · **EquivariantTensors** 0.4.3

**Model parameters used throughout**: `totaldegree = 8`, `wL = 1.5`, ACE1 degree
convention. **Correlation order 3 unless stated otherwise** — sections Q1-Q5 and
N1-N3 are all order 3; orders 2 and 4 are measured in
[Correlation-order dependence](#correlation-order-dependence-orders-2-3-and-4).

Tests the design in `docs/plans/jax_ace_port_plan.md`, "Long-term directions …",
subsection (a): replace ACE's categorical species index with a **frozen** element
embedding imported from a MACE foundation model, folded into the radial index

```
R'[(k,n), l](r_ij, z_j)  =  emb[z_j, k] * R[n, l](r_ij)
```

so that `abasis`, `aabasis` and `A2B` see only a wider radial basis.

---

## Executive summary

The design **works and is cheaper to build than the plan assumed** — but it is a
*different and smaller* model class than the plan's framing suggests, and the
economics are worse for small element counts than hoped.

| # | Question | Answer |
|---|----------|--------|
| 1 | Can ET express the channel-diagonal AA restriction cleanly? | **Yes, trivially** — it is a property of `mb_spec`, which ET takes as an explicit list. Zero ET changes needed. **But** the spike hit a **latent ET correctness bug** on the way (below), which must be fixed first. |
| 2 | What does it cost to build a spec over `d·n_rnl` radial functions? | **Essentially nothing** if enumerated channel-diagonally: 0.001 s and `n_B = 54·d` **exactly**, linear in `d` out to `d=128`. Generate-then-filter costs `O(d^ν)` (16.3 s at d=32) and must be avoided. |
| 3 | Where is the crossover? | **Confirmed, and it moves strongly with correlation order.** **At order 3** / degree 8, embedding-`d` breaks even with categorical-`S` at `d* = n_B(S)/54`: `d*` = 5.7 (S=2), 17.0 (S=3), 70.8 (S=5), 522 (S=10); with MACE-MP-0 small's full `d=128` the embedding loses up to S=6 and wins from S=7. **Measured at orders 2 and 4 as well**: at `d=128` the crossover sits between S=12 and 13 (order 2), S=6 and 7 (order 3), S=4 and 5 (order 4). |
| 4 | How small can `d` be? | **The SVD does not decay, and it is the wrong measurement anyway.** Restricted to an S-element model the table has rank exactly S; `d < S` destroys element distinguishability and `d > S` provably makes the basis **linearly dependent** (measured, exactly as predicted). The plan's hoped-for "4 or 8 leading components carry most of it" is **refuted**. |
| 5 | Does it actually work? | **Yes.** A real model built from the real MACE-MP-0-small table is rotation-invariant, permutation-invariant and exactly linear in the coefficients, all to ~1e-15. |

**Recommendation: the Julia route is viable, but needs modification** — see
[Recommendation](#recommendation). Do not abandon it for the JAX fallback.

---

## Negative results, stated first

### N1. A latent correctness bug in EquivariantTensors blocks this feature today

`EquivariantTensors.sparse_equivariant_tensor` (the **singular** one, which
`ACEpotentials.Models._generate_ace_model` calls) **silently returns a
non-equivariant basis** when `mb_spec` is not already grouped by correlation
order.

Mechanism (source-read, then confirmed by measurement):

* `sparse_ace_utils.jl:83` builds `𝔸spec` in symmetrisation order and passes it
  straight to `SparseSymmProd`.
* `sparsesymmprod.jl:38-41`:
  ```julia
  function SparseSymmProd(spec; kwargs...)
     if !issorted(spec, by=length)
        spec = sort(spec, by=length)      # <-- silently reorders
     end
  ```
  It re-sorts by correlation order and returns `AA` in that layout.
* The `A2Bmaps` columns are **not** permuted to match. Columns and `AA` entries
  are then misaligned, and `B = A2B * AA` is garbage — but garbage of the right
  *shape*, so nothing errors.

The **plural** `sparse_equivariant_tensors` avoids this at `sparse_ace_utils.jl:22`
with `𝔸spec = sort(union(𝔸specs...), by = bb -> (length(bb), bb))` — the sort its
own comment calls "very hacky and brittle". That comment is correct; the singular
path is missing the same guard.

Measured (`spike_et_order_bug.jl`, below), relative rotation error of the site basis:

| spec | n_B | rot. error, `mb_spec` **not** order-grouped | rot. error, order-grouped |
|---|---|---|---|
| d=2 channel-diagonal | 108 | **6.82e-01** | 5.22e-16 |
| d=3 channel-diagonal | 162 | **6.68e-01** | 3.89e-16 |
| d=4 channel-diagonal | 216 | **8.64e-01** | 5.41e-16 |
| single channel (d=1), order-grouping deliberately scrambled | 54 | **3.97e-02** | 3.33e-16 |

The last row is the important one: this is **not** about embeddings. Any caller
that hands `sparse_equivariant_tensor` a spec in a different order gets a wrong
basis. ACEpotentials is safe today only by accident — `Models.sparse_AA_spec`
emits its spec grouped by correlation order (measured: `issorted(length.(mb1)) == true`),
because `gensparse` loops `NU` outermost.

*Measured on ET 0.4.3.* `~/.julia/dev/EquivariantTensors` (0.5.0, incl. PR #143)
has a **byte-identical** `src/ace/sparse_ace_utils.jl`, so the defect is
present there too (inferred by file diff, not re-measured).

**Suggested upstream fix:** in `sparse_equivariant_tensor`, sort `𝔸spec` by
`(length(bb), bb)` and permute `symm`'s columns to match, exactly as the plural
version does. A cheap belt-and-braces addition: have `SparseSymmProd` *reject* an
unsorted spec rather than silently reordering it.

### N2. The embedding table has no low-rank structure to exploit

SVD of the extracted MACE-MP-0-small table (89 x 128), and of the rows for
realistic element subsets:

| matrix | shape | rank | cum. energy @4 | @8 | @16 | @32 | @64 |
|---|---|---|---|---|---|---|---|
| all 89 elements | 89x128 | 89 | 0.485 | 0.621 | 0.751 | 0.882 | 0.984 |
| Si,C | 2x128 | 2 | 1.000 | – | – | – | – |
| Si,C,O | 3x128 | 3 | 1.000 | – | – | – | – |
| Si,C,O,H,N | 5x128 | 5 | 0.924 | – | – | – | – |
| 10-element set | 10x128 | 10 | 0.675 | 0.925 | – | – | – |

Singular values of the full table: 15.02, 9.54, 7.72, 6.40, …, 0.32 — a slow,
featureless decay, needing 64 components for 98% of the energy. Row norms
1.14–4.59; mean |cos| between distinct element rows 0.179. This is close to what
an arbitrary near-incoherent set of 89 vectors in 128 dimensions looks like —
unsurprising, since a first layer applied to a one-hot is defined only up to an
invertible transform that downstream layers absorb. **There is no "chemical"
low-rank structure in this table.**

### N3. Channel-diagonal + frozen embedding is a strictly smaller model class

This is the framing correction that matters most, and it is not in the plan.

ACEpotentials **already** folds neighbour species into the radial index. It is
written down explicitly in `src/models/Rnl_learnable.jl:79-99`
(`set_onehot_weights!`):

```
# Rnl(r, Z1, Z2) = ∑_q W[(nl), q, Z1, Z2] * P_q(r)
# For linear models this becomes R(n'z')l(r, Z1, Z2) = Pn'(r) * δ_{z',Z2}
# n    | 1    2    3    4    5    6    7    8    ...
# n'z' | 1,1  1,2  1,3  2,1  2,2  2,3  3,1  3,2  ...
```

and in `src/ace1_compat.jl:104,227`, where the degree is `TotalDegree(1.0*NZ, 1/wL)`
so that the composite index `n = (n'-1)·NZ + z'` runs to `NZ·maxdeg`. That is the
**source of the measured 308 -> 917 growth** — not species appearing in the spec as
a separate label.

So the proposed change is exactly: replace the one-hot `δ_{z',Z2}` by a general
`emb[Z2, k]`, **and additionally restrict the AA product to channel-diagonal
terms**. The first half is a reparameterisation; the second half is a
**restriction**. Today's model has *full* channel mixing.

Consequence, verified numerically: for a fixed `(nn,ll)` block at correlation
order ν, the categorical scheme spans the full symmetric power `Sym^ν(R^S)` of
dimension `C(S+ν-1, ν)`, while channel-diagonal-`d` spans only
`span{ v_k^⊗ν : k=1..d }`, of dimension `min(d, C(S+ν-1,ν))`. Measured rank of
that span for the real MACE rows:

| S | ν | `C(S+ν-1,ν)` | d=2 | d=4 | d=8 | d=16 | d=32 | d=128 | smallest d for full rank |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| 2 | 2 | 3 | 2 | 3 | 3 | 3 | 3 | 3 | 3 |
| 2 | 3 | 4 | 2 | 4 | 4 | 4 | 4 | 4 | 4 |
| 3 | 3 | 10 | 2 | 4 | 8 | 10 | 10 | 10 | 10 |
| 5 | 3 | 35 | 2 | 4 | 8 | 16 | 32 | 35 | 35 |
| 10 | 3 | 220 | 2 | 4 | 8 | 16 | 32 | **128** | **> 128** |

i.e. **rank = min(d, dim)** exactly — each channel buys exactly one independent
species-interaction pattern. The embedding scheme is therefore a **rank-`d`
truncation of the ν-fold species tensor**. For S=10, ν=3, MACE-MP-0 small's full
128 channels still fall short of the 220 the categorical basis carries.

Note also that the singular spectrum *of that span* (the quantity that actually
governs how small `d` can be) is also smooth — for S=10, ν=3 the cumulative
energy is 0.43 at d=4, 0.59 at d=8, 0.89 at d=32. No knee.

**This does not break convexity** — with `emb` frozen the model stays linear in
the coefficients, and Q5 confirms it. But it is a modelling decision with an
accuracy cost that this spike cannot quantify (no fit was run), and it should be
presented as one, not as a free reparameterisation.

---

## Q1. Can EquivariantTensors express the channel-diagonal restriction?

**Yes, cleanly, with no upstream change** (once N1 is fixed).

`sparse_equivariant_tensor` takes `mb_spec` as an explicit `Vector{Vector{(n,l)}}`.
`symmetrisation_matrix` (`src/utils/symmop.jl:22-101`) iterates over
`unique(_vecnt2nnll.(mb_spec))` and calls `O3.coupling_coeffs(L, ll, nn; …)`;
`nn` is carried through untouched and used only for permutation-invariance
bookkeeping. **The coupling coefficients depend only on `ll` and `L`.** So which
`nn` tuples exist is entirely the caller's choice, and "all ν factors share the
same `k`" is a one-line property of the enumeration:

```julia
mb_spec = [ [ (n = (b.n-1)*d + k, l = b.l) for b in bb ]   # composite n = (n',k)
            for k = 1:d for bb in mb_single_channel ]
```

Confirmation that ET really does treat the composite index as opaque: the
resulting basis size is `n_B = 54·d` **exactly** for every `d` tested (1…128) —
`d` verbatim copies of the single-species basis, which is what channel-diagonal
means.

## Q2. Cost of building a spec over `d·n_rnl` radial functions

Order 3, total degree 8, `wL=1.5`.

**Generate-then-filter** (build the full `d`-wide AA spec, then drop
non-diagonal terms) — *do not do this*:

| d | n_rnl | n_mb | n_B | t_spec [s] | t_symm [s] |
|---|---|---|---|---|---|
| 2 | 51 | 186 | 131 | 0.003 | 0.002 |
| 4 | 102 | 407 | 283 | 0.025 | 0.004 |
| 8 | 204 | 837 | 581 | 0.213 | 0.007 |
| 16 | 408 | 1709 | 1183 | 1.993 | 0.015 |
| 32 | 816 | 3441 | 2381 | **16.324** | 0.035 |

For comparison, the *full-mixing* spec at d=8 costs `t_spec = 0.272 s` — i.e. the
filter saves basis size but **not** enumeration time, because the `O(d^ν)`
enumeration has already happened. (These runs also use the ACE1 degree
convention `n/d + l·wL`, which makes channels inequivalent and inflates `n_B`
above `54·d`.)

**Direct channel-diagonal enumeration** (enumerate the single-channel spec once,
replicate it into each channel; channel-free level `n' + l·wL`) — *do this*:

| d | n_rnl | n_mb | n_B | n_B/d | t_spec [s] | t_symm [s] |
|---|---|---|---|---|---|---|
| 1 | 24 | 73 | 54 | 54.0 | 0.001 | 0.001 |
| 2 | 48 | 146 | 108 | 54.0 | 0.001 | 0.002 |
| 4 | 96 | 292 | 216 | 54.0 | 0.000 | 0.002 |
| 8 | 192 | 584 | 432 | 54.0 | 0.000 | 0.005 |
| 16 | 384 | 1168 | 864 | 54.0 | 0.000 | 0.010 |
| 32 | 768 | 2336 | 1728 | 54.0 | 0.000 | 0.020 |
| 64 | 1536 | 4672 | 3456 | 54.0 | 0.001 | 0.040 |
| 128 | 3072 | 9344 | 6912 | 54.0 | 0.001 | 0.082 |

`n_B = 54·d` exactly; spec generation is free; symmetrisation is linear in `d`
(0.082 s at d=128). **This removes the plan's second "unverified" worry
entirely — provided the enumeration, not a post-hoc filter, is what changes.**

## Q3. Crossover (correlation order 3)

Categorical baseline **validated against the shipped constructor**:
`ace1_model(elements=…, order=3, totaldegree=8, rcut=5.5)` gives
`size(WB) = (54,1) / (308,2) / (917,3)` for `[Si] / [Si,C] / [Si,C,O]`,
reproducing the plan's 308 -> 917 exactly; the standalone harness used for the
table below reproduces 54 / 308 / 917 from the same recipe.

Since embedding `n_B = 54·d` exactly, the break-even channel width is
`d* = n_B(cat,S)/54`. (Total fitted parameters are `n_B·S` in both schemes, so
the ratios below are also the parameter-count ratios.)

| S | n_B categorical | **d\*** (break-even) | cost ratio at d=8 | at d=32 | at **d=128** |
|---|---|---|---|---|---|
| 1 | 54 | 1.0 | 0.12x | 0.03x | 0.01x |
| 2 | 308 | **5.7** | 0.71x | 0.18x | 0.04x |
| 3 | 917 | **17.0** | 2.12x | 0.53x | 0.13x |
| 4 | 2037 | 37.7 | 4.72x | 1.18x | 0.29x |
| 5 | 3824 | **70.8** | 8.85x | 2.21x | 0.55x |
| 6 | 6434 | 119.1 | 14.89x | 3.72x | 0.93x |
| 7 | 10022 | 185.6 | 23.20x | 5.80x | 1.45x |
| 8 | 14745 | 273.1 | 34.13x | 8.53x | 2.13x |
| 9 | 20758 | 384.4 | 48.05x | 12.01x | 3.00x |
| 10 | 28217 | **522.5** | 65.32x | 16.33x | 4.08x |
| 11 | 37278 | 690.3 | 86.29x | 21.57x | 5.39x |
| 12 | 48097 | 890.7 | 111.34x | 27.83x | 6.96x |

(cost ratio > 1 = embedding wins by that factor; < 1 = embedding is that much
*worse*.) Categorical growth beyond the table, same recipe: S=15 -> 92 658,
S=20 -> 216 626.

**The plan's claim is confirmed**: embeddings lose badly for two elements and win
for many. Quantitatively, with MACE-MP-0 small used at its native `d = 128`, the
**crossover is at S ≈ 6-7** (S=6: 0.93x, still a small loss; S=7: 1.45x).

Read together with N3, the honest summary is: `d` is a truncation rank, the
saving is `C(S+ν-1,ν)/d` per correlation order, and there is no free lunch below
`d = S` because the ν=1 (two-body) part needs `d ≥ S` just to tell the elements
apart.

## Q4. How small can `d` be? (correlation order 3)

**Smaller than S is lossy; larger than S is provably wasteful; the SVD of the
table does not tell you where to stop.** See N2 for the spectrum (no decay) and
N3 for why the table's own SVD is the wrong diagnostic: restricted to an
S-element model it has rank exactly S by construction.

The concrete cost of `d > S` was measured directly, as the numerical rank of the
per-centre basis block over random configurations (Si,C,O, so S=3, order 3,
degree 8, 3·n_B random environments, real MACE-MP-0-small channels 1..d):

| d | n_B | numerical rank | deficiency | predicted deficiency |
|---|---|---|---|---|
| 1 | 54 | 54 | 0 | 0 |
| 2 | 108 | 108 | 0 | 0 |
| 3 | 162 | 162 | 0 | 0 |
| 4 | 216 | **208** | 8 | 8 = (4-3)x8 |
| 6 | 324 | **300** | 24 | 24 = (6-3)x8 |

The prediction is `Σ_ν n_B^{(ν)} · (d − min(d, C(S+ν−1,ν)))`, with 8 order-1
basis functions per channel and `C(3+ν−1,ν)` = 3, 6, 10 for ν = 1, 2, 3. It
matches exactly. A rank-deficient design matrix is not fatal for a regularised
linear fit, but it is wasted width and wasted conditioning.

**Practical guidance: `d` should be chosen in `[S, C(S+ν-1,ν)]`**, and the
honest answer to "how small can it be" is that it is an empirical fitting
question this spike cannot settle — it was not measured, because no fit was run.

## Q5. Does it actually work? (correlation order 3)

Yes. Built with the **real** MACE-MP-0-small table, elements `[Si, C, O]`,
order 3, total degree 8, `Winit` replaced by `set_embedding_weights!` — the exact
embedding analogue of `Models.set_onehot_weights!`:

```
Wnlq[i_nl, n', iz, jz] = emb[jz, k]     where (n', k) is the composite index n
```

| d | n_B | rot. inv. (E) | rot. inv. (B) | perm. inv. | linearity | sees species? | rank(B) |
|---|---|---|---|---|---|---|---|
| 1 | 54 | 6.08e-15 | 3.33e-15 | 8.75e-16 | 5.54e-16 | yes | 54/54 |
| 2 | 108 | 2.18e-15 | 3.02e-15 | 2.02e-15 | 4.05e-16 | yes | 108/108 |
| 3 | 162 | 1.40e-15 | 1.30e-15 | 4.12e-16 | 1.91e-16 | yes | 162/162 |
| 4 | 216 | 1.58e-15 | 1.24e-15 | 8.00e-16 | 7.40e-16 | yes | 208/216 |
| 6 | 324 | 2.57e-15 | 1.91e-15 | 5.46e-16 | 2.84e-16 | yes | 300/324 |

"linearity" is `|(E − E_ref) − dot(B, c)|` with `c = get_basis_params(model, ps)` —
i.e. the model is exactly linear in the fitted coefficients, so `acefit!`, BLR,
committees and the smoothness priors apply unchanged. **The convexity property
that motivates the whole feature survives.**

Control: `ace1_model([Si,C,O], order=3, totaldegree=8)` gives rot(B) = 6e-16 …
2e-15 under the identical test, confirming the harness.

Not tested: forces/virials, a real fit, accuracy, or serialisation.

---

## Provenance of the embedding table

* **Checkpoint**: MACE-MP-0 **small**,
  `https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model`
  sha256 `2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736`
* **Layer**: `node_embedding.linear.weight` (`LinearNodeEmbeddingBlock`, an e3nn
  `Linear` mapping `89 x 0e -> 128 x 0e`), stored flat with length 11 392,
  reshaped `(89, 128)` row-major in (element, channel).
* **Normalisation**: multiplied by the e3nn path weight `1/sqrt(89) = 0.1060`, so
  the stored table is what the network actually applies to the one-hot.
* **Row order**: the checkpoint's own `atomic_numbers` buffer (89 entries,
  Z = 1…94), saved as `Z` in the npz.
* `torch` 2.14.0 was used **only** for the one-off extraction, in a throwaway
  virtualenv. `mace` and `e3nn` were *not* installed — `scripts/extract_mace_embedding.py`
  unpickles the checkpoint with a stub class for anything it cannot import.
  Nothing Python is on any ACEpotentials load or evaluation path.
* The `.npz` is kept strictly numeric (`emb`, `Z`,
  `e3nn_path_normalisation`) so `NPZ.jl` can read it; provenance lives in a JSON
  sidecar. The first attempt stored strings in the npz and `NPZ.jl` could not read
  numpy unicode dtypes — worth knowing before shaping the artifact.

## Reproduction

```bash
# 0. one-off: throwaway venv with torch only (no mace, no e3nn)
uv venv --python 3.11 /tmp/tvenv
VIRTUAL_ENV=/tmp/tvenv uv pip install --python /tmp/tvenv/bin/python torch numpy

# 1. fetch the checkpoint and extract the frozen table (offline, once)
curl -sSL -o /tmp/mace_small.model \
  https://github.com/ACEsuit/mace-mp/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model
/tmp/tvenv/bin/python scripts/extract_mace_embedding.py /tmp/mace_small.model \
  -o /tmp/mace_element_embedding.npz
# -> emb (89, 128), Z 1..94, norm=0.106 ; + /tmp/mace_element_embedding.json

# 2. categorical baseline (validates the harness against the shipped constructor)
julia +1.12.6 --project=. -e '
using ACEpotentials, Printf
for els in ([:Si],[:Si,:C],[:Si,:C,:O])
   p = ace1_model(elements=els, order=3, totaldegree=8, rcut=5.5)
   @printf("%-14s size(WB)=%s\n", string(els), string(size(p.ps.WB)))
end'
# -> (54,1) (308,2) (917,3)
```

The three spike scripts below are throwaway; they were run from a scratch
directory and are reproduced here in full rather than committed.

### `spike_et_order_bug.jl` — N1, and the Q1/Q2 direct enumeration

```julia
using ACEpotentials, Random, LinearAlgebra, StaticArrays, Polynomials4ML, Printf
const M = ACEpotentials.Models
import EquivariantTensors as ET
randrot(rng) = (Q=qr(randn(rng,3,3)).Q; A=SMatrix{3,3}(Matrix(Q)); det(A)<0 ? -A : A)

function rot_err(mb, rsp; seed=5)
   maxl = maximum(maximum(b.l for b in bb) for bb in mb)
   t = ET.sparse_equivariant_tensor(L=0, mb_spec=mb, Rnl_spec=rsp,
                                    Ylm_spec=M._make_Y_spec(maxl), basis=real)
   yb = M._make_Y_basis(:solid, maxl); rng = MersenneTwister(seed); Nat = 8
   c = randn(rng, length(rsp)); e = 0.0
   for _ = 1:3
      Rs = [(r=2.0+rand(rng); u=randn(rng,SVector{3,Float64}); r*u/norm(u)) for _=1:Nat]
      Q = randrot(rng)
      f(R) = (rs = norm.(R);
              ET.evaluate(t, [c[j]*exp(-rs[i])*rs[i]^(j%3) for i=1:Nat, j=1:length(rsp)],
                          Polynomials4ML.evaluate(yb, R), NamedTuple(), NamedTuple())[1])
      B, Br = f(Rs), f([Q*R for R in Rs])
      e = max(e, norm(B-Br, Inf)/max(norm(B, Inf), 1e-30))
   end
   return size(t.A2Bmaps[1],1), e
end

lvl1 = M.TotalDegree(1.0, 1/1.5); r1 = M.oneparticle_spec(lvl1, 8)
AA1 = M.sparse_AA_spec(; order=3, r_spec=r1, level=lvl1, max_level=8)
mb1 = unique([[(n=b.n, l=b.l) for b in bb] for bb in AA1])
println("shipped spec is order-grouped: ", issorted(length.(mb1)))   # true

for d in [2,3,4]
   rsp = sort([(n=(b.n-1)*d+k, l=b.l) for b in r1 for k=1:d], by=x->(x.l,x.n))
   mbu = [[(n=(b.n-1)*d+k, l=b.l) for b in bb] for k=1:d for bb in mb1]
   mbs = sort(mbu, by=length)
   n1,e1 = rot_err(mbu, rsp); n2,e2 = rot_err(mbs, rsp)
   @printf("d=%d  unsorted: n_B=%-5d rot=%.3e  |  order-grouped: n_B=%-5d rot=%.3e\n",
           d, n1, e1, n2, e2)
end
# d=1 with order-grouping deliberately broken -> also wrong:
mb_scram = mb1[sortperm([b[1].n for b in mb1])]
@printf("d=1 scrambled: rot=%.3e\n", rot_err(mb_scram, r1)[2])
```

### `spike_specscale.jl` — Q2/Q3 tables

```julia
using ACEpotentials, Printf
const M = ACEpotentials.Models
import EquivariantTensors as ET
const WL = 1.5
ace1_level(w) = M.TotalDegree(1.0*w, 1/WL)

function build_from_mb(mb, rsp)
   maxl = maximum(maximum(b.l for b in bb) for bb in mb)
   t = @elapsed tensor = ET.sparse_equivariant_tensor(L=0, mb_spec=mb, Rnl_spec=rsp,
                             Ylm_spec=M._make_Y_spec(maxl), basis=real)
   size(tensor.A2Bmaps[1],1), size(tensor.A2Bmaps[1],2), t
end

function build_categorical(S, order, deg)          # == what ace1_model builds
   lvl = ace1_level(S); r_spec = M.oneparticle_spec(lvl, deg)
   t_spec = @elapsed begin
      AA = M.sparse_AA_spec(; order=order, r_spec=r_spec, level=lvl, max_level=deg)
      mb = unique([[(n=b.n,l=b.l) for b in bb] for bb in AA])
   end
   n_B, n_AA, t_symm = build_from_mb(mb, r_spec)
   (; n_B, n_rnl=length(r_spec), n_mb=length(mb), n_AA, t_spec, t_symm)
end

function build_diag_direct(d, order, deg)          # channel-diagonal, direct
   lvl1 = ace1_level(1); r1 = M.oneparticle_spec(lvl1, deg)
   t_spec = @elapsed begin
      AA1 = M.sparse_AA_spec(; order=order, r_spec=r1, level=lvl1, max_level=deg)
      mb1 = unique([[(n=b.n,l=b.l) for b in bb] for bb in AA1])
      mb  = [[(n=(b.n-1)*d+k, l=b.l) for b in bb] for k=1:d for bb in mb1]
      mb  = sort(mb, by=length)                    # REQUIRED: see N1
   end
   rsp = sort([(n=(b.n-1)*d+k, l=b.l) for b in r1 for k=1:d], by=x->(x.l,x.n))
   n_B, n_AA, t_symm = build_from_mb(mb, rsp)
   (; n_B, n_rnl=length(rsp), n_mb=length(mb), n_AA, t_spec, t_symm)
end

build_categorical(2,3,8); build_diag_direct(2,3,8)   # warm up
for S in [1,2,3,5,10,15,20]; r=build_categorical(S,3,8)
   @printf("cat  S=%-3d n_B=%-7d t_spec=%.3f t_symm=%.3f\n", S, r.n_B, r.t_spec, r.t_symm) end
for d in [1,2,4,8,16,32,64,128]; r=build_diag_direct(d,3,8)
   @printf("emb  d=%-3d n_B=%-7d (n_B/d=%.1f) t_spec=%.3f t_symm=%.3f\n",
           d, r.n_B, r.n_B/d, r.t_spec, r.t_symm) end
```

### `spike_q5_model.jl` — Q5 (needs `NPZ` stacked onto the load path)

```julia
using ACEpotentials, Printf, Random, LinearAlgebra, StaticArrays, NPZ
const M = ACEpotentials.Models
const WL = 1.5
z = npzread(ENV["EMB_NPZ"]); EMB_FULL = z["emb"]; Z_ROWS = Int.(z["Z"])
sym2z = Dict(:H=>1,:C=>6,:N=>7,:O=>8,:Si=>14)
emb_table(els, d) = EMB_FULL[[findfirst(==(sym2z[e]), Z_ROWS) for e in els], 1:d]
chan(n,d) = mod1(n,d); nprime(n,d) = div(n-1,d)+1

function embedding_ace_model(elements, d; order=3, deg=8)
   lvl1 = M.TotalDegree(1.0, 1/WL)
   r1 = M.oneparticle_spec(lvl1, deg); maxnp = maximum(b.n for b in r1)
   AA1 = M.sparse_AA_spec(; order=order, r_spec=r1, level=lvl1, max_level=deg)
   r_spec = sort([(n=(b.n-1)*d+k, l=b.l) for b in r1 for k=1:d], by=x->(x.l,x.n))
   AA = sort([[(n=(b.n-1)*d+k, l=b.l, m=b.m) for b in bb] for k=1:d for bb in AA1],
             by = length)                                    # REQUIRED: see N1
   rin0cuts = M._default_rin0cuts(elements)
   rb = M.ace_learnable_Rnlrzz(; spec=r_spec, maxq=maxnp, elements=elements,
            rin0cuts=rin0cuts, polys=:legendre, envelopes=:poly2sx, Winit=:zero)
   pb = M.ace_learnable_Rnlrzz(; spec=[(n=n,l=0) for n=1:maxnp], maxq=maxnp,
            elements=elements, rin0cuts=rin0cuts, polys=:legendre,
            envelopes=:poly2sx, Winit=:zero)
   m = M.ace_model(rb, :solid, AA, lvl1, pb, M._make_Vref(elements, nothing, false))
   m.meta["init_WB"]="glorot_normal"; m.meta["init_Wpair"]="glorot_normal"; m
end

"the embedding analogue of Models.set_onehot_weights!"
function set_embedding_weights!(rbasis, ps, emb, d)
   ps.Wnlq .= 0
   for iz = 1:M._get_nz(rbasis), jz = 1:M._get_nz(rbasis),
       (i_nl, nl) in enumerate(rbasis.spec)
      k, np = chan(nl.n, d), nprime(nl.n, d)
      np <= size(ps.Wnlq, 2) && (ps.Wnlq[i_nl, np, iz, jz] = emb[jz, k])
   end
   ps
end
# then: rotation / permutation / linearity / rank checks as tabulated in Q5.
```

---

## Recommendation

**Pursue the Julia route, with three modifications. Do not fall back to JAX for
this feature.**

The two things the plan flagged as unverified both came out *better* than
feared: ET needs no change to express the restriction (Q1), and the spec build is
free rather than expensive (Q2). The plan's 1-2 week estimate looks plausible for
the mechanics. What changes is the framing and the prerequisites.

1. **Fix the ET ordering bug first (N1), and treat it as the gating item.** It is
   independent of this feature, it silently corrupts any non-order-grouped spec,
   and it is a two-line fix in `sparse_equivariant_tensor` plus a guard in
   `SparseSymmProd`. Ship it upstream with a regression test that builds a
   deliberately scrambled spec. Until it lands, any work here is building on a
   trap.

2. **Rewrite the framing from "lift O(S^ν) scaling" to "rank-`d` truncation of
   the species tensor" (N3).** ACEpotentials *already* folds species into the
   radial index (`set_onehot_weights!` documents it in the source); the new part is
   the channel-diagonal restriction, which strictly shrinks the model class. The
   plan should say so, because it changes what has to be measured before shipping:
   **an accuracy comparison at matched `n_B`**, which this spike did not do. That
   experiment — fit categorical S=5..10 vs embedding at several `d`, compare RMSE
   at equal basis size — is the right next spike, and it is cheap now that the
   construction works. Run it at order 4 as well as order 3: that is where the
   cost win is largest and the truncation loss is worst, so it is where the
   trade-off actually has to be decided.

3. **Set expectations on where it pays, and state the correlation order when you
   do.** At degree 8 with MACE-MP-0 small's native `d=128`, the crossover is
   **S ≈ 12-13 at order 2, S ≈ 6-7 at order 3, S ≈ 4-5 at order 4** — it roughly
   halves per increment in ν, so the feature gets more attractive exactly where
   ACE hurts most. But the truncation sharpens at the same time: at S=10 a d=128
   embedding spans 100% of the species tensor at ν=2, 58% at ν=3 and only 18% at
   ν=4. The window where it is both cheaper *and* lossless is narrow — S=13-15
   (ν=2), S=7-8 (ν=3), S=5-6 (ν=4) — and above it the feature is cheap but
   approximate. Truncating `d` below `S` is not a cheap win either; it removes the
   model's ability to distinguish elements. The useful band remains
   `S ≤ d ≤ C(S+ν-1,ν)`.

Two smaller points for the plan's constraint list, both confirmed by this spike:

* **Constraint 1 ("species behind one narrow interface") is already half-met.**
  `Models.set_onehot_weights!` *is* the "edge species -> radial coefficients"
  function; the embedding version is a sibling of it, ~10 lines. The node-species
  gather `WB[:, i_z0]` scales linearly in S and is not the bottleneck — this
  feature only needs the edge one.
* **Constraint 3 (species-scheme flag in the export schema) stands**, and should
  carry the channel width `d`, the composite-index convention
  (`n = (n'-1)·d + k`), the checkpoint sha256 and the layer name. The npz must be
  numeric-only for `NPZ.jl`, with provenance in a sidecar.

## Correlation-order dependence (orders 2, 3 and 4)

Follow-up to Q3/Q4: the `O(S^ν)` problem worsens with ν, so a crossover measured
at one order may not transfer. Two effects pull in opposite directions; both are
measured below. Same degree 8, `wL = 1.5`, same harness.

### Single-channel baselines

`n_B(emb, d) = n_B(d=1) · d` **exactly** at every order tested, out to d=128, so
the break-even width is always `d* = n_B(cat,S) / n_B(d=1)`.

| order ν | n_B at d=1 | per-order breakdown (ν=1,2,3,4) | n_B at d=128 | t_spec [s] | t_symm at d=128 [s] |
|---|---|---|---|---|---|
| 2 | 31 | 8, 23 | 3 968 | 0.000 | 0.043 |
| 3 | 54 | 8, 23, 23 | 6 912 | 0.001 | 0.079 |
| 4 | 69 | 8, 23, 23, 15 | 8 832 | 0.001 | 0.103 |

The Q2 conclusion holds unchanged at every order: direct channel-diagonal
enumeration is free (≤ 0.001 s) and symmetrisation is linear in `d`.

### Effect 1: the crossover moves in the feature's favour at higher ν

Categorical `n_B` and the gain from an embedding of width `d`
(gain > 1 = embedding is that many times smaller; **bold** = the bracket
containing the `d=128` crossover).

| S | **ν=2** n_B | d\* | gain d=8 | d=32 | d=128 | **ν=3** n_B | d\* | gain d=8 | d=32 | d=128 | **ν=4** n_B | d\* | gain d=8 | d=32 | d=128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 31 | 1.0 | 0.12 | 0.03 | 0.01 | 54 | 1.0 | 0.12 | 0.03 | 0.01 | 69 | 1.0 | 0.12 | 0.03 | 0.01 |
| 2 | 109 | 3.5 | 0.44 | 0.11 | 0.03 | 308 | 5.7 | 0.71 | 0.18 | 0.04 | 546 | 7.9 | 0.99 | 0.25 | 0.06 |
| 3 | 233 | 7.5 | 0.94 | 0.23 | 0.06 | 917 | 17.0 | 2.12 | 0.53 | 0.13 | 2 106 | 30.5 | 3.82 | 0.95 | 0.24 |
| 4 | 404 | 13.0 | 1.63 | 0.41 | 0.10 | 2 037 | 37.7 | 4.72 | 1.18 | 0.29 | 5 772 | 83.7 | 10.46 | 2.61 | **0.65** |
| 5 | 621 | 20.0 | 2.50 | 0.63 | 0.16 | 3 824 | 70.8 | 8.85 | 2.21 | 0.55 | 12 899 | 186.9 | 23.37 | 5.84 | **1.46** |
| 6 | 885 | 28.5 | 3.57 | 0.89 | 0.22 | 6 434 | 119.1 | 14.89 | 3.72 | **0.93** | 25 194 | 365.1 | 45.64 | 11.41 | 2.85 |
| 7 | 1 195 | 38.5 | 4.82 | 1.20 | 0.30 | 10 022 | 185.6 | 23.20 | 5.80 | **1.45** | 44 691 | 647.7 | 80.96 | 20.24 | 5.06 |
| 8 | 1 552 | 50.1 | 6.26 | 1.56 | 0.39 | 14 745 | 273.1 | 34.13 | 8.53 | 2.13 | 73 782 | 1 069 | 133.66 | 33.42 | 8.35 |
| 9 | 1 955 | 63.1 | 7.88 | 1.97 | 0.49 | 20 758 | 384.4 | 48.05 | 12.01 | 3.00 | 115 180 | 1 669 | 208.66 | 52.16 | 13.04 |
| 10 | 2 405 | 77.6 | 9.70 | 2.42 | 0.61 | 28 217 | 522.5 | 65.32 | 16.33 | 4.08 | 171 961 | 2 492 | 311.52 | 77.88 | 19.47 |
| 11 | 2 901 | 93.6 | 11.70 | 2.92 | 0.73 | 37 278 | 690.3 | 86.29 | 21.57 | 5.39 | 247 517 | 3 587 | 448.40 | 112.10 | 28.03 |
| 12 | 3 444 | 111.1 | 13.89 | 3.47 | **0.87** | 48 097 | 890.7 | 111.34 | 27.83 | 6.96 | 345 611 | 5 009 | 626.11 | 156.53 | 39.13 |
| 13 | 4 033 | 130.1 | 16.26 | 4.07 | **1.02** | 60 829 | 1 127 | 140.81 | 35.20 | 8.80 | *not reached* | | | | |
| 14 | 4 669 | 150.6 | 18.83 | 4.71 | 1.18 | 75 631 | 1 401 | 175.07 | 43.77 | 10.94 | | | | | |
| 15 | 5 351 | 172.6 | 21.58 | 5.39 | 1.35 | 92 658 | 1 716 | 214.49 | 53.62 | 13.41 | | | | | |
| 16 | 6 080 | 196.1 | 24.52 | 6.13 | 1.53 | 112 066 | 2 075 | 259.41 | 64.85 | 16.21 | | | | | |
| 17 | 6 855 | 221.1 | 27.64 | 6.91 | 1.73 | 134 011 | 2 482 | 310.21 | 77.55 | 19.39 | | | | | |
| 18 | 7 677 | 247.6 | 30.96 | 7.74 | 1.93 | 158 649 | 2 938 | 367.24 | 91.81 | 22.95 | | | | | |
| 19 | 8 545 | 275.6 | 34.46 | 8.61 | 2.15 | 186 135 | 3 447 | 430.87 | 107.72 | 26.93 | | | | | |
| 20 | 9 460 | 305.2 | 38.15 | 9.54 | 2.38 | 216 626 | 4 012 | 501.45 | 125.36 | 31.34 | | | | | |

**Range reached at order 4: S = 1…12.** The scan stopped there because the S=12
build took 270 s against a 180 s-per-build budget (spec 8.2 s + symmetrisation
9.2 s + the surrounding basis bookkeeping); `n_B` is already 345 611 and the next
step is roughly another ×1.4 in S-growth. **S ≥ 13 at order 4 is not reported and
has not been extrapolated.** Orders 2 and 3 reached S=20 comfortably (≤ 4.6 s per
build).

**Crossover brackets at d = 128** (bracketed by measured points, not interpolated):

| ν | gain crosses 1 between | at d=32 | at d=8 |
|---|---|---|---|
| 2 | **S=12 (0.87) and S=13 (1.02)** | S=6 (0.89) / S=7 (1.20) | S=3 (0.94) / S=4 (1.63) |
| 3 | **S=6 (0.93) and S=7 (1.45)** | S=3 (0.53) / S=4 (1.18) | S=2 (0.71) / S=3 (2.12) |
| 4 | **S=4 (0.65) and S=5 (1.46)** | S=3 (0.95) / S=4 (2.61) | S=2 (0.99) / S=3 (3.82) |

**One-line answer:** at fixed `d = 128`, the crossover element count falls from
**S≈12-13 at order 2, to S≈6-7 at order 3, to S≈4-5 at order 4** — effect 1 is
real and large, roughly halving the crossover for each increment in ν.

### Effect 2: the model-class restriction gets sharper at higher ν

`dim Sym^ν(R^S) = C(S+ν-1, ν)` is the species-tensor dimension the categorical
basis carries; channel-diagonal-`d` spans `min(d, dim)` of it. Measured rank of
the diagonal span built from the real MACE-MP-0-small rows:

| S | ν | `C(S+ν-1,ν)` | d=8 | d=16 | d=32 | **d=128** | fraction captured at d=128 |
|---|---|---|---|---|---|---|---|
| 5 | 2 | 15 | 8 | 15 | 15 | 15 | 100% |
| 5 | 3 | 35 | 8 | 16 | 32 | 35 | 100% |
| 5 | 4 | 70 | 8 | 16 | 32 | 70 | 100% |
| 10 | 2 | 55 | 8 | 16 | 32 | 55 | 100% |
| 10 | 3 | 220 | 8 | 16 | 32 | **128** | **58%** |
| 10 | 4 | **715** | 8 | 16 | 32 | **128** | **18%** |

Rank is exactly `min(d, dim)` throughout. **At S=10 the fraction of the species
tensor a d=128 embedding can represent drops from 100% (ν=2) to 58% (ν=3) to 18%
(ν=4)** — effect 2 confirmed, and it is severe at order 4. The singular spectrum
of that span is smooth, with no knee (S=10, ν=4: cumulative energy 0.46 at 4
components, 0.63 at 8, 0.79 at 16, 0.91 at 32, 0.98 at 64).

*Derived, not measured* (pure combinatorics): `d=128` spans the **full** species
tensor up to **S=15 at ν=2** (`C(16,2)=120`), **S=8 at ν=3** (`C(10,3)=120`) and
**S=6 at ν=4** (`C(9,4)=126`); beyond those it is a strict truncation.

### The two effects together

Putting the brackets beside the lossless ceilings gives the window in which a
`d=128` embedding is **both cheaper than categorical and still lossless**:

| ν | crossover S (cheaper from) | lossless up to S | **window** |
|---|---|---|---|
| 2 | 13 | 15 | S = 13-15 |
| 3 | 7 | 8 | S = 7-8 |
| 4 | 5 | 6 | S = 5-6 |

The window is narrow at every order and it **moves down in S as ν rises**. Above
it the embedding is cheaper but lossy — which is the regime the feature is
actually for, and exactly the regime where an accuracy measurement is
indispensable. Higher correlation order therefore makes the feature *more*
attractive on cost and *less* safe on accuracy, at the same time.

### Reading the two effects together differently: `d` is a free parameter

The "window" table above fixes `d = 128` and concludes the safe region is narrow.
That conclusion is an artefact of the constraint, not a property of the method.
**128 is MACE-MP-0-small's neural channel width; it has no connection to how many
channels ACE needs to resolve species.** The quantity that matters is
`dim Sym^nu(R^S) = C(S+nu-1, nu)`, and `d` is ours to choose.

Choosing `d = C(S+nu-1, nu)` makes the diagonal span **exactly full rank**
(measured: rank is `min(d, dim)` throughout), i.e. **lossless** — and it is still
cheaper than categorical whenever `C(S+nu-1,nu) < n_B(S)/n_B(1) = d*`. It always
is, and by a margin that grows with `nu`:

| nu | S | n_B categorical | `dim` = lossless `d` | `d*` break-even | lossless saving |
|---|---|---|---|---|---|
| 2 | 10 | 2 405 | 55 | 77.6 | **1.41x** |
| 2 | 20 | 9 460 | 210 | 305.2 | **1.45x** |
| 3 | 10 | 28 217 | 220 | 522.5 | **2.38x** |
| 3 | 20 | 216 626 | 1 540 | 4 011.6 | **2.60x** |
| 4 | 5 | 12 899 | 70 | 186.9 | **2.67x** |
| 4 | 10 | 171 961 | 715 | 2 492.2 | **3.49x** |

So there *is* a free lunch after all, just a modest one: **a guaranteed-lossless
1.3-3.5x, no accuracy question to answer, growing with correlation order.**
Derived from the measured `n_B` and the combinatorial `dim`; the saving with
per-order widths (below) is larger still, and is not yet measured.

This reframes the feature. The lossy regime is not the only regime, and the
lossless one needs no fit to justify — it is a *reparameterisation*, not an
approximation. Everything beyond `d = dim` is the trade the degeneracy probe has
to price.

### Correction: "d > S is wasted" holds only for the two-body block

Q4 concluded the embedding table "has rank exactly S, so `d < S` destroys element
resolution and `d > S` is provably wasted". The first half stands; **the second
half is superseded by the order-scan result.** The order-4 deficiency formula,
exact at every `d` measured, is

```
deficit = sum_nu  n_B^(nu) * ( d - min(d, C(S+nu-1, nu)) )
```

so each correlation order saturates at its *own* dimension. `d > S` is wasted
only in the `nu = 1` block, where `dim = S`; the `nu >= 2` blocks keep absorbing
channels up to `C(S+nu-1,nu)`, which is far larger. Q4 measured the two-body
waste (`8` and `24` deficient functions at `d = 4, 6`, `S = 3`) and generalised
it too far.

**Consequence — use a different width per correlation order.** A single `d`
either starves the high orders or pads the low ones: at `d = 16`, order 4,
`S = 3`, a 1104-function basis carries only **617** independent functions, 44%
redundant. Setting `d_nu = min(d_max, C(S+nu-1, nu))` removes that redundancy by
construction.

That is not cosmetic. The measured design matrix reaches **cond ~ 3e21**, and a
rank-deficient design matrix is exactly what a convex linear solve must not be
handed — it undermines QR/LSQR conditioning and the BLR posterior, which are the
reasons for preferring the linear route in the first place. **Per-order widths
should be treated as part of the construction, not an optimisation.**

### Rank deficiency of the real basis at order 4

The Q4 measurement repeated at order 4, `[Si,C,O]` (S=3), real MACE-MP-0-small
channels 1..d, numerical rank of the per-centre design matrix over `3·n_B`
random environments. Prediction is
`Σ_ν n_B^{(ν)} · (d − min(d, C(S+ν−1,ν)))` with the measured per-order counts
`n_B^{(ν)} = 8, 23, 23, 15` and `C(3+ν−1,ν) = 3, 6, 10, 15`.

| d | n_B | cond(B) | rank | deficit | predicted | match | rot(B) |
|---|---|---|---|---|---|---|---|
| 1 | 69 | 4.3e+06 | 69 | 0 | 0 | yes | 1.4e-15 |
| 3 | 207 | 3.9e+07 | 207 | 0 | 0 | yes | 5.1e-16 |
| 4 | 276 | 1.7e+19 | 268 | 8 | 8 | yes | 2.7e-16 |
| 6 | 414 | 3.5e+18 | 390 | 24 | 24 | yes | 1.3e-15 |
| 10 | 690 | 5.6e+19 | 542 | 148 | 148 | yes | 4.0e-15 |
| 12 | 828 | 8.7e+19 | 572 | 256 | 256 | yes | 9.8e-16 |
| 16 | 1104 | 3.0e+21 | 617 | **487** | **487** | yes | 1.5e-15 |

**The prediction is exact at every `d`.** At d=16 a 1104-function basis carries
only 617 independent functions — 44% of it is redundant. Rotation invariance
holds throughout, so the order-4 construction is sound; the deficiency is the
model class, not a bug.

Note on method: the design matrix is severely ill-conditioned (up to 3e21), so
the numerical-rank tolerance matters. Ranks were counted at `1e-12 · σ₁`, which
is where the count plateaus — at d=16 the rank reads 568 / 598 / 611 / 616 /
617 / 617 for tolerances 1e-8 … 1e-13. The earlier order-3 table used
`1e-10 · σ₁`; at that tolerance the order-4 deficits read 151 / 258 / 494
instead of 148 / 256 / 487, i.e. the tighter threshold over-counts deficiency by
about 1%. The order-3 numbers are unaffected (their `d` values are small enough
that the matrices are far better conditioned).

### Reproduction

```bash
# order scan (orders 2, 3, 4) -- ~35 min wall on this Mac, order 4 dominates
BUDGET=180 julia +1.12.6 --project=. spike_order_scan.jl
# order-4 rank deficiency (needs NPZ on the load path, as for spike_q5_model.jl)
EMB_NPZ=/tmp/mace_element_embedding.npz ORDER=4 julia +1.12.6 spike_rank_order4.jl
```

`spike_order_scan.jl` is `spike_specscale.jl` above with `build_categorical` /
`build_diag_direct` wrapped in a loop over `order in [2,3,4]`, a per-build time
budget that breaks out of the S loop, and `byord` read off
`ET.get_nnll_spec(tensor, 1)`. `spike_rank_order4.jl` is `spike_q5_model.jl`
above with `order=4`, `d in [1,3,4,6,10,12,16]`, `ncfg = 3·n_B`, rank counted at
`1e-12·σ₁`, and the prediction column added.

## What could not be determined

* **Any accuracy statement.** No fit was run. Whether channel-diagonal-`d`
  reaches categorical accuracy at equal or smaller `n_B` is unmeasured, and is the
  single biggest open question.
* **Whether a truncated `d < S` is usable in practice.** The rank argument says it
  loses element resolution; how much that costs is an empirical question.
* **Evaluation cost.** Only basis *sizes* and *construction* times were measured.
  Per-atom evaluation time for a `d`-wide radial basis was not benchmarked, and
  the radial basis grows `d`-fold, so it is not simply proportional to `n_B`.
* **Forces, virials, serialisation, `acefit!` end-to-end.** Not exercised.
* **Whether ET's dev checkout (0.5.0 / PR #143) behaves identically.** Measured on
  0.4.3 only; the relevant source file is byte-identical, so this is inferred.
* **Order 4 beyond S=12, and any order above 4.** The order-4 scan stopped at
  S=12 on a 180 s-per-build budget; larger S was not built and has not been
  extrapolated. Order 5+ was not attempted.
