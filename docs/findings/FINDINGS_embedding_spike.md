# Frozen MACE element embeddings in ACEpotentials: one-day spike

**Branch** `jax-eval` · **Date** 2026-09-11 · **Machine** Apple Silicon Mac, CPU only
**Julia** 1.12.6 · **ACEpotentials** 0.10.2 (repo `a82a7a92`) · **EquivariantTensors** 0.4.3

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
| 3 | Where is the crossover? | **Confirmed.** At order 3 / degree 8, embedding-`d` breaks even with categorical-`S` at `d* = n_B(S)/54`: `d*` = 5.7 (S=2), 17.0 (S=3), 70.8 (S=5), 522 (S=10). With MACE-MP-0 small's full `d=128`, the embedding **loses up to S≈6 and wins from S≈7**. |
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

## Q3. Crossover

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

## Q4. How small can `d` be?

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

## Q5. Does it actually work?

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
   construction works.

3. **Set expectations on where it pays.** At order 3 / degree 8 the crossover with
   MACE-MP-0 small's native `d=128` is **S ≈ 6-7**. This is a many-element
   feature, exactly as the plan says, but "many" starts around seven, not three.
   Truncating `d` below `S` is not a cheap win — it removes the model's ability to
   distinguish elements — and `d > C(S+ν-1,ν)` is provably wasted width. The
   useful band is `S ≤ d ≤ C(S+ν-1,ν)`.

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
* **Higher correlation orders.** Everything here is order 3.
