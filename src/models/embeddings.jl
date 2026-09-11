# Frozen element embeddings: lifting ACE's O(S^ν) species scaling.
#
# ACE folds the neighbour species into the radial index (Rnl_learnable.jl:83):
#
#    categorical:  R(n'z')l(r, Z1, Z2) = P_n'(r) * δ_{z', Z2}
#
# Replacing the one-hot δ with a frozen element embedding of width d gives
#
#    embedding:    R(n'k)l (r, Z1, Z2) = P_n'(r) * emb[Z2, k]
#
# and nothing downstream changes: `abasis`, `aabasis` and the symmetrisation
# matrix see a wider radial basis and nothing else.  With `emb` FROZEN the model
# stays linear in its coefficients, so `acefit!`, the smoothness priors, BLR and
# the committees all apply unchanged.  Training `emb` would break that and is
# deliberately not offered here.
#
# See docs/findings/FINDINGS_embedding_spike.md and the "Stage 1E" section of
# docs/plans/jax_ace_port_plan.md.  Prior art: Darby, Kovács, Batatia, Caro,
# Hart, Ortner, Csányi, "Tensor-reduced atomic density representations",
# PRL 131, 028001 (2023), arXiv:2210.01705.

using JSON

"""
   ElementEmbedding

A frozen `(S, d)` element-embedding table, its atomic numbers, and the
provenance of the checkpoint it came from.  `emb[i, :]` is the vector for
element `Z[i]`.
"""
struct ElementEmbedding
   Z::Vector{Int}
   emb::Matrix{Float64}
   meta::Dict{String, Any}
end

Base.show(io::IO, e::ElementEmbedding) =
   print(io, "ElementEmbedding($(length(e.Z)) elements, d = $(size(e.emb, 2)), ",
             "source = $(get(e.meta, "checkpoint", "unknown")))")

"""
   read_mace_embedding(path) -> ElementEmbedding

Read a frozen embedding artefact written by `scripts/extract_mace_embedding.py`:
a single JSON file carrying `Z`, `emb` and the provenance of the checkpoint.

JSON rather than npz deliberately: ACEpotentials already depends on JSON, and a
frozen table of a few hundred kB does not justify adding a binary-format
dependency to the whole package.  Nothing here runs Python — the table is an
artefact, extracted once, offline.  A Python dependency on the load path would
recreate exactly the cross-codebase fragility this design avoids.
"""
function read_mace_embedding(path::AbstractString)
   d = JSON.parsefile(path)
   haskey(d, "emb") && haskey(d, "Z") ||
      error("$path is not an embedding artefact: expected keys `emb` and `Z`")
   rows = d["emb"]
   emb = Matrix{Float64}(reduce(vcat, [reshape(Float64.(r), 1, :) for r in rows]))
   Z = Int.(d["Z"])
   size(emb, 1) == length(Z) ||
      error("emb has $(size(emb,1)) rows but Z has $(length(Z)) entries")
   meta = Dict{String, Any}(k => v for (k, v) in d if k ∉ ("emb", "Z"))
   return ElementEmbedding(Z, emb, meta)
end

"""
   embedding_rows(e::ElementEmbedding, zlist; d = size(e.emb, 2))

The `(length(zlist), d)` block of rows for `zlist` (atomic numbers), in that
order, truncated to the leading `d` channels.
"""
function embedding_rows(e::ElementEmbedding, zlist; d = size(e.emb, 2),
                        normalise = true)
   1 <= d <= size(e.emb, 2) ||
      error("d = $d out of range for a width-$(size(e.emb, 2)) table")
   idx = map(zlist) do z
      i = findfirst(==(Int(z)), e.Z)
      i === nothing && error("element Z = $z is not in the embedding table")
      i
   end
   rows = e.emb[idx, 1:d]
   if normalise
      # Normalise AFTER truncation, and this matters far more than it looks.
      # A frozen per-row scaling is absorbed by the linear coefficients, so the
      # model spans the same space either way -- but the raw MACE entries are
      # O(0.1), so at correlation order ν the basis is scaled by ~1e-3, and BLR's
      # prior on coefficient magnitude is NOT scale-invariant.  Measured: the raw
      # table fits forces ~16x worse than `ace1_model` at identical n_B, purely
      # through the regularisation, with bases that are exactly proportional.
      # Normalising the FULL row and then truncating is not enough -- that leaves
      # ~1/sqrt(d_full) per channel, which was the same bug one step removed.
      for i = 1:size(rows, 1)
         nrm = norm(@view rows[i, :])
         nrm > 0 || error("element $(zlist[i]) has a zero embedding row at d = $d")
         rows[i, :] ./= nrm
      end
   end
   return rows
end

"""
   embedding_widths(S, order; d_max = nothing)

Per-correlation-order channel widths `d_ν = min(d_max, C(S+ν-1, ν))`.

Each order saturates at its own species-tensor dimension `dim Sym^ν(R^S)`, so a
single width either starves the high orders or pads the low ones.  Padding is
not merely wasteful: it makes the design matrix rank-deficient, and measured
condition numbers already reach ~1e21, which is precisely what a convex linear
solve must not be handed.

With `d_max = nothing` every order gets its full dimension, which is **lossless**
— a reparameterisation of the categorical basis, not an approximation.
"""
function embedding_widths(S::Integer, order::Integer; d_max = nothing)
   dims = [binomial(S + ν - 1, ν) for ν = 1:order]
   d_max === nothing ? dims : min.(d_max, dims)
end

"""
   set_embedding_weights!(rbasis, ps, emb)

Freeze `ps.Wnlq` so that `R(n'k)l(r, Z1, Z2) = P_n'(r) * emb[Z2, k]`.

The index convention matches `set_onehot_weights!`, with the embedding width `d`
in place of `NZ`:

    n    | 1    2    3    4    5    6    ...
    n'k  | 1,1  1,2  1,3  2,1  2,2  2,3  ...   (d = 3)

so `k = mod1(n, d)` and `n' = div(n-1, d) + 1`.
"""
function set_embedding_weights!(rbasis::LearnableRnlrzzBasis, ps,
                                emb::AbstractMatrix)
   NZ = _get_nz(rbasis)
   size(emb, 1) == NZ ||
      error("embedding has $(size(emb, 1)) rows but the basis has $NZ elements")
   d = size(emb, 2)
   ps.Wnlq[:] .= 0
   for iz1 = 1:NZ, iz2 = 1:NZ
      for (i_nl, nl) in enumerate(rbasis.spec)
         k  = mod1(nl.n, d)
         n_ = div(nl.n - 1, d) + 1
         if n_ <= size(ps.Wnlq, 2)
            ps.Wnlq[i_nl, n_, iz1, iz2] = emb[iz2, k]
         end
      end
   end
   return ps
end


"""
   ace_embedding_model(; elements, order, totaldegree, embedding, d_max = nothing, ...)

An ACE model whose species dependence enters through a **frozen** element
embedding rather than a categorical index.

The channel is folded into the radial index, `n = (n'-1)*d + k`, so the radial
basis is `d` copies of the single-channel one and `abasis` / `aabasis` / `A2B`
are untouched.  The many-body spec is **channel-diagonal** — all `ν` factors of a
product share one `k` — because full channel mixing would reintroduce `d^ν`,
which is the same combinatorial wall in a new variable.

Widths are per correlation order, `d_ν = min(d_max, C(S+ν-1, ν))`.  With
`d_max = nothing` every order gets its full species-tensor dimension, which is
**lossless**: a reparameterisation of the categorical basis, measured 1.4-10x
smaller (see docs/findings/FINDINGS_embedding_spike.md).

`embedding` is an `ElementEmbedding`; it is frozen, so the model stays linear in
its coefficients and `acefit!` applies unchanged.
"""
function ace_embedding_model(; elements, order, totaldegree,
                               embedding::ElementEmbedding,
                               d_max = nothing,
                               wL = 1.5, maxl = nothing, Ytype = :solid,
                               rcut = nothing, E0s = nothing, ZBL = false,
                               pair_maxn = nothing, ace1_compat = true,
                               normalise = true,
                               rng = Random.default_rng())
   zlist = _convert_zlist(elements)
   S = length(zlist)
   widths = embedding_widths(S, order; d_max = d_max)
   d = maximum(widths)
   emb = embedding_rows(embedding, [Int(z) for z in zlist]; d = d,
                        normalise = normalise)

   # single-channel one-particle spec, then widened by d
   level1 = TotalDegree(1.0, 1 / wL)
   r1 = oneparticle_spec(level1, totaldegree)
   maxl === nothing || (r1 = [b for b in r1 if b.l <= maxl])
   rspec = [ (n = (b.n - 1) * d + k, l = b.l) for b in r1 for k = 1:d ]

   rin0cuts = _default_rin0cuts(zlist)
   rcut === nothing ||
      (rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts))

   # Match `ace1_model`'s radial heuristics, not `ace_learnable_Rnlrzz`'s
   # defaults.  They differ in three ways that matter, and using the defaults
   # gave forces ~16x worse than ace1_model at identical n_B:
   #   * the Agnesi transform is (p,q) = (2,4) in ACE1, (2,2) by default;
   #   * ACE1 folds the envelope into the orthogonality, so the polynomials are
   #     Jacobi(2*pin, 2*pcut) = Jacobi(4,4) for the (:x,2,2) envelope, not
   #     Legendre (ace1_compat.jl:255-261);
   #   * ACE1 splines the basis afterwards.
   # `_default_rin0cuts`' rcutfactor = 2.5 already matches ACE1's
   # rcut = (:bondlen, 2.5), so the cutoffs need no adjustment.
   trans = ace1_compat ? agnesi_transform.(rin0cuts, 2, 4) :
                         agnesi_transform.(rin0cuts, 2, 2)
   polys = ace1_compat ? (:jacobi, 4.0, 4.0) : :legendre
   rbasis = ace_learnable_Rnlrzz(; elements = zlist, spec = rspec,
                                   maxq = maximum(b.n for b in rspec),
                                   rin0cuts = rin0cuts, transforms = trans,
                                   polys = polys, Winit = :glorot_normal)

   # Freeze the embedding into the radial weights, then spline -- the same order
   # `ace1_model` uses for its one-hot weights (ace1_compat.jl:281-285).  After
   # splining the basis carries the embedding and has no Wnlq left to fit, which
   # is what keeps the model linear in WB.
   ps_r = initialparameters(rng, rbasis)
   set_embedding_weights!(rbasis, ps_r, emb)
   rbasis_eval = ace1_compat ? splinify(rbasis, ps_r) : rbasis

   # channel-diagonal many-body spec: every factor of a product shares one k,
   # and order ν only draws on its own d_ν channels
   AA1 = sparse_AA_spec(; order = order, r_spec = r1,
                          level = level1, max_level = totaldegree)
   AA_spec = [ [ (n = (b.n - 1) * d + k, l = b.l, m = b.m) for b in bb ]
               for bb in AA1 for k = 1:widths[length(bb)] ]
   # sparse_equivariant_tensor mis-couples a spec that is not grouped by
   # correlation order; see N1 in FINDINGS_embedding_spike.md
   AA_spec = sort(AA_spec, by = length)

   pmaxn = pair_maxn === nothing ? totaldegree : pair_maxn
   pair_basis = ace_learnable_Rnlrzz(; elements = zlist, level = TotalDegree(),
                     max_level = pmaxn, maxl = 0, maxn = pmaxn,
                     rin0cuts = rbasis.rin0cuts,
                     transforms = (:agnesi, 1, 4), envelopes = :poly1sr)
   pair_basis.meta["Winit"] = "onehot"
   pair_basis = splinify(pair_basis, initialparameters(rng, pair_basis))

   rcut_max = maximum([x.rcut for x in rin0cuts])
   Vref = _make_Vref(zlist, E0s, ZBL, rcut_max)
   raw = ace_model(rbasis_eval, Ytype, AA_spec, level1, pair_basis, Vref)
   raw.meta["init_WB"] = "zeros"
   raw.meta["embedding"] = Dict("d_max" => d, "widths" => widths,
                                "provenance" => embedding.meta)

   ps, st = LuxCore.setup(rng, raw)
   # on the non-splined path the weights live in ps and must be frozen here
   ace1_compat || set_embedding_weights!(rbasis, ps.rbasis, emb)
   return ACEPotential(raw, ps, st)
end
