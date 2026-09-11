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
function embedding_rows(e::ElementEmbedding, zlist; d = size(e.emb, 2))
   1 <= d <= size(e.emb, 2) ||
      error("d = $d out of range for a width-$(size(e.emb, 2)) table")
   idx = map(zlist) do z
      i = findfirst(==(Int(z)), e.Z)
      i === nothing && error("element Z = $z is not in the embedding table")
      i
   end
   return e.emb[idx, 1:d]
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
