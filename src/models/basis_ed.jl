
# ------------------------------------------------------------
#   Basis evaluation with derivatives (design-matrix assembly)
#
#   `evaluate_basis_ed(model, Rs, Zs, Z0, ps, st)` returns the site basis
#   B (length `length_basis(model)`) and its Jacobian with respect to the
#   neighbour positions, as a `Matrix{SVector{3,T}}` of size
#   (length_basis, length(Rs)); dB[k, j] = ∂B[k] / ∂Rs[j].
#
#   This used to be a `ForwardDiff.jacobian` over the full basis vector
#   followed by a chain of collect / reshape / permutedims, which inferred
#   as `Any` and allocated ~100 MB per site.  It is now a hand-written
#   forward-mode (pushforward) pass with `SVector{3}` tangents through
#      rs -> Rnl, Ylm -> A (pooled product) -> AA (symmetric product)
#         -> B = A2B * AA,
#   plus the pair basis.  Only the species block of B is nonzero for a
#   given centre atom, so only that block (n_B + n_pair entries) is
#   differentiated; the public wrapper scatters it into the full layout.
#
#   All intermediates live in a `BasisEDWorkspace` that is allocated once
#   per call (or reused across sites / structures if the caller passes one).
# ------------------------------------------------------------

using SparseArrays: SparseMatrixCSC, nzrange, rowvals, nonzeros
using LinearAlgebra: mul!
using StaticArrays: SVector, SMatrix

# ------------------------------------------------------------
#  workspace

mutable struct BasisEDWorkspace{T}
   maxneigh::Int
   rs::Vector{T}
   ∇rs::Vector{SVector{3, T}}
   Rnl::Matrix{T}                      # (maxneigh, nR)
   dRnl::Matrix{T}                     # (maxneigh, nR)   ∂Rnl/∂r
   ∂Rnl::Matrix{SVector{3, T}}         # (maxneigh, nR)   ∂Rnl/∂𝐫
   Ylm::Matrix{T}                      # (maxneigh, nY)
   ∂Ylm::Matrix{SVector{3, T}}         # (maxneigh, nY)
   A::Vector{T}                        # (nA,)
   ∂A::Matrix{SVector{3, T}}           # (maxneigh, nA)
   AA::Vector{T}                       # (nAA,)
   ∂AA::Matrix{SVector{3, T}}          # (maxneigh, nAA)
   Bi::Vector{T}                       # (nBi,)
   ∂Bi::Matrix{SVector{3, T}}          # (maxneigh, nBi)  NB: neighbour-major
   Rpair::Matrix{T}                    # (maxneigh, npair)
   dRpair::Matrix{T}                   # (maxneigh, npair)
end

function BasisEDWorkspace(model::ACEModel, maxneigh::Integer; T = Float64)
   nR = length(model.rbasis)
   nY = length(model.ybasis)
   nA = length(model.tensor.abasis)
   nAA = length(model.tensor.aabasis)
   nBi = length(model.tensor)
   npair = model.pairbasis === nothing ? 0 : length(model.pairbasis)
   z = zero(SVector{3, T})
   return BasisEDWorkspace{T}(maxneigh,
         zeros(T, maxneigh), fill(z, maxneigh),
         zeros(T, maxneigh, nR), zeros(T, maxneigh, nR), fill(z, maxneigh, nR),
         zeros(T, maxneigh, nY), fill(z, maxneigh, nY),
         zeros(T, nA), fill(z, maxneigh, nA),
         zeros(T, nAA), fill(z, maxneigh, nAA),
         zeros(T, nBi), fill(z, maxneigh, nBi),
         zeros(T, maxneigh, npair), zeros(T, maxneigh, npair))
end

# grow the workspace if a site has more neighbours than it was built for
function _ensure_capacity!(ws::BasisEDWorkspace{T}, model::ACEModel, nneigh::Integer) where {T}
   if nneigh > ws.maxneigh
      ws2 = BasisEDWorkspace(model, max(nneigh, 2 * ws.maxneigh); T = T)
      for f in fieldnames(BasisEDWorkspace)
         setfield!(ws, f, getfield(ws2, f))
      end
   end
   return ws
end

# ------------------------------------------------------------
#  pushforward kernels

# product of a tuple and its gradient: (∏ b_t, (∂/∂b_t ∏ b_s)_t)
_prod_ed(b::NTuple{1, T}) where {T} = b[1], (one(T),)

function _prod_ed(b::NTuple{N, T}) where {N, T}
   p2, g2 = _prod_ed(b[2:N])
   return b[1] * p2, (p2, ntuple(i -> b[1] * g2[i], N - 1)...)
end

# A[iA] = ∑_j Rnl[j, n] Ylm[j, l]
# ∂A[j, iA] = ∂Rnl[j, n] Ylm[j, l] + Rnl[j, n] ∂Ylm[j, l]
function _pf_A!(A::AbstractVector{T}, ∂A::AbstractMatrix{SVector{3, T}},
                spec::AbstractVector{NTuple{2, Int}}, nneigh::Int,
                Rnl, ∂Rnl, Ylm, ∂Ylm) where {T}
   @inbounds for (iA, (n, l)) in enumerate(spec)
      a = zero(T)
      @simd ivdep for j = 1:nneigh
         r = Rnl[j, n]; y = Ylm[j, l]
         a += r * y
         ∂A[j, iA] = ∂Rnl[j, n] * y + r * ∂Ylm[j, l]
      end
      A[iA] = a
   end
   return nothing
end

# AA[iAA] = ∏_t A[ϕ_t];   ∂AA[j, iAA] = ∑_t (∏_{s≠t} A[ϕ_s]) ∂A[j, ϕ_t]
function _pf_AA_N!(AA::AbstractVector{T}, ∂AA::AbstractMatrix{SVector{3, T}},
                   range::UnitRange{Int}, spec::Vector{NTuple{N, Int}}, nneigh::Int,
                   A::AbstractVector{T}, ∂A::AbstractMatrix{SVector{3, T}}) where {T, N}
   @inbounds for (iAA, ϕ) in zip(range, spec)
      aa, ∇aa = _prod_ed(ntuple(t -> A[ϕ[t]], N))
      AA[iAA] = aa
      @simd ivdep for j = 1:nneigh
         d = ∇aa[1] * ∂A[j, ϕ[1]]
         for t = 2:N
            d += ∇aa[t] * ∂A[j, ϕ[t]]
         end
         ∂AA[j, iAA] = d
      end
   end
   return nothing
end

@generated function _pf_AA!(AA, ∂AA, basis::EquivariantTensors.SparseSymmProd{ORD},
                            nneigh::Int, A, ∂A) where {ORD}
   quote
      if basis.hasconst
         AA[1] = one(eltype(AA))
         @inbounds for j = 1:nneigh
            ∂AA[j, 1] = zero(eltype(∂AA))
         end
      end
      Base.Cartesian.@nexprs $ORD N -> _pf_AA_N!(AA, ∂AA, basis.ranges[N],
                                                 basis.specs[N], nneigh, A, ∂A)
      return nothing
   end
end

# B = A2B * AA;   ∂B[j, k] = ∑_iAA A2B[k, iAA] ∂AA[j, iAA]
function _pf_A2B!(B::AbstractVector{T}, ∂B::AbstractMatrix{SVector{3, T}},
                  A2B::SparseMatrixCSC, nneigh::Int,
                  AA::AbstractVector{T}, ∂AA::AbstractMatrix{SVector{3, T}}) where {T}
   fill!(B, zero(T))
   @inbounds for k = 1:size(A2B, 1), j = 1:nneigh
      ∂B[j, k] = zero(SVector{3, T})
   end
   rv = rowvals(A2B); nz = nonzeros(A2B)
   @inbounds for iAA = 1:size(A2B, 2)
      aa = AA[iAA]
      for p in nzrange(A2B, iAA)
         k = rv[p]; c = nz[p]
         B[k] += c * aa
         @simd ivdep for j = 1:nneigh
            ∂B[j, k] += c * ∂AA[j, iAA]
         end
      end
   end
   return nothing
end

# generic fallback for a dense (or otherwise non-CSC) coupling matrix
function _pf_A2B!(B::AbstractVector{T}, ∂B::AbstractMatrix{SVector{3, T}},
                  A2B::AbstractMatrix, nneigh::Int,
                  AA::AbstractVector{T}, ∂AA::AbstractMatrix{SVector{3, T}}) where {T}
   mul!(B, A2B, AA)
   @inbounds for k = 1:size(A2B, 1), j = 1:nneigh
      d = zero(SVector{3, T})
      for iAA = 1:size(A2B, 2)
         d += A2B[k, iAA] * ∂AA[j, iAA]
      end
      ∂B[j, k] = d
   end
   return nothing
end

# ------------------------------------------------------------
#  the site pushforward: fills ws.Bi, ws.∂Bi (block of the many-body basis
#  for the centre species) and ws.Rpair, ws.dRpair (pair basis; the pair
#  block of B is ∑_j Rpair[j, :] and its derivative dRpair[j, k] * ∇rs[j]).

function evaluate_basis_ed!(ws::BasisEDWorkspace{T}, model::ACEModel,
                            Rs::AbstractVector{SVector{3, T}}, Zs, Z0,
                            ps, st) where {T}
   n = length(Rs)
   _ensure_capacity!(ws, model, n)
   rs = view(ws.rs, 1:n)
   ∇rs = view(ws.∇rs, 1:n)
   radii_ed!(rs, ∇rs, Rs)

   # radial basis: r-derivative -> vector derivative
   Rnl = view(ws.Rnl, 1:n, :)
   dRnl = view(ws.dRnl, 1:n, :)
   evaluate_ed_batched!(Rnl, dRnl, model.rbasis, rs, Z0, Zs, ps.rbasis, st.rbasis)
   ∂Rnl = view(ws.∂Rnl, 1:n, :)
   @inbounds for t = 1:size(Rnl, 2), j = 1:n
      ∂Rnl[j, t] = dRnl[j, t] * ∇rs[j]
   end

   # spherical harmonics with gradients
   Ylm = view(ws.Ylm, 1:n, :)
   ∂Ylm = view(ws.∂Ylm, 1:n, :)
   P4ML.evaluate_ed!(Ylm, ∂Ylm, model.ybasis, Rs)

   # A -> AA -> B
   ∂A = view(ws.∂A, 1:n, :)
   _pf_A!(ws.A, ∂A, model.tensor.abasis.spec, n, Rnl, ∂Rnl, Ylm, ∂Ylm)
   ∂AA = view(ws.∂AA, 1:n, :)
   _pf_AA!(ws.AA, ∂AA, model.tensor.aabasis, n, ws.A, ∂A)
   ∂Bi = view(ws.∂Bi, 1:n, :)
   _pf_A2B!(ws.Bi, ∂Bi, model.tensor.A2Bmaps[1], n, ws.AA, ∂AA)

   # pair basis
   if model.pairbasis !== nothing
      Rpair = view(ws.Rpair, 1:n, :)
      dRpair = view(ws.dRpair, 1:n, :)
      evaluate_ed_batched!(Rpair, dRpair, model.pairbasis, rs, Z0, Zs,
                           ps.pairbasis, st.pairbasis)
   end
   return ws
end

# ------------------------------------------------------------
#  public interface

"""
   evaluate_basis_ed(model::ACEModel, Rs, Zs, Z0, ps, st; ws = nothing)

Evaluate the site basis `B` (as `evaluate_basis`) together with its
Jacobian with respect to the neighbour positions,
`dB::Matrix{SVector{3,T}}` of size `(length_basis(model), length(Rs))`,
`dB[k, j] = ∂B[k]/∂Rs[j]`.  Only the block of `B` belonging to the centre
species `Z0` (and its pair block) is nonzero.

A `BasisEDWorkspace` can be passed via `ws` to avoid re-allocating the
intermediates when many sites are evaluated.
"""
function evaluate_basis_ed(model::ACEModel,
                           Rs::AbstractVector{SVector{3, T}}, Zs, Z0,
                           ps, st;
                           ws::Union{Nothing, BasisEDWorkspace{T}} = nothing) where {T}
   nB = length_basis(model)
   n = length(Rs)
   B = zeros(T, nB)
   dB = zeros(SVector{3, T}, nB, n)
   if n == 0
      return B, dB
   end
   if ws === nothing
      ws = BasisEDWorkspace(model, n; T = T)
   end
   evaluate_basis_ed!(ws, model, Rs, Zs, Z0, ps, st)
   _scatter_basis_ed!(B, dB, ws, model, Z0, n)
   return B, dB
end

# write the block result from the workspace into the full basis layout
function _scatter_basis_ed!(B::AbstractVector{T}, dB::AbstractMatrix{SVector{3, T}},
                            ws::BasisEDWorkspace{T}, model::ACEModel, Z0,
                            n::Int) where {T}
   iB = get_basis_inds(model, Z0)
   @inbounds for (kk, k) in enumerate(iB)
      B[k] = ws.Bi[kk]
      for j = 1:n
         dB[k, j] = ws.∂Bi[j, kk]
      end
   end
   if model.pairbasis !== nothing
      iP = get_pairbasis_inds(model, Z0)
      @inbounds for (kk, k) in enumerate(iP)
         b = zero(T)
         for j = 1:n
            b += ws.Rpair[j, kk]
            dB[k, j] = ws.dRpair[j, kk] * ws.∇rs[j]
         end
         B[k] = b
      end
   end
   return B, dB
end
