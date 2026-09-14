
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
#   as `Any` and allocated ~100 MB per site.  It is now a forward-mode
#   (pushforward) pass with `SVector{3}` tangents through
#      rs -> Rnl, Ylm -> A (pooled product) -> AA (symmetric product)
#         -> B = A2B * AA,
#   plus the pair basis.  The radial and spherical-harmonic parts are
#   evaluated here; the tensor part (A -> AA -> B) is
#   `EquivariantTensors.pushforward_rows!`, the row-wise (one tangent per
#   neighbour) pushforward of `SparseACEbasis` (ET >= 0.5.2).  Only the
#   species block of B is nonzero for a given centre atom, so only that
#   block (n_B + n_pair entries) is differentiated; the public wrapper
#   scatters it into the full layout.
#
#   The embedding intermediates live in a `BasisEDWorkspace` that is
#   allocated once per call (or reused across sites / structures if the
#   caller passes one); the A / AA intermediates are on ET's Bumper stack.
# ------------------------------------------------------------

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
   Bi::Vector{T}                       # (nBi,)
   ∂Bi::Matrix{SVector{3, T}}          # (maxneigh, nBi)  NB: neighbour-major
   Rpair::Matrix{T}                    # (maxneigh, npair)
   dRpair::Matrix{T}                   # (maxneigh, npair)
end

function BasisEDWorkspace(model::ACEModel, maxneigh::Integer; T = Float64)
   nR = length(model.rbasis)
   nY = length(model.ybasis)
   nBi = length(model.tensor)
   npair = model.pairbasis === nothing ? 0 : length(model.pairbasis)
   z = zero(SVector{3, T})
   return BasisEDWorkspace{T}(maxneigh,
         zeros(T, maxneigh), fill(z, maxneigh),
         zeros(T, maxneigh, nR), zeros(T, maxneigh, nR), fill(z, maxneigh, nR),
         zeros(T, maxneigh, nY), fill(z, maxneigh, nY),
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

   # A -> AA -> B: row-wise pushforward through the tensor (ET >= 0.5.2)
   ∂Bi = view(ws.∂Bi, 1:n, :)
   EquivariantTensors.pushforward_rows!(ws.Bi, ∂Bi, model.tensor,
                                        Rnl, Ylm, ∂Rnl, ∂Ylm)

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
