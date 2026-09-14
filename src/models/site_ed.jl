
# ------------------------------------------------------------
#   Allocation-free site energy + gradient for a linear ACEModel
#
#   `evaluate_ed(model, Rs, Zs, Z0, ps, st)` (the kernel behind
#   `energy_forces_virial`) used to allocate ~60 arrays per site and ~6 per
#   edge, recompute the A basis twice, and go through two sparse matvecs
#   with the A2B map per site (see docs/findings/FINDINGS_forward_profile.md).
#
#   Here the readout weights are folded through the coupling map once per
#   call, wAA[iz] = A2B' * WB[:, iz], so that
#       E_i = wAA[iz] . AA(A(Rnl, Ylm))      and      ∂E_i/∂AA = wAA[iz],
#   every intermediate lives in a `SiteEDWorkspace`, the A basis is computed
#   once and reused by the pullback, and the radial part of the gradient is
#   reduced to one scalar per edge before it is turned into a vector.
# ------------------------------------------------------------

using LinearAlgebra: dot, norm
using StaticArrays: SVector
using SparseArrays: SparseMatrixCSC

mutable struct SiteEDWorkspace{T}
   maxneigh::Int
   rs::Vector{T}
   ∇rs::Vector{SVector{3, T}}
   Rnl::Matrix{T}                   # (maxneigh, nR)
   dRnl::Matrix{T}                  # (maxneigh, nR)   ∂Rnl/∂r
   Ylm::Matrix{T}                   # (maxneigh, nY)
   dYlm::Matrix{SVector{3, T}}      # (maxneigh, nY)
   A::Vector{T}                     # (nA,)
   AA::Vector{T}                    # (nAA,)
   ∂A::Vector{T}                    # (nA,)
   ∂Rnl::Matrix{T}                  # (maxneigh, nR)   ∂E/∂Rnl
   ∂Ylm::Matrix{T}                  # (maxneigh, nY)   ∂E/∂Ylm
   ∇Ei::Vector{SVector{3, T}}       # (maxneigh,)
   sbuf::Vector{T}                  # (maxneigh,)
   Rpair::Matrix{T}                 # (maxneigh, npair)
   dRpair::Matrix{T}                # (maxneigh, npair)
end

function SiteEDWorkspace(model::ACEModel, maxneigh::Integer; T = Float64)
   nR = length(model.rbasis)
   nY = length(model.ybasis)
   nA = length(model.tensor.abasis)
   nAA = length(model.tensor.aabasis)
   npair = model.pairbasis === nothing ? 0 : length(model.pairbasis)
   z = zero(SVector{3, T})
   return SiteEDWorkspace{T}(maxneigh,
         zeros(T, maxneigh), fill(z, maxneigh),
         zeros(T, maxneigh, nR), zeros(T, maxneigh, nR),
         zeros(T, maxneigh, nY), fill(z, maxneigh, nY),
         zeros(T, nA), zeros(T, nAA), zeros(T, nA),
         zeros(T, maxneigh, nR), zeros(T, maxneigh, nY),
         fill(z, maxneigh), zeros(T, maxneigh),
         zeros(T, maxneigh, npair), zeros(T, maxneigh, npair))
end

function _ensure_capacity!(ws::SiteEDWorkspace{T}, model::ACEModel, nneigh::Integer) where {T}
   if nneigh > ws.maxneigh
      ws2 = SiteEDWorkspace(model, max(nneigh, 2 * ws.maxneigh); T = T)
      for f in fieldnames(SiteEDWorkspace)
         setfield!(ws, f, getfield(ws2, f))
      end
   end
   return ws
end

# readout weights folded through the coupling map, one vector per element:
#    wAA[iz] = A2B' * WB[:, iz]
fold_readout_weights(model::ACEModel, ps, iz::Integer) =
      Vector(model.tensor.A2Bmaps[1]' * (@view ps.WB[:, iz]))

fold_readout_weights(model::ACEModel, ps) =
      [ fold_readout_weights(model, ps, iz) for iz = 1:_get_nz(model) ]

# radial part reduced to one scalar per edge before forming the vector, then
# the Ylm part; function barrier as in `_assemble_grad_ed!`
function _assemble_grad_fast!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs, sbuf, nX::Int)
   @inbounds for j = 1:nX
      sbuf[j] = zero(eltype(sbuf))
   end
   @inbounds for t = 1:size(∂Rnl, 2)
      @simd ivdep for j = 1:nX
         sbuf[j] = muladd(∂Rnl[j, t], dRnl[j, t], sbuf[j])
      end
   end
   @inbounds for j = 1:nX
      ∇Ei[j] = sbuf[j] * ∇rs[j]
   end
   @inbounds for t = 1:size(∂Ylm, 2)
      for j = 1:nX
         ∇Ei[j] += ∂Ylm[j, t] * dYlm[j, t]
      end
   end
   return ∇Ei
end

"""
   evaluate_ed!(ws, model, Rs, Zs, Z0, ps, st, wAA) -> Ei

Site energy of `model` at a centre of species `Z0` with neighbours `(Rs, Zs)`
and its gradient with respect to the neighbour positions, written into
`ws.∇Ei[1:length(Rs)]`.  `wAA` is the vector of folded readout weights of
the centre species (`fold_readout_weights(model, ps)[_z2i(model, Z0)]`).
Allocation-free once the workspace is large enough.
"""
function evaluate_ed!(ws::SiteEDWorkspace{T}, model::ACEModel,
                      Rs::AbstractVector{SVector{3, T}}, Zs, Z0,
                      ps, st, wAA::AbstractVector) where {T}
   n = length(Rs)
   i_z0 = _z2i(model.rbasis, Z0)
   if n == 0
      return T(model.Vref.E0[Z0])
   end
   _ensure_capacity!(ws, model, n)

   rs = view(ws.rs, 1:n)
   ∇rs = view(ws.∇rs, 1:n)
   radii_ed!(rs, ∇rs, Rs)

   # embeddings (forward mode)
   Rnl = view(ws.Rnl, 1:n, :)
   dRnl = view(ws.dRnl, 1:n, :)
   evaluate_ed_batched!(Rnl, dRnl, model.rbasis, rs, Z0, Zs, ps.rbasis, st.rbasis)
   Ylm = view(ws.Ylm, 1:n, :)
   dYlm = view(ws.dYlm, 1:n, :)
   P4ML.evaluate_ed!(Ylm, dYlm, model.ybasis, Rs)

   # A, AA and the readout
   A = ws.A; AA = ws.AA
   EquivariantTensors.evaluate!(A, model.tensor.abasis, (Rnl, Ylm))
   EquivariantTensors.evaluate!(AA, model.tensor.aabasis, A)
   Ei = dot(wAA, AA)

   # pullback: ∂E/∂AA = wAA -> ∂A -> (∂Rnl, ∂Ylm)
   ∂A = ws.∂A
   EquivariantTensors.pullback!(∂A, wAA, model.tensor.aabasis, A)
   ∂Rnl = view(ws.∂Rnl, 1:n, :)
   ∂Ylm = view(ws.∂Ylm, 1:n, :)
   EquivariantTensors.pullback!((∂Rnl, ∂Ylm), ∂A, model.tensor.abasis, (Rnl, Ylm))
   ∇Ei = ws.∇Ei
   _assemble_grad_fast!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs, ws.sbuf, n)

   # pair potential
   if model.pairbasis !== nothing
      Rpair = view(ws.Rpair, 1:n, :)
      dRpair = view(ws.dRpair, 1:n, :)
      evaluate_ed_batched!(Rpair, dRpair, model.pairbasis, rs, Z0, Zs,
                           ps.pairbasis, st.pairbasis)
      Wp = ps.Wpair
      @inbounds for k = 1:size(Rpair, 2)
         w = Wp[k, i_z0]
         e = zero(T)
         for j = 1:n
            e += Rpair[j, k]
            ∇Ei[j] += (w * dRpair[j, k]) * ∇rs[j]
         end
         Ei += w * e
      end
   end

   # one-body reference
   @assert model.Vref isa OneBody
   Ei += model.Vref.E0[Z0]
   return Ei
end
