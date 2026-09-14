# Scratch re-implementations of the basis-with-derivatives path.  NOTHING in
# src/ is modified; these are standalone functions that call the same model
# components.  Three levels:
#
#   efv_basis_v1 : the ORIGINAL evaluate_basis_ed (ForwardDiff over the full
#                  length_basis vector), but a type-stable accumulation loop
#                  (function barrier; no Unitful inside; no dv[k, :] copies; and
#                  only the species block + pair block of the basis, which is
#                  the only non-zero part).
#   efv_basis_v2 : v1 + ForwardDiff on the SPECIES BLOCK only, fixed chunk,
#                  writing directly into an (n_Bi x nneigh) SVector matrix.
#   efv_basis_v3 : v1 + hand-written forward-mode (pushforward) through
#                  Rnl / Ylm / A / AA / A2B, using EquivariantTensors'
#                  existing-but-unused `_jacobian_X` kernels.
#   efv_basis_v3t: v3 with Threads.@threads over sites.
#
# All return unit-less (E::Vector, F::Matrix{SVector{3}}, V::Vector{SMatrix})
# with F of size (natoms, length_basis) exactly like the original.
using ACEpotentials, StaticArrays, LinearAlgebra, LuxCore, Random
const ForwardDiff = ACEpotentials.Models.ForwardDiff
using ACEpotentials.Models: ACEModel, ACEPotential, evaluate_basis, evaluate_basis_ed,
      get_basis_inds, get_pairbasis_inds, length_basis, _z2i, radii_ed!,
      evaluate_ed_batched, __vec, __svecs
using AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
const ET = ACEpotentials.Models.EquivariantTensors
const P4ML = ACEpotentials.Models.P4ML

# ---------------------------------------------------------------------------
# type-stable accumulation (function barrier).  v is the full basis vector,
# dv is (nbasis_rows x nneigh) of SVector{3}; `rows` says which rows of the
# full basis these correspond to (all of them for the original path).
function _accumulate!(E::Vector{T}, F::Matrix{SVector{3, T}}, V::Vector{SMatrix{3, 3, T, 9}},
                      v::AbstractVector{T}, dv::AbstractMatrix{SVector{3, T}},
                      rows::AbstractVector{Int}, Js::Vector{Int},
                      Rs::Vector{SVector{3, T}}, i::Int) where {T}
   nneigh = length(Js)
   @inbounds for (kk, k) in enumerate(rows)
      E[k] += v[k]
      fi = zero(SVector{3, T})
      vk = zero(SMatrix{3, 3, T, 9})
      for α = 1:nneigh
         d = dv[kk, α]
         F[Js[α], k] -= d
         fi += d
         vk -= d * Rs[α]'
      end
      F[i, k] += fi
      V[k] += vk
   end
   return nothing
end

_rows(model, z0) = vcat(get_basis_inds(model, z0), get_pairbasis_inds(model, z0))

# ---------------------------------------------------------------------------
# v1: original derivative path, type-stable accumulation
function efv_basis_v1(at, calc::ACEPotential{<: ACEModel}; nlist = PairList(at, cutoff_radius(calc)))
   model = calc.model; ps = calc.ps; st = calc.st
   nB = length_basis(model); nat = length(at)
   E = zeros(nB); F = zeros(SVector{3, Float64}, nat, nB); V = zeros(SMatrix{3, 3, Float64, 9}, nB)
   for i = 1:nat
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      v, dv = evaluate_basis_ed(model, Rs, Zs, z0, ps, st)
      rows = _rows(model, z0)
      _accumulate!(E, F, V, v, view(dv, rows, :), rows, Js, Rs, i)
   end
   return (energy = E, forces = F, virial = V)
end

# ---------------------------------------------------------------------------
# v2: ForwardDiff on the species block only (Bi and Apair), fixed chunk size,
# no reshape/permutedims copies.
function _block_basis(model, Rs, Zs, z0, ps, st)
   # the non-zero part of evaluate_basis: [Bi; Apair]
   rs = [norm(r) for r in Rs]
   Rnl = ACEpotentials.Models.evaluate_batched(model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
   Ylm = P4ML.evaluate(model.ybasis, Rs)
   BB = ET.evaluate(model.tensor, Rnl, Ylm, NamedTuple(), NamedTuple())
   Bi = BB[1]
   Rpair = ACEpotentials.Models.evaluate_batched(model.pairbasis, rs, z0, Zs, ps.pairbasis, st.pairbasis)
   Apair = vec(sum(Rpair, dims = 1))
   return vcat(Bi, Apair)
end

function basis_ed_fd(model, Rs::Vector{SVector{3, T}}, Zs, z0, ps, st; chunk = 12) where {T}
   x = __vec(Rs)
   f = _x -> _block_basis(model, __svecs(_x), Zs, z0, ps, st)
   cfg = ForwardDiff.JacobianConfig(f, x, ForwardDiff.Chunk{chunk}())
   J = ForwardDiff.jacobian(f, x, cfg)          # (nrows, 3 nneigh)
   nrows = size(J, 1); nneigh = length(Rs)
   dB = Matrix{SVector{3, T}}(undef, nrows, nneigh)
   @inbounds for j = 1:nneigh, k = 1:nrows
      dB[k, j] = SVector(J[k, 3j-2], J[k, 3j-1], J[k, 3j])
   end
   B = f(x)
   return B, dB
end

function efv_basis_v2(at, calc::ACEPotential{<: ACEModel}; nlist = PairList(at, cutoff_radius(calc)), chunk = 12)
   model = calc.model; ps = calc.ps; st = calc.st
   nB = length_basis(model); nat = length(at)
   E = zeros(nB); F = zeros(SVector{3, Float64}, nat, nB); V = zeros(SMatrix{3, 3, Float64, 9}, nB)
   for i = 1:nat
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      rows = _rows(model, z0)
      v_blk, dv = basis_ed_fd(model, Rs, Zs, z0, ps, st; chunk = chunk)
      v = zeros(nB); v[rows] .= v_blk
      _accumulate!(E, F, V, v, dv, rows, Js, Rs, i)
   end
   return (energy = E, forces = F, virial = V)
end

# ---------------------------------------------------------------------------
# v3: hand-written pushforward through A (pooled sparse product), AA (sparse
# symmetric product) and A2B (sparse matrix), with SVector{3} tangents per
# neighbour.  Structure follows EquivariantTensors' `_jacobian_X` kernels
# (sparseprodpool.jl:567, sparsesymmprod.jl:349), which cannot be called
# directly because they take promote_type(Float64, SVector{3}) = Any.
struct PFState{TA2B}
   A2B::TA2B              # SparseMatrixCSC, n_Bi x nAA
end
PFState(model::ACEModel) = PFState(model.tensor.A2Bmaps[1])

# A[iA] = sum_j Rnl[j,n] Ylm[j,l];  ∂A[j, iA] = ∂Rnl[j,n] Ylm[j,l] + Rnl[j,n] ∂Ylm[j,l]
function _pf_A!(A::AbstractVector{T}, ∂A::AbstractMatrix{SVector{3, T}}, spec, Rnl, ∂Rnl, Ylm, ∂Ylm) where {T}
   nneigh = size(Rnl, 1)
   @inbounds for (iA, (n, l)) in enumerate(spec)
      a = zero(T)
      for j = 1:nneigh
         r = Rnl[j, n]; y = Ylm[j, l]
         a += r * y
         ∂A[j, iA] = ∂Rnl[j, n] * y + r * ∂Ylm[j, l]
      end
      A[iA] = a
   end
   return nothing
end

# ∂AA[j, iAA] = sum_t (prod_{s != t} A[ϕ_s]) ∂A[j, ϕ_t]
function _pf_AA_N!(AA::AbstractVector{T}, ∂AA::AbstractMatrix{SVector{3, T}}, iiAA, spec::Vector{NTuple{N, Int}},
                   A::AbstractVector{T}, ∂A::AbstractMatrix{SVector{3, T}}) where {T, N}
   nneigh = size(∂A, 1)
   @inbounds for (iAA, ϕ) in zip(iiAA, spec)
      Avals = ntuple(t -> A[ϕ[t]], N)
      aa, ∇aa = ET._static_prod_ed(Avals)
      AA[iAA] = aa
      for j = 1:nneigh
         d = zero(SVector{3, T})
         for t = 1:N
            d += ∇aa[t] * ∂A[j, ϕ[t]]
         end
         ∂AA[j, iAA] = d
      end
   end
   return nothing
end

@generated function _pf_AA!(AA, ∂AA, basis::ET.SparseSymmProd{ORD}, A, ∂A) where {ORD}
   quote
      if basis.hasconst
         AA[1] = one(eltype(AA)); ∂AA[:, 1] .= Ref(zero(eltype(∂AA)))
      end
      Base.Cartesian.@nexprs $ORD N -> _pf_AA_N!(AA, ∂AA, basis.ranges[N], basis.specs[N], A, ∂A)
      return nothing
   end
end

function basis_ed_pf(model, Rs::Vector{SVector{3, T}}, Zs, z0, ps, st, pf::PFState) where {T}
   nneigh = length(Rs)
   rs = zeros(T, nneigh); ∇rs = zeros(SVector{3, T}, nneigh)
   radii_ed!(rs, ∇rs, Rs)
   # radial basis and its r-derivative -> vector derivative
   Rnl, dRnl_r = evaluate_ed_batched(model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
   nR = size(Rnl, 2)
   ∂Rnl = [dRnl_r[j, k] * ∇rs[j] for j = 1:nneigh, k = 1:nR]
   # spherical harmonics with gradients
   nY = length(model.ybasis)
   Ylm = zeros(T, nneigh, nY); ∂Ylm = zeros(SVector{3, T}, nneigh, nY)
   P4ML.evaluate_ed!(Ylm, ∂Ylm, model.ybasis, Rs)
   # A basis
   abasis = model.tensor.abasis; aabasis = model.tensor.aabasis
   nA = length(abasis); nAA = length(aabasis)
   A = zeros(T, nA); ∂A = Matrix{SVector{3, T}}(undef, nneigh, nA)
   _pf_A!(A, ∂A, abasis.spec, Rnl, ∂Rnl, Ylm, ∂Ylm)
   # AA basis
   AA = zeros(T, nAA); ∂AA = Matrix{SVector{3, T}}(undef, nneigh, nAA)
   _pf_AA!(AA, ∂AA, aabasis, A, ∂A)
   # B = A2B * AA ;  ∂B (nBi x nneigh) = A2B * ∂AA'
   Bi = pf.A2B * AA
   ∂Bi = pf.A2B * permutedims(∂AA)
   nBi = length(Bi)
   # pair basis
   Rpair, dRpair = evaluate_ed_batched(model.pairbasis, rs, z0, Zs, ps.pairbasis, st.pairbasis)
   npair = size(Rpair, 2)
   B = vcat(Bi, vec(sum(Rpair, dims = 1)))
   dB = Matrix{SVector{3, T}}(undef, nBi + npair, nneigh)
   @inbounds for j = 1:nneigh
      for k = 1:nBi
         dB[k, j] = ∂Bi[k, j]
      end
      for k = 1:npair
         dB[nBi + k, j] = dRpair[j, k] * ∇rs[j]
      end
   end
   return B, dB
end

function efv_basis_v3(at, calc::ACEPotential{<: ACEModel}; nlist = PairList(at, cutoff_radius(calc)),
                      pf = PFState(calc.model))
   model = calc.model; ps = calc.ps; st = calc.st
   nB = length_basis(model); nat = length(at)
   E = zeros(nB); F = zeros(SVector{3, Float64}, nat, nB); V = zeros(SMatrix{3, 3, Float64, 9}, nB)
   for i = 1:nat
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      rows = _rows(model, z0)
      v_blk, dv = basis_ed_pf(model, Rs, Zs, z0, ps, st, pf)
      v = zeros(nB); v[rows] .= v_blk
      _accumulate!(E, F, V, v, dv, rows, Js, Rs, i)
   end
   return (energy = E, forces = F, virial = V)
end

# threaded over sites: each thread accumulates into its own (E, F, V) and they
# are summed at the end (F is natoms x nB per thread, so memory is nthreads x F).
function efv_basis_v3t(at, calc::ACEPotential{<: ACEModel}; nlist = PairList(at, cutoff_radius(calc)),
                       pf = PFState(calc.model))
   model = calc.model; ps = calc.ps; st = calc.st
   nB = length_basis(model); nat = length(at)
   nt = Threads.maxthreadid()
   Es = [zeros(nB) for _ in 1:nt]
   Fs = [zeros(SVector{3, Float64}, nat, nB) for _ in 1:nt]
   Vs = [zeros(SMatrix{3, 3, Float64, 9}, nB) for _ in 1:nt]
   Threads.@threads :static for i = 1:nat
      tid = Threads.threadid()
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      rows = _rows(model, z0)
      v_blk, dv = basis_ed_pf(model, Rs, Zs, z0, ps, st, pf)
      v = zeros(nB); v[rows] .= v_blk
      _accumulate!(Es[tid], Fs[tid], Vs[tid], v, dv, rows, Js, Rs, i)
   end
   for t = 2:nt
      Es[1] .+= Es[t]; Fs[1] .+= Fs[t]; Vs[1] .+= Vs[t]
   end
   return (energy = Es[1], forces = Fs[1], virial = Vs[1])
end

# ---------------------------------------------------------------------------
# the ACEfit.feature_matrix layout, from an (E, F, V) triple (unit-less)
function feature_matrix_from_efv(efv, nat, nB; has_E = true, has_F = true, has_V = true)
   nrows = has_E + 3nat * has_F + 6 * has_V
   dm = Matrix{Float64}(undef, nrows, nB)
   i = 1
   if has_E
      dm[i, :] .= efv.energy; i += 1
   end
   if has_F
      dm[i:i+3nat-1, :] .= reinterpret(Float64, efv.forces); i += 3nat
   end
   if has_V
      idx = SVector(1, 5, 9, 6, 3, 2)
      @inbounds for k = 1:nB
         vk = efv.virial[k]
         for (r, q) in enumerate(idx)
            dm[i + r - 1, k] = vk[q]
         end
      end
   end
   return dm
end

# ---------------------------------------------------------------------------
# v4: v3 with a reusable per-process workspace so that the big intermediates
# (∂A, ∂AA, ∂B) are allocated once per (model, max neighbours) instead of once
# per site.  Same arithmetic as v3.
mutable struct PFWorkspace{T}
   maxneigh::Int
   rs::Vector{T}; ∇rs::Vector{SVector{3, T}}
   Rnl::Matrix{T}; dRnl_r::Matrix{T}; ∂Rnl::Matrix{SVector{3, T}}
   Ylm::Matrix{T}; ∂Ylm::Matrix{SVector{3, T}}
   A::Vector{T}; ∂A::Matrix{SVector{3, T}}
   AA::Vector{T}; ∂AA::Matrix{SVector{3, T}}; ∂AAt::Matrix{SVector{3, T}}
   Bi::Vector{T}; ∂Bi::Matrix{SVector{3, T}}
   Rpair::Matrix{T}; dRpair::Matrix{T}
   v::Vector{T}
end

function PFWorkspace(model::ACEModel, maxneigh::Int; T = Float64)
   nR = length(model.rbasis); nY = length(model.ybasis)
   nA = length(model.tensor.abasis); nAA = length(model.tensor.aabasis)
   nBi = size(model.tensor.A2Bmaps[1], 1); npair = length(model.pairbasis)
   z = zero(SVector{3, T})
   PFWorkspace{T}(maxneigh,
      zeros(T, maxneigh), fill(z, maxneigh),
      zeros(T, maxneigh, nR), zeros(T, maxneigh, nR), fill(z, maxneigh, nR),
      zeros(T, maxneigh, nY), fill(z, maxneigh, nY),
      zeros(T, nA), fill(z, maxneigh, nA),
      zeros(T, nAA), fill(z, maxneigh, nAA), fill(z, nAA, maxneigh),
      zeros(T, nBi), fill(z, nBi, maxneigh),
      zeros(T, maxneigh, npair), zeros(T, maxneigh, npair),
      zeros(T, length_basis(model)))
end

# same as _accumulate! but reading the block derivative from ∂Bi (nBi x nneigh)
# and the pair derivative from dRpair/∇rs, without forming dB.
function _accumulate_pf!(E, F, V, ws::PFWorkspace{T}, rows_B, rows_pair, Js, Rs, i, nneigh) where {T}
   @inbounds for (kk, k) in enumerate(rows_B)
      E[k] += ws.Bi[kk]
      fi = zero(SVector{3, T}); vk = zero(SMatrix{3, 3, T, 9})
      for α = 1:nneigh
         d = ws.∂Bi[kk, α]
         F[Js[α], k] -= d; fi += d; vk -= d * Rs[α]'
      end
      F[i, k] += fi; V[k] += vk
   end
   @inbounds for (kk, k) in enumerate(rows_pair)
      e = zero(T); fi = zero(SVector{3, T}); vk = zero(SMatrix{3, 3, T, 9})
      for α = 1:nneigh
         e += ws.Rpair[α, kk]
         d = ws.dRpair[α, kk] * ws.∇rs[α]
         F[Js[α], k] -= d; fi += d; vk -= d * Rs[α]'
      end
      E[k] += e; F[i, k] += fi; V[k] += vk
   end
   return nothing
end

function efv_basis_v4(at, calc::ACEPotential{<: ACEModel}; nlist = PairList(at, cutoff_radius(calc)),
                      ws = nothing)
   model = calc.model; ps = calc.ps; st = calc.st
   nB = length_basis(model); nat = length(at)
   maxneigh = maximum(length(get_neighbours(at, calc, nlist, i)[1]) for i in 1:nat)
   if ws === nothing || ws.maxneigh < maxneigh
      ws = PFWorkspace(model, maxneigh)
   end
   E = zeros(nB); F = zeros(SVector{3, Float64}, nat, nB); V = zeros(SMatrix{3, 3, Float64, 9}, nB)
   A2B = model.tensor.A2Bmaps[1]
   for i = 1:nat
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      n = length(Rs)
      rs = view(ws.rs, 1:n); ∇rs = view(ws.∇rs, 1:n)
      radii_ed!(rs, ∇rs, Rs)
      Rnl = view(ws.Rnl, 1:n, :); dRnl_r = view(ws.dRnl_r, 1:n, :)
      ACEpotentials.Models.evaluate_ed_batched!(Rnl, dRnl_r, model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
      ∂Rnl = view(ws.∂Rnl, 1:n, :)
      @inbounds for k = 1:size(Rnl, 2), j = 1:n
         ∂Rnl[j, k] = dRnl_r[j, k] * ∇rs[j]
      end
      Ylm = view(ws.Ylm, 1:n, :); ∂Ylm = view(ws.∂Ylm, 1:n, :)
      P4ML.evaluate_ed!(Ylm, ∂Ylm, model.ybasis, Rs)
      ∂A = view(ws.∂A, 1:n, :)
      _pf_A!(ws.A, ∂A, model.tensor.abasis.spec, Rnl, ∂Rnl, Ylm, ∂Ylm)
      ∂AA = view(ws.∂AA, 1:n, :)
      _pf_AA!(ws.AA, ∂AA, model.tensor.aabasis, ws.A, ∂A)
      mul!(ws.Bi, A2B, ws.AA)
      ∂AAt = view(ws.∂AAt, :, 1:n); permutedims!(∂AAt, ∂AA, (2, 1))
      ∂Bi = view(ws.∂Bi, :, 1:n)
      mul!(∂Bi, A2B, ∂AAt)
      Rpair = view(ws.Rpair, 1:n, :); dRpair = view(ws.dRpair, 1:n, :)
      ACEpotentials.Models.evaluate_ed_batched!(Rpair, dRpair, model.pairbasis, rs, z0, Zs, ps.pairbasis, st.pairbasis)
      _accumulate_pf!(E, F, V, ws, get_basis_inds(model, z0), get_pairbasis_inds(model, z0), Js, Rs, i, n)
   end
   return (energy = E, forces = F, virial = V)
end
