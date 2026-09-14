# Scratch prototype of an optimised forward evaluator (energy, forces, virial)
# for a fitted ACEModel with SPLINE radial bases.  Nothing in src/ is modified;
# this file builds a `FastEFV` object from an ACEPotential and evaluates it.
#
# What it changes relative to the production path (see FINDINGS_forward_profile.md):
#   R1  radial splines: a hand-written cubic B-spline kernel on a plain
#       coefficient matrix, value+derivative in one pass, transform/envelope as
#       scalar Duals once per edge (no SVector{LEN, Dual} objects).
#   R2  factorised radial: the spline table is reduced to its distinct columns
#       (P_{n'} polynomials) and each Rnl column is  c[jz, n] * P[:, u(n)]  --
#       this is exact for both the categorical (one-hot, c ∈ {0,1}) and the
#       embedded (c = emb[jz, k]) models.
#   T1  fold WB through A2B once per element: wAA_z = A2B' * WB[:, z]; then
#       E_i = wAA ⋅ AA and ∂AA = wAA (no B, no sparse matvecs, no KA launches).
#   T2  A basis computed once (production computes it twice).
#   W1  all per-site buffers live in a reusable workspace (one per task).
#   L1  site loop reads the PairList directly, species pre-gathered once per
#       call, forces accumulated unit-free and units attached at the end.
#   L2  threading with Threads.@spawn over chunks, per-task workspace + force
#       accumulator, reduced at the end.

using ACEpotentials, AtomsBase, AtomsCalculators, LinearAlgebra, StaticArrays, Unitful
using ACEpotentials.Models.ForwardDiff: Dual, value, partials
const M = ACEpotentials.Models
const ET = M.EquivariantTensors
const P4ML = M.P4ML
const Interp = M.Interpolations
using ACEpotentials.Models.NeighbourLists: PairList
const NeighbourLists = ACEpotentials.Models.NeighbourLists

# ---------------------------------------------------------------------------
#  R1/R2: spline tables
# ---------------------------------------------------------------------------

struct FastRadial{NZ, TT, TENV, T}
   i2z::NTuple{NZ, Int}
   transforms::Matrix{TT}         # NZ x NZ  (plain Matrix: no SMatrix tuple-indexing)
   envelopes::Matrix{TENV}
   P::Array{T, 3}                 # (npad, NU, NZ*NZ): distinct spline columns per pair
   c::Matrix{T}                   # (LEN, NZ*NZ)   : Rnl[:, n] = c[n, pair] * P[:, u[n, pair], pair]
   u::Matrix{Int}                 # (LEN, NZ*NZ)   : column of P used by Rnl column n for this pair
   x0::T; h::T                    # spline node range  x0 : h : ...
   ax::UnitRange{Int}             # axes of the coefficient vector (e.g. 0:101)
   LEN::Int; NU::Int
end

# factorise the columns of C (npad x LEN) into distinct columns up to scale
function _factorise_columns(C::Matrix{T}; rtol = 1e-12) where {T}
   npad, LEN = size(C)
   P = Vector{Vector{T}}(); u = zeros(Int, LEN); c = zeros(T, LEN)
   for n = 1:LEN
      col = @view C[:, n]
      nc = norm(col)
      if nc == 0
         u[n] = 1; c[n] = 0; isempty(P) && push!(P, zeros(T, npad)); continue
      end
      found = false
      for (k, p) in enumerate(P)
         np = norm(p); np == 0 && continue
         s = dot(col, p) / dot(p, p)
         if norm(col .- s .* p) <= rtol * nc
            u[n] = k; c[n] = s; found = true; break
         end
      end
      if !found
         push!(P, collect(col)); u[n] = length(P); c[n] = one(T)
      end
   end
   return P, u, c
end

function FastRadial(basis::M.SplineRnlrzzBasis{NZ}) where {NZ}
   spl11 = basis.splines[1, 1]
   rng = spl11.itp.ranges[1]
   coefs = spl11.itp.itp.coefs
   ax = axes(coefs, 1); ax = first(ax):last(ax)
   LEN = length(basis.spec)
   npad = length(ax)
   # raw coefficient tables per pair
   Cs = Matrix{Float64}[]
   for jz = 1:NZ, iz = 1:NZ       # column-major over (iz, jz)
      spl = basis.splines[iz, jz]
      @assert spl.itp.ranges[1] == rng
      cf = spl.itp.itp.coefs
      C = zeros(npad, LEN)
      for (r, i) in enumerate(ax); C[r, :] .= cf[i]; end
      push!(Cs, C)
   end
   # factorise each pair table separately: P[:, k, p] are the distinct columns
   # of pair p, u[n, p] the column used by Rnl column n, c[n, p] its scale
   facs = [ _factorise_columns(C) for C in Cs ]
   NU = maximum(length(f[1]) for f in facs)
   P = zeros(npad, NU, NZ * NZ)
   c = zeros(LEN, NZ * NZ)
   u = ones(Int, LEN, NZ * NZ)
   for p = 1:NZ*NZ
      Pcols, up, cp = facs[p]
      for k = 1:length(Pcols); P[:, k, p] .= Pcols[k]; end
      u[:, p] .= up; c[:, p] .= cp
   end
   # verify the factorisation is exact
   err = 0.0
   for p = 1:NZ*NZ, n = 1:LEN
      err = max(err, maximum(abs, Cs[p][:, n] .- c[n, p] .* P[:, u[n, p], p]))
   end
   err > 1e-10 && error("column factorisation is not exact: err = $err")
   transforms = Matrix(basis.transforms); envelopes = Matrix(basis.envelopes)
   return FastRadial(basis._i2z, transforms, envelopes, P, c, u,
                     Float64(first(rng)), Float64(step(rng)), ax, LEN, NU)
end

_z2i(fr::FastRadial{NZ}, z) where {NZ} = findfirst(==(z), fr.i2z)::Int

const CUBIC = Interp.Cubic(Interp.Line(Interp.OnGrid()))

# per-edge scalar data for the spline kernel
struct EdgeSpl{T}
   i::Int                     # base coefficient row (1-based into P)
   wv::NTuple{4, T}
   wg::NTuple{4, T}
   e::T; de::T; s::T          # envelope, d envelope / dr, (dx/dr)/h
   pair::Int
end

@inline function _edge_spl(fr::FastRadial, r, iz, jz)
   Tij = fr.transforms[iz, jz]; env = fr.envelopes[iz, jz]
   d_r = Dual{Nothing}(r, one(r))
   x_d = Tij(d_r)
   e_d = M.evaluate(env, d_r, x_d)
   x = value(x_d); dx = partials(x_d, 1)
   e = value(e_d); de = partials(e_d, 1)
   xl = (x - fr.x0) / fr.h + 1
   (pos, δx) = Interp.positions(CUBIC, fr.ax, xl)
   wv = Float64.(Interp.value_weights(CUBIC, δx))
   wg = Float64.(Interp.gradient_weights(CUBIC, δx))
   i = pos - first(fr.ax) + 1
   return EdgeSpl(i, wv, wg, e, de, dx / fr.h, (jz - 1) * length(fr.i2z) + iz)
end

# evaluate Rnl, dRnl (nX x LEN) for the edges; es and Pv/Pg are workspaces
function radial_ed!(Rnl, dRnl, fr::FastRadial{NZ}, rs, iz, jzs, es, Pv, Pg, pr) where {NZ}
   nX = length(rs)
   @inbounds for j = 1:nX
      es[j] = _edge_spl(fr, rs[j], iz, jzs[j])
   end
   P = fr.P
   # distinct columns: value and d/dx for every edge
   @inbounds for k = 1:fr.NU
      for j = 1:nX
         s = es[j]; i = s.i; p = s.pair
         v = s.wv[1] * P[i, k, p] + s.wv[2] * P[i+1, k, p] + s.wv[3] * P[i+2, k, p] + s.wv[4] * P[i+3, k, p]
         g = s.wg[1] * P[i, k, p] + s.wg[2] * P[i+1, k, p] + s.wg[3] * P[i+2, k, p] + s.wg[4] * P[i+3, k, p]
         Pv[j, k] = v
         Pg[j, k] = g
      end
   end
   # fold the envelope into the distinct columns once (Pv <- e*Pv, Pg <- de*Pv + e*s*Pg)
   @inbounds for k = 1:fr.NU
      @simd ivdep for j = 1:nX
         s = es[j]
         v = Pv[j, k]; g = Pg[j, k]
         Pv[j, k] = s.e * v
         Pg[j, k] = s.de * v + s.e * s.s * g
      end
   end
   # expand: Rnl[j, n] = c[n, pair_j] * Pv[j, u[n, pair_j]]
   c = fr.c; u = fr.u
   @inbounds for j = 1:nX; pr[j] = es[j].pair; end
   @inbounds for n = 1:fr.LEN
      @simd ivdep for j = 1:nX
         p = pr[j]
         cn = c[n, p]; k = u[n, p]
         Rnl[j, n] = cn * Pv[j, k]
         dRnl[j, n] = cn * Pg[j, k]
      end
   end
   return Rnl, dRnl
end

# radial part: s_j = Σ_t ∂Rnl[j,t]*dRnl[j,t] (SIMD over j), then ∇Ei[j] = s_j * ∇rs[j];
# Ylm part as before (dYlm entries are SVectors).
function _assemble_grad_fast!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs, sbuf, nX)
   @inbounds for j = 1:nX; sbuf[j] = 0.0; end
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

# ---------------------------------------------------------------------------
#  workspaces + folded weights
# ---------------------------------------------------------------------------

mutable struct Workspace{T}
   nmax::Int
   rs::Vector{T}; ∇rs::Vector{SVector{3, T}}
   Rs::Vector{SVector{3, T}}; jzs::Vector{Int}; Js::Vector{Int}
   es::Vector{EdgeSpl{T}}
   Pv::Matrix{T}; Pg::Matrix{T}
   Rnl::Matrix{T}; dRnl::Matrix{T}
   Ylm::Matrix{T}; dYlm::Matrix{SVector{3, T}}
   A::Vector{T}; AA::Vector{T}; ∂A::Vector{T}
   ∂Rnl::Matrix{T}; ∂Ylm::Matrix{T}
   ∇Ei::Vector{SVector{3, T}}
   rs_scratch::Vector{T}; pr::Vector{Int}
   Ppv::Matrix{T}; Ppg::Matrix{T}; Rp::Matrix{T}; dRp::Matrix{T}
end

function Workspace(fe, nmax)
   T = Float64
   NU = fe.rad.NU; LEN = fe.rad.LEN; nY = length(fe.model.ybasis)
   nA = length(fe.model.tensor.abasis); nAA = length(fe.model.tensor.aabasis)
   NUp = fe.pair.NU; LENp = fe.pair.LEN
   Workspace{T}(nmax,
      zeros(T, nmax), zeros(SVector{3,T}, nmax), zeros(SVector{3,T}, nmax), zeros(Int, nmax), zeros(Int, nmax),
      Vector{EdgeSpl{T}}(undef, nmax),
      zeros(T, nmax, NU), zeros(T, nmax, NU),
      zeros(T, nmax, LEN), zeros(T, nmax, LEN),
      zeros(T, nmax, nY), zeros(SVector{3,T}, nmax, nY),
      zeros(T, nA), zeros(T, nAA), zeros(T, nA),
      zeros(T, nmax, LEN), zeros(T, nmax, nY),
      zeros(SVector{3,T}, nmax), zeros(T, nmax), zeros(Int, nmax),
      zeros(T, nmax, NUp), zeros(T, nmax, NUp), zeros(T, nmax, LENp), zeros(T, nmax, LENp))
end

struct FastEFV{TM, TR, TP, T}
   model::TM
   rad::TR
   pair::TP
   wAA::Vector{Vector{T}}       # per element: A2B' * WB[:, iz]
   Wpair::Matrix{T}
   E0::Vector{T}
   rcut::T
   i2z::Vector{Int}
end

function FastEFV(V::M.ACEPotential)
   model = V.model
   rad = FastRadial(model.rbasis)
   pair = FastRadial(model.pairbasis)
   A2B = model.tensor.A2Bmaps[1]
   NZ = length(model._i2z)
   wAA = [ Vector(A2B' * V.ps.WB[:, iz]) for iz = 1:NZ ]
   E0 = [ model.Vref.E0[z] for z in model._i2z ]
   rcut = ustrip(u"Å", M.cutoff_radius(V))
   return FastEFV(model, rad, pair, wAA, Matrix(V.ps.Wpair), E0, rcut, collect(model._i2z))
end

# ---------------------------------------------------------------------------
#  site evaluation: Ei, ∇Ei into ws.∇Ei[1:nX]
# ---------------------------------------------------------------------------

# use views of the (nmax x ...) workspaces sized to the actual nX
_rows(A::AbstractMatrix, nX) = view(A, 1:nX, :)

function site_ed!(fe::FastEFV, ws::Workspace, nX, iz)
   model = fe.model
   Rs = view(ws.Rs, 1:nX); jzs = view(ws.jzs, 1:nX)
   rs = view(ws.rs, 1:nX); ∇rs = view(ws.∇rs, 1:nX)
   @inbounds for j = 1:nX
      r = norm(Rs[j]); rs[j] = r; ∇rs[j] = Rs[j] / r
   end
   Rnl = _rows(ws.Rnl, nX); dRnl = _rows(ws.dRnl, nX)
   radial_ed!(Rnl, dRnl, fe.rad, rs, iz, jzs, ws.es, ws.Pv, ws.Pg, ws.pr)
   Ylm = _rows(ws.Ylm, nX); dYlm = _rows(ws.dYlm, nX)
   P4ML.evaluate_ed!(Ylm, dYlm, model.ybasis, Rs)
   A = ws.A; AA = ws.AA; ∂A = ws.∂A
   ET.evaluate!(A, model.tensor.abasis, (Rnl, Ylm))
   ET.evaluate!(AA, model.tensor.aabasis, A)
   wAA = fe.wAA[iz]
   Ei = dot(wAA, AA)
   ET.pullback!(∂A, wAA, model.tensor.aabasis, A)
   ∂Rnl = _rows(ws.∂Rnl, nX); ∂Ylm = _rows(ws.∂Ylm, nX)
   ET.pullback!((∂Rnl, ∂Ylm), ∂A, model.tensor.abasis, (Rnl, Ylm))
   ∇Ei = ws.∇Ei
   _assemble_grad_fast!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs, ws.rs_scratch, nX)
   # pair
   Rp = _rows(ws.Rp, nX); dRp = _rows(ws.dRp, nX)
   radial_ed!(Rp, dRp, fe.pair, rs, iz, jzs, ws.es, ws.Ppv, ws.Ppg, ws.pr)
   Wp = fe.Wpair
   @inbounds for n = 1:size(Rp, 2)
      w = Wp[n, iz]
      for j = 1:nX
         Ei += w * Rp[j, n]
         ∇Ei[j] += (w * dRp[j, n]) * ∇rs[j]
      end
   end
   Ei += fe.E0[iz]
   return Ei
end

# ---------------------------------------------------------------------------
#  system-level driver
# ---------------------------------------------------------------------------

function _gather!(ws::Workspace, fe::FastEFV, nlist::PairList, izs::Vector{Int}, i)
   n1, n2 = nlist.first[i], nlist.first[i+1]-1
   nX = n2 - n1 + 1
   if nX > ws.nmax
      error("workspace too small: nX = $nX > nmax = $(ws.nmax)")
   end
   @inbounds for (a, n) in enumerate(n1:n2)
      j = nlist.j[n]
      ws.Js[a] = j
      ws.Rs[a] = NeighbourLists._getR(nlist, n)
      ws.jzs[a] = izs[j]
   end
   return nX
end

function _chunk!(fe::FastEFV, ws::Workspace, nlist, izs, F::Vector{SVector{3,Float64}}, sites)
   E = 0.0
   Vir = zero(SMatrix{3, 3, Float64})
   for i in sites
      nX = _gather!(ws, fe, nlist, izs, i)
      Ei = site_ed!(fe, ws, nX, izs[i])
      E += Ei
      ∇Ei = ws.∇Ei; Js = ws.Js; Rs = ws.Rs
      fi = zero(SVector{3, Float64})
      @inbounds for a = 1:nX
         g = ∇Ei[a]
         F[Js[a]] -= g
         fi += g
         Vir -= g * Rs[a]'
      end
      F[i] += fi
   end
   return E, Vir
end

mutable struct FastEFVState
   nlist::Any
   wss::Vector{Workspace{Float64}}
   Fs::Vector{Vector{SVector{3,Float64}}}
end

function max_nneigh(nlist::PairList)
   m = 0
   for i = 1:length(nlist.first)-1
      m = max(m, nlist.first[i+1] - nlist.first[i])
   end
   return m
end

# nlist can be passed in to test reuse; ntasks = 1 => serial
function fast_efv(sys, fe::FastEFV; nlist = PairList(sys, fe.rcut * u"Å"),
                  ntasks = Threads.nthreads(), wss = nothing)
   nat = length(sys)
   izs = [ findfirst(==(atomic_number(sys, i)), fe.i2z)::Int for i = 1:nat ]
   nmax = max_nneigh(nlist)
   if wss === nothing || length(wss) < ntasks || wss[1].nmax < nmax
      wss = [ Workspace(fe, nmax + 8) for _ = 1:ntasks ]
   end
   chunks = collect(Iterators.partition(1:nat, cld(nat, ntasks)))
   if ntasks == 1
      F = zeros(SVector{3,Float64}, nat)
      E, Vir = _chunk!(fe, wss[1], nlist, izs, F, 1:nat)
   else
      Fs = [ zeros(SVector{3,Float64}, nat) for _ = 1:length(chunks) ]
      tasks = [ Threads.@spawn _chunk!(fe, wss[c], nlist, izs, Fs[c], chunks[c]) for c = 1:length(chunks) ]
      res = fetch.(tasks)
      E = sum(r[1] for r in res); Vir = sum(r[2] for r in res)
      F = Fs[1]
      for c = 2:length(chunks); F .+= Fs[c]; end
   end
   return (energy = E * u"eV", forces = F .* u"eV/Å", virial = Vir * u"eV"), wss
end
