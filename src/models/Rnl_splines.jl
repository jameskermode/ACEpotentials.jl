

# ------------------------------------------------------------ 
#      CONSTRUCTORS AND UTILITIES 
# ------------------------------------------------------------ 


Base.length(basis::SplineRnlrzzBasis) = length(basis.spec)

function initialparameters(rng::AbstractRNG, 
                           basis::SplineRnlrzzBasis)
   return NamedTuple()
end                           

function initialstates(rng::AbstractRNG, 
                       basis::SplineRnlrzzBasis)
   return NamedTuple()                       
end
                  

# ------------------------------------------------------------ 
#      EVALUATION INTERFACE
# ------------------------------------------------------------ 

(l::SplineRnlrzzBasis)(args...) = evaluate(l, args...)


# function evaluate!(Rnl, basis::SplineRnlrzzBasis, r::Real, Zi, Zj, ps, st)
#    Rnl[:] .= evaluate(basis, r, Zi, Zj, ps, st)
#    return Rnl, st 
# end


function evaluate(basis::SplineRnlrzzBasis, r::Real, Zi, Zj, ps, st)
   iz = _z2i(basis, Zi)
   jz = _z2i(basis, Zj)
   T_ij = basis.transforms[iz, jz]
   env_ij = basis.envelopes[iz, jz]
   spl_ij = basis.splines[iz, jz]

   x_ij = T_ij(r)
   e_ij = evaluate(env_ij, r, x_ij)

   return spl_ij(x_ij) * e_ij
end


function evaluate_batched!(Rnl, basis::SplineRnlrzzBasis,
                           rs, zi, zjs, ps, st)
   @assert length(rs) == length(zjs)
   @assert size(Rnl, 1) >= length(rs) && size(Rnl, 2) >= length(basis)
   _spline_tables_batched!(Rnl, nothing, basis.tables, _z2i(basis, zi), rs, zjs, basis)
   return Rnl
end

function whatalloc(::typeof(evaluate_batched!), 
                   basis::SplineRnlrzzBasis, 
                   rs, zi, zjs, ps, st)
   T = eltype(rs)
   return (T, length(rs), length(basis))
end                   


function evaluate_batched(basis::SplineRnlrzzBasis, 
                           rs, zi, zjs, ps, st)
   Rnl = zeros(whatalloc(evaluate_batched!, basis, rs, zi, zjs, ps, st)...)
   return evaluate_batched!(Rnl, basis, rs, zi, zjs, ps, st)
end

# ----- gradients 
# because the typical scenario is that we have few r, then moderately 
# many q and then many (n, l), this seems to be best done in Forward-mode. 


import ForwardDiff
using ForwardDiff: Dual

function evaluate_ed(basis::SplineRnlrzzBasis, r::T, Zi, Zj, ps, st) where {T <: Real}
   d_r = Dual{T}(r, one(T))
   d_Rnl = evaluate(basis, d_r, Zi, Zj, ps, st)
   Rnl = ForwardDiff.value.(d_Rnl)
   Rnl_d = ForwardDiff.extract_derivative(T, d_Rnl) 
   return Rnl, Rnl_d 
end



function evaluate_ed_batched!(Rnl, Rnl_d,
                             basis::SplineRnlrzzBasis,
                             rs::AbstractVector{T}, Zi, Zs, ps, st
                             ) where {T <: Real}
   @assert length(rs) == length(Zs)
   @assert size(Rnl, 1) >= length(rs) && size(Rnl, 2) >= length(basis)
   @assert size(Rnl_d, 1) >= length(rs) && size(Rnl_d, 2) >= length(basis)
   _spline_tables_batched!(Rnl, Rnl_d, basis.tables, _z2i(basis, Zi), rs, Zs, basis)
   return Rnl, Rnl_d
end

function whatalloc(::typeof(evaluate_ed_batched!), 
                  basis::SplineRnlrzzBasis, 
                  rs::AbstractVector, Zi, Zs, ps, st)
   T = eltype(rs)
   return (T, length(rs), length(basis)), (T, length(rs), length(basis))
end


function evaluate_ed_batched(basis::SplineRnlrzzBasis, 
                             rs::AbstractVector, Zi, Zs, ps, st)
   alc_Rnl, alc_Rnl_d = whatalloc(evaluate_ed_batched!, basis, rs, Zi, Zs, ps, st)
   Rnl = zeros(alc_Rnl...)
   Rnl_d = zeros(alc_Rnl_d...)
   return evaluate_ed_batched!(Rnl, Rnl_d, basis, rs, Zi, Zs, ps, st)
end


function rrule(::typeof(evaluate_batched), 
               basis::SplineRnlrzzBasis, 
               rs, zi, zjs, ps, st)
   Rnl = evaluate_batched(basis, rs, zi, zjs, ps, st)

   return Rnl, 
         Δ -> (NoTangent(), NoTangent(), NoTangent(), NoTangent(), 
              NamedTuple(), NoTangent())
end


# ------------------------------------------------------------
#      COMPACT SPLINE TABLES + FAST BATCHED KERNEL
# ------------------------------------------------------------
#
# `SplineRnlrzzBasis` stores one cubic B-spline per species pair whose value
# is an `SVector{LEN}`.  Evaluating it per edge through `Interpolations` with
# `Dual` numbers builds `SVector{LEN, Dual}` objects and dominates the cost of
# a site evaluation (see docs/findings/FINDINGS_forward_profile.md).
#
# `RnlSplineTables` holds the same B-spline coefficients as plain arrays, and
# exploits that for the bases produced by `splinify` of a one-hot
# (`set_onehot_weights!`) or otherwise l-independent weight tensor every
# column `Rnl[:, n]` is a scalar multiple of one of a few distinct columns:
#
#     Rnl[:, n] = c[n, pair] * P[:, u[n, pair], pair]
#
# with NU = size(P, 2) << LEN distinct columns per species pair.  The NU
# distinct splines are evaluated once per edge (value and derivative in one
# pass, on plain floats) and then expanded.  The factorisation is detected
# numerically when the tables are built; a basis whose columns are not
# proportional gets the trivial factorisation (NU = LEN, c = 1, u = n) and
# goes through exactly the same kernel.
#
# The B-spline arithmetic follows `Interpolations.jl`'s
# `BSpline(Cubic(Line(OnGrid())))` on a uniform grid: for the index
# coordinate xl = (x - x0)/h + 1 in [1, n], the four coefficients
# xf-1 .. xf+2 (padded axes 0:n+1) are combined with the cubic weights
# below (`Interpolations.value_weights` / `gradient_weights`).

using LinearAlgebra: norm, dot

_pair_index(NZ, iz, jz) = (jz - 1) * NZ + iz

# factorise the columns of C (npad x LEN) into distinct columns up to scale
function _factorise_columns(C::AbstractMatrix{T}; rtol = 1e-12) where {T}
   npad, LEN = size(C)
   P = Vector{Vector{T}}()
   u = zeros(Int, LEN)
   c = zeros(T, LEN)
   for n = 1:LEN
      col = @view C[:, n]
      nc = norm(col)
      if nc == 0
         isempty(P) && push!(P, zeros(T, npad))
         u[n] = 1; c[n] = zero(T)
         continue
      end
      found = false
      for (k, p) in enumerate(P)
         np = norm(p)
         np == 0 && continue
         s = dot(col, p) / dot(p, p)
         if norm(col .- s .* p) <= rtol * nc
            u[n] = k; c[n] = s; found = true
            break
         end
      end
      if !found
         push!(P, collect(col)); u[n] = length(P); c[n] = one(T)
      end
   end
   return P, u, c
end

function RnlSplineTables(_i2z::NTuple{NZ, Int},
                         transforms::AbstractMatrix, envelopes::AbstractMatrix,
                         splines::AbstractMatrix{<: SPL_OF_SVEC{LEN, T}};
                         factorise = true, rtol = 1e-12) where {NZ, LEN, T}
   spl11 = splines[1, 1]
   rng = spl11.itp.ranges[1]
   nnodes = length(rng)
   ax = axes(spl11.itp.itp.coefs, 1)
   npad = length(ax)
   @assert first(ax) == 0 && npad == nnodes + 2
   # raw coefficient tables per pair, column-major over (iz, jz)
   Cs = Matrix{T}[]
   for jz = 1:NZ, iz = 1:NZ
      spl = splines[iz, jz]
      @assert spl.itp.ranges[1] == rng
      cf = spl.itp.itp.coefs
      C = zeros(T, npad, LEN)
      for (r, i) in enumerate(ax)
         C[r, :] .= cf[i]
      end
      push!(Cs, C)
   end
   # factorise (and verify), else the trivial factorisation
   facs = factorise ? [ _factorise_columns(C; rtol = rtol) for C in Cs ] : nothing
   ok = factorise
   if ok
      for (p, (Pc, up, cp)) in enumerate(facs)
         Cmax = max(one(T), maximum(abs, Cs[p]; init = zero(T)))
         for n = 1:LEN
            err = maximum(abs, Cs[p][:, n] .- cp[n] .* Pc[up[n]])
            if err > 100 * rtol * Cmax
               ok = false
            end
         end
      end
   end
   if ok
      NU = maximum(length(f[1]) for f in facs)
      P = zeros(T, npad, NU, NZ * NZ)
      c = zeros(T, LEN, NZ * NZ)
      u = ones(Int, LEN, NZ * NZ)
      for p = 1:NZ*NZ
         Pc, up, cp = facs[p]
         for k = 1:length(Pc)
            P[:, k, p] .= Pc[k]
         end
         u[:, p] .= up
         c[:, p] .= cp
      end
   else
      NU = LEN
      P = zeros(T, npad, LEN, NZ * NZ)
      for p = 1:NZ*NZ
         P[:, :, p] .= Cs[p]
      end
      c = ones(T, LEN, NZ * NZ)
      u = repeat(collect(1:LEN), 1, NZ * NZ)
   end
   return RnlSplineTables(P, c, u, T(first(rng)), T(last(rng)), T(step(rng)), nnodes, NU, LEN, ok,
                          Matrix(transforms), Matrix(envelopes))
end

# cubic B-spline position and weights on the index grid 1:n, following
# Interpolations.positions / value_weights / gradient_weights
@inline function _cubic_pos(xl, n::Int)
   xf = floor(ForwardDiff.value(xl))    # plain float also for Dual xl
   xf = ifelse(xf > n - 1, xf - one(xf), xf)
   δ = xl - xf
   return Int(xf), δ       # coefficient rows (padded, 1-based): xf, xf+1, xf+2, xf+3
end

@inline function _cubic_value_weights(δ)
   δc = 1 - δ
   return (δc^3 / 6,
           2/3 - δ^2 + δ^3 / 2,
           2/3 - δc^2 + δc^3 / 2,
           δ^3 / 6)
end

@inline function _cubic_gradient_weights(δ)
   δc = 1 - δ
   return (-δc^2 / 2,
           -2 * δ + 3 * δ^2 / 2,
           2 * δc - 3 * δc^2 / 2,
           δ^2 / 2)
end

@inline function _check_spline_domain(tab::RnlSplineTables, x)
   if !(tab.x0 <= x <= tab.x1)
      throw(DomainError(x, "spline argument outside [$(tab.x0), $(tab.x1)]"))
   end
   return nothing
end

# Rnl (and, if Rnl_d !== nothing, Rnl_d) for the edges (rs, zjs) of a centre
# of species index iz.  Generic in the element type of rs so that Dual
# numbers can be pushed through (as the previous implementation allowed).
function _spline_tables_batched!(Rnl, Rnl_d, tab::RnlSplineTables,
                                 iz::Int, rs::AbstractVector{T}, zjs,
                                 basis::SplineRnlrzzBasis{NZ}) where {T, NZ}
   nX = length(rs)
   NU = tab.NU; LEN = tab.LEN
   P = tab.P; c = tab.c; u = tab.u
   x0 = tab.x0; h = tab.h; nn = tab.nnodes
   withgrad = !(Rnl_d === nothing)
   @no_escape begin
      Pv = @alloc(T, nX, NU)
      Pg = @alloc(T, nX, NU)
      pr = @alloc(Int, nX)
      # (1) distinct columns per edge: value and derivative w.r.t. r
      @inbounds for j = 1:nX
         jz = _z2i(basis, zjs[j])
         p = _pair_index(NZ, iz, jz)
         pr[j] = p
         Tij = tab.transforms[iz, jz]
         env = tab.envelopes[iz, jz]
         if withgrad
            d_r = Dual{T}(rs[j], one(T))
            x_d = Tij(d_r)
            e_d = evaluate(env, d_r, x_d)
            x = ForwardDiff.value(x_d); dx = ForwardDiff.extract_derivative(T, x_d)
            e = ForwardDiff.value(e_d); de = ForwardDiff.extract_derivative(T, e_d)
         else
            x = Tij(rs[j])
            e = evaluate(env, rs[j], x)
            dx = zero(T); de = zero(T)
         end
         _check_spline_domain(tab, x)
         xl = (x - x0) / h + 1
         i, δ = _cubic_pos(xl, nn)
         wv = _cubic_value_weights(δ)
         if withgrad
            wg = _cubic_gradient_weights(δ)
            sg = dx / h
            for k = 1:NU
               v = wv[1] * P[i, k, p] + wv[2] * P[i+1, k, p] + wv[3] * P[i+2, k, p] + wv[4] * P[i+3, k, p]
               g = wg[1] * P[i, k, p] + wg[2] * P[i+1, k, p] + wg[3] * P[i+2, k, p] + wg[4] * P[i+3, k, p]
               Pv[j, k] = e * v
               Pg[j, k] = de * v + e * sg * g
            end
         else
            for k = 1:NU
               v = wv[1] * P[i, k, p] + wv[2] * P[i+1, k, p] + wv[3] * P[i+2, k, p] + wv[4] * P[i+3, k, p]
               Pv[j, k] = e * v
            end
         end
      end
      # (2) expand to the full basis
      @inbounds for n = 1:LEN
         @simd ivdep for j = 1:nX
            p = pr[j]
            Rnl[j, n] = c[n, p] * Pv[j, u[n, p]]
         end
      end
      if withgrad
         @inbounds for n = 1:LEN
            @simd ivdep for j = 1:nX
               p = pr[j]
               Rnl_d[j, n] = c[n, p] * Pg[j, u[n, p]]
            end
         end
      end
   end
   return nothing
end
