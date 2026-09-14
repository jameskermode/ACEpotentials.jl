# Variable-projection learning of the species weights INSIDE a Finnis-Sinclair
# density, with the ACE part of the design matrix fixed (cached assemblies).
#
# Extra columns for P densities p = 1..P, per central species a:
#     rho_i^{(p)} = sum_{s,k} w^{(p)}_{s,k} rho_i^{s,k},  w = exp(theta)
#     rho_i^{s,k} = sum_{j: z_j = s, r_ij < rcut} g_k(r_ij),  g_k as in fs_columns.jl
#     column (a,p): sum_{i: z_i = a} phi(rho_i^{(p)}),  phi(x) = sqrt(x+eps) - sqrt(eps)
# with E / F / V rows in ACEfit's row layout.  The builder is generic in the
# element type so ForwardDiff can differentiate w.r.t. theta (and log alpha).
#
# Inner problem (fixed theta): Tikhonov LS in the weighted, prior-scaled
# variables, lambda fixed.  Solved by projecting the extra columns out of the
# range of the (once-factorised) ACE block, which is exact for the augmented
# problem [A X; lam I].  Outer gradient by Kaufman/VarPro: dL/dtheta =
# 2 r' (dX/dtheta) c_X at c = c*(theta), which is the exact gradient of the
# reduced objective (dL/dc = 0 at c*).
using ACEpotentials, ACEfit
using AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Serialization, Statistics, Unitful
include(joinpath(@__DIR__, "fs_columns.jl"))       # g, dg, fcut, NL, FSSpec
include(joinpath(@__DIR__, "fs_embed.jl"))         # EmbedSpec, wsqrt_fun (hand tilts)
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_THREADS", "4")))   # the gemms here are small; 12 threads thrash when the box is shared
const M = ACEpotentials.Models
const FDiff = ACEpotentials.Models.ForwardDiff
const Optim = Base.require(Base.PkgId(Base.UUID("429524aa-4258-5aef-a3af-852621145aeb"), "Optim"))

const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
const S = 5
const K = 3
const KS = S * K
const ALPHAS0 = [2.0, 4.0, 6.0]
const R0 = 2.5
const RCUT = 6.25
const EPSR = 1e-8
const LAMS = 10.0 .^ (0:-1:-8)
const ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]

# ---------------------------------------------------------------- data & splits
load_all() = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
function split0(data_all)
   p = shuffle(MersenneTwister(0), 1:length(data_all))
   return data_all[p[1:200]], data_all[p[201:300]]
end
function split1(data_all)
   p0 = shuffle(MersenneTwister(0), 1:length(data_all))
   p1 = shuffle(MersenneTwister(1), sort(p0[301:end]))
   return data_all[p1[1:200]], data_all[p1[201:300]]
end
function row_layout(data)
   kind = Int[]; nat = Int[]; sid = Int[]
   for (is, sys) in enumerate(data)
      n = length(sys)
      push!(kind, 1); push!(nat, n); push!(sid, is)
      append!(kind, fill(2, 3n)); append!(nat, fill(n, 3n)); append!(sid, fill(is, 3n))
      append!(kind, fill(3, 6)); append!(nat, fill(n, 6)); append!(sid, fill(is, 6))
   end
   return (kind = kind, nat = nat, sid = sid)
end
function rmse_efv(resid, lay)
   r = copy(resid); r[lay.kind .!= 2] ./= lay.nat[lay.kind .!= 2]
   f(k) = sqrt(sum(abs2, r[lay.kind .== k]) / count(==(k), lay.kind))
   return (F = f(2), E = f(1), V = f(3))
end
function load_asm(name)
   A, Y, W = deserialize(joinpath(CACHEDIR, name))
   return (A = A, Y = Y, W = W)
end
function prior_diag(D)
   m = ace1_model(elements = ELS, order = 3, totaldegree = D)
   return M.algebraic_smoothness_prior(m.model; p = 4).diag
end

# ---------------------------------------------------------------- pair data
struct StructPairs
   nat::Int
   zidx::Vector{Int}
   pi::Vector{Int}
   pj::Vector{Int}
   r::Vector{Float64}
   R::Vector{SVector{3, Float64}}
end
function StructPairs(sys)
   nat = length(sys)
   zidx = [findfirst(==(atomic_number(sys, i)), ZS) for i in 1:nat]
   any(isnothing, zidx) && error("species not in ZS")
   nlist = NL.PairList(sys, RCUT * u"Å")
   pi = Int[]; pj = Int[]; rr = Float64[]; RR = SVector{3, Float64}[]
   for (i, j, R) in NL.pairs(nlist)
      r = norm(R); r >= RCUT && continue
      push!(pi, i); push!(pj, j); push!(rr, r); push!(RR, R)
   end
   return StructPairs(nat, zidx, pi, pj, rr, RR)
end
nrows(sp::StructPairs) = 1 + 3sp.nat + 6

# ---------------------------------------------------------------- column builder
# W :: KS x P (species-major channel index (s-1)*K + k), alphas :: K.
# Returns X :: nrows x (S*P), column (a,p) at (a-1)*P + p.  Generic eltype.
function extra_X(sp::StructPairs, W::AbstractMatrix, alphas::AbstractVector)
   T = promote_type(eltype(W), eltype(alphas))
   P = size(W, 2); nat = sp.nat; np = length(sp.pi)
   rho = zeros(T, nat, P)
   gk = Matrix{T}(undef, np, K); dgk = Matrix{T}(undef, np, K)
   @inbounds for q in 1:np
      r = sp.r[q]; sj = sp.zidx[sp.pj[q]]; i = sp.pi[q]
      fc = fcut(r, RCUT); dfc = dfcut(r, RCUT)
      for k in 1:K
         e = exp(-alphas[k] * (r / R0 - 1))
         gk[q, k] = e * fc
         dgk[q, k] = e * (-alphas[k] / R0 * fc + dfc)
         for p in 1:P
            rho[i, p] += W[(sj - 1) * K + k, p] * gk[q, k]
         end
      end
   end
   ncol = S * P
   X = zeros(T, nrows(sp), ncol)
   dphi = Matrix{T}(undef, nat, P)
   @inbounds for i in 1:nat, p in 1:P
      x = rho[i, p] + EPSR
      s = sqrt(x)
      X[1, (sp.zidx[i] - 1) * P + p] += s - sqrt(EPSR)
      dphi[i, p] = 1 / (2s)
   end
   @inbounds for q in 1:np
      i = sp.pi[q]; j = sp.pj[q]; sj = sp.zidx[j]; a = sp.zidx[i]
      Rhat = sp.R[q] / sp.r[q]
      for p in 1:P
         dgw = zero(T)
         for k in 1:K
            dgw += W[(sj - 1) * K + k, p] * dgk[q, k]
         end
         c = (a - 1) * P + p
         dV = (dphi[i, p] * dgw) * Rhat
         for b in 1:3
            X[1 + 3(j - 1) + b, c] -= dV[b]
            X[1 + 3(i - 1) + b, c] += dV[b]
         end
         V = dV * sp.R[q]'                       # V -= dV (x) R
         X[1 + 3nat + 1, c] -= V[1, 1]; X[1 + 3nat + 2, c] -= V[2, 2]; X[1 + 3nat + 3, c] -= V[3, 3]
         X[1 + 3nat + 4, c] -= V[2, 3]; X[1 + 3nat + 5, c] -= V[1, 3]; X[1 + 3nat + 6, c] -= V[1, 2]
      end
   end
   return X
end
function extra_X(sps::Vector{StructPairs}, W, alphas)
   T = promote_type(eltype(W), eltype(alphas))
   X = Matrix{T}(undef, sum(nrows, sps), S * size(W, 2))
   r0 = 0
   for sp in sps
      n = nrows(sp)
      X[r0+1:r0+n, :] .= extra_X(sp, W, alphas)
      r0 += n
   end
   return X
end

# Parametrisations of W (KS x Pcol, Pcol output densities = columns per species):
#   :mixed     P densities, each a free mix of all S*K channels          (15 params, S*P cols)
#   :perwidth  P x K densities, density (p,k) mixes only the width-k channels
#              over species                                              (15 params, S*K*P cols)
#              -> theta = 0 is exactly the sqrt(rho_tot) K=3 reference, and the
#                 hand tilts sqrt(rho_tot + rho^s) are nested
#   :full      P x K densities, each a free mix of all channels          (45 params, S*K*P cols)
struct ParamSpec
   mode::Symbol
   P::Int
   learn_alpha::Bool
end
ncol_dens(ps::ParamSpec) = ps.mode == :mixed ? ps.P : ps.P * K
nfree_W(ps::ParamSpec) = ps.mode == :mixed ? KS * ps.P : ps.mode == :perwidth ? KS * ps.P : KS * K * ps.P
ntheta(ps::ParamSpec) = nfree_W(ps) + (ps.learn_alpha ? K : 0)
function unpack(theta, ps::ParamSpec)
   T = eltype(theta)
   nW = nfree_W(ps)
   if ps.mode == :mixed
      W = reshape(exp.(theta[1:nW]), KS, ps.P)
   elseif ps.mode == :full
      W = reshape(exp.(theta[1:nW]), KS, K * ps.P)
   else
      W = zeros(T, KS, K * ps.P)
      for p in 1:ps.P, k in 1:K, s in 1:S
         W[(s - 1) * K + k, (p - 1) * K + k] = exp(theta[((p - 1) * K + (k - 1)) * S + s])
      end
   end
   alphas = ps.learn_alpha ? exp.(theta[nW+1:nW+K]) : T.(ALPHAS0)
   return W, alphas
end
# per-width theta for given species weight vectors ws[p][s] (same for every width)
function theta_perwidth(wss::Vector{<:AbstractVector}; learn_alpha = false)
   th = Float64[]
   for ws in wss, k in 1:K, s in 1:S
      push!(th, log(ws[s]))
   end
   learn_alpha && append!(th, log.(ALPHAS0))
   return th
end

# ---------------------------------------------------------------- inner solve
# Fixed ACE block factorised once.  Aw: weighted, prior-scaled ACE columns.
struct ACEFactor
   Q::Matrix{Float64}                    # thin Q of [Aw; lam I]  ((m+nA) x nA)
   R::Matrix{Float64}                    # nA x nA
   nA::Int
   m::Int
   lam::Float64
   Qty::Vector{Float64}                  # Q' [y; 0]
   Py::Vector{Float64}                   # (I - QQ') [y; 0]  (length m+nA)
   y::Vector{Float64}
end
function ACEFactor(Aw::Matrix{Float64}, y::Vector{Float64}, lam::Float64)
   m, nA = size(Aw)
   At = vcat(Aw, lam * Matrix{Float64}(I, nA, nA))
   F = qr!(At)
   R = Matrix(F.R)
   Q = Matrix(F.Q)                       # thin
   At = nothing; F = nothing; GC.gc()
   yt = vcat(y, zeros(nA))
   Qty = Q' * yt
   Py = yt - Q * Qty
   return ACEFactor(Q, R, nA, m, lam, Qty, Py, y)
end
# project columns Xw (m x nX) out of range(Q), including the lam I_X rows
function inner_solve(af::ACEFactor, Xw::Matrix{Float64})
   m, nA, lam = af.m, af.nA, af.lam
   nX = size(Xw, 2)
   Xt = vcat(Xw, zeros(nA, nX))
   QtX = af.Q' * Xt
   PX = Xt - af.Q * QtX
   # reduced LS: min || [PX; lam I] c - [Py; 0] ||
   Bred = vcat(PX, lam * Matrix{Float64}(I, nX, nX))
   yred = vcat(af.Py, zeros(nX))
   cX = Bred \ yred
   cA = UpperTriangular(af.R) \ (af.Qty - QtX * cX)
   return cA, cX
end
# reduced objective (weighted) at theta; returns objective, residual (m), cA, cX
function reduced(af::ACEFactor, Aw, Xw)
   cA, cX = inner_solve(af, Xw)
   r = Aw * cA + Xw * cX - af.y
   L = sum(abs2, r) + af.lam^2 * (sum(abs2, cA) + sum(abs2, cX))
   return L, r, cA, cX
end

# ---------------------------------------------------------------- VarPro problem
mutable struct VarPro
   af::ACEFactor
   Aw::Matrix{Float64}
   Wrow::Vector{Float64}          # row weights
   sps::Vector{StructPairs}
   ps::ParamSpec
   xscale::Vector{Float64}        # fixed per-column scale (set at init)
   lay::NamedTuple
   Y::Vector{Float64}             # unweighted targets
   Araw::Matrix{Float64}
   Pdiag::Vector{Float64}
   trace::Vector{Any}
   last::Any
end
function VarPro(af, Aw, Wrow, sps, ps::ParamSpec, theta0, lay, Y, Araw, Pdiag)
   W, al = unpack(theta0, ps)
   X = extra_X(sps, W, al)
   med = sort([norm(c) for c in eachcol(Aw)])[div(end, 2)]
   xn = [norm(Wrow .* c) for c in eachcol(X)]
   xscale = med ./ max.(xn, 1e-12)
   return VarPro(af, Aw, Wrow, sps, ps, xscale, lay, Y, Araw, Pdiag, Any[], nothing)
end
weighted_X(vp::VarPro, X) = (vp.Wrow .* X) .* reshape(vp.xscale, 1, :)
function objective(vp::VarPro, theta)
   W, al = unpack(theta, vp.ps)
   X = extra_X(vp.sps, W, al)
   Xw = weighted_X(vp, X)
   L, r, cA, cX = reduced(vp.af, vp.Aw, Xw)
   vp.last = (theta = copy(theta), L = L, r = r, cA = cA, cX = cX, X = X)
   return L
end
# Kaufman gradient: 2 r' (dXw/dtheta) cX with r, cX at the current theta
function gradient(vp::VarPro, theta)
   (vp.last === nothing || vp.last.theta != theta) && objective(vp, theta)
   r, cX = vp.last.r, vp.last.cX
   cs = vp.xscale .* cX
   rw = vp.Wrow .* r
   f(th) = begin
      W, al = unpack(th, vp.ps)
      acc = zero(eltype(th)); r0 = 0
      for sp in vp.sps
         n = nrows(sp)
         Xs = extra_X(sp, W, al)
         acc += dot(view(rw, r0+1:r0+n), Xs * cs)
         r0 += n
      end
      acc
   end
   return 2 .* FDiff.gradient(f, theta)
end
# unweighted train/test errors at the current solution
function errors(vp::VarPro, cA, cX, X; test = nothing)
   c_ace = cA ./ vp.Pdiag
   cx = vp.xscale .* cX
   pred_tr = vp.Araw * c_ace + X * cx
   etr = rmse_efv(pred_tr - vp.Y, vp.lay)
   test === nothing && return (tr = etr,)
   Ate, Yte, Xte, layte = test
   pred_te = Ate * c_ace + Xte * cx
   return (tr = etr, te = rmse_efv(pred_te - Yte, layte), resid_te = pred_te - Yte)
end

# ---------------------------------------------------------------- optimiser driver
# L-BFGS via Optim; the trace records the objective every f-evaluation that Optim
# accepts (iteration end).  `callback` may return true to stop early.
function optimise!(vp::VarPro, theta0; iters = 100, callback = nothing, show = true)
   fg!(Fv, G, th) = begin
      if G !== nothing
         G .= gradient_analytic(vp, th)
         return vp.last.L
      end
      return objective(vp, th)
   end
   it = Ref(0)
   cb = st -> begin
      it[] += 1
      x = st.metadata["x"]
      (vp.last === nothing || vp.last.theta != x) && objective(vp, x)
      e = errors(vp, vp.last.cA, vp.last.cX, vp.last.X)
      push!(vp.trace, (it = it[], L = vp.last.L, trainF = e.tr.F, theta = copy(st.metadata["x"])))
      show && @printf("  it %3d  L=%.6e  train F=%.5f  |g|=%.2e\n", it[], vp.last.L, e.tr.F, st.g_norm)
      flush(stdout)
      callback === nothing ? false : callback(vp, st.metadata["x"], it[])
   end
   res = Optim.optimize(Optim.only_fg!(fg!), theta0, Optim.LBFGS(),
                        Optim.Options(iterations = iters, callback = cb, extended_trace = true,
                                      store_trace = false, show_trace = false,
                                      g_tol = 1e-10, f_reltol = 1e-12, x_reltol = 0.0))
   return Optim.minimizer(res), res
end

# ---------------------------------------------------------------- reporting
# learned weights: for each output density, the S x K channel table normalised to max = 1
function weight_table(theta, ps::ParamSpec)
   W, al = unpack(theta, ps)
   io = IOBuffer()
   @printf(io, "  alphas = %s\n", join([@sprintf("%.3f", a) for a in al], ", "))
   for c in 1:size(W, 2)
      w = reshape(W[:, c], K, S)'          # S x K
      w = w ./ maximum(w)
      label = ps.mode == :mixed ? "density $c" : "density p=$((c - 1) ÷ K + 1), width k=$((c - 1) % K + 1)"
      share = vec(sum(w, dims = 2)); share ./= sum(share)
      @printf(io, "  %-28s %s | species share %s\n", label,
              join([@sprintf("%s:%s", ELS[s], join([@sprintf("%5.2f", w[s, k]) for k in 1:K], "/")) for s in 1:S], "  "),
              join([@sprintf("%4.1f%%", 100share[s]) for s in 1:S], " "))
   end
   return String(take!(io))
end
# compact: species share per output density (row) as percentages
function species_shares(theta, ps::ParamSpec)
   W, _ = unpack(theta, ps)
   [ (sh = [sum(W[(s - 1) * K + 1:s * K, c]) for s in 1:S]; sh ./ sum(sh)) for c in 1:size(W, 2) ]
end

# convex fit with a lambda sweep on fixed extra columns (the earlier spikes' protocol)
function sweep_fit(Aw, Yw, Wrow, Pdiag, Araw, Ytr, lay_tr, Ate, Yte, lay_te, Xtr, Xte; lams = LAMS)
   nace = size(Aw, 2)
   if size(Xtr, 2) == 0
      T = TikhonovFactor(Aw, Yw); xscale = Float64[]
   else
      med = sort([norm(c) for c in eachcol(Aw)])[div(end, 2)]
      xn = [norm(Wrow .* c) for c in eachcol(Xtr)]; xscale = med ./ max.(xn, 1e-12)
      T = TikhonovFactor(hcat(Aw, (Wrow .* Xtr) .* reshape(xscale, 1, :)), Yw)
   end
   out = []
   for lam in lams
      z = tikhonov_solve(T, lam)
      c_ace = z[1:nace] ./ Pdiag
      cx = z[nace+1:end] .* xscale
      pred_te = Ate * c_ace + Xte * cx; pred_tr = Araw * c_ace + Xtr * cx
      push!(out, (lam = lam, te = rmse_efv(pred_te - Yte, lay_te), tr = rmse_efv(pred_tr - Ytr, lay_tr),
                  resid_te = pred_te - Yte))
   end
   best = out[argmin([o.te.F for o in out])]
   return (best = best, all = out)
end

# ---------------------------------------------------------------- analytic gradient
# g(theta) = sum_rows rw . (X(theta) c)  with rw = W .* r and c = xscale .* cX fixed.
# Per structure, with beta_q = Rhat_q.(rF_i - rF_j) - sum_ab Rv_ab Rhat_a R_b per pair,
#     g = sum_i c[a_i,col] [ rE phi(rho_i^col) + phi'(rho_i^col) D_i^col ]
#     rho_i^col = sum_ch W[ch,col] rho_ch[i,ch],   D_i^col = sum_ch W[ch,col] Dch[i,ch]
#     rho_ch[i,(s,k)] = sum_{q: i, s_j = s} g_k(r_q),   Dch[i,(s,k)] = sum_{q: i, s_j = s} dg_k(r_q) beta_q
# so dg/dW[ch,col] = sum_i c [ rE phi' rho_ch + phi'' rho_ch D_i + phi' Dch ]  and the
# alpha derivatives follow from dg_k/dalpha, d(dg_k)/dalpha.  Verified against the
# ForwardDiff gradient in varpro_gradcheck.jl.
function gradient_struct!(gW::AbstractMatrix, galpha::AbstractVector, sp::StructPairs, W, alphas, c::AbstractVector,
                          rw::AbstractVector, learn_alpha::Bool)
   nat = sp.nat; np = length(sp.pi); Pcol = size(W, 2)
   rE = rw[1]
   rF = [SVector{3}(rw[1 + 3(j - 1) + 1], rw[1 + 3(j - 1) + 2], rw[1 + 3(j - 1) + 3]) for j in 1:nat]
   o = 1 + 3nat
   Rv = @SMatrix [rw[o+1] rw[o+6] rw[o+5]; 0.0 rw[o+2] rw[o+4]; 0.0 0.0 rw[o+3]]
   rho_ch = zeros(nat, KS); Dch = zeros(nat, KS)
   drho_ch = learn_alpha ? zeros(nat, KS) : rho_ch; dDch = learn_alpha ? zeros(nat, KS) : Dch
   @inbounds for q in 1:np
      r = sp.r[q]; i = sp.pi[q]; j = sp.pj[q]; sj = sp.zidx[j]
      Rhat = sp.R[q] / r
      beta = dot(Rhat, rF[i] - rF[j]) - dot(Rhat, Rv * sp.R[q])
      fc = fcut(r, RCUT); dfc = dfcut(r, RCUT)
      for k in 1:K
         e = exp(-alphas[k] * (r / R0 - 1))
         gk = e * fc; dgk = e * (-alphas[k] / R0 * fc + dfc)
         ch = (sj - 1) * K + k
         rho_ch[i, ch] += gk
         Dch[i, ch] += dgk * beta
         if learn_alpha
            drho_ch[i, ch] += -(r / R0 - 1) * gk
            dDch[i, ch] += (-(r / R0 - 1) * dgk - e * fc / R0) * beta
         end
      end
   end
   @inbounds for i in 1:nat, col in 1:Pcol
      cc = c[(sp.zidx[i] - 1) * Pcol + col]
      cc == 0 && continue
      rho = zero(eltype(W)); D = zero(eltype(W))
      for ch in 1:KS
         rho += W[ch, col] * rho_ch[i, ch]; D += W[ch, col] * Dch[i, ch]
      end
      x = rho + EPSR; s = sqrt(x)
      p1 = 1 / (2s); p2 = -1 / (4 * x * s)
      for ch in 1:KS
         gW[ch, col] += cc * (rE * p1 * rho_ch[i, ch] + p2 * rho_ch[i, ch] * D + p1 * Dch[i, ch])
      end
      if learn_alpha
         for k in 1:K
            drho = 0.0; dD = 0.0
            for s_ in 1:S
               ch = (s_ - 1) * K + k
               drho += W[ch, col] * drho_ch[i, ch]; dD += W[ch, col] * dDch[i, ch]
            end
            galpha[k] += cc * (rE * p1 * drho + p2 * drho * D + p1 * dD)
         end
      end
   end
   return nothing
end
function gradient_analytic(vp::VarPro, theta)
   (vp.last === nothing || vp.last.theta != theta) && objective(vp, theta)
   W, al = unpack(theta, vp.ps)
   c = vp.xscale .* vp.last.cX
   rw = vp.Wrow .* vp.last.r
   gW = zeros(size(W)); galpha = zeros(K)
   r0 = 0
   for sp in vp.sps
      n = nrows(sp)
      gradient_struct!(gW, galpha, sp, W, al, c, view(rw, r0+1:r0+n), vp.ps.learn_alpha)
      r0 += n
   end
   # chain rule through the parametrisation (W = exp(theta) entries; alpha = exp)
   ps = vp.ps; nW = nfree_W(ps)
   g = zeros(length(theta))
   if ps.mode == :mixed || ps.mode == :full
      g[1:nW] .= vec(gW) .* vec(W)
   else
      for p in 1:ps.P, k in 1:K, s in 1:S
         idx = ((p - 1) * K + (k - 1)) * S + s
         g[idx] = gW[(s - 1) * K + k, (p - 1) * K + k] * W[(s - 1) * K + k, (p - 1) * K + k]
      end
   end
   ps.learn_alpha && (g[nW+1:nW+K] .= galpha .* al)
   return 2 .* g
end
