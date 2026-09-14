# General fixed-embedding columns: arbitrary functions of the K-vector of
# fixed TOTAL densities per site,
#     rho_i = (rho_i^1, ..., rho_i^K),  rho_i^k = sum_{j, r_ij < rcut} g_k(r_ij)
# (species-blind, the best variant in the first spike), with a per-central-
# species readout:
#     B_{a,f}(R) = sum_{i : z_i = a} Phi_f(rho_i)
# for a list of scalar functions Phi_f with gradients dPhi_f/drho (K-vectors).
# Forces / virials follow from the chain rule exactly as in fs_columns.jl:
#     dE_i/dR_ij = sum_k (dPhi_f/drho_k)(rho_i) * g_k'(r_ij) * R_ij / r_ij.
# Same row layout as fs_columns.jl / ACEfit.feature_matrix.
include(joinpath(@__DIR__, "fs_columns.jl"))     # g, dg, _displaced, _strained, NL

struct EmbedSpec
   species::Vector{Int}
   alphas::Vector{Float64}
   r0::Float64
   rcut::Float64
   funs::Vector{Any}            # rho::SVector{K} -> (value::Float64, grad::SVector{K})
   names::Vector{String}
   per_species::Bool            # true: density channels are (species, width), K*S of them
end
EmbedSpec(species, funs, names; alphas = [2.0, 4.0, 6.0], r0 = 2.5, rcut = 6.25,
          per_species = false) =
   EmbedSpec(Int.(species), Float64.(alphas), r0, rcut, collect(Any, funs), String.(names),
             per_species)

# number of density channels: K widths, or K*S (species s, width k) -> (s-1)*K + k
nK(es::EmbedSpec) = length(es.alphas) * (es.per_species ? length(es.species) : 1)
nw(es::EmbedSpec) = length(es.alphas)
chan(es::EmbedSpec, sj, k) = es.per_species ? (sj - 1) * nw(es) + k : k
nfun(es::EmbedSpec) = length(es.funs)
ncolumns(es::EmbedSpec) = length(es.species) * nfun(es)
colidx(es::EmbedSpec, a, f) = (a - 1) * nfun(es) + f

# nat x K matrix of total densities for one structure
function site_densities(es::EmbedSpec, sys)
   nat = length(sys); K = nK(es)
   zidx = [findfirst(==(atomic_number(sys, i)), es.species) for i in 1:nat]
   rho = zeros(nat, K)
   nlist = NL.PairList(sys, es.rcut * u"Å")
   for (i, j, R) in NL.pairs(nlist)
      r = norm(R); r >= es.rcut && continue
      for k in 1:nw(es)
         rho[i, chan(es, zidx[j], k)] += g(r, es.alphas[k], es.r0, es.rcut)
      end
   end
   return rho
end

function fs_efv(es::EmbedSpec, sys)
   nat = length(sys); K = nK(es); nf = nfun(es); ncol = ncolumns(es)
   zidx = [findfirst(==(atomic_number(sys, i)), es.species) for i in 1:nat]
   any(isnothing, zidx) && error("species not in EmbedSpec")
   nlist = NL.PairList(sys, es.rcut * u"Å")
   rho = zeros(nat, K)
   for (i, j, R) in NL.pairs(nlist)
      r = norm(R); r >= es.rcut && continue
      for k in 1:nw(es)
         rho[i, chan(es, zidx[j], k)] += g(r, es.alphas[k], es.r0, es.rcut)
      end
   end
   E = zeros(ncol)
   grads = Matrix{SVector{K, Float64}}(undef, nat, nf)
   for i in 1:nat
      ρ = SVector{K}(rho[i, :])
      for f in 1:nf
         v, gr = es.funs[f](ρ)
         E[colidx(es, zidx[i], f)] += v
         grads[i, f] = gr
      end
   end
   F = fill(zero(SVector{3, Float64}), nat, ncol)
   V = fill(zero(SMatrix{3, 3, Float64}), ncol)
   for (i, j, R) in NL.pairs(nlist)
      r = norm(R); r >= es.rcut && continue
      Rhat = R / r
      dgv = SVector{K}(ntuple(q -> (es.per_species && (q - 1) ÷ nw(es) + 1 != zidx[j]) ? 0.0 :
                                   dg(r, es.alphas[(q - 1) % nw(es) + 1], es.r0, es.rcut), K))
      for f in 1:nf
         c = colidx(es, zidx[i], f)
         dV = dot(grads[i, f], dgv) * Rhat
         F[j, c] -= dV; F[i, c] += dV
         V[c] -= dV * R'
      end
   end
   return E, F, V
end

function fs_feature_matrix(es::EmbedSpec, sys)
   nat = length(sys)
   E, F, V = fs_efv(es, sys)
   X = zeros(1 + 3nat + 6, ncolumns(es))
   X[1, :] .= E
   X[2:1+3nat, :] .= reinterpret(Float64, F)
   for c in 1:ncolumns(es)
      X[2+3nat:end, c] .= V[c][SVector(1, 5, 9, 6, 3, 2)]
   end
   return X
end
fs_feature_matrix(es::EmbedSpec, data::AbstractVector) =
   reduce(vcat, [fs_feature_matrix(es, sys) for sys in data])

# ---------------------------------------------------------------- function families
const EPS = 1e-8
# (A) power of one width:  (rho_k + eps)^m - eps^m
function power_fun(K, k, m)
   ρ -> begin
      x = ρ[k] + EPS
      (x^m - EPS^m, SVector{K}(ntuple(q -> q == k ? m * x^(m - 1) : 0.0, K)))
   end
end
# (C) geometric cross term between widths: sqrt(rho_k1 rho_k2 + eps)
function cross_fun(K, k1, k2)
   ρ -> begin
      p = ρ[k1] * ρ[k2] + EPS
      s = sqrt(p)
      (s - sqrt(EPS), SVector{K}(ntuple(q -> (q == k1 ? ρ[k2] : 0.0) + (q == k2 ? ρ[k1] : 0.0), K)) / (2s))
   end
end
# (C) sqrt of a weighted combination of widths
function wsqrt_fun(K, w::SVector)
   ρ -> begin
      x = dot(w, ρ) + EPS
      s = sqrt(x)
      (s - sqrt(EPS), w / (2s))
   end
end
# (B) Chebyshev T_n(u(rho_k)), n = 1..Kc, u a monotone map of rho_k to [-1,1]
#     over the training range.  transform = :sqrt (u linear in sqrt(rho)) or
#     :log (u linear in log(rho)).  Returns a vector of Kc functions.
function cheb_T_dT(u, Kc)
   T = zeros(Kc + 1); dT = zeros(Kc + 1)
   T[1] = 1.0; dT[1] = 0.0
   Kc >= 1 && (T[2] = u; dT[2] = 1.0)
   for n in 2:Kc
      T[n+1] = 2u * T[n] - T[n-1]
      dT[n+1] = 2T[n] + 2u * dT[n] - dT[n-1]
   end
   return T, dT
end
function cheb_funs(K, k, Kc, transform, lo, hi)
   tf(x) = transform == :sqrt ? sqrt(x + EPS) : log(x + EPS)
   dtf(x) = transform == :sqrt ? 1 / (2sqrt(x + EPS)) : 1 / (x + EPS)
   scale = 2 / (hi - lo)
   [ρ -> begin
        s = tf(ρ[k]); u = scale * (s - lo) - 1
        T, dT = cheb_T_dT(u, Kc)
        (T[n+1], SVector{K}(ntuple(q -> q == k ? dT[n+1] * scale * dtf(ρ[k]) : 0.0, K)))
     end for n in 1:Kc]
end
# training-set range of the transformed density per width (with a small margin)
function density_ranges(es::EmbedSpec, data, transform; margin = 0.02)
   tf(x) = transform == :sqrt ? sqrt(x + EPS) : log(x + EPS)
   los = fill(Inf, nK(es)); his = fill(-Inf, nK(es))
   for sys in data
      rho = site_densities(es, sys)
      for k in 1:nK(es)
         v = tf.(rho[:, k])
         los[k] = min(los[k], minimum(v)); his[k] = max(his[k], maximum(v))
      end
   end
   w = his .- los
   return los .- margin .* w, his .+ margin .* w
end

# ---------------------------------------------------------------- FD check
function fd_check(es::EmbedSpec, sys; h = 1e-5, natoms_check = 4)
   E0, F0, V0 = fs_efv(es, sys)
   nat = length(sys); ncol = ncolumns(es)
   X0 = [_sv(position(sys, i)) for i in 1:nat]
   maxerr_F = 0.0
   for j in 1:min(nat, natoms_check), a in 1:3
      e = SVector{3}(a == 1, a == 2, a == 3) * h * u"Å"
      Xp = copy(X0); Xp[j] += e
      Xm = copy(X0); Xm[j] -= e
      dE = (fs_efv(es, _displaced(sys, Xp))[1] - fs_efv(es, _displaced(sys, Xm))[1]) / (2h)
      for c in 1:ncol
         maxerr_F = max(maxerr_F, abs(-dE[c] - F0[j, c][a]))
      end
   end
   maxerr_V = 0.0
   for a in 1:3, b in 1:3
      eps = zeros(3, 3); eps[a, b] = h
      dE = (fs_efv(es, _strained(sys, eps))[1] - fs_efv(es, _strained(sys, -eps))[1]) / (2h)
      for c in 1:ncol
         maxerr_V = max(maxerr_V, abs(-dE[c] - V0[c][a, b]))
      end
   end
   return (maxerr_F = maxerr_F, maxerr_V = maxerr_V,
           maxabs_F = maximum(norm, F0), maxabs_V = maximum(x -> maximum(abs, x), V0))
end
