# More nonlinear functions of the SAME fixed total density, still linear in
# the coefficients.  Solve-only on the cached Cantor assemblies.
#   (A) powers rho^m added to the sqrt-only model, and all powers together
#   (B) Chebyshev embedding F(rho) = sum_n a_n T_n(u), u = map of sqrt(rho) or
#       log(rho) to [-1,1] over the training range, Kc = 4, 6, 8
#   (C) cross terms sqrt(rho_k1 rho_k2) and sqrt of weighted width combinations
# Reference for the marginal gains: sqrt(rho_tot), K=3 (15 columns).
#   julia --project=acejax/julia acejax/spike_fs/embed_spike.jl
using ACEpotentials, ACEfit
using AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Serialization, Statistics
include(joinpath(@__DIR__, "fs_embed.jl"))
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)
M = ACEpotentials.Models

const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const DEGS = parse.(Int, split(get(ENV, "DEGS", "4,5,6"), ","))
const LAMS = 10.0 .^ (0:-1:-8)

data_all = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
tr, te = data_all[p[1:200]], data_all[p[201:300]]
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]
mkmodel(D) = ace1_model(elements = ELS, order = 3, totaldegree = D)
const K = 3

function row_layout(data)
   kind = Int[]; nat = Int[]; sid = Int[]
   for (is, sys) in enumerate(data)
      n = length(sys)
      push!(kind, 1); push!(nat, n); push!(sid, is)
      append!(kind, fill(2, 3n)); append!(nat, fill(n, 3n)); append!(sid, fill(is, 3n))
      append!(kind, fill(3, 6)); append!(nat, fill(n, 6)); append!(sid, fill(is, 6))
   end
   return kind, nat, sid
end
function rmse_efv(resid, kind, nat)
   r = copy(resid); r[kind .!= 2] ./= nat[kind .!= 2]
   f(k) = sqrt(sum(abs2, r[kind .== k]) / count(==(k), kind))
   return (F = f(2), E = f(1), V = f(3))
end
kind_tr, nat_tr, sid_tr = row_layout(tr)
kind_te, nat_te, sid_te = row_layout(te)

# ---------------------------------------------------------------- column sets
base = EmbedSpec(ZS, [], [])
lo_s, hi_s = density_ranges(base, tr, :sqrt)
lo_l, hi_l = density_ranges(base, tr, :log)
@info "training range of sqrt(rho_k): $(round.(lo_s, digits=3)) .. $(round.(hi_s, digits=3))"
@info "training range of log(rho_k):  $(round.(lo_l, digits=3)) .. $(round.(hi_l, digits=3))"
pw(ms...) = [power_fun(K, k, m) for m in ms for k in 1:K]
cheb(Kc, tf, lo, hi) = vcat([cheb_funs(K, k, Kc, tf, lo[k], hi[k]) for k in 1:K]...)
spec(funs) = EmbedSpec(ZS, funs, ["" for _ in funs])
variants = [
   ("sqrt only [ref]",             spec(pw(1/2))),
   # (A) powers
   ("sqrt + rho^1/8",              spec(pw(1/2, 1/8))),
   ("sqrt + rho^1/4",              spec(pw(1/2, 1/4))),
   ("sqrt + rho^3/4",              spec(pw(1/2, 3/4))),
   ("sqrt + rho^1 (in span)",      spec(pw(1/2, 1))),
   ("sqrt + rho^2",                spec(pw(1/2, 2))),
   ("all powers 1/8..2",           spec(pw(1/8, 1/4, 1/2, 3/4, 2))),
   # (B) Chebyshev embeddings
   ("cheb[sqrt] Kc=4",             spec(cheb(4, :sqrt, lo_s, hi_s))),
   ("cheb[sqrt] Kc=6",             spec(cheb(6, :sqrt, lo_s, hi_s))),
   ("cheb[sqrt] Kc=8",             spec(cheb(8, :sqrt, lo_s, hi_s))),
   ("cheb[log] Kc=4",              spec(cheb(4, :log, lo_l, hi_l))),
   ("cheb[log] Kc=6",              spec(cheb(6, :log, lo_l, hi_l))),
   ("cheb[log] Kc=8",              spec(cheb(8, :log, lo_l, hi_l))),
   # (C) cross terms
   ("sqrt + cross sqrt(rho_k1 rho_k2)", spec(vcat(pw(1/2), [cross_fun(K, 1, 2), cross_fun(K, 1, 3), cross_fun(K, 2, 3)]))),
   ("sqrt + 4 weighted sqrt",      spec(vcat(pw(1/2), [wsqrt_fun(K, SVector(w...)) for w in
                                       ((1, 1, 1), (1, 0.5, 0.25), (0.25, 0.5, 1), (0, 1, 1))]))),
]
Xc = Dict{String, Any}()
for (name, es) in variants
   t = @elapsed Xc[name] = (fs_feature_matrix(es, tr), fs_feature_matrix(es, te))
   @info @sprintf("  columns %-34s ncol=%3d  %.1f s", name, ncolumns(es), t)
end

function solve_sweep(Bw, Yw, nace, Pdiag, xscale, Atr, Ate, Xtr, Xte, Ytr, Yte)
   T = TikhonovFactor(Bw, Yw)
   best = nothing
   for lam in LAMS
      z = tikhonov_solve(T, lam)
      c_ace = z[1:nace] ./ Pdiag
      pred_te = Ate * c_ace; pred_tr = Atr * c_ace
      if Xtr !== nothing
         c_x = z[nace+1:end] .* xscale
         pred_te .+= Xte * c_x; pred_tr .+= Xtr * c_x
      end
      ete = rmse_efv(pred_te .- Yte, kind_te, nat_te)
      etr = rmse_efv(pred_tr .- Ytr, kind_tr, nat_tr)
      if best === nothing || ete.F < best.te.F
         best = (te = ete, tr = etr, lam = lam, resid_te = pred_te .- Yte)
      end
   end
   return best
end

nte = length(te)
fmask = kind_te .== 2
sse(resid) = [sum(abs2, resid[fmask .& (sid_te .== s)]) for s in 1:nte]
nF = [3 * length(sys) for sys in te]
function boot_ci(s0, s1)
   brng = MersenneTwister(1); diffs = Float64[]
   for b in 1:2000
      idx = rand(brng, 1:nte, nte)
      push!(diffs, 100 * (sqrt(sum(s1[idx]) / sum(nF[idx])) / sqrt(sum(s0[idx]) / sum(nF[idx])) - 1))
   end
   return quantile(diffs, 0.025), quantile(diffs, 0.975)
end

for D in DEGS
   Atr, Ytr, Wtr = deserialize(joinpath(CACHEDIR, "asm_cantor_D$(D)_train200.jls"))
   Ate, Yte, Wte = deserialize(joinpath(CACHEDIR, "asm_cantor_D$(D)_test100.jls"))
   m = mkmodel(D)
   Pdiag = M.algebraic_smoothness_prior(m.model; p = 4).diag
   Aw = (Atr ./ reshape(Pdiag, 1, :)) .* Wtr
   Yw = Wtr .* Ytr
   nace = size(Aw, 2)
   med = sort([norm(c) for c in eachcol(Aw)])[div(end, 2)]
   println("\n==== D=$D  (nace=$nace) ====")
   base = solve_sweep(Aw, Yw, nace, Pdiag, 1.0, Atr, Ate, nothing, nothing, Ytr, Yte)
   @printf("%-36s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g\n",
           "linear ACE", nace, base.te.F, base.te.E, base.te.V, base.tr.F, base.lam)
   ref = nothing; s_ref = nothing
   for (name, es) in variants
      Xtr, Xte = Xc[name]
      xn = [norm(Wtr .* c) for c in eachcol(Xtr)]
      xscale = med ./ max.(xn, 1e-12)
      Xw = (Wtr .* Xtr) .* reshape(xscale, 1, :)
      condX = (sv = svdvals(Xw); sv[1] / sv[end])
      Bw = hcat(Aw, Xw)
      t = @elapsed r = solve_sweep(Bw, Yw, nace, Pdiag, xscale, Atr, Ate, Xtr, Xte, Ytr, Yte)
      Bw = nothing; GC.gc()
      s1 = sse(r.resid_te)
      if ref === nothing
         ref = r; s_ref = s1
         @printf("%-36s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g  dF(vs linear)=%+.1f%%  cond(X)=%.1e  [%.0fs]\n",
                 name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam,
                 100 * (r.te.F / base.te.F - 1), condX, t)
      else
         lo, hi = boot_ci(s_ref, s1)
         @printf("%-36s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g  dF(vs sqrt)=%+.1f%% CI[%+.1f,%+.1f]  cond(X)=%.1e  [%.0fs]\n",
                 name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam,
                 100 * (r.te.F / ref.te.F - 1), lo, hi, condX, t)
      end
      flush(stdout)
   end
end
