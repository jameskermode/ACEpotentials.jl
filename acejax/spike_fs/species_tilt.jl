# Linear probe of "the shape of rho": fixed species-tilted total densities
#   sqrt(rho_tot + rho^s)  and  sqrt(rho_tot - rho^s)   for each species s, width k
# added to the sqrt(rho_tot) reference.  These are the first-order directions a
# fitted species weighting w_s in sqrt(sum_s w_s rho^s) could move in.
# Solve-only on the cached assemblies; also FD-checks the per-species channels.
using ACEpotentials, ACEfit
using AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Serialization, Statistics
include(joinpath(@__DIR__, "fs_embed.jl"))
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)
M = ACEpotentials.Models
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const DEGS = parse.(Int, filter(!isempty, split(get(ENV, "DEGS", "4,5,6"), ",")))
const LAMS = 10.0 .^ (0:-1:-8)
data_all = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]; S = 5; K = 3
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
tr, te = data_all[p[1:200]], data_all[p[201:300]]
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]
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

# weight vectors over the K*S channels: w[(s-1)*K + k]
KS = K * S
wvec(k, ws) = SVector{KS}(ntuple(q -> ((q - 1) % K + 1 == k) ? ws[(q - 1) ÷ K + 1] : 0.0, KS))
ones5 = ones(S)
ref_funs = [wsqrt_fun(KS, wvec(k, ones5)) for k in 1:K]                       # sqrt(rho_tot)
plus_funs = [wsqrt_fun(KS, wvec(k, ones5 .+ (1:S .== s))) for s in 1:S for k in 1:K]   # sqrt(rho_tot + rho^s)
minus_funs = [wsqrt_fun(KS, wvec(k, ones5 .- (1:S .== s))) for s in 1:S for k in 1:K]  # sqrt(rho_tot - rho^s)
spec(funs) = EmbedSpec(ZS, funs, ["" for _ in funs]; per_species = true)
variants = [
   ("sqrt(rho_tot) [ref]",                 spec(ref_funs)),
   ("+ sqrt(rho_tot + rho^s)",             spec(vcat(ref_funs, plus_funs))),
   ("+ sqrt(rho_tot - rho^s)",             spec(vcat(ref_funs, minus_funs))),
   ("+ both tilts",                        spec(vcat(ref_funs, plus_funs, minus_funs))),
]
# FD check of the per-species channel machinery on structure 1
r = fd_check(variants[4][2], te[1]; h = 1e-5, natoms_check = 4)
@printf("FD check (per-species channels, both tilts, %d cols): |F|max=%.2e |V|max=%.2e  err F=%.2e V=%.2e\n",
        ncolumns(variants[4][2]), r.maxabs_F, r.maxabs_V, r.maxerr_F, r.maxerr_V)
# consistency: ref via per-species channels == FSSpec(total = true)
X1 = fs_feature_matrix(variants[1][2], te[1]); X2 = fs_feature_matrix(FSSpec(ZS; phi = :sqrt, total = true), te[1])
println("max |per-species-channel sqrt(rho_tot) - FSSpec(total)| = ", maximum(abs, X1 - X2))
Xc = Dict(name => (fs_feature_matrix(es, tr), fs_feature_matrix(es, te)) for (name, es) in variants)

function solve_sweep(Bw, Yw, nace, Pdiag, xscale, Atr, Ate, Xtr, Xte, Ytr, Yte)
   T = TikhonovFactor(Bw, Yw); best = nothing
   for lam in LAMS
      z = tikhonov_solve(T, lam)
      c_ace = z[1:nace] ./ Pdiag
      pred_te = Ate * c_ace + Xte * (z[nace+1:end] .* xscale)
      pred_tr = Atr * c_ace + Xtr * (z[nace+1:end] .* xscale)
      ete = rmse_efv(pred_te .- Yte, kind_te, nat_te); etr = rmse_efv(pred_tr .- Ytr, kind_tr, nat_tr)
      (best === nothing || ete.F < best.te.F) && (best = (te = ete, tr = etr, lam = lam, resid_te = pred_te .- Yte))
   end
   return best
end
nte = length(te); fmask = kind_te .== 2
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
   m = ace1_model(elements = ELS, order = 3, totaldegree = D)
   Pdiag = M.algebraic_smoothness_prior(m.model; p = 4).diag
   Aw = (Atr ./ reshape(Pdiag, 1, :)) .* Wtr; Yw = Wtr .* Ytr
   nace = size(Aw, 2)
   med = sort([norm(c) for c in eachcol(Aw)])[div(end, 2)]
   println("\n==== D=$D  (nace=$nace) ====")
   ref = nothing; s_ref = nothing
   for (name, es) in variants
      Xtr, Xte = Xc[name]
      xn = [norm(Wtr .* c) for c in eachcol(Xtr)]; xscale = med ./ max.(xn, 1e-12)
      Xw = (Wtr .* Xtr) .* reshape(xscale, 1, :)
      condX = (sv = svdvals(Xw); sv[1] / sv[end])
      r = solve_sweep(hcat(Aw, Xw), Yw, nace, Pdiag, xscale, Atr, Ate, Xtr, Xte, Ytr, Yte)
      GC.gc()
      s1 = sse(r.resid_te)
      if ref === nothing
         ref = r; s_ref = s1
         @printf("%-30s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g  cond(X)=%.1e\n",
                 name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam, condX)
      else
         lo, hi = boot_ci(s_ref, s1)
         @printf("%-30s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g  dF(vs sqrt)=%+.1f%% CI[%+.1f,%+.1f]  cond(X)=%.1e\n",
                 name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam, 100 * (r.te.F / ref.te.F - 1), lo, hi, condX)
      end
      flush(stdout)
   end
end
