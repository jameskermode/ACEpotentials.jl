# Solve-only follow-ups on the cached Cantor assemblies (no re-assembly):
#   * extra column variants: total-density sqrt, wider K, total-density control
#   * paired bootstrap over test structures for the D=4 sqrt(rho) gain
#   * where the gain comes from: test force RMSE by density tercile
#
#   julia --project=acejax/julia acejax/spike_fs/analysis.jl
using ACEpotentials, ACEfit
using AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Serialization, Statistics
include(joinpath(@__DIR__, "fs_columns.jl"))
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)
M = ACEpotentials.Models

const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const DEGS = parse.(Int, split(get(ENV, "DEGS", "4,5"), ","))
const LAMS = 10.0 .^ (0:-1:-8)

DATAFILE = joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz")
data_all = ACEpotentials.ExtXYZ.load(DATAFILE)
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
tr, te = data_all[p[1:200]], data_all[p[201:300]]
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]
mkmodel(D) = ace1_model(elements = ELS, order = 3, totaldegree = D)

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

variants = [
   ("+ sqrt(rho), K=3 [main]",     FSSpec(ZS; phi = :sqrt)),
   ("+ sqrt(rho_tot), K=3",        FSSpec(ZS; phi = :sqrt, total = true)),
   ("+ rho_tot, K=3 (ctrl)",       FSSpec(ZS; phi = :linear, total = true)),
   ("+ sqrt(rho), K=6",            FSSpec(ZS; phi = :sqrt, alphas = [1, 2, 3, 4, 6, 8])),
   ("+ sqrt(rho_tot), K=6",        FSSpec(ZS; phi = :sqrt, total = true, alphas = [1, 2, 3, 4, 6, 8])),
   ("+ sqrt(rho), K=3, r0=2.0",    FSSpec(ZS; phi = :sqrt, r0 = 2.0)),
   ("+ sqrt(rho), K=3, r0=3.0",    FSSpec(ZS; phi = :sqrt, r0 = 3.0)),
]
Xc = Dict(name => (fs_feature_matrix(fs, tr), fs_feature_matrix(fs, te)) for (name, fs) in variants)

# per-structure mean total density (alpha = 4) on the test set, for the tercile split
fs_dens = FSSpec(ZS; phi = :linear, total = true, alphas = [4.0])
dens_te = [fs_efv(fs_dens, sys)[1][1] / length(sys) for sys in te]

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
   @printf("%-32s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%g\n",
           "linear ACE", nace, base.te.F, base.te.E, base.te.V, base.tr.F, base.lam)
   res = Dict{String, Any}()
   for (name, fs) in variants
      Xtr, Xte = Xc[name]
      xn = [norm(Wtr .* c) for c in eachcol(Xtr)]
      xscale = med ./ max.(xn, 1e-12)
      Bw = hcat(Aw, (Wtr .* Xtr) .* reshape(xscale, 1, :))
      r = solve_sweep(Bw, Yw, nace, Pdiag, xscale, Atr, Ate, Xtr, Xte, Ytr, Yte)
      res[name] = r
      @printf("%-32s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%g   dF=%+.1f%%\n",
              name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam,
              100 * (r.te.F / base.te.F - 1))
   end
   # ---- paired bootstrap over test structures: baseline vs two variants
   nte = length(te)
   fmask = kind_te .== 2
   sse(resid) = [sum(abs2, resid[fmask .& (sid_te .== s)]) for s in 1:nte]
   nF = [3 * length(sys) for sys in te]
   s0 = sse(base.resid_te)
   local s1
   for vname in ("+ sqrt(rho), K=3 [main]", "+ sqrt(rho_tot), K=3")
      main = res[vname]
      s1 = sse(main.resid_te)
      brng = MersenneTwister(1)
      diffs = Float64[]
      for b in 1:2000
         idx = rand(brng, 1:nte, nte)
         f0 = sqrt(sum(s0[idx]) / sum(nF[idx])); f1 = sqrt(sum(s1[idx]) / sum(nF[idx]))
         push!(diffs, 100 * (f1 / f0 - 1))
      end
      @printf("bootstrap (2000 resamples of test structures): %s vs baseline, test F change = %+.1f%%  95%% CI [%+.1f%%, %+.1f%%]\n",
              vname, 100 * (main.te.F / base.te.F - 1), quantile(diffs, 0.025), quantile(diffs, 0.975))
      # ---- by density tercile
      ord = sortperm(dens_te)
      terc = [ord[1:33], ord[34:66], ord[67:100]]
      for (t, idx) in enumerate(terc)
         f0 = sqrt(sum(s0[idx]) / sum(nF[idx])); f1 = sqrt(sum(s1[idx]) / sum(nF[idx]))
         @printf("  density tercile %d (mean rho_tot %.2f..%.2f): baseline F=%.4f  variant F=%.4f  (%+.1f%%)\n",
                 t, minimum(dens_te[idx]), maximum(dens_te[idx]), f0, f1, 100 * (f1 / f0 - 1))
      end
      @printf("  structures with lower force SSE under the variant: %d / %d\n", count(s1 .< s0), nte)
   end
end
