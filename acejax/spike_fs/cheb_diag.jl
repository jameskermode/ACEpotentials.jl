# Why does cheb[sqrt] Kc=8 blow up on the test set at D=4?  Solve-only, D=4.
#   * which test structures carry the extra error, and how far into the tails
#     of the density distribution their sites sit
#   * does widening the u-range margin (less of the polynomial's range is used)
#     or a lambda sweep restricted to the FS block fix it
using ACEpotentials, ACEfit
using AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Serialization, Statistics
include(joinpath(@__DIR__, "fs_embed.jl"))
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(Sys.CPU_THREADS)
M = ACEpotentials.Models
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const LAMS = 10.0 .^ (0:-1:-8)
data_all = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
tr, te = data_all[p[1:200]], data_all[p[201:300]]
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]
const K = 3; D = 4
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
rmseF(resid, kind) = sqrt(sum(abs2, resid[kind .== 2]) / count(==(2), kind))
kind_tr, nat_tr, sid_tr = row_layout(tr)
kind_te, nat_te, sid_te = row_layout(te)
Atr, Ytr, Wtr = deserialize(joinpath(CACHEDIR, "asm_cantor_D$(D)_train200.jls"))
Ate, Yte, Wte = deserialize(joinpath(CACHEDIR, "asm_cantor_D$(D)_test100.jls"))
m = ace1_model(elements = ELS, order = 3, totaldegree = D)
Pdiag = M.algebraic_smoothness_prior(m.model; p = 4).diag
Aw = (Atr ./ reshape(Pdiag, 1, :)) .* Wtr; Yw = Wtr .* Ytr
nace = size(Aw, 2)
med = sort([norm(c) for c in eachcol(Aw)])[div(end, 2)]
base = EmbedSpec(ZS, [], [])
spec(funs) = EmbedSpec(ZS, funs, ["" for _ in funs])

function fit(es; lam_list = LAMS)
   Xtr, Xte = fs_feature_matrix(es, tr), fs_feature_matrix(es, te)
   xn = [norm(Wtr .* c) for c in eachcol(Xtr)]
   xscale = med ./ max.(xn, 1e-12)
   T = TikhonovFactor(hcat(Aw, (Wtr .* Xtr) .* reshape(xscale, 1, :)), Yw)
   best = nothing
   for lam in lam_list
      z = tikhonov_solve(T, lam)
      pred = Ate * (z[1:nace] ./ Pdiag) + Xte * (z[nace+1:end] .* xscale)
      r = pred .- Yte; f = rmseF(r, kind_te)
      (best === nothing || f < best.F) && (best = (F = f, lam = lam, resid = r, z = z))
   end
   return best
end

ref = fit(spec([power_fun(K, k, 0.5) for k in 1:K]))
@printf("sqrt only: test F=%.4f lam=%g\n", ref.F, ref.lam)
sse(r) = [sum(abs2, r[(kind_te .== 2) .& (sid_te .== s)]) for s in 1:length(te)]
nF = [3length(sys) for sys in te]
for margin in (0.02, 0.1, 0.3, 1.0)
   lo, hi = density_ranges(base, tr, :sqrt; margin = margin)
   es = spec(vcat([cheb_funs(K, k, 8, :sqrt, lo[k], hi[k]) for k in 1:K]...))
   r = fit(es)
   s0, s1 = sse(ref.resid), sse(r.resid)
   worst = sortperm(s1 .- s0; rev = true)[1:3]
   @printf("cheb[sqrt] Kc=8 margin=%.2f: test F=%.4f lam=%-6g  dF vs sqrt=%+.1f%%; excluding worst 3 structures: F=%.4f vs sqrt %.4f\n",
           margin, r.F, r.lam, 100 * (r.F / ref.F - 1),
           sqrt(sum(s1[setdiff(1:100, worst)]) / sum(nF[setdiff(1:100, worst)])),
           sqrt(sum(s0[setdiff(1:100, worst)]) / sum(nF[setdiff(1:100, worst)])))
   for s in worst
      rho = site_densities(base, te[s])
      u = [2 .* (sqrt.(rho[:, k] .+ EPS) .- lo[k]) ./ (hi[k] - lo[k]) .- 1 for k in 1:K]
      @printf("    structure %3d (nat=%2d): force RMSE sqrt-only %.3f -> cheb %.3f; max|u| per width %s; sites with |u|>0.9: %d\n",
              s, length(te[s]), sqrt(s0[s] / nF[s]), sqrt(s1[s] / nF[s]),
              string(round.([maximum(abs, u[k]) for k in 1:K], digits = 2)), count(any(abs.(hcat(u...)) .> 0.9, dims = 2)))
   end
end
# Fraction of TRAINING sites in the outer 10% of the u range, per width
lo, hi = density_ranges(base, tr, :sqrt)
for k in 1:K
   us = vcat([2 .* (sqrt.(site_densities(base, sys)[:, k] .+ EPS) .- lo[k]) ./ (hi[k] - lo[k]) .- 1 for sys in tr]...)
   @printf("training sites width %d: |u|>0.9: %d / %d (%.2f%%); |u|>0.8: %.2f%%\n", k, count(abs.(us) .> 0.9), length(us),
           100count(abs.(us) .> 0.9) / length(us), 100count(abs.(us) .> 0.8) / length(us))
end
