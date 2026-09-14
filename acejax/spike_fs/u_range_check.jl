# How far do test-set densities fall outside the training range used to map
# u(rho) to [-1,1]?  (Explains the Chebyshev Kc=8 test blow-up.)
using ACEpotentials, AtomsBase, StaticArrays, LinearAlgebra, Random, Printf, Statistics
include(joinpath(@__DIR__, "fs_embed.jl"))
SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
data_all = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
tr, te = data_all[p[1:200]], data_all[p[201:300]]
ZS = [24, 25, 26, 27, 28]
es = EmbedSpec(ZS, [], [])
for tf in (:sqrt, :log)
   lo, hi = density_ranges(es, tr, tf)
   f(x) = tf == :sqrt ? sqrt(x + EPS) : log(x + EPS)
   for k in 1:3
      us = Float64[]; sids = Int[]
      for (is, sys) in enumerate(te)
         rho = site_densities(es, sys)
         u = 2 .* (f.(rho[:, k]) .- lo[k]) ./ (hi[k] - lo[k]) .- 1
         append!(us, u); append!(sids, fill(is, length(u)))
      end
      out = abs.(us) .> 1
      @printf("%-5s width k=%d: test sites with |u|>1: %4d / %5d (%.2f%%), in %2d structures; max |u| = %.3f; T_8(max u) = %.1f\n",
              tf, k, count(out), length(us), 100 * count(out) / length(us), length(unique(sids[out])),
              maximum(abs, us), cheb_T_dT(maximum(abs, us), 8)[1][9])
   end
end
