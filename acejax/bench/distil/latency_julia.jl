# Per-frame latency of the JULIA evaluator (ACEpotentials, CPU) on the same
# frames as latency.py, so the two evaluators of the same model can be compared
# on one host.  Random weights (cost is weight-independent).
#
#   FRAMES=cantor16_k1.xyz EMB=... DEG=6 DMAX=16 julia -t 1 --project=acejax/julia latency_julia.jl
using ACEpotentials, AtomsCalculators, AtomsBase, Random, Printf, LinearAlgebra
M = ACEpotentials.Models
BLAS.set_num_threads(1)
frames = ACEpotentials.ExtXYZ.load(ENV["FRAMES"])
nat = length.(frames)
emb = M.read_mace_embedding(ENV["EMB"])
DEG = parse(Int, get(ENV, "DEG", "6")); DMAX = parse(Int, get(ENV, "DMAX", "16"))
ELS = (:Cr, :Mn, :Fe, :Co, :Ni)
Random.seed!(11)
models = Any[
   ("ACE embedded d<=$DMAX deg $DEG", M.ace_embedding_model(elements = ELS, order = 3,
                                        totaldegree = DEG, embedding = emb, d_max = DMAX)),
]
get(ENV, "CATEGORICAL", "0") == "1" &&
   push!(models, ("ACE categorical deg $DEG", ace1_model(elements = ELS, order = 3, totaldegree = DEG)))
println("$(length(frames)) frames, $(minimum(nat))-$(maximum(nat)) atoms, threads=$(Threads.nthreads())")
for (name, m) in models
   m.ps.WB .= 0.02 .* randn(size(m.ps.WB)); m.ps.Wpair .= 0.02 .* randn(size(m.ps.Wpair))
   efv(sys) = AtomsCalculators.energy_forces_virial(sys, m)
   efv(frames[1])                                     # compile
   ts = Float64[]
   for rep = 1:3
      t0 = time_ns()
      for f in frames; efv(f); end
      push!(ts, (time_ns() - t0) / 1e9 / length(frames))
   end
   t = sort(ts)[2]                                     # median of 3
   @printf("%-28s frame %8.3f ms/frame  %.3e atom-steps/s\n", name, t * 1e3, sum(nat) / length(nat) / t)
end
