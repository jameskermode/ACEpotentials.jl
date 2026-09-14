# Is the MP-0 / MH-1 disagreement smooth or rough?  Same basis, same A, three
# targets: MP-0 labels, MH-1 labels, and their DIFFERENCE.  If the difference
# fits well, the disagreement is a smooth function of structure and tells us
# nothing about label noise; if it fits about as badly as its own magnitude,
# it is rough.  Also answers: is MH-1 more learnable than MP-0?
using Distributed
@everywhere using ACEpotentials, ACEfit
using Random, LinearAlgebra, Printf, Serialization, Statistics
include(joinpath(@__DIR__, "tikhonov.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", "16")))
M = ACEpotentials.Models

CACHEDIR = ENV["ASMCACHE"]; DEG = parse(Int, get(ENV, "DEG", "6")); ORD = 3
kw = (energy_key = "mace_energy", force_key = "mace_force", virial_key = "mace_virial")
ELS = (:Cr, :Mn, :Fe, :Co, :Ni)
LAMS = 10.0 .^ (0:-0.5:-9)

function split(file)
   d = ACEpotentials.ExtXYZ.load(file)
   rng = MersenneTwister(0); p = shuffle(rng, 1:length(d)); ntr = round(Int, 0.8 * length(d))
   d[p[1:ntr]], d[p[ntr+1:end]]
end
tr0, te0 = split(ENV["DATA_MP0"])
tr1, te1 = split(ENV["DATA_MH1"])

m = ace1_model(elements = collect(ELS), order = ORD, totaldegree = DEG)
c0 = joinpath(CACHEDIR, "asm_categorical_ace1_model_o$(ORD)_d$(DEG)_$(basename(ENV["DATA_MP0"])).jls")
c1 = joinpath(CACHEDIR, "asm_categorical_ace1_model_o$(ORD)_d$(DEG)_$(basename(ENV["DATA_MH1"])).jls")
A0, Y0, W0 = deserialize(c0)
if isfile(c1)
   A1, Y1, W1 = deserialize(c1)
else
   A1, Y1, W1 = ACEpotentials.assemble(tr1, m; kw...); serialize(c1, (A1, Y1, W1))
end
# same structures, same basis => same design matrix.  Assert rather than assume.
@assert size(A0) == size(A1)
@printf("max|A0 - A1| = %.2e   max|W0 - W1| = %.2e   (must both be ~0)\n",
        maximum(abs.(A0 .- A1)), maximum(abs.(W0 .- W1)))
P = ACEpotentials.Models.algebraic_smoothness_prior(m.model; p = 4.0)
A0 ./= reshape(P.diag, 1, :); A0 .*= W0
rmsY(y) = sqrt(mean((y ./ W0) .^ 2))     # unweighted RMS of a target, for scale

# the difference target as its own labelled file (same structures, same order),
# written by Python -- AtomsData is immutable, so it cannot be built in place
trd, ted = split(ENV["DATA_DIFF"])
targets = (("MP-0",       Y0,      tr0, te0),
           ("MH-1",       Y1,      tr1, te1),
           ("MP-0 - MH-1", Y0 .- Y1, trd, ted))
for (name, Y, trs, tes) in targets
   T = TikhonovFactor(A0, W0 .* Y)
   best = (Inf, Inf, Inf, 0.0)
   for lam in LAMS
      ACEpotentials.Models.set_linear_parameters!(m, P \ tikhonov_solve(T, lam))
      rt  = ACEpotentials.compute_errors(tes, m; kw..., verbose = false)["rmse"]["set"]
      rtr = ACEpotentials.compute_errors(trs, m; kw..., verbose = false)["rmse"]["set"]
      rt["F"] < best[1] && (best = (rt["F"], rt["E"], rtr["F"], lam))
   end
   @printf("TARGET %-12s  rms(target F)=%.3f   test F=%.4f  train F=%.4f  test E=%.4f  lambda=%g\n",
           name, rmsY(Y), best[1], best[3], best[2], best[4])
   flush(stdout)
end
