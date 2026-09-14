# Per-structure error breakdown on the held-out set: is the 0.43 eV/A force RMSE
# a heavy tail of a few structures (=> label quality) or uniform (=> the basis
# cannot represent the labels)?  Refit from the cached assembly, then evaluate
# each test structure separately and relate its error to what it looks like.
using Distributed
@everywhere using ACEpotentials, ACEfit
using Random, LinearAlgebra, Printf, Serialization, Statistics
include(joinpath(@__DIR__, "tikhonov.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", "16")))
M = ACEpotentials.Models

const DATAFILE = ENV["DATA"]; const CACHEDIR = ENV["ASMCACHE"]
data_all = ACEpotentials.ExtXYZ.load(DATAFILE)
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))
ntr = round(Int, 0.8 * length(data_all))
tr, te = data_all[p[1:ntr]], data_all[p[ntr+1:end]]
kw = (energy_key = "mace_energy", force_key = "mace_force", virial_key = "mace_virial")
ELS = (:Cr, :Mn, :Fe, :Co, :Ni)
ORD, DEG = 3, parse(Int, get(ENV, "DEG", "6"))
LAM = parse(Float64, get(ENV, "LAM", "3.16228e-5"))    # degree-6 categorical optimum

m = ace1_model(elements = collect(ELS), order = ORD, totaldegree = DEG)
cache = joinpath(CACHEDIR, "asm_categorical_ace1_model_o$(ORD)_d$(DEG)_$(basename(DATAFILE)).jls")
A, Y, W = deserialize(cache)
P = ACEpotentials.Models.algebraic_smoothness_prior(m.model; p = 4.0)
A ./= reshape(P.diag, 1, :); A .*= W; Yw = W .* Y
T = TikhonovFactor(A, Yw)
ACEpotentials.Models.set_linear_parameters!(m, P \ tikhonov_solve(T, LAM))

# per-structure force RMSE, plus descriptors of the structure
using AtomsBase, Unitful
rows = map(te) do at
   r = ACEpotentials.compute_errors([at], m; kw..., verbose = false)["rmse"]["set"]
   F = reduce(hcat, at.atom_data.mace_force)
   pos = reduce(hcat, [ustrip.(u"Å", x) for x in position(at, :)])
   cell = reduce(hcat, [ustrip.(u"Å", v) for v in cell_vectors(at)])
   # nearest-neighbour distance and cell volume per atom as "how far from equilibrium"
   nat = length(at)
   (F = r["F"], E = r["E"], nat = nat,
    fmax = maximum(abs.(F)), frms = sqrt(mean(F .^ 2)),
    vol = abs(det(cell)) / nat)
end

Fs = [r.F for r in rows]
@printf("test structures: %d   force RMSE overall: %.4f\n", length(Fs), sqrt(mean(Fs .^ 2)))
@printf("per-structure force RMSE quantiles: p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f p99=%.3f max=%.3f\n",
        quantile(Fs, [0.1, 0.25, 0.5, 0.75, 0.9, 0.99])..., maximum(Fs))
# how much of the total squared error comes from the worst k structures?
o = sortperm(Fs, rev = true)
tot = sum(Fs .^ 2)
for k in (1, 5, 10, 20, 50)
   @printf("worst %2d structures carry %5.1f%% of squared force error\n", k, 100 * sum(Fs[o[1:k]] .^ 2) / tot)
end
# is error explained by how hard the structure is?  correlate with |F| scale
frms = [r.frms for r in rows]; fmax = [r.fmax for r in rows]; vol = [r.vol for r in rows]
@printf("corr(error, rms|F_ref|) = %.3f   corr(error, max|F_ref|) = %.3f   corr(error, vol/atom) = %.3f\n",
        cor(Fs, frms), cor(Fs, fmax), cor(Fs, vol))
println("RELERR relative error (rmse / rms|F_ref|) quantiles: ",
        round.(quantile(Fs ./ frms, [0.1, 0.5, 0.9]), digits = 3))
println("--- worst 8 ---")
for i in o[1:8]
   r = rows[i]
   @printf("  F=%.3f  E=%.4f  nat=%d  rms|F|=%.3f  max|F|=%.3f  vol/atom=%.2f\n",
           r.F, r.E, r.nat, r.frms, r.fmax, r.vol)
end
println("--- best 4 ---")
for i in o[end-3:end]
   r = rows[i]
   @printf("  F=%.3f  E=%.4f  nat=%d  rms|F|=%.3f  max|F|=%.3f  vol/atom=%.2f\n",
           r.F, r.E, r.nat, r.frms, r.fmax, r.vol)
end
