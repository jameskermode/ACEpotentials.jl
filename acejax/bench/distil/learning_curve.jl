# Learning curves from the CACHED assembly: test error vs training-set size, per
# model, on the same held-out structures.  Distinguishes data-limited (error
# still falling with N) from basis-limited (error flat in N).
#
# ACEfit assembles each structure's observations as a contiguous row block in
# data order (1 energy, 3*natoms forces, 6 virials), so the first N training
# structures are the first sum(rows) rows of the cached A -- no reassembly.
using Distributed
@everywhere using ACEpotentials, ACEfit
using Random, LinearAlgebra, Printf, Serialization
include(joinpath(@__DIR__, "tikhonov.jl"))
BLAS.set_num_threads(parse(Int, get(ENV, "BLASTHREADS", "16")))
M = ACEpotentials.Models

const DATAFILE = ENV["DATA"]; const CACHEDIR = ENV["ASMCACHE"]
data_all = ACEpotentials.ExtXYZ.load(DATAFILE)
rng = MersenneTwister(0); p = shuffle(rng, 1:length(data_all))    # SAME split
ntr = round(Int, 0.8 * length(data_all))
tr, te = data_all[p[1:ntr]], data_all[p[ntr+1:end]]
kw = (energy_key = "mace_energy", force_key = "mace_force", virial_key = "mace_virial")
emb = M.read_mace_embedding(ENV["EMB"])
const EMBTAG = get(ENV, "EMBTAG", "")
ELS = (:Cr, :Mn, :Fe, :Co, :Ni)
ORD, DEG = 3, parse(Int, ENV["DEG"])
SMOOTH = 4.0
LAMS = 10.0 .^ (0:-0.5:-9)
NS = (50, 100, 200, 400, ntr)

rows_per(at) = 1 + 3 * length(at) + 6
cum = cumsum(rows_per.(tr))

function curve(name, mk)
   m = mk()
   tag = replace(name, r"[^A-Za-z0-9]" => "_")
   # same key as fit_distilled.jl: the embedded matrices depend on the table
   # and the reduction (EMBTAG), the categorical one does not
   embtag = (startswith(tag, "embedded") && !isempty(EMBTAG)) ? "_" * EMBTAG : ""
   cache = joinpath(CACHEDIR, "asm_$(tag)_o$(ORD)_d$(DEG)$(embtag)_$(basename(DATAFILE)).jls")
   if isfile(cache)
      A, Y, W = deserialize(cache)
   else
      A, Y, W = ACEpotentials.assemble(tr, m; kw...)
      mkpath(CACHEDIR); serialize(cache, (A, Y, W))
   end
   size(A, 1) == cum[end] || error("row count $(size(A,1)) != expected $(cum[end]) -- ordering assumption broken")
   P = ACEpotentials.Models.algebraic_smoothness_prior(m.model; p = SMOOTH)
   A ./= reshape(P.diag, 1, :); A .*= W; Yw = W .* Y
   for N in NS
      r = cum[N]
      An, Yn = view(A, 1:r, :), view(Yw, 1:r)
      T = TikhonovFactor(Matrix(An), Vector(Yn))     # once per N
      best = (Inf, Inf, Inf, 0.0)
      for lam in LAMS
         ACEpotentials.Models.set_linear_parameters!(m, P \ tikhonov_solve(T, lam))
         rt = ACEpotentials.compute_errors(te, m; kw..., verbose = false)["rmse"]["set"]
         rtr = ACEpotentials.compute_errors(tr[1:N], m; kw..., verbose = false)["rmse"]["set"]
         rt["F"] < best[1] && (best = (rt["F"], rt["E"], rtr["F"], lam))
      end
      @printf("LC deg=%d %-22s N=%4d rows=%6d  testF=%.4f testE=%.4f  trainF=%.4f  lambda=%g\n",
              DEG, name, N, r, best[1], best[2], best[3], best[4])
      flush(stdout)
   end
end

curve("categorical ace1_model", () -> ace1_model(elements = collect(ELS), order = ORD, totaldegree = DEG))
curve("embedded d_max=16", () -> M.ace_embedding_model(elements = ELS, order = ORD,
          totaldegree = DEG, embedding = emb, d_max = 16))
