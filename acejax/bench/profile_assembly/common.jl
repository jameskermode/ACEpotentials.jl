# Shared setup for the assembly profiling scripts.  Run with
#   julia --project=acejax/julia acejax/bench/profile_assembly/<script>.jl
using ACEpotentials, ACEfit, LinearAlgebra, StaticArrays, Printf
using ACEpotentials: AtomsData
M = ACEpotentials.Models

const SP = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad/distil"
const DATAFILE = joinpath(SP, "cantor1k_b_mh1.xyz")
const EMBFILE  = joinpath(SP, "mace_mh1_embedding.json")
const ELS = (:Cr, :Mn, :Fe, :Co, :Ni)
const KW = (energy_key = "mace_energy", force_key = "mace_force",
            virial_key = "mace_virial")

load_structs(n = 32) = ACEpotentials.ExtXYZ.load(DATAFILE)[1:n]

function make_models(degs = (6, 8); emb = M.read_mace_embedding(EMBFILE))
   models = Pair{String, Any}[]
   for D in degs
      push!(models, "cat_D$D" => ace1_model(elements = collect(ELS), order = 3,
                                             totaldegree = D))
      push!(models, "emb16_D$D" => M.ace_embedding_model(elements = ELS, order = 3,
                                  totaldegree = D, embedding = emb, d_max = 16))
   end
   return models
end

# give the model non-trivial (fitted-style) linear parameters so that the
# plain energy_forces_virial does real work
function randomise_linear!(m)
   nB = size(m.ps.WB, 1) * size(m.ps.WB, 2) + length(m.ps.Wpair)
   M.set_linear_parameters!(m, 0.01 * randn(nB))
   return m
end

function model_info(name, m)
   nB = size(m.ps.WB, 1); NZ = size(m.ps.WB, 2)
   nA = length(m.model.tensor.abasis); nAA = length(m.model.tensor.aabasis)
   nR = length(m.model.rbasis); nY = length(m.model.ybasis)
   npair = length(m.model.pairbasis)
   rc = maximum(x.rcut for x in m.model.rbasis.rin0cuts)
   @printf("%-10s n_B/elem=%5d  NZ=%d  length_basis=%6d  nR=%3d nY=%3d nA=%5d nAA=%6d npair=%d rcut=%.2f\n",
           name, nB, NZ, M.length_basis(m), nR, nY, nA, nAA, npair, rc)
end

# a quick timer: min of n runs, returns (seconds, bytes allocated, gc fraction)
function bench(f, n = 3)
   f()   # warm-up
   best = (Inf, 0, 0.0)
   for _ in 1:n
      GC.gc()
      stats = @timed f()
      t = stats.time; b = stats.bytes; g = stats.gctime / stats.time
      t < best[1] && (best = (t, b, g))
   end
   return best
end
