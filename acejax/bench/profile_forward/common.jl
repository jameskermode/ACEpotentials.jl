# Shared setup for the FORWARD-path (energy/forces/virial) profiling scripts.
#   julia -t 1 --project=acejax/julia acejax/bench/profile_forward/<script>.jl
using ACEpotentials, AtomsBase, AtomsCalculators, AtomsBuilder, LinearAlgebra,
      StaticArrays, Printf, Random, Unitful
const M = ACEpotentials.Models
BLAS.set_num_threads(1)

const SP = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad/distil"
const DATAFILE = joinpath(SP, "cantor1k_b_mh1.xyz")
const EMBFILE  = joinpath(SP, "mace_mh1_embedding.json")
const ELS = (:Cr, :Mn, :Fe, :Co, :Ni)

# Frames as FlexibleSystem so that AtomsBuilder's `repeat` works.  All the
# measurements below convert to FlexibleSystem once, up front.
function to_flexible(sys)
   atoms = [Atom(atomic_number(sys, i), position(sys, i)) for i = 1:length(sys)]
   return FlexibleSystem(atoms; cell_vectors = cell_vectors(sys),
                         periodicity = periodicity(sys))
end
function load_frames(n = 8)
   fr = ACEpotentials.ExtXYZ.load(DATAFILE)[1:n]
   return [to_flexible(f) for f in fr]
end
supercell(sys::FlexibleSystem, n = 2) = repeat(sys, (n, n, n))

function si_frame()
   ds = ACEpotentials.example_dataset("Si_tiny")[1]
   # take the largest cell in the tiny set, then repeat if needed
   sys = ds[argmax(length.(ds))]
   return to_flexible(sys)
end

function make_models(degs = (6, 8); emb = M.read_mace_embedding(EMBFILE),
                     si = true)
   models = Pair{String, Any}[]
   for D in degs
      push!(models, "cat_D$D" => ace1_model(elements = collect(ELS), order = 3,
                                             totaldegree = D))
      push!(models, "emb16_D$D" => M.ace_embedding_model(elements = ELS, order = 3,
                                  totaldegree = D, embedding = emb, d_max = 16))
   end
   si && push!(models, "Si_D10" => ace1_model(elements = [:Si], order = 3,
                                              totaldegree = 10))
   for (_, m) in models; randomise_linear!(m); end
   return models
end

function randomise_linear!(m; seed = 11)
   Random.seed!(seed)
   m.ps.WB .= 0.02 .* randn(size(m.ps.WB))
   m.ps.Wpair .= 0.02 .* randn(size(m.ps.Wpair))
   return m
end

function model_info(name, m)
   nB = size(m.ps.WB, 1); NZ = size(m.ps.WB, 2)
   nA = length(m.model.tensor.abasis); nAA = length(m.model.tensor.aabasis)
   nR = length(m.model.rbasis); nY = length(m.model.ybasis)
   npair = length(m.model.pairbasis)
   rc = maximum(x.rcut for x in m.model.rbasis.rin0cuts)
   @printf("%-10s n_B/elem=%5d  NZ=%d  nR=%3d nY=%3d nA=%5d nAA=%6d npair=%d rcut=%.2f rbasis=%s\n",
           name, nB, NZ, nR, nY, nA, nAA, npair, rc, nameof(typeof(m.model.rbasis)))
end

# min-of-n timer, returns (seconds, bytes, gc fraction, allocs)
function bench(f, n = 5)
   f()   # warm-up
   best = (Inf, 0, 0.0, 0)
   for _ in 1:n
      GC.gc()
      stats = @timed f()
      t = stats.time; b = stats.bytes; g = stats.gctime / stats.time
      a = Base.gc_alloc_count(stats.gcstats)
      t < best[1] && (best = (t, b, g, a))
   end
   return best
end

mean_nneigh(sys, rc) = begin
   nl = ACEpotentials.Models.PairList(sys, rc * u"Å")
   length(nl.i) / length(sys)
end

efv(sys, m) = AtomsCalculators.energy_forces_virial(sys, m)

function report(label, sys, m, t, b, g, a)
   nat = length(sys)
   @printf("%-44s nat=%4d  %9.3f ms  %9.3e atom-steps/s  %8.2f MB  %6d allocs  GC %4.1f%%\n",
           label, nat, t * 1e3, nat / t, b / 1e6, a, 100g)
end

# exactness check between two efv results
function maxdiff(r1, r2)
   dE = abs(ustrip(r1.energy) - ustrip(r2.energy))
   dF = maximum(norm(ustrip.(f1 - f2)) for (f1, f2) in zip(r1.forces, r2.forces))
   dV = maximum(abs.(ustrip.(r1.virial - r2.virial)))
   return dE, dF, dV
end
