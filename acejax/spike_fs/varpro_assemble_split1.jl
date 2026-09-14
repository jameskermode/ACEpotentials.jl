# Assemble the degree-4 ACE design matrix for the TRANSFER split (Exp. 5 of the
# VarPro spike): a fresh 200/100 split drawn with MersenneTwister(1) from the
# 700 structures NOT used by the MersenneTwister(0) split of the earlier
# spikes, so train and test are disjoint from everything the densities were
# learned on.  Cached to the scratchpad like run_spike.jl does.
#
#   julia -p 4 --project=acejax/julia acejax/spike_fs/varpro_assemble_split1.jl
using Distributed
@everywhere using ACEpotentials, ACEfit
using AtomsBase, Random, Printf, Serialization
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const D = parse(Int, get(ENV, "DEG", "4"))
data_all = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
kw = (energy_key = "mace_energy", force_key = "mace_force", virial_key = "mace_virial")
p0 = shuffle(MersenneTwister(0), 1:length(data_all))
rest = sort(p0[301:end])                                   # the 700 unused structures
p1 = shuffle(MersenneTwister(1), rest)
tr1, te1 = data_all[p1[1:200]], data_all[p1[201:300]]
@info "split1: train 200 / test 100, all disjoint from split0's 300; first ids $(p1[1:5])"
m = ace1_model(elements = ELS, order = 3, totaldegree = D)
for (tag, data) in (("train200", tr1), ("test100", te1))
   f = joinpath(CACHEDIR, "asm_cantor_D$(D)_split1_$(tag).jls")
   isfile(f) && (@info "exists: $f"; continue)
   t = @elapsed (A, Y, W) = ACEpotentials.assemble(data, m; kw...)
   @info @sprintf("assembled %s %s in %.0f s", tag, size(A), t)
   serialize(f, (Matrix(A), Y, W))
end
@info "done"
