# (b) flat profile of energy_forces_virial_basis + type-stability audit
include(joinpath(@__DIR__, "common.jl"))
using Profile
using AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius

structs = load_structs(4)
sys = structs[1]
which_models = get(ENV, "MODELS", "cat_D6,emb16_D8")
models = [p for p in make_models() if p.first in split(which_models, ",")]

function flat_profile(name, f; mincount = 30, noisefloor = 2.0)
   f()                          # warm-up (compile)
   Profile.clear()
   Profile.init(n = 10^7, delay = 0.001)
   Profile.@profile f()
   println("\n==== flat profile: $name (sorted by count, mincount=$mincount) ====")
   Profile.print(format = :flat, sortedby = :count, mincount = mincount,
                 noisefloor = noisefloor, C = false)
   println("\n==== tree profile: $name (top-heavy, maxdepth 22) ====")
   Profile.print(format = :tree, mincount = mincount, noisefloor = noisefloor,
                 maxdepth = 22, C = false)
end

for (name, m) in models
   randomise_linear!(m)
   model_info(name, m)
   flat_profile("energy_forces_virial_basis $name",
                () -> M.energy_forces_virial_basis(sys, m); mincount = 40)
   nl = PairList(sys, cutoff_radius(m))
   Js, Rs, Zs, z0 = get_neighbours(sys, m, nl, 1)
   flat_profile("evaluate_basis_ed (200 sites) $name",
                () -> (for _ in 1:200; M.evaluate_basis_ed(m.model, Rs, Zs, z0, m.ps, m.st); end);
                mincount = 30)
   flush(stdout)
end

