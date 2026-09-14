# Measurement table: energy_forces_virial per structure, all models, two sizes.
#   julia -t 1 --project=acejax/julia acejax/bench/profile_forward/01_measure.jl
#   julia -t 4 --project=acejax/julia acejax/bench/profile_forward/01_measure.jl
include(joinpath(@__DIR__, "common.jl"))

println("threads = $(Threads.nthreads()), julia $(VERSION)")
frames = load_frames(4)
small = frames[1]
big   = supercell(frames[1], 2)
si    = si_frame()
si_big = supercell(si, 2)
println("small: $(length(small)) atoms, big: $(length(big)) atoms, Si: $(length(si)) / $(length(si_big))")

models = make_models((6, 8))
for (name, m) in models; model_info(name, m); end

for (name, m) in models
   rc = ustrip(u"Å", M.cutoff_radius(m))
   systems = name == "Si_D10" ? [("Si", si), ("Si 2x2x2", si_big)] :
                                 [("cantor", small), ("cantor 2x2x2", big)]
   for (slabel, sys) in systems
      nn = mean_nneigh(sys, rc)
      t, b, g, a = bench(() -> efv(sys, m), 5)
      report("$name $slabel (nneigh=$(round(nn, digits=1)))", sys, m, t, b, g, a)
   end
end

# neighbour-list cost on its own
println("\nneighbour list (PairList) cost:")
for (slabel, sys) in [("cantor", small), ("cantor 2x2x2", big)]
   rc = M.cutoff_radius(models[1][2])
   t, b, g, a = bench(() -> M.PairList(sys, rc), 5)
   @printf("  %-16s nat=%4d  %8.3f ms  %6.2f MB  %d allocs\n", slabel, length(sys), t*1e3, b/1e6, a)
end

# precomputed nlist + serial path vs default (what the nlist rebuild costs per call)
println("\nnlist reuse (energy_forces_virial with nlist kwarg) vs default:")
for (name, m) in models[1:2]
   for (slabel, sys) in [("cantor", small), ("cantor 2x2x2", big)]
      nlist = M.PairList(sys, M.cutoff_radius(m))
      t1, = bench(() -> efv(sys, m), 5)
      t2, = bench(() -> AtomsCalculators.energy_forces_virial(sys, m; nlist = nlist), 5)
      @printf("  %-10s %-14s default %8.3f ms   reuse %8.3f ms  (%.1f%% saved)\n",
              name, slabel, t1*1e3, t2*1e3, 100*(1 - t2/t1))
   end
end
