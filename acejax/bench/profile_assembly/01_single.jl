# (a) per-structure timings: energy_forces_virial_basis vs energy_forces_virial
# (c) ACEfit.feature_matrix per structure
include(joinpath(@__DIR__, "common.jl"))
using AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius


structs = load_structs(32)
sys = structs[1]
println("structure 1: $(length(sys)) atoms; 32 structs: $(sum(length, structs)) atoms")
models = make_models()

for (name, m) in models
   randomise_linear!(m)
   model_info(name, m)
   nl = PairList(sys, cutoff_radius(m))
   nn = [length(get_neighbours(sys, m, nl, i)[1]) for i in 1:length(sys)]
   @printf("   neighbours/site: mean %.1f  max %d\n", sum(nn)/length(nn), maximum(nn))
   # neighbour list
   t = bench(() -> PairList(sys, cutoff_radius(m)))
   @printf("   PairList                 %8.4f s  %8.1f MB\n", t[1], t[2]/2^20)
   # one plain evaluation (energy+forces+virial) -- threaded default
   t = bench(() -> M.energy_forces_virial(sys, m))
   @printf("   energy_forces_virial     %8.4f s  %8.1f MB  gc %4.1f%%  (nthreads=%d)\n",
           t[1], t[2]/2^20, 100t[3], Threads.nthreads())
   t = bench(() -> M.energy_forces_virial_serial(sys, m, m.ps, m.st))
   @printf("   energy_forces_virial_ser %8.4f s  %8.1f MB  gc %4.1f%%\n", t[1], t[2]/2^20, 100t[3])
   tefv = t[1]
   # per-site basis pieces
   Js, Rs, Zs, z0 = get_neighbours(sys, m, nl, 1)
   t = bench(() -> M.evaluate_basis(m.model, Rs, Zs, z0, m.ps, m.st))
   @printf("   evaluate_basis (1 site)  %8.5f s  %8.2f MB\n", t[1], t[2]/2^20)
   t = bench(() -> M.evaluate_ed(m.model, Rs, Zs, z0, m.ps, m.st))
   @printf("   evaluate_ed    (1 site)  %8.5f s  %8.2f MB\n", t[1], t[2]/2^20)
   t = bench(() -> M.evaluate_basis_ed(m.model, Rs, Zs, z0, m.ps, m.st), 2)
   @printf("   evaluate_basis_ed (1 site) %8.4f s  %8.1f MB  gc %4.1f%%\n", t[1], t[2]/2^20, 100t[3])
   tbed = t[1]
   # the basis path
   t = bench(() -> M.energy_forces_virial_basis(sys, m), 2)
   @printf("   energy_forces_virial_basis %8.3f s  %8.1f MB  gc %4.1f%%   ratio to efv: %.0fx ; per-site ed sum %.3f s (%.0f%% of total)\n",
           t[1], t[2]/2^20, 100t[3], t[1]/tefv, tbed*length(sys), 100*tbed*length(sys)/t[1])
   tbasis = t[1]
   # (c) the ACEfit wrapper
   d = AtomsData(sys; KW..., weights = ACEpotentials.default_weights(), v_ref = m.model.Vref)
   t = bench(() -> ACEfit.feature_matrix(d, m), 2)
   @printf("   ACEfit.feature_matrix    %8.3f s  %8.1f MB  gc %4.1f%%   overhead over efv_basis: %.3f s (%.1f%%)\n",
           t[1], t[2]/2^20, 100t[3], t[1]-tbasis, 100*(t[1]-tbasis)/tbasis)
   flush(stdout)
end
