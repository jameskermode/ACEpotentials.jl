# thread scaling of the scratch evaluator vs production, 256-atom cell, ntasks = 1,2,4
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fast_efv.jl"))
frames = load_frames(1); big = supercell(frames[1], 2); big3 = supercell(frames[1], 3)
models = make_models((6,); si = false)
println("threads = $(Threads.nthreads()) (M3 Pro: 6 performance + 6 efficiency cores; load avg $(Sys.loadavg()[1]))")
for (name, m) in models
   fe = FastEFV(m)
   for (slabel, sys) in [("cantor 2x2x2", big), ("cantor 3x3x3", big3)]
      nl = PairList(sys, fe.rcut * u"Å")
      t0, = bench(() -> AtomsCalculators.energy_forces_virial(sys, m; nlist = nl), 5)
      @printf("%-9s %-13s nat=%4d production (Folds, %d threads, nlist reused): %8.3f ms  %.3e atom-steps/s\n",
              name, slabel, length(sys), Threads.nthreads(), t0*1e3, length(sys)/t0)
      for nt in (1, 2, 4)
         nt > Threads.nthreads() && continue
         _, wss = fast_efv(sys, fe; ntasks = nt, nlist = nl)
         t, b, g, a = bench(() -> fast_efv(sys, fe; ntasks = nt, wss = wss, nlist = nl), 5)
         @printf("%-9s %-13s nat=%4d fast ntasks=%d: %8.3f ms  %.3e atom-steps/s  (%.2fx vs 1 task)\n",
                 name, slabel, length(sys), nt, t*1e3, length(sys)/t, nt == 1 ? 1.0 : NaN)
      end
   end
end
