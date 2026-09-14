# Before/after + exactness for the scratch optimised evaluator (fast_efv.jl).
#   julia -t 1 --project=acejax/julia acejax/bench/profile_forward/04_opt.jl
#   julia -t 4 --project=acejax/julia acejax/bench/profile_forward/04_opt.jl
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fast_efv.jl"))

println("threads = $(Threads.nthreads())")
frames = load_frames(2)
small = frames[1]; big = supercell(frames[1], 2)
si = si_frame(); si_big = supercell(si, 2)
models = make_models((6, 8))

for (name, m) in models
   fe = FastEFV(m)
   @printf("\n=== %s: radial LEN=%d -> NU=%d distinct columns; pair LEN=%d -> NU=%d\n",
           name, fe.rad.LEN, fe.rad.NU, fe.pair.LEN, fe.pair.NU)
   systems = name == "Si_D10" ? [("Si", si), ("Si 2x2x2", si_big)] :
                                 [("cantor", small), ("cantor 2x2x2", big)]
   for (slabel, sys) in systems
      ref = efv(sys, m)
      res, wss = fast_efv(sys, fe; ntasks = 1)
      dE, dF, dV = maxdiff(ref, res)
      @printf("  exactness %-14s |ΔE|=%.2e  max|ΔF|=%.2e  max|ΔV|=%.2e  %s\n", slabel, dE, dF, dV,
              (dE < 1e-12 * max(1, abs(ustrip(ref.energy))) && dF < 1e-10) ? "OK" : "FAIL")
      t0, b0, g0, a0 = bench(() -> efv(sys, m), 5)
      report("  production $name $slabel", sys, m, t0, b0, g0, a0)
      # serial, fresh workspaces and nlist each call
      t1, b1, g1, a1 = bench(() -> fast_efv(sys, fe; ntasks = 1), 5)
      report("  fast serial (nlist+ws per call) $slabel", sys, m, t1, b1, g1, a1)
      # serial, reuse workspaces, nlist per call
      t2, b2, g2, a2 = bench(() -> fast_efv(sys, fe; ntasks = 1, wss = wss), 5)
      report("  fast serial (ws reused) $slabel", sys, m, t2, b2, g2, a2)
      nl = PairList(sys, fe.rcut * u"Å")
      t3, b3, g3, a3 = bench(() -> fast_efv(sys, fe; ntasks = 1, wss = wss, nlist = nl), 5)
      report("  fast serial (ws+nlist reused) $slabel", sys, m, t3, b3, g3, a3)
      if Threads.nthreads() > 1
         _, wss4 = fast_efv(sys, fe; ntasks = Threads.nthreads())
         res4, _ = fast_efv(sys, fe; ntasks = Threads.nthreads(), wss = wss4)
         dE, dF, dV = maxdiff(ref, res4)
         t4, b4, g4, a4 = bench(() -> fast_efv(sys, fe; ntasks = Threads.nthreads(), wss = wss4), 5)
         report("  fast $(Threads.nthreads()) tasks (ws reused) $slabel  [ΔF=$(round(dF, sigdigits=2))]", sys, m, t4, b4, g4, a4)
      end
      @printf("  speed-up serial: %.2fx (ws reused: %.2fx, +nlist: %.2fx)\n", t0/t1, t0/t2, t0/t3)
   end
end
