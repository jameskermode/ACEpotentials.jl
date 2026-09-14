# Demonstrate the optimisations in basis_opt.jl: correctness vs the original
# energy_forces_virial_basis, and timings.  Run with `julia -t N` for v3t.
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "basis_opt.jl"))
using Unitful: ustrip

structs = load_structs(2)
sys = structs[1]
which_models = get(ENV, "MODELS", "cat_D6,emb16_D6,cat_D8,emb16_D8")
models = [p for p in make_models() if p.first in split(which_models, ",")]
println("threads: $(Threads.nthreads())")

relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), 1e-300)
function check(name, efv, ref)
   eE = relerr(efv.energy, ustrip.(ref.energy))
   eF = relerr(reinterpret(Float64, efv.forces), reinterpret(Float64, ustrip.(ref.forces)))
   eV = relerr(reinterpret(Float64, efv.virial), reinterpret(Float64, ustrip.(ref.virial)))
   @printf("   %-14s max rel err  E %.1e  F %.1e  V %.1e\n", name, eE, eF, eV)
end

for (name, m) in models
   randomise_linear!(m)
   model_info(name, m)
   ref = M.energy_forces_virial_basis(sys, m)
   t0 = bench(() -> M.energy_forces_virial_basis(sys, m), 2)
   @printf("   %-14s %8.3f s  %9.1f MB  gc %4.1f%%\n", "original", t0[1], t0[2]/2^20, 100t0[3])
   for (label, f) in ("v1 stable-acc" => () -> efv_basis_v1(sys, m),
                      "v2 fd-block"   => () -> efv_basis_v2(sys, m),
                      "v3 pushfwd"    => () -> efv_basis_v3(sys, m),
                      "v3t threaded"  => () -> efv_basis_v3t(sys, m))
      out = f()
      check(label, out, ref)
      t = bench(f, 3)
      @printf("   %-14s %8.3f s  %9.1f MB  gc %4.1f%%   speed-up %6.1fx\n",
              label, t[1], t[2]/2^20, 100t[3], t0[1]/t[1])
   end
   # per-site cost of the three derivative kernels
   nl = PairList(sys, cutoff_radius(m))
   Js, Rs, Zs, z0 = get_neighbours(sys, m, nl, 1)
   pf = PFState(m.model)
   t = bench(() -> M.evaluate_basis_ed(m.model, Rs, Zs, z0, m.ps, m.st), 3)
   @printf("   per-site  evaluate_basis_ed (orig) %8.2f ms  %7.1f MB\n", 1e3t[1], t[2]/2^20)
   t = bench(() -> basis_ed_fd(m.model, Rs, Zs, z0, m.ps, m.st), 3)
   @printf("   per-site  basis_ed_fd (block)      %8.2f ms  %7.1f MB\n", 1e3t[1], t[2]/2^20)
   t = bench(() -> basis_ed_pf(m.model, Rs, Zs, z0, m.ps, m.st, pf), 3)
   @printf("   per-site  basis_ed_pf (pushfwd)    %8.2f ms  %7.1f MB\n", 1e3t[1], t[2]/2^20)
   t = bench(() -> M.evaluate_ed(m.model, Rs, Zs, z0, m.ps, m.st), 3)
   @printf("   per-site  evaluate_ed (1 eval+grad)%8.2f ms  %7.1f MB\n", 1e3t[1], t[2]/2^20)
   flush(stdout)
end
