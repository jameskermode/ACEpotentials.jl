# v4 (workspace reuse) vs v3, correctness and allocation; run with -t 1
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "basis_opt.jl"))
structs = load_structs(2); sys = structs[1]
models = [p for p in make_models() if p.first in split(get(ENV, "MODELS", "cat_D6,cat_D8,emb16_D8"), ",")]
relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
for (name, m) in models
   randomise_linear!(m); model_info(name, m)
   r3 = efv_basis_v3(sys, m)
   ws = PFWorkspace(m.model, 90)
   r4 = efv_basis_v4(sys, m; ws = ws)
   @printf("   v4 vs v3: E %.1e  F %.1e  V %.1e\n", relerr(r4.energy, r3.energy),
           relerr(reinterpret(Float64, r4.forces), reinterpret(Float64, r3.forces)),
           relerr(reinterpret(Float64, r4.virial), reinterpret(Float64, r3.virial)))
   t3 = bench(() -> efv_basis_v3(sys, m), 5)
   t4 = bench(() -> efv_basis_v4(sys, m; ws = ws), 5)
   @printf("   v3 %.4f s %7.1f MB | v4 (workspace) %.4f s %7.1f MB\n", t3[1], t3[2]/2^20, t4[1], t4[2]/2^20)
   # sustained: 32 structures back to back, no explicit GC, to see the GC share
   s32 = load_structs(32)
   efv_basis_v4(s32[2], m; ws = ws)
   st3 = @timed for s in s32; efv_basis_v3(s, m); end
   st4 = @timed for s in s32; efv_basis_v4(s, m; ws = ws); end
   @printf("   32 structs back-to-back: v3 %.2f s (gc %.0f%%, %.1f GB) | v4 %.2f s (gc %.0f%%, %.1f GB)\n",
           st3.time, 100st3.gctime/st3.time, st3.bytes/2^30, st4.time, 100st4.gctime/st4.time, st4.bytes/2^30)
end
