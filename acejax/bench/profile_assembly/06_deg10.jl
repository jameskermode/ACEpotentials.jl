# one structure at degree 10 categorical (the 46,635-column production case)
# with the pushforward path, to anchor the projection.
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "basis_opt.jl"))
structs = load_structs(4)
sys = structs[end]     # 48 atoms? report whichever
tb = @elapsed m = ace1_model(elements = collect(ELS), order = 3, totaldegree = 10)
model_info("cat_D10", m); println("   model build: $(round(tb, digits=1)) s; natoms = $(length(sys))")
ws = PFWorkspace(m.model, 100)
efv_basis_v4(sys, m; ws = ws)
t = bench(() -> efv_basis_v4(sys, m; ws = ws), 3)
@printf("   efv_basis_v4 (workspace): %.3f s / structure of %d atoms  (%.2f ms/site), %.1f MB\n",
        t[1], length(sys), 1e3 * t[1] / length(sys), t[2]/2^20)
t = bench(() -> M.energy_forces_virial_serial(sys, m, m.ps, m.st), 3)
@printf("   energy_forces_virial_serial: %.4f s / structure\n", t[1])
nl = PairList(sys, cutoff_radius(m)); Js, Rs, Zs, z0 = get_neighbours(sys, m, nl, 1)
t = bench(() -> M.evaluate_basis_ed(m.model, Rs, Zs, z0, m.ps, m.st), 1)
@printf("   ORIGINAL evaluate_basis_ed, one site: %.2f s, %.0f MB  -> x%d sites = %.0f s/structure before the accumulation loop\n",
        t[1], t[2]/2^20, length(sys), t[1] * length(sys))
