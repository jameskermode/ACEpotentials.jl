# type-stability + allocation audit of the basis-with-derivatives path
include(joinpath(@__DIR__, "common.jl"))
using Profile, InteractiveUtils
const ThreadedEx = ACEpotentials.Models.ThreadedEx
const EquivariantTensors = ACEpotentials.Models.EquivariantTensors
using AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
structs = load_structs(1)
sys = structs[1]
models = [p for p in make_models((6,)) if p.first == "cat_D6"]

name, m = models[1]; randomise_linear!(m)

println("\n\n==== TYPE STABILITY AUDIT ====")

nl = PairList(sys, cutoff_radius(m))
Js, Rs, Zs, z0 = get_neighbours(sys, m, nl, 1)
d = AtomsData(sys; KW..., weights = ACEpotentials.default_weights(), v_ref = m.model.Vref)

function audit(label, f, args)
   rts = Base.return_types(f, typeof.(args))
   io = IOBuffer()
   code_warntype(io, f, typeof.(args))
   s = String(take!(io))
   # count the "red" entries: locals or ssa values inferred as Any / Union
   lines = split(s, '\n')
   bad = [l for l in lines if occursin(r"::(Any|Union\{)", l) || occursin("::ANY", l)]
   println("\n--- $label")
   println("    return type(s): ", rts)
   println("    lines with ::Any or ::Union in code_warntype: $(length(bad)) of $(length(lines))")
   for l in bad[1:min(end, 25)]
      println("      ", first(strip(l), 160))
   end
end

audit("evaluate_basis", M.evaluate_basis, (m.model, Rs, Zs, z0, m.ps, m.st))
audit("evaluate_basis_ed", M.evaluate_basis_ed, (m.model, Rs, Zs, z0, m.ps, m.st))
audit("evaluate_ed", M.evaluate_ed, (m.model, Rs, Zs, z0, m.ps, m.st))
audit("ET.evaluate(tensor)", EquivariantTensors.evaluate,
      (m.model.tensor, zeros(length(Rs), length(m.model.rbasis)),
       zeros(length(Rs), length(m.model.ybasis)), NamedTuple(), NamedTuple()))
audit("energy_forces_virial_basis(at, calc, ps, st)", M.energy_forces_virial_basis,
      (sys, m, m.ps, m.st))
audit("ACEfit.feature_matrix", ACEfit.feature_matrix, (d, m))
# the kwarg body of energy_forces_virial_basis (the wrapper above hides it)
let meth = which(M.energy_forces_virial_basis, typeof.((sys, m, m.ps, m.st)))
   body = Base.bodyfunction(meth)
   ex = ThreadedEx(); nt = Threads.nthreads()
   audit("energy_forces_virial_basis BODY", body,
         (1:length(sys), ex, nt, nl, pairs((;)), M.energy_forces_virial_basis, sys, m, m.ps, m.st))
end
audit("get_neighbours", get_neighbours, (sys, m, nl, 1))

# ---------------- allocation profile ------------------------------------
println("\n\n==== ALLOCATION PROFILE (Profile.Allocs, sample_rate=1e-4) ====")
M.energy_forces_virial_basis(sys, m)
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 1e-4 M.energy_forces_virial_basis(sys, m)
res = Profile.Allocs.fetch()
allocs = res.allocs
println("sampled allocations: $(length(allocs))")
# group by type
bytype = Dict{Any, Tuple{Int, Int}}()
for a in allocs
   c, b = get(bytype, a.type, (0, 0))
   bytype[a.type] = (c + 1, b + a.size)
end
top = sort(collect(bytype), by = x -> -x[2][1])[1:min(end, 15)]
println("top allocation types (count, bytes) [sampled]:")
for (t, (c, b)) in top
   @printf("   %8d  %10.1f MB  %s\n", c, b/2^20, first(string(t), 110))
end
# attribute to the first user frame in the stack
bysite = Dict{String, Int}()
for a in allocs
   for fr in a.stacktrace
      f = string(fr.file)
      if occursin("ACEpotentials/src", f) || occursin("EquivariantTensors", f) || occursin("ForwardDiff", f)
         key = "$(basename(f)):$(fr.line) $(fr.func)"
         bysite[key] = get(bysite, key, 0) + 1
         break
      end
   end
end
println("top allocation sites (first ACEpotentials/ET/ForwardDiff frame) [sampled count]:")
for (k, c) in sort(collect(bysite), by = x -> -x[2])[1:min(end, 20)]
   @printf("   %8d  %s\n", c, k)
end
