# Type-stability + allocation audit of the forward path.
#   julia -t 1 --project=acejax/julia acejax/bench/profile_forward/03_typeaudit.jl
include(joinpath(@__DIR__, "common.jl"))
using InteractiveUtils, Profile, Test
const ET = ACEpotentials.Models.EquivariantTensors
const P4ML = ACEpotentials.Models.P4ML
using ACEpotentials.Models: radii_ed!, evaluate_ed_batched!, evaluate_ed_batched,
      evaluate_ed, _z2i, get_neighbours, PairList

frames = load_frames(1)
sys = frames[1]
models = make_models((6,))
m = models[1].second      # cat_D6
model, ps, st = m.model, m.ps, m.st
nlist = PairList(sys, M.cutoff_radius(m))
Js, Rs, Zs, z0 = get_neighbours(sys, m, nlist, 1)
rs, ∇rs = radii_ed!(zeros(length(Rs)), zeros(SVector{3,Float64}, length(Rs)), Rs)
Rnl, dRnl = evaluate_ed_batched(model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
Ylm, dYlm = P4ML.evaluate_ed(model.ybasis, Rs)
A = zeros(length(model.tensor.abasis)); ET.evaluate!(A, model.tensor.abasis, (Rnl, Ylm))
∂B = ps.WB[:, 1]

function rt(label, f, args...)
   T = Base.return_types(f, typeof.(args))
   ok = length(T) == 1 && Base.isconcretetype(T[1]) 
   @printf("  %-50s -> %s %s\n", label, ok ? "OK  " : "BAD ", first(string(T[1]), 120))
end

println("=== return_types (inferred return type of the call) ===")
rt("get_neighbours",  get_neighbours, sys, m, nlist, 1)
rt("radii_ed!",       radii_ed!, rs, ∇rs, Rs)
rt("evaluate_ed_batched! (spline Rnl)", evaluate_ed_batched!, Rnl, dRnl, model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
rt("M.evaluate (spline, one edge)", M.evaluate, model.rbasis, rs[1], z0, Zs[1], ps.rbasis, st.rbasis)
rt("P4ML.evaluate_ed (Ylm)", P4ML.evaluate_ed, model.ybasis, Rs)
rt("ET.evaluate! (A)", ET.evaluate!, A, model.tensor.abasis, (Rnl, Ylm))
rt("ET.ka_evaluate (A)", ET.ka_evaluate, model.tensor.abasis, (Rnl, Ylm))
rt("ET.ka_evaluate (AA)", ET.ka_evaluate, model.tensor.aabasis, A)
rt("ET.evaluate (tensor, as used)", ET.evaluate, model.tensor, Rnl, Ylm, NamedTuple(), NamedTuple())
rt("ET.pullback (tensor, as used)", ET.pullback, [(@view ps.WB[:, 1])], model.tensor, Rnl, Ylm, A)
rt("evaluate_ed (site E + grad)", evaluate_ed, model, Rs, Zs, z0, ps, st)
rt("M.evaluate (site E)", M.evaluate, model, Rs, Zs, z0, ps, st)
rt("evaluate_ed_batched (pair)", evaluate_ed_batched, model.pairbasis, rs, z0, Zs, ps.pairbasis, st.pairbasis)
rt("eval_grad_site", M.eval_grad_site, m, Rs, Zs, z0)
rt("energy_forces_virial(sys, m)", AtomsCalculators.energy_forces_virial, sys, m)

println("\n=== @code_warntype evaluate_ed: lines mentioning Any / Union ===")
io = IOBuffer()
code_warntype(io, evaluate_ed, typeof.((model, Rs, Zs, z0, ps, st)))
txt = String(take!(io))
for l in split(txt, '\n')
   (occursin("::Any", l) || occursin("Union{", l) || occursin("Body::", l)) && println("  ", first(strip(l), 160))
end

println("\n=== @code_warntype ET.pullback: Body ===")
io = IOBuffer()
code_warntype(io, ET.pullback, typeof.(([(@view ps.WB[:, 1])], model.tensor, Rnl, Ylm, A)))
txt = String(take!(io))
for l in split(txt, '\n')
   (occursin("Body::", l) || occursin("::Any", l)) && println("  ", first(strip(l), 160))
end

println("\n=== @code_warntype energy_forces_virial (SitePotential driver): Body ===")
io = IOBuffer()
code_warntype(io, AtomsCalculators.energy_forces_virial, typeof.((sys, m)))
txt = String(take!(io))
for l in split(txt, '\n')
   (occursin("Body::", l) || occursin("::Any", l)) && println("  ", first(strip(l), 160))
end

println("\n=== ACEPotential field types (mutable struct with untyped ps/st) ===")
println("  typeof(m.ps) = ", first(string(typeof(m.ps)), 200))
println("  fieldtypes(ACEPotential) = ", fieldtypes(typeof(m)))

# ---------------------------------------------------------------------------
# allocation scaling: allocs per site vs number of neighbours
# ---------------------------------------------------------------------------
println("\n=== allocations of evaluate_ed vs nneigh (cat_D6) ===")
for nn in (10, 20, 40, 80)
   Rs_ = Rs[1:nn]; Zs_ = Zs[1:nn]
   evaluate_ed(model, Rs_, Zs_, z0, ps, st)
   st_ = @timed evaluate_ed(model, Rs_, Zs_, z0, ps, st)
   @printf("  nneigh=%3d  %7d bytes  %5d allocs\n", nn, st_.bytes, Base.gc_alloc_count(st_.gcstats))
end
println("=== allocations of evaluate_ed vs nneigh (emb16_D6) ===")
m2 = models[2].second
for nn in (10, 20, 40, 80)
   Rs_ = Rs[1:nn]; Zs_ = Zs[1:nn]
   evaluate_ed(m2.model, Rs_, Zs_, z0, m2.ps, m2.st)
   st_ = @timed evaluate_ed(m2.model, Rs_, Zs_, z0, m2.ps, m2.st)
   @printf("  nneigh=%3d  %7d bytes  %5d allocs\n", nn, st_.bytes, Base.gc_alloc_count(st_.gcstats))
end

println("\n=== Profile.Allocs: top allocation sites in energy_forces_virial (cat_D6, 32 atoms) ===")
efv(sys, m)
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate = 0.1 for _ = 1:5; efv(sys, m); end
res = Profile.Allocs.fetch()
# aggregate by (type, first non-Base frame)
agg = Dict{String, Tuple{Int, Int}}()
for a in res.allocs
   fr = nothing
   for f in a.stacktrace
      s = string(f.file)
      if occursin("ACEpotentials", s) || occursin("EquivariantTensors", s) || occursin("AtomsCalculatorsUtilities", s) || occursin("Polynomials4ML", s) || occursin("Interpolations", s)
         fr = "$(basename(s)):$(f.line) $(f.func)"; break
      end
   end
   fr === nothing && (fr = "<other>")
   key = "$(first(string(a.type), 40)) @ $fr"
   c, b = get(agg, key, (0, 0))
   agg[key] = (c + 1, b + a.size)
end
for (k, (c, b)) in sort(collect(agg), by = x -> -x[2][2])[1:25]
   @printf("  %6d allocs %9.2f MB  %s\n", c, b / 1e6, first(k, 130))
end
