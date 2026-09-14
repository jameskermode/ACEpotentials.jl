# Phase 0 spike: Julia baseline timings for the ETACE site_basis (descriptor)
# and site_grads, plus an edge-list dump for the JAX side. SPIKE CODE.
using ACEpotentials, StaticArrays, Lux, Random, LuxCore, AtomsBuilder, Unitful,
      JSON, BenchmarkTools, Printf
M = ACEpotentials.Models; ETM = ACEpotentials.ETModels
import EquivariantTensors as ET

rcut = 5.5
rin0cuts = M._default_rin0cuts((:Si,))
rin0cuts = (x -> (rin=x.rin, r0=x.r0, rcut=rcut)).(rin0cuts)
model = M.ace_model(; elements=(:Si,), order=3, Ytype=:solid, level=M.TotalDegree(),
        max_level=10, maxl=6, pair_maxn=10, rin0cuts=rin0cuts,
        init_WB=:glorot_normal, init_Wpair=:glorot_normal)
ps, st = Lux.setup(MersenneTwister(1234), model)
et = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et)
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]

out = Dict{String,Any}()
@printf("%-8s %-8s %12s %12s\n", "atoms", "edges", "site_basis/ms", "jacobian/ms")
for nc in (2, 4)                      # 64, 512 atoms
   Random.seed!(20260909)
   sys = AtomsBuilder.bulk(:Si, cubic=true) * nc
   rattle!(sys, 0.1u"Å")
   G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
   n = length(sys)

   ETM.site_basis(et, G, et_ps, et_st)                     # warmup
   t_b = minimum(@benchmark ETM.site_basis($et, $G, $et_ps, $et_st) samples=20).time / 1e6

   ETM.site_basis_jacobian(et, G, et_ps, et_st)            # warmup
   t_j = minimum(@benchmark ETM.site_basis_jacobian($et, $G, $et_ps, $et_st) samples=10).time / 1e6

   @printf("%-8d %-8d %12.3f %12.3f\n", n, ET.nedges(G), t_b, t_j)
   out["n$(n)"] = Dict("n_atoms"=>n, "n_edges"=>ET.nedges(G),
                       "edge_i"=>G.ii .- 1, "edge_j"=>G.jj .- 1,
                       "edge_rij"=>[Vector(e.𝐫) for e in G.edge_data],
                       "julia_site_basis_ms"=>t_b, "julia_jacobian_ms"=>t_j)
end
open(joinpath(@__DIR__, "bench_data.json"), "w") do io; JSON.print(io, out); end
println("wrote bench_data.json  (threads=$(Threads.nthreads()))")
