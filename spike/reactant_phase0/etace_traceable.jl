# THE question the earlier spike should have asked: is the STANDARD ETACE code
# path traceable by Reactant? The ETACE rewrite was designed with that in mind;
# lammps-jax's ace_export.jl hand-writes arrays instead, but that is its choice,
# not a demonstration that hand-writing is required.
#
# Escalating attempts, each reporting exactly where it fails. SPIKE CODE.

using ACEpotentials, StaticArrays, Lux, LuxCore, AtomsBuilder, AtomsBase,
      Unitful, Random, LinearAlgebra, Printf
import EquivariantTensors as ET
import DecoratedParticles as DP
using Reactant
Reactant.set_default_backend("cpu")
M = ACEpotentials.Models; ETM = ACEpotentials.ETModels
println("Reactant ", pkgversion(Reactant), "  ACEpotentials ", pkgversion(ACEpotentials))

rng = MersenneTwister(1234); Random.seed!(1234)
elements = (:Si,); rcut = 5.5
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin=x.rin, r0=x.r0, rcut=rcut)).(rin0cuts)
model = M.ace_model(; elements=elements, order=3, Ytype=:solid,
                    level=M.TotalDegree(), max_level=10, maxl=6, pair_maxn=10,
                    rin0cuts=rin0cuts, init_WB=:glorot_normal,
                    init_Wpair=:glorot_normal)
ps, st = Lux.setup(rng, model)
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(rng, et_model)
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]

sys = AtomsBuilder.bulk(:Si, cubic=true) * 2
rattle!(sys, 0.1u"Å")
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
println("system: $(length(sys)) atoms, $(ET.nedges(G)) edges, maxneigs=$(G.maxneigs)")

B_ref = ETM.site_basis(et_model, G, et_ps, et_st)
println("eager site_basis: size $(size(B_ref))  max|B| = $(maximum(abs, B_ref))\n")

function attempt(f, name)
    print(rpad(name, 46))
    try
        r = f()
        println("OK   ", r)
    catch err
        s = sprint(showerror, err)
        s = replace(first(split(s, "\n")), r"\s+" => " ")
        println("FAIL: ", first(s, 150))
    end
end

# A: compile as a function of the parameters, graph captured as a constant
attempt("A. @compile f(ps), graph captured") do
    f(p) = sum(ETM.site_basis(et_model, G, p, et_st))
    rp = Reactant.to_rarray(et_ps)
    c = @compile f(rp)
    v = Float64(c(rp)); @sprintf("sum(B) = %.6f (eager %.6f)", v, sum(B_ref))
end

# B: the realistic export shape -- trace through the edge geometry
_rebuild(Rmat) = ET.ETGraph(G.ii, G.jj, G.first, G.node_data,
    [DP.PState(𝐫=SVector{3}(Rmat[:, i]), z0=x.z0, z1=x.z1, 𝐒=x.𝐒)
     for (i, x) in enumerate(G.edge_data)], G.graph_data, G.maxneigs)
Rmat0 = Matrix(reinterpret(reshape, Float64, [x.𝐫 for x in G.edge_data]))
attempt("B. @compile f(edge positions)") do
    f(R) = sum(ETM.site_basis(et_model, _rebuild(R), et_ps, et_st))
    rR = Reactant.to_rarray(Rmat0)
    c = @compile f(rR)
    v = Float64(c(rR)); @sprintf("sum(B) = %.6f (eager %.6f)", v, sum(B_ref))
end

# C: the Lux model call itself (site energies)
attempt("C. @compile et_model(G, ps, st)") do
    f(p) = sum(et_model(G, p, et_st)[1])
    rp = Reactant.to_rarray(et_ps)
    c = @compile f(rp); @sprintf("sum(E) = %.6f", Float64(c(rp)))
end

# D: the ET kernels in isolation -- do THEY trace?
Rnl, _ = et_model.rembed(G, et_ps.rembed, et_st.rembed)
Ylm, _ = et_model.yembed(G, et_ps.yembed, et_st.yembed)
attempt("D. @compile A-basis (PooledSparseProduct)") do
    f(r, y) = ET.ka_evaluate(et_model.basis.abasis, (r, y))
    rr, ry = Reactant.to_rarray(Array(Rnl)), Reactant.to_rarray(Array(Ylm))
    c = @compile f(rr, ry)
    A = Array(c(rr, ry)); Aref = ET.ka_evaluate(et_model.basis.abasis, (Array(Rnl), Array(Ylm)))
    @sprintf("max|c-e| = %.3e", maximum(abs.(A .- Aref)))
end
A0 = ET.ka_evaluate(et_model.basis.abasis, (Array(Rnl), Array(Ylm)))
attempt("E. @compile AA-basis (SparseSymmProd)") do
    f(a) = ET.ka_evaluate(et_model.basis.aabasis, a)
    ra = Reactant.to_rarray(A0)
    c = @compile f(ra)
    AA = Array(c(ra)); AAref = ET.ka_evaluate(et_model.basis.aabasis, A0)
    @sprintf("max|c-e| = %.3e", maximum(abs.(AA .- AAref)))
end
