# Does the STANDARD ETACE path trace under Reactant on a CUDA host?
# One case per process, so an OOM-kill is attributable and cannot swallow the rest.
#   julia --project=. etace_one.jl <case> <cpu|gpu>
# cases: yembed rembed_nosel rembed abasis aabasis site_basis site_basis_R sitee misc
const CASE = ARGS[1]; const BACKEND = length(ARGS) >= 2 ? ARGS[2] : "cpu"

using ACEpotentials, StaticArrays, Lux, LuxCore, AtomsBuilder, AtomsBase,
      Unitful, Random, LinearAlgebra, Printf
import EquivariantTensors as ET
import DecoratedParticles as DP
using CUDA
using Reactant
Reactant.set_default_backend(BACKEND)
M = ACEpotentials.Models; ETM = ACEpotentials.ETModels
say(args...) = (println(args...); flush(stdout))
say("## case=", CASE, " backend=", BACKEND, " Julia ", VERSION,
    " Reactant ", pkgversion(Reactant), " ACEpotentials ", pkgversion(ACEpotentials),
    " ET ", pkgversion(ET), " CUDA ", pkgversion(CUDA), " CUDA.functional=", CUDA.functional())
say("## ka_with_reactant methods: ",
    isdefined(Reactant, :ka_with_reactant) ? length(methods(getfield(Reactant, :ka_with_reactant))) : "NOT DEFINED")

rng = MersenneTwister(1234); Random.seed!(1234)
elements = (:Si,); rcut = 5.5
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin=x.rin, r0=x.r0, rcut=rcut)).(rin0cuts)
model = M.ace_model(; elements=elements, order=3, Ytype=:solid,
                    level=M.TotalDegree(), max_level=10, maxl=6, pair_maxn=10,
                    rin0cuts=rin0cuts, init_WB=:glorot_normal, init_Wpair=:glorot_normal)
ps, st = Lux.setup(rng, model)
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(rng, et_model)
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]

sys = AtomsBuilder.bulk(:Si, cubic=true) * 2
rattle!(sys, 0.1u"Å")
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
say("## system: $(length(sys)) atoms, $(ET.nedges(G)) edges, maxneigs=$(G.maxneigs)")

function run(case)
    if case == "yembed"
        ref = sum(et_model.yembed(G, et_ps.yembed, et_st.yembed)[1])
        fy(p) = sum(et_model.yembed(G, p, et_st.yembed)[1])
        rp = Reactant.to_rarray(et_ps.yembed)
        v = Float64((@compile fy(rp))(rp)); return @sprintf("|c-e| = %.3e  (eager %.6f)", abs(v-ref), ref)
    elseif case == "rembed_nosel"
        inner = et_model.rembed.layer
        sub = ET.EmbedDP(inner.trans, inner.basis)
        emb = ET.EdgeEmbed(sub)
        psS, stS = LuxCore.setup(MersenneTwister(0), emb)
        ref = sum(emb(G, psS, stS)[1])
        fr(p) = sum(emb(G, p, stS)[1])
        rp = Reactant.to_rarray(psS)
        v = Float64((@compile fr(rp))(rp)); return @sprintf("|c-e| = %.3e  (eager %.6f)", abs(v-ref), ref)
    elseif case == "rembed"
        ref = sum(et_model.rembed(G, et_ps.rembed, et_st.rembed)[1])
        fR(p) = sum(et_model.rembed(G, p, et_st.rembed)[1])
        rp = Reactant.to_rarray(et_ps.rembed)
        v = Float64((@compile fR(rp))(rp)); return @sprintf("|c-e| = %.3e  (eager %.6f)", abs(v-ref), ref)
    elseif case == "abasis"
        Rnl, _ = et_model.rembed(G, et_ps.rembed, et_st.rembed)
        Ylm, _ = et_model.yembed(G, et_ps.yembed, et_st.yembed)
        fa(r, y) = ET.ka_evaluate(et_model.basis.abasis, (r, y))
        rr, ry = Reactant.to_rarray(Array(Rnl)), Reactant.to_rarray(Array(Ylm))
        A = Array((@compile fa(rr, ry))(rr, ry))
        Aref = ET.ka_evaluate(et_model.basis.abasis, (Array(Rnl), Array(Ylm)))
        return @sprintf("max|c-e| = %.3e  (size %s)", maximum(abs.(A .- Aref)), string(size(A)))
    elseif case == "aabasis"
        Rnl, _ = et_model.rembed(G, et_ps.rembed, et_st.rembed)
        Ylm, _ = et_model.yembed(G, et_ps.yembed, et_st.yembed)
        A0 = ET.ka_evaluate(et_model.basis.abasis, (Array(Rnl), Array(Ylm)))
        faa(a) = ET.ka_evaluate(et_model.basis.aabasis, a)
        ra = Reactant.to_rarray(A0)
        AA = Array((@compile faa(ra))(ra))
        AAref = ET.ka_evaluate(et_model.basis.aabasis, A0)
        return @sprintf("max|c-e| = %.3e  (size %s)", maximum(abs.(AA .- AAref)), string(size(AA)))
    elseif case == "site_basis"
        B_ref = ETM.site_basis(et_model, G, et_ps, et_st)
        fb(p) = sum(ETM.site_basis(et_model, G, p, et_st))
        rp = Reactant.to_rarray(et_ps)
        v = Float64((@compile fb(rp))(rp))
        return @sprintf("sum = %.8f  eager %.8f  |c-e| = %.3e", v, sum(B_ref), abs(v-sum(B_ref)))
    elseif case == "site_basis_R"
        B_ref = ETM.site_basis(et_model, G, et_ps, et_st)
        rebuild(Rmat) = ET.ETGraph(G.ii, G.jj, G.first, G.node_data,
            [DP.PState(𝐫=SVector{3}(Rmat[:, i]), z0=x.z0, z1=x.z1, 𝐒=x.𝐒)
             for (i, x) in enumerate(G.edge_data)], G.graph_data, G.maxneigs)
        Rmat0 = Matrix(reinterpret(reshape, Float64, [x.𝐫 for x in G.edge_data]))
        fbr(R) = sum(ETM.site_basis(et_model, rebuild(R), et_ps, et_st))
        rR = Reactant.to_rarray(Rmat0)
        v = Float64((@compile fbr(rR))(rR))
        return @sprintf("sum = %.8f  eager %.8f  |c-e| = %.3e", v, sum(B_ref), abs(v-sum(B_ref)))
    elseif case == "sitee"
        ref = sum(et_model(G, et_ps, et_st)[1])
        fe(p) = sum(et_model(G, p, et_st)[1])
        rp = Reactant.to_rarray(et_ps)
        v = Float64((@compile fe(rp))(rp)); return @sprintf("|c-e| = %.3e  (eager %.6f)", abs(v-ref), ref)
    elseif case == "misc"
        h(u) = hcat(u[:,3] .* u[:,3], u[:,2] .* u[:,3])
        u = [1.0 2.0 3.0; 4.0 5.0 6.0]; ru = Reactant.to_rarray(u)
        c = Array((@compile h(ru))(ru)); e = h(u); d = maximum(abs.(c .- e))
        return @sprintf("max|c-e| = %.3e %s", d, d < 1e-12 ? "(#3267 NOT present)" : "(#3267 PRESENT $(c))")
    end
    error("unknown case $case")
end

try
    say("RESULT ", CASE, " ", BACKEND, "  TRACES  ", run(CASE))
catch err
    s = replace(first(split(sprint(showerror, err), "\n")), r"\s+" => " ")
    say("RESULT ", CASE, " ", BACKEND, "  FAIL  ", first(s, 220))
    bt = stacktrace(catch_backtrace())
    keep = filter(fr -> occursin("ACEpotentials", string(fr.file)) ||
                        occursin("EquivariantTensors", string(fr.file)) ||
                        occursin("Polynomials4ML", string(fr.file)) ||
                        occursin("DecoratedParticles", string(fr.file)), bt)
    for fr in first(keep, 6); say("   AT ", fr.func, " @ ", basename(string(fr.file)), ":", fr.line); end
end
