include(joinpath(@__DIR__, "etace_traceable_setup.jl"))
function tryit(f, name)
    print(rpad(name, 40))
    try
        r = f(); println("TRACES OK   ", r)
    catch err
        println("FAIL: ", first(replace(first(split(sprint(showerror,err),"\n")), r"\s+"=>" "), 90))
    end
end
tryit("yembed (solid harmonics)") do
    ref = sum(et_model.yembed(G, et_ps.yembed, et_st.yembed)[1])
    f(p) = sum(et_model.yembed(G, p, et_st.yembed)[1])
    rp = Reactant.to_rarray(et_ps.yembed)
    v = Float64((@compile f(rp))(rp))
    "max|c-e| = $(abs(v-ref))"
end
tryit("rembed WITHOUT SelectLinL") do
    inner = et_model.rembed.layer            # EmbedDP(trans, Penv, linl)
    sub = ET.EmbedDP(inner.trans, inner.basis)   # drop the SelectLinL post-layer
    emb = ET.EdgeEmbed(sub)
    psS, stS = LuxCore.setup(MersenneTwister(0), emb)
    ref = sum(emb(G, psS, stS)[1])
    f(p) = sum(emb(G, p, stS)[1])
    rp = Reactant.to_rarray(psS)
    v = Float64((@compile f(rp))(rp)); "max|c-e| = $(abs(v-ref))"
end
