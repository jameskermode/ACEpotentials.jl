include(joinpath(@__DIR__, "etace_traceable_setup.jl"))
function show_err(f, name)
    println("\n########## ", name, " ##########")
    try f() catch err
        bt = catch_backtrace()
        println(first(split(sprint(showerror, err), "\n")))
        frames = stacktrace(bt)
        keep = filter(fr -> occursin("ACEpotentials", string(fr.file)) ||
                            occursin("EquivariantTensors", string(fr.file)) ||
                            occursin("Polynomials4ML", string(fr.file)) ||
                            occursin("DecoratedParticles", string(fr.file)), frames)
        for fr in first(keep, 8)
            println("   ", fr.func, "  @  ", basename(string(fr.file)), ":", fr.line)
        end
        isempty(keep) && for fr in first(frames, 8); println("   ", fr); end
    end
end
show_err("A. site_basis(ps traced)") do
    f(p) = sum(ETM.site_basis(et_model, G, p, et_st))
    rp = Reactant.to_rarray(et_ps); (@compile f(rp))(rp)
end
show_err("rembed only") do
    f(p) = sum(et_model.rembed(G, p, et_st.rembed)[1])
    rp = Reactant.to_rarray(et_ps.rembed); (@compile f(rp))(rp)
end
show_err("yembed only") do
    f(p) = sum(et_model.yembed(G, p, et_st.yembed)[1])
    rp = Reactant.to_rarray(et_ps.yembed); (@compile f(rp))(rp)
end
