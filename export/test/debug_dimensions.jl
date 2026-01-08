#=
Debug Dimension Mismatch
========================
Investigate the n_rnl mismatch (19 vs 25) and spec_R indices.
=#

using Random
using Printf
using LinearAlgebra

# Load ACEpotentials
using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

# Load extraction modules
include(joinpath(@__DIR__, "..", "src", "reactant_state.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_embeddings.jl"))

using Lux

println("="^60)
println("DEBUG DIMENSION MISMATCH")
println("="^60)

## Create ACE model
elements = (:Si,)
level = M.TotalDegree()
max_level = 8
order = 2
maxl = 2
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(rin0cuts)
rng = Random.MersenneTwister(1234)

ace_model = M.ace_model(; elements = elements, order = order,
                        Ytype = :solid, level = level, max_level = max_level,
                        maxl = maxl, pair_maxn = max_level,
                        rin0cuts = rin0cuts,
                        pair_learnable = true,
                        init_WB = :glorot_normal, init_Wpair = :glorot_normal)

ps, st = Lux.setup(rng, ace_model)
full_stacked_calc = ETM.convert2et_full(ace_model, ps, st; rng=rng)

## Find ACE calculator
local ace_calc = nothing
for c in full_stacked_calc.calcs
    if c isa ETM.WrappedSiteCalculator{<:ETM.ETACE}
        global ace_calc = c
    end
end

## Extract state
ace_state = prepare_reactant_state(ace_calc; T=Float32)

println("\n1. Extracted state:")
@printf("   n_rnl (our extraction): %d\n", ace_state.n_rnl)
@printf("   n_ylm: %d\n", ace_state.nYlm)
@printf("   spec_R length: %d\n", length(ace_state.spec_R))
@printf("   spec_Y length: %d\n", length(ace_state.spec_Y))
@printf("   spec_R range: [%d, %d]\n", minimum(ace_state.spec_R), maximum(ace_state.spec_R))
@printf("   spec_Y range: [%d, %d]\n", minimum(ace_state.spec_Y), maximum(ace_state.spec_Y))

println("\n2. Original model structure:")
rembed = ace_calc.model.rembed
println("   rembed type: ", typeof(rembed))

# Try to get output dimension
try
    if hasproperty(rembed, :layer)
        inner = rembed.layer
        println("   rembed.layer type: ", typeof(inner))
        if hasproperty(inner, :post)
            println("   rembed.layer.post type: ", typeof(inner.post))
            if hasproperty(inner.post, :in_dim)
                println("   rembed.layer.post.in_dim: ", inner.post.in_dim)
            end
        end
    end
catch e
    println("   Error: ", e)
end

println("\n3. W_radial shape:")
@printf("   W_radial: %s\n", size(ace_state.W_radial))
@printf("   Expected: (n_rnl, n_polys, n_pairs) = (%d, %d, %d)\n",
        ace_state.n_rnl, ace_state.n_polys, ace_state.n_species^2)

println("\n4. Original rembed parameter check:")
println("   ps.rembed type: ", typeof(ace_calc.ps.rembed))
if hasproperty(ace_calc.ps.rembed, :post) && hasproperty(ace_calc.ps.rembed.post, :W)
    W = ace_calc.ps.rembed.post.W
    @printf("   ps.rembed.post.W shape: %s\n", size(W))
    @printf("   Actual n_rnl from W: %d\n", size(W, 1))
end

println("\n5. Original basis state:")
basis_st = ace_calc.st.basis
println("   aspec length: ", length(basis_st.aspec))
println("   aspec[1:5]: ", basis_st.aspec[1:min(5, length(basis_st.aspec))])

# Check if our spec_R/spec_Y match
spec_R = [s[1] for s in basis_st.aspec]
spec_Y = [s[2] for s in basis_st.aspec]
@printf("   Original spec_R range: [%d, %d]\n", minimum(spec_R), maximum(spec_R))
@printf("   Original spec_Y range: [%d, %d]\n", minimum(spec_Y), maximum(spec_Y))

println("\n" * "="^60)
println("DEBUG COMPLETE")
println("="^60)
