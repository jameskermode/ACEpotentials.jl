#=
Debug Parameter Extraction
===========================
Compare extracted parameters with original model to find discrepancies.
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
include(joinpath(@__DIR__, "..", "src", "reactant_ace_kernel.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_stacked.jl"))

using Lux

println("="^60)
println("DEBUG PARAMETER EXTRACTION")
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

## Convert to ETACE
full_stacked_calc = ETM.convert2et_full(ace_model, ps, st; rng=rng)

println("\n1. StackedCalculator structure:")
for (i, c) in enumerate(full_stacked_calc.calcs)
    println("   [$i] $(typeof(c))")
end

## Find pair and ACE calculators
pair_calc = nothing
ace_calc = nothing
for c in full_stacked_calc.calcs
    if c isa ETM.WrappedSiteCalculator{<:ETM.ETPairModel}
        pair_calc = c
    elseif c isa ETM.WrappedSiteCalculator{<:ETM.ETACE}
        ace_calc = c
    end
end

println("\n2. ACE Calculator Structure:")
println("   model type: $(typeof(ace_calc.model))")
println("   rembed type: $(typeof(ace_calc.model.rembed))")

## Extract ACE state
ace_state = prepare_reactant_state(ace_calc; T=Float32)

println("\n3. Extracted ACE Agnesi params (7 rows: pin, pcut, a, b0, b1, rin, req):")
for (row, name) in enumerate(["pin", "pcut", "a", "b0", "b1", "rin", "req"])
    @printf("   %s: %.6f\n", name, ace_state.agnesi_params[row, 1])
end

println("\n4. Extracted ACE Chebyshev coefficients (first 5):")
@printf("   A: %s\n", ace_state.poly_A[1:min(5, length(ace_state.poly_A))])
@printf("   B: %s\n", ace_state.poly_B[1:min(5, length(ace_state.poly_B))])
@printf("   C: %s\n", ace_state.poly_C[1:min(5, length(ace_state.poly_C))])

## Check original ACE transform params in state
println("\n5. Original ACE state structure:")
println("   st.rembed type: $(typeof(ace_calc.st.rembed))")
if hasproperty(ace_calc.st.rembed, :trans)
    trans_st = ace_calc.st.rembed.trans
    println("   st.rembed.trans type: $(typeof(trans_st))")
    if hasproperty(trans_st, :params)
        println("   st.rembed.trans.params: $(trans_st.params)")
    end
end

## Test Agnesi transform with extracted params vs recomputed
println("\n6. Testing Agnesi transform at r=2.35:")
r = Float32(2.35)
pin = Int(ace_state.agnesi_params[1, 1])
pcut = Int(ace_state.agnesi_params[2, 1])
a = ace_state.agnesi_params[3, 1]
b0 = ace_state.agnesi_params[4, 1]
b1 = ace_state.agnesi_params[5, 1]
rin = ace_state.agnesi_params[6, 1]
req = ace_state.agnesi_params[7, 1]

y_extracted = compute_agnesi_transform(r, pin, pcut, a, b0, b1, rin, req)
@printf("   y (extracted params): %.6f\n", y_extracted)

## Pair model analysis
if pair_calc !== nothing
    println("\n7. Pair Calculator Structure:")
    println("   model type: $(typeof(pair_calc.model))")
    println("   rembed type: $(typeof(pair_calc.model.rembed))")

    pair_state = Main._extract_pair_state(pair_calc, 1, Float32)

    println("\n8. Extracted Pair Agnesi params (5 rows: pcut, pin, rin, req, rcut):")
    for (row, name) in enumerate(["pcut", "pin", "rin", "req", "rcut"])
        @printf("   %s: %.6f\n", name, pair_state.agnesi_params[row, 1])
    end

    println("\n9. Testing Pair Agnesi transform at r=2.35:")
    pcut_p = pair_state.agnesi_params[1, 1]
    pin_p = pair_state.agnesi_params[2, 1]
    rin_p = pair_state.agnesi_params[3, 1]
    req_p = pair_state.agnesi_params[4, 1]
    rcut_p = pair_state.agnesi_params[5, 1]

    y_pair = compute_agnesi_transform(r, pcut_p, pin_p, rin_p, req_p, rcut_p)
    @printf("   y (pair params): %.6f\n", y_pair)

    # Check original pair state
    println("\n10. Original Pair state structure:")
    if hasproperty(pair_calc.model.rembed, :layer)
        outer = pair_calc.model.rembed.layer
        println("   outer (layer) type: $(typeof(outer))")
        if hasproperty(outer, :rbasis)
            rbasis = outer.rbasis
            println("   rbasis type: $(typeof(rbasis))")
            if hasproperty(rbasis, :trans)
                println("   rbasis.trans type: $(typeof(rbasis.trans))")
            end
        end
    end

    # Check state
    println("\n11. Pair st.rembed structure:")
    pair_st = pair_calc.st.rembed
    println("   st.rembed type: $(typeof(pair_st))")
    if hasproperty(pair_st, :layer)
        layer_st = pair_st.layer
        println("   st.rembed.layer type: $(typeof(layer_st))")
        if hasproperty(layer_st, :rbasis) && hasproperty(layer_st.rbasis, :trans)
            trans_st = layer_st.rbasis.trans
            println("   st.rembed.layer.rbasis.trans type: $(typeof(trans_st))")
            if hasproperty(trans_st, :params)
                println("   params: $(trans_st.params)")
            end
        end
    end
end

println("\n" * "="^60)
println("DEBUG COMPLETE")
println("="^60)
