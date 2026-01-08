#=
Debug Pair Energy
=================
Compare pair energy computation in detail.
=#

using Random
using Printf
using LinearAlgebra
using StaticArrays
using Statistics: mean

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
import EquivariantTensors as ET

println("="^60)
println("DEBUG PAIR ENERGY")
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
model_state = ReactantStackedModel(full_stacked_calc; T=Float32)

## Find pair calculator
local pair_calc = nothing
for c in full_stacked_calc.calcs
    if c isa ETM.WrappedSiteCalculator{<:ETM.ETPairModel}
        global pair_calc = c
    end
end

if pair_calc === nothing
    println("No pair calculator found!")
    exit(1)
end

println("\n1. Pair model structure:")
println("   Model type: ", typeof(pair_calc.model))
println("   rcut: ", pair_calc.rcut)

## Create test system - single pair at 2.35 Å
r = 2.35f0
rij = SVector{3,Float32}(r, 0.0f0, 0.0f0)
rhat = SVector{3,Float32}(1.0f0, 0.0f0, 0.0f0)

println("\n2. Test pair at r = ", r, " Å")

## Our pair computation
pair_state = model_state.pair_state
species_Z = model_state.species_Z
zi = 1  # Si
zj = 1  # Si
pair_idx = zz_to_pair_index(zi, zj, pair_state.n_species)

println("\n3. Our pair parameters:")
@printf("   n_polys: %d\n", pair_state.n_polys)
@printf("   n_basis: %d\n", pair_state.n_basis)
@printf("   rcut: %.4f\n", pair_state.rcut)

# Extract Agnesi parameters
pin = Int(pair_state.agnesi_params[1, pair_idx])
pcut = Int(pair_state.agnesi_params[2, pair_idx])
a = pair_state.agnesi_params[3, pair_idx]
b0 = pair_state.agnesi_params[4, pair_idx]
b1 = pair_state.agnesi_params[5, pair_idx]
rin = pair_state.agnesi_params[6, pair_idx]
req = pair_state.agnesi_params[7, pair_idx]

@printf("   Agnesi: pin=%d, pcut=%d, a=%.4f, b0=%.4f, b1=%.4f, rin=%.4f, req=%.4f\n",
        pin, pcut, a, b0, b1, rin, req)

# Our distance transform
y = compute_agnesi_transform(r, pin, pcut, a, b0, b1, rin, req)
@printf("   Our y (Agnesi transform): %.6f\n", y)

# Our envelope
env = compute_envelope(y)
@printf("   Our envelope: %.6f\n", env)

# Our polynomials
P = compute_chebyshev_basis(y, pair_state.n_polys, pair_state.poly_A, pair_state.poly_B, pair_state.poly_C)
@printf("   Our P[1:5]: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", P[1], P[2], P[3], P[4], P[5])

# Enveloped basis
P_env = P .* env
@printf("   Our P_env[1:5]: [%.4f, %.4f, %.4f, %.4f, %.4f]\n", P_env[1], P_env[2], P_env[3], P_env[4], P_env[5])

# Pair features
pair_features = pair_state.W_radial[:, :, pair_idx] * P_env
@printf("   Our pair_features[1:5]: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
        pair_features[1], pair_features[2], pair_features[3], pair_features[4], pair_features[5])

# Site energy
site_energy = dot(pair_features, pair_state.W_readout[:, zi])
@printf("   Our pair site energy: %.6f\n", site_energy)

## Now compute with original pair model
println("\n4. Original pair computation:")

# Build graph for original model
using AtomsBase
using Unitful

positions = Float32[0.0 0.0 0.0; 2.35 0.0 0.0]
n_atoms = 2

cell_vec = 100.0 * u"Å" .* [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
atoms_sys = AtomsBase.FlexibleSystem(
    [AtomsBase.Atom(:Si, Float64.(positions[i, :]) * u"Å") for i in 1:n_atoms];
    cell_vectors = cell_vec,
    periodicity = (false, false, false)
)

import AtomsCalculators
E_pair_orig = AtomsCalculators.potential_energy(atoms_sys, pair_calc)
E_pair_orig_val = Float32(ustrip(u"eV", E_pair_orig))
@printf("   Original pair energy (total): %.6f eV\n", E_pair_orig_val)
@printf("   Original pair energy (per edge): %.6f eV\n", E_pair_orig_val / 2)

## Compare intermediate values
println("\n5. Comparing intermediate values:")

# Build ETGraph
G = ET.Atoms.interaction_graph(atoms_sys, pair_calc.rcut * u"Å")
@printf("   ETGraph edges: %d\n", length(G.edge_data))

# Original rembed
rembed_out, _ = pair_calc.model.rembed(G, pair_calc.ps.rembed, pair_calc.st.rembed)
println("   Original rembed shape: ", size(rembed_out))
println("   Original rembed[1,1,:5]: ", rembed_out[1, 1, 1:min(5, size(rembed_out, 3))])

# Extract original parameters for debugging
println("\n6. Original pair model state:")
ps_rembed = pair_calc.ps.rembed
st_rembed = pair_calc.st.rembed

# Check state structure
println("   st_rembed keys: ", keys(st_rembed))

# Check outer envelope
if hasproperty(st_rembed, :envelope)
    env_st = st_rembed.envelope
    println("   envelope_st keys: ", keys(env_st))
    if hasproperty(env_st, :rcut)
        @printf("   Outer envelope: rcut=%.4f, p=%d\n", env_st.rcut, env_st.p)
        # Compute outer envelope for our test distance
        r_test = 2.35f0
        outer_env = (1.0f0 - r_test / Float32(env_st.rcut))^env_st.p
        @printf("   Outer envelope at r=%.4f: %.6f\n", r_test, outer_env)
    end
end

if hasproperty(st_rembed, :rbasis)
    rbasis_st = st_rembed.rbasis
    println("   rbasis_st keys: ", keys(rbasis_st))
    if hasproperty(rbasis_st, :trans)
        trans_st = rbasis_st.trans
        println("   trans_st keys: ", keys(trans_st))
        if hasproperty(trans_st, :params)
            orig_params = trans_st.params[1]
            @printf("   Original Agnesi: pin=%d, pcut=%d, a=%.4f, b0=%.4f, b1=%.4f, rin=%.4f, req=%.4f\n",
                    orig_params.pin, orig_params.pcut, orig_params.a,
                    orig_params.b0, orig_params.b1, orig_params.rin, orig_params.req)
        end
    end
    if hasproperty(rbasis_st, :basis)
        basis_st = rbasis_st.basis
        println("   basis_st type: ", typeof(basis_st))
        if hasproperty(basis_st, :A)
            @printf("   Original poly A[1:5]: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                    basis_st.A[1], basis_st.A[2], basis_st.A[3], basis_st.A[4], basis_st.A[5])
            @printf("   Our poly A[1:5]: [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                    pair_state.poly_A[1], pair_state.poly_A[2], pair_state.poly_A[3],
                    pair_state.poly_A[4], pair_state.poly_A[5])
        end
    end
end

# Check feature ratio
println("\n6b. Feature ratio analysis:")
orig_feat = Float32.(rembed_out[1, 1, :])
our_feat = pair_features
for i in 1:min(5, length(orig_feat))
    ratio = abs(orig_feat[i]) > 1e-8 ? our_feat[i] / orig_feat[i] : 0.0
    @printf("   Feature %d: orig=%.4f, ours=%.4f, ratio=%.4f\n", i, orig_feat[i], our_feat[i], ratio)
end

# Test: what if outer envelope multiplies pair_features?
println("\n6c. Testing outer envelope effects:")
outer_env = (1.0f0 - 2.35f0 / 5.5f0)^1
println("   Outer envelope: ", outer_env)
println("   If features *= outer_env:")
for i in 1:min(3, length(orig_feat))
    adjusted = pair_features[i] * outer_env
    @printf("      Feature %d: orig=%.4f, adjusted=%.4f\n", i, orig_feat[i], adjusted)
end
println("   If features /= outer_env:")
for i in 1:min(3, length(orig_feat))
    adjusted = pair_features[i] / outer_env
    @printf("      Feature %d: orig=%.4f, adjusted=%.4f\n", i, orig_feat[i], adjusted)
end

# Check what factor makes them equal
avg_ratio = mean([our_feat[i] / orig_feat[i] for i in 1:length(orig_feat)])
println("   Average ratio: ", avg_ratio)
println("   Needed multiplier: ", 1/avg_ratio)

# Check if it's related to envelope differently
println("\n6d. Envelope analysis:")
inner_env = compute_envelope(y)
println("   Inner env (1-y²)²: ", inner_env)
# What if pair uses different inner envelope?
alt_inner_env = (1.0f0 - y)^2 * (1.0f0 + y)^2  # Same as (1-y²)²
println("   Alt inner env: ", alt_inner_env)

# Test without inner envelope
pair_features_no_env = pair_state.W_radial[:, :, pair_idx] * P  # P without envelope
println("   Features WITHOUT inner env [1:3]: ", pair_features_no_env[1:3])
println("   Ratio with orig if no inner env: ", pair_features_no_env[1] / orig_feat[1])

# Test with different envelope powers
println("\n6e. Testing different envelope formulas:")
# (1-y²)^1 instead of (1-y²)²
env_pow1 = (1.0f0 - y^2)
P_env1 = P .* env_pow1
feat_env1 = pair_state.W_radial[:, :, pair_idx] * P_env1
println("   (1-y²)^1: ", env_pow1, ", ratio: ", feat_env1[1] / orig_feat[1])

# (1-y²)^0.5
env_pow05 = sqrt(1.0f0 - y^2)
P_env05 = P .* env_pow05
feat_env05 = pair_state.W_radial[:, :, pair_idx] * P_env05
println("   (1-y²)^0.5: ", env_pow05, ", ratio: ", feat_env05[1] / orig_feat[1])

# Combined with outer envelope (1-r/rcut)^p
full_env = inner_env * outer_env  # both envelopes
P_full = P .* full_env
feat_full = pair_state.W_radial[:, :, pair_idx] * P_full
println("   inner * outer: ", full_env, ", ratio: ", feat_full[1] / orig_feat[1])

# Compare our pair energy with original
edge_rij = Float32[2.35 0.0 0.0; -2.35 0.0 0.0]
atomic_numbers = Int32[14, 14]
edge_i = Int32[1, 2]
edge_j = Int32[2, 1]

E_pair_ours = compute_pair_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                                   Int32(2), Int32(2), pair_state, species_Z)
@printf("\n7. Comparison:\n")
@printf("   Our pair energy: %.6f\n", E_pair_ours)
@printf("   Original pair energy: %.6f\n", E_pair_orig_val)
@printf("   Difference: %.6e\n", abs(E_pair_ours - E_pair_orig_val))
@printf("   Relative error: %.2f%%\n", 100 * abs(E_pair_ours - E_pair_orig_val) / abs(E_pair_orig_val))

println("\n" * "="^60)
println("DEBUG COMPLETE")
println("="^60)
