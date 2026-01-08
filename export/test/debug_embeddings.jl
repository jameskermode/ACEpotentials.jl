#=
Debug Embeddings Comparison
===========================
Compare our radial/angular embeddings with the original ETACE.
=#

using Random
using Printf
using LinearAlgebra
using StaticArrays

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
println("DEBUG EMBEDDINGS COMPARISON")
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

## Find ACE calculator
local ace_calc = nothing
for c in full_stacked_calc.calcs
    if c isa ETM.WrappedSiteCalculator{<:ETM.ETACE}
        global ace_calc = c
    end
end

## Extract state
ace_state = prepare_reactant_state(ace_calc; T=Float32)

## Create test system
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

println("\n1. Building ETGraph from system...")

# Build graph using interaction_graph
G = ET.Atoms.interaction_graph(atoms_sys, ace_calc.rcut * u"Å")

println("   Nodes: ", length(G.node_data))
println("   Edges: ", length(G.edge_data))
println("   Edge data type: ", typeof(G.edge_data))

# Print edge info
for (i, e) in enumerate(G.edge_data)
    @printf("   Edge %d: r_norm=%.4f\n", i, norm(e.𝐫))
end

println("\n2. Computing original radial embedding...")

# Get original rembed output
rembed_out, rembed_st_out = ace_calc.model.rembed(G, ace_calc.ps.rembed, ace_calc.st.rembed)
println("   rembed output type: ", typeof(rembed_out))
println("   rembed output size: ", size(rembed_out))
println("   rembed output norm: ", norm(rembed_out))
println("   rembed range: [", minimum(rembed_out), ", ", maximum(rembed_out), "]")

println("\n3. Computing original angular embedding...")

# Get original yembed output
yembed_out, yembed_st_out = ace_calc.model.yembed(G, ace_calc.ps.yembed, ace_calc.st.yembed)
println("   yembed output type: ", typeof(yembed_out))
println("   yembed output size: ", size(yembed_out))
println("   yembed output norm: ", norm(yembed_out))
println("   yembed range: [", minimum(yembed_out), ", ", maximum(yembed_out), "]")

println("\n4. Computing our embeddings...")

# Parameters
r = Float32(2.35)
rhat = SVector{3,Float32}(1.0, 0.0, 0.0)
zi = 1
zj = 1

# Our radial embedding
our_Rnl = compute_radial_embedding(r, zi, zj, ace_state)
println("   Our Rnl size: ", length(our_Rnl))
println("   Our Rnl norm: ", norm(our_Rnl))
println("   Our Rnl range: [", minimum(our_Rnl), ", ", maximum(our_Rnl), "]")

# Our angular embedding (solid harmonics, same as ACE uses)
our_Ylm = compute_solid_harmonics_reactant(r, rhat, ace_state.maxl)
println("   Our Ylm (solid) size: ", length(our_Ylm))
println("   Our Ylm (solid) norm: ", norm(our_Ylm))
println("   Our Ylm (solid) range: [", minimum(our_Ylm), ", ", maximum(our_Ylm), "]")

println("\n5. Comparing embeddings (first edge)...")

# Extract first edge embedding from original
if ndims(rembed_out) == 3
    orig_Rnl = Float32.(rembed_out[1, 1, :])  # First edge
elseif ndims(rembed_out) == 2
    orig_Rnl = Float32.(rembed_out[1, :])  # First edge
else
    orig_Rnl = Float32.(vec(rembed_out))
end

if ndims(yembed_out) == 3
    orig_Ylm = Float32.(yembed_out[1, 1, :])
elseif ndims(yembed_out) == 2
    orig_Ylm = Float32.(yembed_out[1, :])
else
    orig_Ylm = Float32.(vec(yembed_out))
end

println("   Original Rnl[1:5]: ", orig_Rnl[1:min(5, length(orig_Rnl))])
println("   Our Rnl[1:5]: ", our_Rnl[1:min(5, length(our_Rnl))])

if length(orig_Rnl) == length(our_Rnl)
    rnl_diff = norm(orig_Rnl - our_Rnl)
    @printf("   Rnl difference (norm): %.6e\n", rnl_diff)
else
    println("   Rnl lengths differ: ", length(orig_Rnl), " vs ", length(our_Rnl))
end

println("   Original Ylm[1:5]: ", orig_Ylm[1:min(5, length(orig_Ylm))])
println("   Our Ylm[1:5]: ", our_Ylm[1:min(5, length(our_Ylm))])

if length(orig_Ylm) == length(our_Ylm)
    ylm_diff = norm(orig_Ylm - our_Ylm)
    @printf("   Ylm difference (norm): %.6e\n", ylm_diff)
else
    println("   Ylm lengths differ: ", length(orig_Ylm), " vs ", length(our_Ylm))
end

println("\n6. Computing original basis output...")

# Basis computation (pooled sparse product + symm prod + coupling)
basis_out_tuple, basis_st_out = ace_calc.model.basis((rembed_out, yembed_out),
                                                ace_calc.ps.basis, ace_calc.st.basis)
println("   basis output type: ", typeof(basis_out_tuple))

# basis_out is a tuple of matrices (one per L value), unwrap for L=0
basis_out = basis_out_tuple[1]  # L=0 component
println("   L=0 basis size: ", size(basis_out))
println("   L=0 basis norm: ", norm(basis_out))
println("   L=0 basis[1:5]: ", basis_out[1, 1:min(5, size(basis_out, 2))])

println("\n7. Computing our basis (single atom)...")

# Build 3D tensors for our computation
Rnl_3 = zeros(Float32, 1, 1, ace_state.n_rnl)
Ylm_3 = zeros(Float32, 1, 1, ace_state.nYlm)
Rnl_3[1, 1, :] .= our_Rnl
Ylm_3[1, 1, :] .= our_Ylm

# Our ACE evaluation
BB, A, AA = ace_evaluate_reactant(Rnl_3, Ylm_3, ace_state.spec_R, ace_state.spec_Y,
                                  ace_state.specs_mats, ace_state.A2Bmap)
println("   Our A: ", size(A), ", norm=", norm(A))
println("   Our AA: ", size(AA), ", norm=", norm(AA))
println("   Our BB: ", size(BB), ", norm=", norm(BB))

println("\n8. Readout comparison...")

# Our readout
our_energy = dot(vec(BB), ace_state.W_readout[:, 1])
@printf("   Our site energy: %.6f\n", our_energy)

# Original readout
readout_out, _ = ace_calc.model.readout((basis_out, G.node_data),
                                         ace_calc.ps.readout, ace_calc.st.readout)
println("   Original readout output: ", readout_out)
println("   Original readout sum: ", sum(readout_out))

println("\n" * "="^60)
println("DEBUG COMPLETE")
println("="^60)
