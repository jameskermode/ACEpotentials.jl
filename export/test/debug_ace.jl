#=
Debug ACE Energy Computation
============================
Compare ACE computation intermediate values with original.
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
println("DEBUG ACE ENERGY COMPUTATION")
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

println("\n1. State dimensions:")
@printf("   n_rnl=%d, n_ylm=%d, n_basis=%d\n", ace_state.n_rnl, ace_state.nYlm, ace_state.n_basis)
@printf("   spec_R length=%d, spec_Y length=%d\n", length(ace_state.spec_R), length(ace_state.spec_Y))
@printf("   specs_mats: %d orders\n", length(ace_state.specs_mats))
for (i, sm) in enumerate(ace_state.specs_mats)
    @printf("     order %d: %s\n", i, size(sm))
end
@printf("   A2Bmap: %s\n", size(ace_state.A2Bmap))
@printf("   W_readout: %s\n", size(ace_state.W_readout))

## Create test structure (Si dimer)
positions = Float32[0.0 0.0 0.0; 2.35 0.0 0.0]
atomic_numbers = Int32[14, 14]
n_atoms = Int32(2)
edge_i = Int32[1, 2]
edge_j = Int32[2, 1]
n_edges = Int32(2)
edge_rij = Float32[
    2.35 0.0 0.0;
    -2.35 0.0 0.0
]

println("\n2. Test structure:")
@printf("   Atoms: %d, Edges: %d\n", n_atoms, n_edges)
@printf("   r = %.2f Å\n", norm(edge_rij[1, :]))

## Compute embeddings for first edge
r = Float32(2.35)
rhat = SVector{3,Float32}(1.0, 0.0, 0.0)
zi = 1  # Silicon
zj = 1

println("\n3. Single edge embeddings:")

# Radial embedding
Rnl = compute_radial_embedding(r, zi, zj, ace_state)
@printf("   Rnl: length=%d, norm=%.6f, range=[%.6f, %.6f]\n",
        length(Rnl), norm(Rnl), minimum(Rnl), maximum(Rnl))

# Angular embedding
Ylm = compute_ylm_reactant(rhat, ace_state.maxl)
@printf("   Ylm: length=%d, norm=%.6f, range=[%.6f, %.6f]\n",
        length(Ylm), norm(Ylm), minimum(Ylm), maximum(Ylm))

println("\n4. Full ACE computation:")

## Build 3D tensors (for single atom with one neighbor)
Rnl_3 = zeros(Float32, 1, 1, ace_state.n_rnl)
Ylm_3 = zeros(Float32, 1, 1, ace_state.nYlm)
Rnl_3[1, 1, :] .= Rnl
Ylm_3[1, 1, :] .= Ylm

# Pooled sparse product
A = pooled_sparse_product_reactant(Rnl_3, Ylm_3, ace_state.spec_R, ace_state.spec_Y)
@printf("   A: size=%s, norm=%.6f\n", size(A), norm(A))

# Symmetric product
AA = sparse_symm_prod_reactant(A, ace_state.specs_mats)
@printf("   AA: size=%s, norm=%.6f\n", size(AA), norm(AA))

# Coupling
BB = AA * transpose(ace_state.A2Bmap)
@printf("   BB: size=%s, norm=%.6f\n", size(BB), norm(BB))

# Site energy (no readout for now, just dot with weights)
site_E = dot(vec(BB), ace_state.W_readout[:, 1])
@printf("   Site energy (single atom, one neighbor): %.6f\n", site_E)

## Now compute full ACE energy with all edges
println("\n5. Full energy computation with all edges:")

E_ace = Main.compute_ace_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                                 n_atoms, n_edges, ace_state)
@printf("   ACE energy (our implementation): %.6f\n", E_ace)

## Compare with original
println("\n6. Original ACE computation:")

# Create ETGraph for original computation
using AtomsBase
using Unitful

cell_vec = 100.0 * u"Å" .* [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
atoms_sys = AtomsBase.FlexibleSystem(
    [AtomsBase.Atom(:Si, Float64.(positions[i, :]) * u"Å") for i in 1:n_atoms];
    cell_vectors = cell_vec,
    periodicity = (false, false, false)
)

# Get original ACE contribution
import AtomsCalculators
E_orig_full = AtomsCalculators.potential_energy(atoms_sys, full_stacked_calc)
E_orig_full_val = Float32(ustrip(u"eV", E_orig_full))
@printf("   Original total energy: %.6f eV\n", E_orig_full_val)

# Try to get ACE-only contribution
E_orig_ace_only = AtomsCalculators.potential_energy(atoms_sys, ace_calc)
E_orig_ace_val = Float32(ustrip(u"eV", E_orig_ace_only))
@printf("   Original ACE-only energy: %.6f eV\n", E_orig_ace_val)

println("\n7. Analysis:")
@printf("   Ratio (original/ours): %.2f\n", E_orig_ace_val / E_ace)

## Try to trace original computation
println("\n8. Tracing original ETACE computation:")

# Build ETGraph from system
G = ET.etgraph(ace_calc.model, atoms_sys)
println("   ETGraph: n_nodes=", length(G.node_data), ", n_edges=", length(G.edge_data))

# Get basis output from original
basis_out, _ = ace_calc.model.basis(G, ace_calc.ps.basis, ace_calc.st.basis)
println("   Original basis output:")
if basis_out isa Tuple
    for (i, b) in enumerate(basis_out)
        @printf("     [%d] shape=%s, norm=%.6f\n", i, size(b), norm(b))
    end
else
    @printf("     shape=%s, norm=%.6f\n", size(basis_out), norm(basis_out))
end

println("\n" * "="^60)
println("DEBUG COMPLETE")
println("="^60)
