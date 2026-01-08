#=
Debug Tetrahedron Energy
========================
Trace through the tetrahedron computation step by step.
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
println("DEBUG TETRAHEDRON")
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

## Find ACE calculator
local ace_calc = nothing
for c in full_stacked_calc.calcs
    if c isa ETM.WrappedSiteCalculator{<:ETM.ETACE}
        global ace_calc = c
    end
end

## Create tetrahedron
a = 2.35f0
tetra_positions = Float32[
    0.0      0.0             0.0;
    a        0.0             0.0;
    a/2      a*sqrt(3)/2     0.0;
    a/2      a*sqrt(3)/6     a*sqrt(2/3)
]
tetra_Z = Int32[14, 14, 14, 14]
n_atoms = Int32(4)

println("\n1. Tetrahedron setup:")
@printf("   Atoms: %d\n", n_atoms)
for i in 1:n_atoms
    @printf("   Atom %d: (%.4f, %.4f, %.4f)\n", i, tetra_positions[i, 1], tetra_positions[i, 2], tetra_positions[i, 3])
end

## Build edge list
rcut = model_state.rcut
edge_i_list = Int32[]
edge_j_list = Int32[]
edge_rij_list = Float32[]

for i in 1:n_atoms
    for j in 1:n_atoms
        i == j && continue
        rij = tetra_positions[j, :] - tetra_positions[i, :]
        r = norm(rij)
        if r < rcut
            push!(edge_i_list, i)
            push!(edge_j_list, j)
            append!(edge_rij_list, rij)
        end
    end
end

n_edges = Int32(length(edge_i_list))
# Note: append! creates [x1,y1,z1,x2,y2,z2,...], so reshape(flat, 3, n_edges) gives
# columns [x1,y1,z1], [x2,y2,z2], ..., then permutedims transposes to rows
edge_rij = permutedims(reshape(Float32.(edge_rij_list), 3, Int(n_edges)), (2, 1))
edge_i = Int32.(edge_i_list)
edge_j = Int32.(edge_j_list)

println("\n2. Edge list:")
@printf("   Total edges: %d\n", n_edges)
for e in 1:n_edges
    r = norm(edge_rij[e, :])
    @printf("   Edge %d: %d→%d, r=%.4f\n", e, edge_i[e], edge_j[e], r)
end

## Count neighbors per atom
neig_counts = zeros(Int, n_atoms)
for e in 1:n_edges
    neig_counts[edge_i[e]] += 1
end
println("\n3. Neighbor counts:")
for i in 1:n_atoms
    @printf("   Atom %d: %d neighbors\n", i, neig_counts[i])
end

## Our computation
println("\n4. Our ACE computation:")
state = model_state.ace_state
E_ace_ours = Main.compute_ace_energy(edge_rij, tetra_Z, edge_i, edge_j, n_atoms, n_edges, state)
@printf("   Our ACE energy: %.6f\n", E_ace_ours)

## Original computation
using AtomsBase
using Unitful

cell_vec = 100.0 * u"Å" .* [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
atoms_tetra = AtomsBase.FlexibleSystem(
    [AtomsBase.Atom(:Si, Float64.(tetra_positions[i, :]) * u"Å") for i in 1:n_atoms];
    cell_vectors = cell_vec,
    periodicity = (false, false, false)
)

import AtomsCalculators
E_ace_orig = AtomsCalculators.potential_energy(atoms_tetra, ace_calc)
E_ace_orig_val = Float32(ustrip(u"eV", E_ace_orig))
@printf("   Original ACE energy: %.6f\n", E_ace_orig_val)
@printf("   Ratio: %.4f\n", E_ace_orig_val / E_ace_ours)

## Build ETGraph and compare intermediate results
println("\n5. Original intermediate values:")
G = ET.Atoms.interaction_graph(atoms_tetra, ace_calc.rcut * u"Å")
@printf("   ETGraph nodes: %d, edges: %d\n", length(G.node_data), length(G.edge_data))

# Embeddings
rembed_out, _ = ace_calc.model.rembed(G, ace_calc.ps.rembed, ace_calc.st.rembed)
yembed_out, _ = ace_calc.model.yembed(G, ace_calc.ps.yembed, ace_calc.st.yembed)
@printf("   rembed out shape: %s, norm: %.6f\n", size(rembed_out), norm(rembed_out))
@printf("   yembed out shape: %s, norm: %.6f\n", size(yembed_out), norm(yembed_out))

# Basis output
basis_out, _ = ace_calc.model.basis((rembed_out, yembed_out), ace_calc.ps.basis, ace_calc.st.basis)
println("   basis out type: ", typeof(basis_out))
if basis_out isa Tuple
    for (i, b) in enumerate(basis_out)
        @printf("   basis_out[%d] shape: %s, norm: %.6f\n", i, size(b), norm(b))
    end
end

# Readout
readout_out, _ = ace_calc.model.readout((basis_out, G.node_data), ace_calc.ps.readout, ace_calc.st.readout)
println("   readout out: ", readout_out)
@printf("   readout sum: %.6f\n", sum(readout_out))

## Our intermediate values
println("\n6. Our intermediate values (detailed):")

# Build 3D tensors like compute_ace_energy does
n_rnl = state.n_rnl
n_ylm = state.nYlm

neig_count = zeros(Int, n_atoms)
for e in 1:n_edges
    neig_count[edge_i[e]] += 1
end
max_neigs = maximum(neig_count)
@printf("   max_neigs: %d\n", max_neigs)

Rnl_3 = zeros(Float32, max_neigs, n_atoms, n_rnl)
Ylm_3 = zeros(Float32, max_neigs, n_atoms, n_ylm)

neig_idx = zeros(Int, n_atoms)
for e in 1:n_edges
    i = edge_i[e]
    j_atom = edge_j[e]

    rij_vec = SVector{3,Float32}(edge_rij[e, 1], edge_rij[e, 2], edge_rij[e, 3])
    r = norm(rij_vec)
    r > state.rcut && continue
    rhat = r > eps(Float32) ? rij_vec / r : SVector{3,Float32}(0.0, 0.0, 1.0)

    zi = z_to_species_index(Int(tetra_Z[i]), state.species_Z)
    zj = z_to_species_index(Int(tetra_Z[j_atom]), state.species_Z)

    Rnl = compute_radial_embedding(r, zi, zj, state)
    Ylm = compute_solid_harmonics_reactant(r, rhat, state.maxl)

    neig_idx[i] += 1
    idx = neig_idx[i]
    for r_idx in 1:n_rnl
        Rnl_3[idx, i, r_idx] = Rnl[r_idx]
    end
    for y_idx in 1:n_ylm
        Ylm_3[idx, i, y_idx] = Ylm[y_idx]
    end
end

@printf("   Rnl_3 shape: %s, norm: %.6f\n", size(Rnl_3), norm(Rnl_3))
@printf("   Ylm_3 shape: %s, norm: %.6f\n", size(Ylm_3), norm(Ylm_3))

# ACE evaluation
BB, A, AA = ace_evaluate_reactant(Rnl_3, Ylm_3, state.spec_R, state.spec_Y,
                                  state.specs_mats, state.A2Bmap)
@printf("   A shape: %s, norm: %.6f\n", size(A), norm(A))
@printf("   AA shape: %s, norm: %.6f\n", size(AA), norm(AA))
@printf("   BB shape: %s, norm: %.6f\n", size(BB), norm(BB))

# Site energies
for i in 1:n_atoms
    site_E = dot(BB[i, :], state.W_readout[:, 1])
    @printf("   Site energy[%d]: %.6f\n", i, site_E)
end

println("\n" * "="^60)
println("DEBUG COMPLETE")
println("="^60)
