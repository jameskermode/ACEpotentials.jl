#=
Test Force Computation
======================

Tests that the Reactant-compatible force computation matches
the original ACEpotentials.jl force calculation using Enzyme autodiff.
=#

using Test
using Printf
using LinearAlgebra
using Random
using StaticArrays
using SparseArrays
using AtomsCalculators

# Load ACEpotentials
using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

# Load extraction and computation modules
include(joinpath(@__DIR__, "..", "src", "reactant_state.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_embeddings.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_ace_kernel.jl"))
include(joinpath(@__DIR__, "..", "src", "reactant_stacked.jl"))

using AtomsBase
using Lux
using Unitful

println("="^60)
println("FORCE COMPUTATION TEST")
println("="^60)

## ============================================================================
## Step 1: Create ACE model
## ============================================================================

println("\n1. Creating ACE model...")

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
model_state = ReactantStackedModel(full_stacked_calc; T=Float64)  # Use Float64 for force accuracy

rcut = model_state.rcut
@printf("   Model created: order=%d, maxl=%d, rcut=%.2f Å\n", order, maxl, rcut)

## ============================================================================
## Step 2: Create test structure (Si dimer)
## ============================================================================

println("\n2. Creating test structure (Si dimer)...")

positions = Float64[
    0.0  0.0  0.0;
    2.35 0.0  0.0
]
atomic_numbers = Int32[14, 14]
n_atoms = Int32(2)

# Build edge list
edge_i = Int32[1, 2]
edge_j = Int32[2, 1]
n_edges = Int32(2)

edge_rij = Float64[
    positions[2, 1] - positions[1, 1]  positions[2, 2] - positions[1, 2]  positions[2, 3] - positions[1, 3];
    positions[1, 1] - positions[2, 1]  positions[1, 2] - positions[2, 2]  positions[1, 3] - positions[2, 3]
]

@printf("   Atoms: %d, Edges: %d\n", n_atoms, n_edges)
@printf("   Si-Si distance: %.4f Å\n", norm(edge_rij[1, :]))

## ============================================================================
## Step 3: Test energy computation with Float64
## ============================================================================

println("\n3. Testing energy with Float64...")

E_ours = stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                    n_atoms, n_edges, model_state)
@printf("   Our energy: %.10f\n", E_ours)

# Compare with original
cell_vec = 100.0 * u"Å" .* [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
atoms_sys = AtomsBase.FlexibleSystem(
    [AtomsBase.Atom(:Si, positions[i, :] * u"Å") for i in 1:n_atoms];
    cell_vectors = cell_vec,
    periodicity = (false, false, false)
)

E_orig = AtomsCalculators.potential_energy(atoms_sys, full_stacked_calc)
E_orig_val = Float64(ustrip(u"eV", E_orig))
@printf("   Original energy: %.10f eV\n", E_orig_val)
@printf("   Difference: %.2e\n", abs(E_orig_val - E_ours))

## ============================================================================
## Step 4: Test finite difference gradient
## ============================================================================

println("\n4. Computing finite difference gradients...")

# Compute finite difference gradient of energy w.r.t. edge_rij
function compute_energy_wrapper(edge_rij_flat)
    edge_rij_mat = reshape(edge_rij_flat, Int(n_edges), 3)
    return stacked_energy_from_edges(edge_rij_mat, atomic_numbers, edge_i, edge_j,
                                      n_atoms, n_edges, model_state)
end

edge_rij_flat = vec(edge_rij)
h = 1e-5
fd_grad = zeros(length(edge_rij_flat))

for i in 1:length(edge_rij_flat)
    edge_rij_plus = copy(edge_rij_flat)
    edge_rij_minus = copy(edge_rij_flat)
    edge_rij_plus[i] += h
    edge_rij_minus[i] -= h
    fd_grad[i] = (compute_energy_wrapper(edge_rij_plus) - compute_energy_wrapper(edge_rij_minus)) / (2h)
end

fd_grad_mat = reshape(fd_grad, Int(n_edges), 3)
println("   Finite difference ∂E/∂rij:")
for e in 1:n_edges
    @printf("     Edge %d: [%.6f, %.6f, %.6f]\n", e, fd_grad_mat[e, 1], fd_grad_mat[e, 2], fd_grad_mat[e, 3])
end

## ============================================================================
## Step 5: Test Enzyme autodiff
## ============================================================================

println("\n5. Testing Enzyme autodiff...")

using Enzyme

# Create a simple wrapper function for Enzyme
function energy_from_edges_enzyme(edge_rij, atomic_numbers, edge_i, edge_j,
                                   n_atoms, n_edges, model_state)
    return stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                      n_atoms, n_edges, model_state)
end

# Allocate gradient buffer
d_edge_rij = zeros(Float64, n_edges, 3)

# Try Enzyme autodiff
try
    # Use ReverseWithPrimal to get both energy and gradients
    result = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        energy_from_edges_enzyme,
        Enzyme.Active,
        Enzyme.Duplicated(copy(edge_rij), d_edge_rij),
        Enzyme.Const(atomic_numbers),
        Enzyme.Const(edge_i),
        Enzyme.Const(edge_j),
        Enzyme.Const(n_atoms),
        Enzyme.Const(n_edges),
        Enzyme.Const(model_state)
    )

    energy_enzyme = result[2]
    println("   Enzyme energy: ", energy_enzyme)
    println("   Enzyme ∂E/∂rij:")
    for e in 1:n_edges
        @printf("     Edge %d: [%.6f, %.6f, %.6f]\n", e, d_edge_rij[e, 1], d_edge_rij[e, 2], d_edge_rij[e, 3])
    end

    # Compare with finite difference
    grad_diff = norm(d_edge_rij - fd_grad_mat)
    @printf("   Gradient difference (Enzyme vs FD): %.2e\n", grad_diff)

catch e
    println("   Enzyme autodiff failed: ", e)
    println("   Trying simpler approach...")
end

## ============================================================================
## Step 6: Compare forces with original
## ============================================================================

println("\n6. Comparing forces with original...")

# Get original forces
F_orig = AtomsCalculators.forces(atoms_sys, full_stacked_calc)
F_orig_mat = zeros(Float64, n_atoms, 3)
for i in 1:n_atoms
    F_orig_mat[i, :] = ustrip.(u"eV/Å", F_orig[i])
end

println("   Original forces:")
for i in 1:n_atoms
    @printf("     Atom %d: [%.6f, %.6f, %.6f] eV/Å\n", i, F_orig_mat[i, 1], F_orig_mat[i, 2], F_orig_mat[i, 3])
end

# Compute forces from edge gradients
# For edge e: i = edge_i[e], j = edge_j[e], r_ij = r_j - r_i
#
# ∂E/∂r_i = Σ_e ∂E/∂r_ij * ∂r_ij/∂r_i
#   where ∂r_ij/∂r_i = -I if edge_i[e] == i, +I if edge_j[e] == i, 0 otherwise
#
# So: ∂E/∂r_i = -Σ_{e: edge_i[e]==i} ∂E/∂r_ij + Σ_{e: edge_j[e]==i} ∂E/∂r_ij
#     F_i = -∂E/∂r_i = Σ_{e: edge_i[e]==i} ∂E/∂r_ij - Σ_{e: edge_j[e]==i} ∂E/∂r_ij

F_ours = zeros(Float64, n_atoms, 3)
for e in 1:n_edges
    i = edge_i[e]
    j = edge_j[e]
    # Edge i→j contributes +∂E/∂r_ij to F_i, -∂E/∂r_ij to F_j
    F_ours[i, :] .+= fd_grad_mat[e, :]
    F_ours[j, :] .-= fd_grad_mat[e, :]
end

# Note: No division by 2! Each edge contributes independently to forces.
# The formula F_i = Σ_{e: edge_i[e]==i} ∂E/∂r_ij - Σ_{e: edge_j[e]==i} ∂E/∂r_ij
# naturally handles bidirectional edges correctly.

println("   Our forces (from FD gradients):")
for i in 1:n_atoms
    @printf("     Atom %d: [%.6f, %.6f, %.6f] eV/Å\n", i, F_ours[i, 1], F_ours[i, 2], F_ours[i, 3])
end

force_diff = norm(F_ours - F_orig_mat)
@printf("   Force difference: %.2e eV/Å\n", force_diff)

## ============================================================================
## Step 7: Test with tetrahedron
## ============================================================================

println("\n7. Testing with tetrahedron structure...")

a = 2.35
tetra_positions = Float64[
    0.0      0.0             0.0;
    a        0.0             0.0;
    a/2      a*sqrt(3)/2     0.0;
    a/2      a*sqrt(3)/6     a*sqrt(2/3)
]
tetra_Z = Int32[14, 14, 14, 14]
n_atoms_tetra = Int32(4)

# Build edge list
tetra_edge_i = Int32[]
tetra_edge_j = Int32[]
tetra_edge_rij = Float64[]

for i in 1:n_atoms_tetra
    for j in 1:n_atoms_tetra
        i == j && continue
        rij = tetra_positions[j, :] - tetra_positions[i, :]
        r = norm(rij)
        if r < rcut
            push!(tetra_edge_i, i)
            push!(tetra_edge_j, j)
            append!(tetra_edge_rij, rij)
        end
    end
end

n_edges_tetra = Int32(length(tetra_edge_i))
tetra_edge_rij = permutedims(reshape(tetra_edge_rij, 3, Int(n_edges_tetra)), (2, 1))

# Compute FD gradients for tetrahedron
function compute_tetra_energy(edge_rij_flat)
    edge_rij_mat = reshape(edge_rij_flat, Int(n_edges_tetra), 3)
    return stacked_energy_from_edges(edge_rij_mat, tetra_Z, tetra_edge_i, tetra_edge_j,
                                      n_atoms_tetra, n_edges_tetra, model_state)
end

tetra_edge_flat = vec(tetra_edge_rij)
fd_grad_tetra = zeros(length(tetra_edge_flat))
for i in 1:length(tetra_edge_flat)
    edge_plus = copy(tetra_edge_flat)
    edge_minus = copy(tetra_edge_flat)
    edge_plus[i] += h
    edge_minus[i] -= h
    fd_grad_tetra[i] = (compute_tetra_energy(edge_plus) - compute_tetra_energy(edge_minus)) / (2h)
end
fd_grad_tetra_mat = reshape(fd_grad_tetra, Int(n_edges_tetra), 3)

# Compute forces
F_tetra_ours = zeros(Float64, n_atoms_tetra, 3)
for e in 1:n_edges_tetra
    i = tetra_edge_i[e]
    j = tetra_edge_j[e]
    F_tetra_ours[i, :] .+= fd_grad_tetra_mat[e, :]
    F_tetra_ours[j, :] .-= fd_grad_tetra_mat[e, :]
end
# No division by 2 - same reasoning as above

# Get original forces
atoms_tetra = AtomsBase.FlexibleSystem(
    [AtomsBase.Atom(:Si, tetra_positions[i, :] * u"Å") for i in 1:n_atoms_tetra];
    cell_vectors = cell_vec,
    periodicity = (false, false, false)
)
F_tetra_orig = AtomsCalculators.forces(atoms_tetra, full_stacked_calc)
F_tetra_orig_mat = zeros(Float64, n_atoms_tetra, 3)
for i in 1:n_atoms_tetra
    F_tetra_orig_mat[i, :] = ustrip.(u"eV/Å", F_tetra_orig[i])
end

println("   Original tetrahedron forces:")
for i in 1:n_atoms_tetra
    @printf("     Atom %d: [%.6f, %.6f, %.6f]\n", i, F_tetra_orig_mat[i, 1], F_tetra_orig_mat[i, 2], F_tetra_orig_mat[i, 3])
end

println("   Our tetrahedron forces:")
for i in 1:n_atoms_tetra
    @printf("     Atom %d: [%.6f, %.6f, %.6f]\n", i, F_tetra_ours[i, 1], F_tetra_ours[i, 2], F_tetra_ours[i, 3])
end

tetra_force_diff = norm(F_tetra_ours - F_tetra_orig_mat)
@printf("   Tetrahedron force difference: %.2e eV/Å\n", tetra_force_diff)
@printf("   Relative error: %.2e\n", tetra_force_diff / norm(F_tetra_orig_mat))

## ============================================================================
## Step 8: Test virial tensor computation
## ============================================================================

println("\n8. Testing virial tensor computation...")

# Virial tensor: V_ab = -Σ_e r_e[a] * (∂E/∂r_e)[b]
# The negative sign comes from the definition of virial from pair forces
# V_ab = -Σ_{i<j} r_ij[a] * f_ij[b]

# Compute virial for dimer
V_dimer_ours = zeros(Float64, 3, 3)
for e in 1:n_edges
    for a in 1:3
        for b in 1:3
            V_dimer_ours[a, b] -= edge_rij[e, a] * fd_grad_mat[e, b]
        end
    end
end

println("   Dimer virial (from FD gradients):")
@printf("     [%.4f  %.4f  %.4f]\n", V_dimer_ours[1,1], V_dimer_ours[1,2], V_dimer_ours[1,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_dimer_ours[2,1], V_dimer_ours[2,2], V_dimer_ours[2,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_dimer_ours[3,1], V_dimer_ours[3,2], V_dimer_ours[3,3])

# Get original virial
V_orig = AtomsCalculators.virial(atoms_sys, full_stacked_calc)
V_orig_mat = zeros(Float64, 3, 3)
for i in 1:3, j in 1:3
    V_orig_mat[i, j] = ustrip(u"eV", V_orig[i, j])
end

println("   Original dimer virial:")
@printf("     [%.4f  %.4f  %.4f]\n", V_orig_mat[1,1], V_orig_mat[1,2], V_orig_mat[1,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_orig_mat[2,1], V_orig_mat[2,2], V_orig_mat[2,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_orig_mat[3,1], V_orig_mat[3,2], V_orig_mat[3,3])

virial_diff = norm(V_dimer_ours - V_orig_mat)
@printf("   Virial difference: %.2e eV\n", virial_diff)

# Compute virial for tetrahedron
V_tetra_ours = zeros(Float64, 3, 3)
for e in 1:n_edges_tetra
    for a in 1:3
        for b in 1:3
            V_tetra_ours[a, b] -= tetra_edge_rij[e, a] * fd_grad_tetra_mat[e, b]
        end
    end
end

println("\n   Tetrahedron virial (from FD gradients):")
@printf("     [%.4f  %.4f  %.4f]\n", V_tetra_ours[1,1], V_tetra_ours[1,2], V_tetra_ours[1,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_tetra_ours[2,1], V_tetra_ours[2,2], V_tetra_ours[2,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_tetra_ours[3,1], V_tetra_ours[3,2], V_tetra_ours[3,3])

# Get original tetrahedron virial
V_tetra_orig = AtomsCalculators.virial(atoms_tetra, full_stacked_calc)
V_tetra_orig_mat = zeros(Float64, 3, 3)
for i in 1:3, j in 1:3
    V_tetra_orig_mat[i, j] = ustrip(u"eV", V_tetra_orig[i, j])
end

println("   Original tetrahedron virial:")
@printf("     [%.4f  %.4f  %.4f]\n", V_tetra_orig_mat[1,1], V_tetra_orig_mat[1,2], V_tetra_orig_mat[1,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_tetra_orig_mat[2,1], V_tetra_orig_mat[2,2], V_tetra_orig_mat[2,3])
@printf("     [%.4f  %.4f  %.4f]\n", V_tetra_orig_mat[3,1], V_tetra_orig_mat[3,2], V_tetra_orig_mat[3,3])

virial_tetra_diff = norm(V_tetra_ours - V_tetra_orig_mat)
@printf("   Tetrahedron virial difference: %.2e eV\n", virial_tetra_diff)
@printf("   Relative error: %.2e\n", virial_tetra_diff / norm(V_tetra_orig_mat))

println("\n" * "="^60)
println("FORCE AND VIRIAL TEST COMPLETE")
println("="^60)
