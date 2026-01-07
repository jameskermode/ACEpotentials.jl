#=
Reactant-Compatible StackedCalculator
=====================================

Support for combined 1-body + 2-body + many-body ACE models.
Exports StackedCalculator as a unified Reactant-compilable function.
=#

using StaticArrays
using LinearAlgebra: norm

## ============================================================================
## ReactantStackedModel - Combined 1+2+many body state
## ============================================================================

"""
    ReactantPairState{T}

State for 2-body (pair) potential component.
"""
struct ReactantPairState{T}
    n_pairs::Int                     # Number of (n,l) pair basis functions
    agnesi_params::Matrix{T}         # (5, n_species_pairs) Agnesi parameters
    poly_A::Vector{T}                # Chebyshev recurrence
    poly_B::Vector{T}
    poly_C::Vector{T}
    n_polys::Int
    W_pair::Array{T,3}               # (n_pairs, n_polys, n_species_pairs) radial weights
    W_readout_pair::Matrix{T}        # (n_pairs, n_species) readout weights
end

"""
    ReactantStackedModel{T}

Complete state for 1+2+many body model.

Components:
- E0: One-body reference energies per species
- pair_state: Two-body pair potential (optional)
- ace_state: Many-body ACE (required)
"""
struct ReactantStackedModel{T}
    # Species configuration
    n_species::Int
    species_Z::Vector{Int}
    rcut::T

    # One-body: E0[species] lookup
    E0::Vector{T}

    # Two-body pair potential (optional)
    has_pair::Bool
    pair_state::Union{ReactantPairState{T}, Nothing}

    # Many-body ACE
    ace_state::ReactantETACEState{T}
end

## ============================================================================
## Model Extraction from StackedCalculator
## ============================================================================

"""
    ReactantStackedModel(calc::ETModels.StackedCalculator)

Convert StackedCalculator to ReactantStackedModel.
"""
function ReactantStackedModel(calc)
    # Extract species info from first calculator
    # TODO: Get actual species info
    species_Z = collect(1:4)  # Placeholder
    n_species = length(species_Z)
    rcut = 6.0  # Placeholder

    # Initialize E0
    E0 = zeros(Float64, n_species)

    # Check for pair model
    has_pair = false
    pair_state = nothing

    # Extract ACE state
    # TODO: Actually extract from calc
    ace_state = ReactantETACEState{Float64}(
        n_species, species_Z, rcut,
        Int[], Int[], Matrix{Int}[], zeros(1, 1),
        10, 5, zeros(5, n_species^2),
        ones(10), zeros(10), zeros(10), zeros(5, 10, n_species^2),
        2, 9,
        10, zeros(10, n_species), E0
    )

    return ReactantStackedModel{Float64}(
        n_species, species_Z, rcut,
        E0,
        has_pair, pair_state,
        ace_state
    )
end

## ============================================================================
## Pair Energy Computation
## ============================================================================

"""
    compute_pair_energy(r::T, iz::Int, jz::Int, pair_state::ReactantPairState{T}) where T

Compute pair energy contribution for a single edge.
"""
function compute_pair_energy(r::T, iz::Int, jz::Int,
                              pair_state::ReactantPairState{T}) where T
    pair_idx = zz_to_pair_index(iz, jz, 4)  # TODO: get n_species

    # Agnesi transform
    pcut = pair_state.agnesi_params[1, pair_idx]
    pin = pair_state.agnesi_params[2, pair_idx]
    rin = pair_state.agnesi_params[3, pair_idx]
    req = pair_state.agnesi_params[4, pair_idx]
    rcut = pair_state.agnesi_params[5, pair_idx]

    y = compute_agnesi_transform(r, pcut, pin, rin, req, rcut)
    env = compute_envelope(y)

    # Polynomial basis
    P = compute_chebyshev_basis(y, pair_state.n_polys,
                                 pair_state.poly_A, pair_state.poly_B, pair_state.poly_C)
    P_env = P .* env

    # Pair radial features
    R_pair = pair_state.W_pair[:, :, pair_idx] * P_env

    # Pair energy (sum over pair basis)
    # Use center atom species for readout
    E_pair = dot(R_pair, pair_state.W_readout_pair[:, iz])

    return E_pair
end

## ============================================================================
## Full Stacked Energy Computation
## ============================================================================

"""
    stacked_energy_from_graph(positions, atomic_numbers, edge_i, edge_j,
                               n_atoms, n_edges, model::ReactantStackedModel)

Compute total energy from atomic graph.

This is the main function to compile with Reactant.

# Arguments
- `positions`: (max_atoms, 3) atomic positions
- `atomic_numbers`: (max_atoms,) atomic numbers (0 for padding)
- `edge_i`, `edge_j`: (max_edges,) neighbor list indices (0 for padding)
- `n_atoms`, `n_edges`: Actual counts (for masking)
- `model`: ReactantStackedModel containing all parameters

# Returns
- `energy`: Total energy (scalar)
"""
function stacked_energy_from_graph(positions::AbstractMatrix{T},
                                    atomic_numbers::AbstractVector{Int},
                                    edge_i::AbstractVector{Int},
                                    edge_j::AbstractVector{Int},
                                    n_atoms::Int,
                                    n_edges::Int,
                                    model::ReactantStackedModel{T}) where T

    # Compute edge vectors
    edge_rij = zeros(T, length(edge_i), 3)
    for e in 1:n_edges
        i, j = edge_i[e], edge_j[e]
        if i > 0 && j > 0
            edge_rij[e, :] = positions[j, :] - positions[i, :]
        end
    end

    # Call energy function with edge vectors
    return stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                      n_atoms, n_edges, model)
end

"""
    stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                               n_atoms, n_edges, model::ReactantStackedModel)

Compute energy from edge vectors (for Enzyme differentiation).
"""
function stacked_energy_from_edges(edge_rij::AbstractMatrix{T},
                                    atomic_numbers::AbstractVector{Int},
                                    edge_i::AbstractVector{Int},
                                    edge_j::AbstractVector{Int},
                                    n_atoms::Int,
                                    n_edges::Int,
                                    model::ReactantStackedModel{T}) where T

    # Initialize energy
    energy = zero(T)

    # 1. One-body energy: sum of E0
    for i in 1:n_atoms
        Z = atomic_numbers[i]
        s = findfirst(==(Z), model.species_Z)
        if !isnothing(s)
            energy += model.E0[s]
        end
    end

    # Early exit if no edges
    n_edges == 0 && return energy

    # Build 3D embedding arrays from edges
    # Need to reshape edge-based embeddings to (maxneigs, nnodes, nfeatures)
    maxneigs = 50  # TODO: Get from model
    n_rnl = model.ace_state.n_rnl
    n_ylm = model.ace_state.nYlm

    Rnl_3 = zeros(T, maxneigs, n_atoms, n_rnl)
    Ylm_3 = zeros(T, maxneigs, n_atoms, n_ylm)

    # Track neighbor count per atom
    neig_count = zeros(Int, n_atoms)

    # Compute embeddings for each edge
    for e in 1:n_edges
        i, j = edge_i[e], edge_j[e]
        i == 0 && continue

        rij = SVector{3,T}(edge_rij[e, 1], edge_rij[e, 2], edge_rij[e, 3])
        r = norm(rij)
        r < 1e-10 && continue

        rhat = rij / r

        # Get species indices
        iz = findfirst(==(atomic_numbers[i]), model.species_Z)
        jz = findfirst(==(atomic_numbers[j]), model.species_Z)
        (isnothing(iz) || isnothing(jz)) && continue

        # Compute embeddings
        Rnl, Ylm = compute_edge_embeddings(r, rhat, iz, jz, model.ace_state)

        # Store in 3D arrays
        neig_count[i] += 1
        ni = neig_count[i]
        if ni <= maxneigs
            Rnl_3[ni, i, :] = Rnl
            Ylm_3[ni, i, :] = Ylm
        end

        # 2. Pair energy (if present)
        if model.has_pair && !isnothing(model.pair_state)
            energy += compute_pair_energy(r, iz, jz, model.pair_state)
        end
    end

    # 3. Many-body ACE energy
    species_indices = zeros(Int, n_atoms)
    for i in 1:n_atoms
        s = findfirst(==(atomic_numbers[i]), model.species_Z)
        species_indices[i] = isnothing(s) ? 0 : s
    end

    ace_energy = ace_energy_from_embeddings(Rnl_3, Ylm_3, species_indices, model.ace_state)
    energy += ace_energy

    return energy
end

## ============================================================================
## Energy, Forces, Virial Computation
## ============================================================================

"""
    stacked_efv_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                           n_atoms, n_edges, model::ReactantStackedModel)

Compute energy, forces, and virial using Enzyme autodiff.

# Returns
- `energy`: Total energy (scalar)
- `forces`: (max_atoms, 3) atomic forces
- `virial`: (3, 3) virial tensor
"""
function stacked_efv_from_edges(edge_rij::AbstractMatrix{T},
                                 atomic_numbers::AbstractVector{Int},
                                 edge_i::AbstractVector{Int},
                                 edge_j::AbstractVector{Int},
                                 n_atoms::Int,
                                 n_edges::Int,
                                 model::ReactantStackedModel{T}) where T

    max_edges = size(edge_rij, 1)
    max_atoms = length(atomic_numbers)

    # Gradient buffer for edge vectors
    d_edge_rij = zeros(T, max_edges, 3)

    # Forward + reverse pass via Enzyme
    _, energy = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        stacked_energy_from_edges,
        Enzyme.Active,
        Enzyme.Duplicated(edge_rij, d_edge_rij),
        Enzyme.Const(atomic_numbers),
        Enzyme.Const(edge_i),
        Enzyme.Const(edge_j),
        Enzyme.Const(n_atoms),
        Enzyme.Const(n_edges),
        Enzyme.Const(model)
    )

    # Assemble forces from edge gradients
    forces = zeros(T, max_atoms, 3)
    for e in 1:n_edges
        i, j = edge_i[e], edge_j[e]
        i == 0 && continue

        # d_edge_rij[e, :] = dE/d(r_j - r_i)
        # Force on i: F_i += d_edge_rij (edge points from i to j)
        # Force on j: F_j -= d_edge_rij
        forces[i, :] .+= d_edge_rij[e, :]
        forces[j, :] .-= d_edge_rij[e, :]
    end
    forces .*= -1  # F = -dE/dr

    # Compute virial: V_ab = -sum_ij (dE/dr_ij)_a * (r_ij)_b
    virial = zeros(T, 3, 3)
    for e in 1:n_edges
        i = edge_i[e]
        i == 0 && continue

        for a in 1:3
            for b in 1:3
                virial[a, b] -= d_edge_rij[e, a] * edge_rij[e, b]
            end
        end
    end

    return energy, forces, virial
end
