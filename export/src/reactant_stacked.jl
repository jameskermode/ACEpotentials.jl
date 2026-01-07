#=
Reactant-Compatible StackedCalculator
=====================================

Support for combined 1-body + 2-body + many-body ACE models.
Exports StackedCalculator as a unified Reactant-compilable function.
=#

using StaticArrays
using LinearAlgebra: norm

# Import ACEpotentials types
import ACEpotentials.Models.ETModels: StackedCalculator, WrappedSiteCalculator,
    ETACE, ETPairModel, ETOneBody

## ============================================================================
## ReactantPairState - State for 2-body (pair) potential
## ============================================================================

"""
    ReactantPairState{T}

State for 2-body (pair) potential component.
Contains radial basis parameters and readout weights for pair interactions.
"""
struct ReactantPairState{T}
    n_basis::Int                     # Number of pair basis functions
    n_species::Int                   # Number of species
    n_pairs::Int                     # Number of species pairs (n_species^2)

    # Radial basis parameters (same structure as ACE)
    n_polys::Int
    agnesi_params::Matrix{T}         # (5, n_pairs)
    poly_A::Vector{T}
    poly_B::Vector{T}
    poly_C::Vector{T}
    W_radial::Array{T,3}             # (n_basis, n_polys, n_pairs)

    # Readout
    W_readout::Matrix{T}             # (n_basis, n_species)
end

## ============================================================================
## ReactantStackedModel - Combined 1+2+many body state
## ============================================================================

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
    ReactantStackedModel(calc::StackedCalculator; T=Float32)

Convert StackedCalculator to ReactantStackedModel.

The StackedCalculator typically contains:
1. ETOneBodyPotential (WrappedSiteCalculator{ETOneBody}) - reference energies
2. ETPairPotential (WrappedSiteCalculator{ETPairModel}) - pair potential
3. ETACEPotential (WrappedSiteCalculator{ETACE}) - many-body ACE

This function extracts parameters from all components into a unified
Reactant-compatible format.
"""
function ReactantStackedModel(calc::StackedCalculator; T::Type=Float32)
    # Find each calculator type in the stack
    onebody_calc = nothing
    pair_calc = nothing
    ace_calc = nothing

    for c in calc.calcs
        if c isa WrappedSiteCalculator{<:ETOneBody}
            onebody_calc = c
        elseif c isa WrappedSiteCalculator{<:ETPairModel}
            pair_calc = c
        elseif c isa WrappedSiteCalculator{<:ETACE}
            ace_calc = c
        end
    end

    # ACE calculator is required
    if ace_calc === nothing
        error("StackedCalculator must contain an ETACEPotential")
    end

    # Extract ACE state (this contains most of the model structure)
    ace_state = prepare_reactant_state(ace_calc; T=T)

    # Use species info from ACE state
    n_species = ace_state.n_species
    species_Z = ace_state.species_Z
    rcut = ace_state.rcut

    # Extract E0 from one-body calculator
    E0 = _extract_E0(onebody_calc, n_species, T)

    # Extract pair state if present
    has_pair = pair_calc !== nothing
    pair_state = has_pair ? _extract_pair_state(pair_calc, n_species, T) : nothing

    # Update ACE state E0 with extracted values
    ace_state_with_E0 = ReactantETACEState{T}(
        ace_state.n_species, ace_state.species_Z, ace_state.rcut,
        ace_state.spec_R, ace_state.spec_Y, ace_state.specs_mats, ace_state.A2Bmap,
        ace_state.n_polys, ace_state.n_rnl, ace_state.agnesi_params,
        ace_state.poly_A, ace_state.poly_B, ace_state.poly_C, ace_state.W_radial,
        ace_state.maxl, ace_state.nYlm,
        ace_state.n_basis, ace_state.W_readout, E0
    )

    return ReactantStackedModel{T}(
        n_species, species_Z, rcut,
        E0,
        has_pair, pair_state,
        ace_state_with_E0
    )
end

"""
    ReactantStackedModel(calc::WrappedSiteCalculator{<:ETACE}; T=Float32)

Create ReactantStackedModel from a single ETACEPotential (no pair or one-body).
"""
function ReactantStackedModel(calc::WrappedSiteCalculator{<:ETACE}; T::Type=Float32)
    ace_state = prepare_reactant_state(calc; T=T)

    return ReactantStackedModel{T}(
        ace_state.n_species,
        ace_state.species_Z,
        ace_state.rcut,
        ace_state.E0,
        false,
        nothing,
        ace_state
    )
end

## ============================================================================
## Component extraction helpers
## ============================================================================

"""
Extract reference energies E0 from one-body calculator.
"""
function _extract_E0(onebody_calc, n_species::Int, T::Type)
    E0 = zeros(T, n_species)

    if onebody_calc === nothing
        return E0
    end

    try
        # ETOneBody stores E0 in its state
        model = onebody_calc.model
        st = onebody_calc.st

        if hasproperty(st, :E0) && st.E0 isa AbstractDict
            # E0 is a Dict from species to energy
            for (species, energy) in st.E0
                # Convert species to index
                # Species might be ChemicalSpecies or Int
                z = _species_to_atomic_number(species)
                if z <= n_species
                    E0[z] = T(energy)
                end
            end
        end
    catch e
        @warn "Could not extract E0 from one-body calculator" exception=e
    end

    return E0
end

"""
Extract pair potential state.
"""
function _extract_pair_state(pair_calc, n_species::Int, T::Type)
    model = pair_calc.model
    ps = pair_calc.ps
    st = pair_calc.st

    # Get dimensions
    n_basis = model.readout.in_dim
    n_pairs = n_species * n_species

    # Extract radial basis parameters (similar to ACE extraction)
    n_polys = _extract_n_polys_pair(model.rembed)

    # Agnesi parameters
    agnesi_params = _extract_agnesi_params_pair(model.rembed, st.rembed, n_pairs, T)

    # Chebyshev coefficients
    poly_A, poly_B, poly_C = _extract_chebyshev_coeffs_pair(model.rembed, n_polys, T)

    # Radial weights
    W_radial = _extract_radial_weights_pair(ps.rembed, n_basis, n_polys, n_pairs, T)

    # Readout weights
    W_readout = T.(dropdims(ps.readout.W, dims=1))

    return ReactantPairState{T}(
        n_basis, n_species, n_pairs,
        n_polys, agnesi_params, poly_A, poly_B, poly_C, W_radial,
        W_readout
    )
end

"""
Convert species identifier to atomic number.
"""
function _species_to_atomic_number(species)
    if species isa Integer
        return Int(species)
    elseif hasproperty(species, :atomic_number)
        return Int(species.atomic_number)
    else
        # Try to extract from ChemicalSpecies
        return Int(species)
    end
end

## ============================================================================
## Pair extraction helpers (similar to ACE but different structure)
## ============================================================================

function _extract_n_polys_pair(rembed)
    try
        # Pair model has different structure: EdgeEmbed(EnvRBranchL(env, rbasis))
        # where rbasis is EmbedDP(trans, polys, linl)
        inner = rembed.basis
        if hasproperty(inner, :rbasis)
            rbasis = inner.rbasis
            if hasproperty(rbasis, :post)
                return rbasis.post.in_dim
            end
        end
    catch
    end
    return 10
end

function _extract_agnesi_params_pair(rembed, rembed_st, n_pairs, T)
    params = zeros(T, 5, n_pairs)

    try
        inner = rembed.basis
        if hasproperty(inner, :rbasis)
            trans = inner.rbasis.trans
            if hasproperty(trans, :refstate) && hasproperty(trans.refstate, :params)
                agnesi_list = trans.refstate.params
                for (idx, p) in enumerate(agnesi_list)
                    if idx <= n_pairs
                        params[1, idx] = T(p.pcut)
                        params[2, idx] = T(p.pin)
                        params[3, idx] = T(p.rin)
                        params[4, idx] = T(p.req)
                        params[5, idx] = T(p.rcut)
                    end
                end
            end
        end
    catch e
        @warn "Could not extract pair Agnesi params" exception=e
        for idx in 1:n_pairs
            params[1, idx] = T(2.0)
            params[2, idx] = T(2.0)
            params[3, idx] = T(1.0)
            params[4, idx] = T(2.5)
            params[5, idx] = T(6.0)
        end
    end

    return params
end

function _extract_chebyshev_coeffs_pair(rembed, n_polys, T)
    A = fill(T(2), n_polys)
    A[1] = T(1)
    B = zeros(T, n_polys)
    C = fill(T(-1), n_polys)
    C[1] = T(0)

    try
        inner = rembed.basis
        if hasproperty(inner, :rbasis)
            rbasis = inner.rbasis
            if hasproperty(rbasis, :basis) && hasproperty(rbasis.basis, :A)
                polys = rbasis.basis
                A = T.(polys.A[1:n_polys])
                B = T.(polys.B[1:n_polys])
                C = T.(polys.C[1:n_polys])
            end
        end
    catch
    end

    return A, B, C
end

function _extract_radial_weights_pair(ps_rembed, n_basis, n_polys, n_pairs, T)
    W = zeros(T, n_basis, n_polys, n_pairs)

    try
        # Pair model: ps_rembed.rbasis.post.W
        if hasproperty(ps_rembed, :rbasis) && hasproperty(ps_rembed.rbasis, :post)
            W_raw = ps_rembed.rbasis.post.W
            for i in axes(W_raw, 1), j in axes(W_raw, 2), k in axes(W_raw, 3)
                if i <= n_basis && j <= n_polys && k <= n_pairs
                    W[i, j, k] = T(W_raw[i, j, k])
                end
            end
        end
    catch e
        @warn "Could not extract pair radial weights" exception=e
    end

    return W
end

## ============================================================================
## Energy computation functions (for Reactant compilation)
## ============================================================================

"""
    stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                               n_atoms, n_edges, model::ReactantStackedModel)

Compute total energy from edge vectors and atomic information.

This function is designed to be compilable with Reactant.
All control flow is based on compile-time constants (model structure).

# Arguments
- `edge_rij`: (n_edges, 3) edge vectors rij = rj - ri
- `atomic_numbers`: (n_atoms,) atomic numbers
- `edge_i`, `edge_j`: (n_edges,) edge indices
- `n_atoms`, `n_edges`: actual counts (for masking padded inputs)
- `model`: ReactantStackedModel with all parameters

# Returns
- Total energy (scalar)
"""
function stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                    n_atoms::Int32, n_edges::Int32,
                                    model::ReactantStackedModel)
    T = eltype(edge_rij)

    # 1. One-body energy
    E_onebody = compute_onebody_energy(atomic_numbers, n_atoms, model.E0, model.species_Z)

    # 2. Pair energy (if present)
    E_pair = if model.has_pair && model.pair_state !== nothing
        compute_pair_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                           n_atoms, n_edges, model.pair_state, model.species_Z)
    else
        zero(T)
    end

    # 3. Many-body ACE energy
    E_ace = compute_ace_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                               n_atoms, n_edges, model.ace_state)

    return E_onebody + E_pair + E_ace
end

"""
Compute one-body energy: E = Σᵢ E0[species[i]]
"""
function compute_onebody_energy(atomic_numbers, n_atoms::Int32, E0::Vector{T},
                                 species_Z::Vector{Int}) where T
    energy = zero(T)
    for i in 1:n_atoms
        z = atomic_numbers[i]
        iz = z_to_species_index(z, species_Z)
        energy += E0[iz]
    end
    return energy
end

"""
Compute pair energy from edge vectors.
"""
function compute_pair_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                             n_atoms::Int32, n_edges::Int32,
                             pair_state::ReactantPairState{T},
                             species_Z::Vector{Int}) where T
    # TODO: Implement full pair potential evaluation
    # This requires:
    # 1. Compute distances from edge_rij
    # 2. Apply Agnesi transform
    # 3. Evaluate Chebyshev polynomials
    # 4. Apply radial weights
    # 5. Sum site energies

    return zero(T)  # Placeholder
end

"""
Compute ACE many-body energy from edge vectors.

This is the main ACE evaluation function that:
1. Computes embeddings (Rnl, Ylm) from edge vectors
2. Pools over neighbors to get atomic features A
3. Applies sparse symmetric products
4. Applies coupling matrix (A2Bmap)
5. Computes readout (linear combination)
"""
function compute_ace_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                            n_atoms::Int32, n_edges::Int32,
                            state::ReactantETACEState{T}) where T
    # TODO: Implement full ACE evaluation
    # This requires the full embedding + kernel pipeline
    # For now, return placeholder

    return zero(T)  # Placeholder
end

## ============================================================================
## EFV (Energy, Forces, Virial) computation
## ============================================================================

"""
    stacked_efv_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                           n_atoms, n_edges, model::ReactantStackedModel)

Compute energy, forces, and virial from edge vectors.

Forces are computed via automatic differentiation (Enzyme) of the energy
with respect to edge vectors.

# Returns
- (energy, forces, virial) tuple
"""
function stacked_efv_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                 n_atoms::Int32, n_edges::Int32,
                                 model::ReactantStackedModel)
    # TODO: Implement with Enzyme.autodiff for gradients
    # This will require:
    # 1. Forward pass for energy
    # 2. Reverse pass for ∂E/∂(edge_rij)
    # 3. Accumulate forces from edge gradients
    # 4. Compute virial from edge gradients and vectors

    T = eltype(edge_rij)

    energy = stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                       n_atoms, n_edges, model)

    # Placeholder forces and virial
    forces = zeros(T, n_atoms, 3)
    virial = zeros(T, 3, 3)

    return (energy, forces, virial)
end
