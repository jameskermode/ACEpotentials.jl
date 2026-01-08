#=
Reactant-Compatible StackedCalculator
=====================================

Support for combined 1-body + 2-body + many-body ACE models.
Exports StackedCalculator as a unified Reactant-compilable function.
=#

using StaticArrays
using LinearAlgebra: norm

# Import ACEpotentials types
import ACEpotentials.ETModels: StackedCalculator, WrappedSiteCalculator,
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
    rcut::T                          # Cutoff radius
    agnesi_params::Matrix{T}         # (7, n_pairs) - pin, pcut, a, b0, b1, rin, req
    poly_A::Vector{T}
    poly_B::Vector{T}
    poly_C::Vector{T}
    W_radial::Array{T,3}             # (n_basis, n_polys, n_pairs)

    # Outer cutoff envelope: (1 - r/rcut_outer)^p_outer
    rcut_outer::T                    # Cutoff radius for outer envelope
    p_outer::Int                     # Power for outer envelope

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

    # Extract cutoff from the pair model
    rcut = T(pair_calc.rcut)

    # Extract radial basis parameters (similar to ACE extraction)
    n_polys = _extract_n_polys_pair(model.rembed)

    # Agnesi parameters (7 params: pin, pcut, a, b0, b1, rin, req) - extract from state
    agnesi_params = _extract_agnesi_params_pair(st.rembed, n_pairs, T)

    # Chebyshev coefficients - extract from state
    poly_A, poly_B, poly_C = _extract_chebyshev_coeffs_pair(st.rembed, n_polys, T)

    # Radial weights
    W_radial = _extract_radial_weights_pair(ps.rembed, n_basis, n_polys, n_pairs, T)

    # Outer envelope parameters from EnvRBranchL
    rcut_outer, p_outer = _extract_outer_envelope_pair(st.rembed, rcut, T)

    # Readout weights
    W_readout = T.(dropdims(ps.readout.W, dims=1))

    return ReactantPairState{T}(
        n_basis, n_species, n_pairs,
        n_polys, rcut, agnesi_params, poly_A, poly_B, poly_C, W_radial,
        rcut_outer, p_outer,
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
        # Pair model has structure: EdgeEmbed(EnvRBranchL(env, rbasis))
        # EdgeEmbed wraps in `layer` field
        # where rbasis is EmbedDP(trans, polys, linl)
        outer = hasproperty(rembed, :layer) ? rembed.layer : rembed.basis
        if hasproperty(outer, :rbasis)
            rbasis = outer.rbasis
            if hasproperty(rbasis, :post)
                return rbasis.post.in_dim
            end
        end
    catch
    end
    return 10
end

"""
Extract Agnesi parameters (7 values) from pair model state.
Pair state structure: st.rembed.layer.rbasis.trans.params (or st.rembed.rbasis.trans.params)
Each param tuple has: pin, pcut, a, b0, b1, rin, req
"""
function _extract_agnesi_params_pair(rembed_st, n_pairs, T)
    params = zeros(T, 7, n_pairs)

    try
        # Navigate to the transform state
        # Pair model structure: st.rembed.layer.rbasis.trans.params
        # or possibly st.rembed.rbasis.trans.params
        trans_st = nothing
        if hasproperty(rembed_st, :layer) && hasproperty(rembed_st.layer, :rbasis)
            trans_st = rembed_st.layer.rbasis.trans
        elseif hasproperty(rembed_st, :rbasis)
            trans_st = rembed_st.rbasis.trans
        end

        if trans_st !== nothing && hasproperty(trans_st, :params)
            agnesi_list = trans_st.params
            for (idx, p) in enumerate(agnesi_list)
                if idx <= n_pairs
                    params[1, idx] = T(p.pin)
                    params[2, idx] = T(p.pcut)
                    params[3, idx] = T(p.a)
                    params[4, idx] = T(p.b0)
                    params[5, idx] = T(p.b1)
                    params[6, idx] = T(p.rin)
                    params[7, idx] = T(p.req)
                end
            end
        else
            @warn "Could not find pair Agnesi params in state, using defaults"
            _set_default_agnesi_params!(params, n_pairs, T)
        end
    catch e
        @warn "Could not extract pair Agnesi params" exception=e
        _set_default_agnesi_params!(params, n_pairs, T)
    end

    return params
end

function _set_default_agnesi_params!(params, n_pairs, T)
    for idx in 1:n_pairs
        params[1, idx] = T(2)     # pin
        params[2, idx] = T(2)     # pcut
        params[3, idx] = T(0.5)   # a
        params[4, idx] = T(-1.0)  # b0
        params[5, idx] = T(2.0)   # b1
        params[6, idx] = T(0.5)   # rin
        params[7, idx] = T(2.5)   # req
    end
end

"""
Extract Chebyshev coefficients from pair model state.
Pair state structure: st.rembed.layer.rbasis.basis has (A, B, C)
"""
function _extract_chebyshev_coeffs_pair(rembed_st, n_polys, T)
    A = fill(T(2), n_polys)
    A[1] = T(1)
    B = zeros(T, n_polys)
    C = fill(T(-1), n_polys)
    C[1] = T(0)

    try
        # Navigate to polynomial coefficients in state
        # Structure: st.rembed.layer.rbasis.basis or st.rembed.rbasis.basis
        basis_st = nothing
        if hasproperty(rembed_st, :layer) && hasproperty(rembed_st.layer, :rbasis)
            basis_st = rembed_st.layer.rbasis.basis
        elseif hasproperty(rembed_st, :rbasis)
            basis_st = rembed_st.rbasis.basis
        end

        if basis_st !== nothing && hasproperty(basis_st, :A)
            A = T.(basis_st.A[1:n_polys])
            B = T.(basis_st.B[1:n_polys])
            C = T.(basis_st.C[1:n_polys])
        else
            @warn "Could not find pair Chebyshev coeffs in state, using standard"
        end
    catch e
        @warn "Could not extract pair Chebyshev coefficients" exception=e
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

"""
Extract outer envelope parameters from pair model state.
EnvRBranchL applies (1 - r/rcut)^p as an outer cutoff envelope.
"""
function _extract_outer_envelope_pair(rembed_st, default_rcut, T)
    rcut_outer = default_rcut
    p_outer = 1

    try
        # Structure: st.rembed.envelope has (rcut, p)
        if hasproperty(rembed_st, :envelope)
            env_st = rembed_st.envelope
            if hasproperty(env_st, :rcut)
                rcut_outer = T(env_st.rcut)
            end
            if hasproperty(env_st, :p)
                p_outer = Int(env_st.p)
            end
        end
    catch e
        @warn "Could not extract outer envelope params, using defaults" exception=e
    end

    return rcut_outer, p_outer
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

Pipeline for each edge:
1. Compute distance from edge vector
2. Apply Agnesi transform: r → y ∈ [-1, 1]
3. Evaluate Chebyshev polynomials (NO inner envelope for pair model)
4. Apply outer cutoff envelope: (s^(-p) - 1) * (1 - s) where s = r/rcut
5. Linear layer to get pair features
6. Apply readout weights for center atom species

Note: The pair model uses a different envelope structure than the ACE many-body model.
The pair model applies EnvRBranchL which uses outer_env * rbasis, where rbasis does NOT
include the inner (1-y²)² envelope.

# Arguments
- `edge_rij`: (n_edges, 3) edge vectors
- `atomic_numbers`: (n_atoms,) atomic numbers
- `edge_i`, `edge_j`: (n_edges,) edge indices
- `n_atoms`, `n_edges`: counts
- `pair_state`: ReactantPairState with parameters
- `species_Z`: species index to atomic number mapping

# Returns
- Total pair energy (scalar)
"""
function compute_pair_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                             n_atoms::Int32, n_edges::Int32,
                             pair_state::ReactantPairState{T},
                             species_Z::Vector{Int}) where T
    energy = zero(T)
    n_polys = pair_state.n_polys
    n_basis = pair_state.n_basis
    rcut = pair_state.rcut

    # Outer envelope parameters
    rcut_outer = pair_state.rcut_outer
    p_outer = pair_state.p_outer

    # Pre-allocate buffers
    P = Vector{T}(undef, n_polys)

    for e in 1:n_edges
        i = edge_i[e]
        j_atom = edge_j[e]

        # Compute distance
        rij = SVector{3,T}(edge_rij[e, 1], edge_rij[e, 2], edge_rij[e, 3])
        r = norm(rij)

        # Skip if outside cutoff
        r > rcut && continue

        # Get species indices
        zi = z_to_species_index(Int(atomic_numbers[i]), species_Z)
        zj = z_to_species_index(Int(atomic_numbers[j_atom]), species_Z)

        # Get pair index (asymmetric: depends on both center and neighbor species)
        pair_idx = zz_to_pair_index(zi, zj, pair_state.n_species)

        # Extract Agnesi parameters for this pair (7 params: pin, pcut, a, b0, b1, rin, req)
        pin = Int(pair_state.agnesi_params[1, pair_idx])
        pcut = Int(pair_state.agnesi_params[2, pair_idx])
        a = pair_state.agnesi_params[3, pair_idx]
        b0 = pair_state.agnesi_params[4, pair_idx]
        b1 = pair_state.agnesi_params[5, pair_idx]
        rin = pair_state.agnesi_params[6, pair_idx]
        req = pair_state.agnesi_params[7, pair_idx]

        # Apply Agnesi transform (7-param version)
        y = compute_agnesi_transform(r, pin, pcut, a, b0, b1, rin, req)

        # Evaluate Chebyshev polynomials (NO inner envelope for pair model!)
        compute_chebyshev_basis!(P, y, pair_state.poly_A, pair_state.poly_B, pair_state.poly_C)

        # Linear layer first (before envelope): W_radial[:, :, pair_idx] * P
        # W_radial is (n_basis, n_polys, n_pairs)
        pair_features = pair_state.W_radial[:, :, pair_idx] * P

        # Apply outer cutoff envelope: (s^(-p) - 1) * (1 - s) where s = r/rcut
        # This is the envelope from EnvRBranchL in the pair model
        s = r / rcut_outer
        outer_env = (s^(-p_outer) - one(T)) * (one(T) - s)
        pair_features = pair_features .* outer_env

        # Readout: pair_features · W_readout[:, zi]
        # W_readout is (n_basis, n_species)
        for b in 1:n_basis
            energy += pair_features[b] * pair_state.W_readout[b, zi]
        end
    end

    return energy
end

"""
Compute ACE many-body energy from edge vectors.

This is the main ACE evaluation function that:
1. Computes embeddings (Rnl, Ylm) from edge vectors
2. Pools over neighbors to get atomic features A
3. Applies sparse symmetric products
4. Applies coupling matrix (A2Bmap)
5. Computes readout (linear combination)

# Arguments
- `edge_rij`: (n_edges, 3) edge vectors rij = rj - ri
- `atomic_numbers`: (n_atoms,) atomic numbers
- `edge_i`, `edge_j`: (n_edges,) edge indices (1-based)
- `n_atoms`, `n_edges`: actual counts
- `state`: ReactantETACEState with all parameters

# Returns
- Total ACE energy (scalar)
"""
function compute_ace_energy(edge_rij, atomic_numbers, edge_i, edge_j,
                            n_atoms::Int32, n_edges::Int32,
                            state::ReactantETACEState{T}) where T
    n_rnl = state.n_rnl
    n_ylm = state.nYlm

    # Step 1: Count neighbors per atom to determine max_neigs
    neig_count = zeros(Int, n_atoms)
    for e in 1:n_edges
        i = edge_i[e]
        neig_count[i] += 1
    end
    max_neigs = maximum(neig_count)

    # Step 2: Build 3D embedding tensors
    # Rnl_3[j, i, r] = radial embedding r for j-th neighbor of atom i
    # Ylm_3[j, i, l] = angular embedding l for j-th neighbor of atom i
    Rnl_3 = zeros(T, max_neigs, n_atoms, n_rnl)
    Ylm_3 = zeros(T, max_neigs, n_atoms, n_ylm)

    # Track current neighbor index per atom
    neig_idx = zeros(Int, n_atoms)

    for e in 1:n_edges
        i = edge_i[e]
        j_atom = edge_j[e]

        # Get edge vector and compute distance
        rij = SVector{3,T}(edge_rij[e, 1], edge_rij[e, 2], edge_rij[e, 3])
        r = norm(rij)

        # Skip if outside cutoff
        r > state.rcut && continue

        # Compute unit vector (avoid division by zero)
        rhat = r > eps(T) ? rij / r : SVector{3,T}(zero(T), zero(T), one(T))

        # Get species indices
        zi = z_to_species_index(Int(atomic_numbers[i]), state.species_Z)
        zj = z_to_species_index(Int(atomic_numbers[j_atom]), state.species_Z)

        # Compute radial embedding
        Rnl = compute_radial_embedding(r, zi, zj, state)

        # Compute angular embedding (solid harmonics: r^l * Y_lm)
        Ylm = compute_solid_harmonics_reactant(r, rhat, state.maxl)

        # Store in 3D tensors
        neig_idx[i] += 1
        idx = neig_idx[i]
        for r_idx in 1:n_rnl
            Rnl_3[idx, i, r_idx] = Rnl[r_idx]
        end
        for l_idx in 1:n_ylm
            Ylm_3[idx, i, l_idx] = Ylm[l_idx]
        end
    end

    # Step 3: Run ACE kernel
    BB, _, _ = ace_evaluate_reactant(Rnl_3, Ylm_3,
                                      state.spec_R, state.spec_Y,
                                      state.specs_mats, state.A2Bmap)

    # Step 4: Compute site energies and sum
    energy = zero(T)
    for i in 1:n_atoms
        zi = z_to_species_index(Int(atomic_numbers[i]), state.species_Z)
        # Site energy: BB[i, :] · W_readout[:, zi]
        for b in 1:state.n_basis
            energy += BB[i, b] * state.W_readout[b, zi]
        end
    end

    return energy
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
