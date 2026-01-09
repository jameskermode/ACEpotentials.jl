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
## Vectorized Energy Computation (Reactant-compatible)
## ============================================================================
## All functions use broadcasting and matrix ops - no scalar loops.
## For single-species models; multi-species requires additional selection matrices.

"""
    stacked_energy_vectorized(rij, pool_matrix, n_atoms, model, ace_state_params)

Compute total stacked energy (E0 + pair + ACE) using vectorized operations.

All operations use broadcasting/matmul for Reactant compatibility.
Single-species version - all atoms use same parameters.

# Arguments
- `rij`: (n_edges, 3) edge displacement vectors
- `pool_matrix`: (n_atoms, n_edges) pooling matrix (1 where edge starts from atom)
- `n_atoms`: number of atoms (for E0 computation)
- `model`: ReactantStackedModel with E0, pair_state, has_pair
- `ace_state_params`: Pre-extracted ACE parameters (see extract_ace_params)

# Returns
- Total energy (scalar)
"""
function stacked_energy_vectorized(
    rij::AbstractMatrix{T},
    pool_matrix::AbstractMatrix{T},
    n_atoms,
    model::ReactantStackedModel{T},
    ace_params::NamedTuple
) where T
    # 1. One-body: E0[1] * n_atoms (single species)
    E_onebody = model.E0[1] * n_atoms

    # 2. Pair energy (vectorized)
    E_pair = if model.has_pair && model.pair_state !== nothing
        compute_pair_energy_vectorized(rij, model.pair_state)
    else
        zero(T)
    end

    # 3. ACE many-body (selection matrix approach)
    E_ace = compute_ace_energy_vectorized(rij, pool_matrix, ace_params)

    return E_onebody + E_pair + E_ace
end

"""Vectorized pair energy for single-species models."""
function compute_pair_energy_vectorized(
    rij::AbstractMatrix{T},
    pair::ReactantPairState{T}
) where T
    # Distance computation
    r2 = sum(rij .^ 2, dims=2)
    r_vec = sqrt.(dropdims(r2, dims=2))

    # Agnesi transform (single species: pair_idx=1)
    a = pair.agnesi_params[3, 1]
    b0 = pair.agnesi_params[4, 1]
    b1 = pair.agnesi_params[5, 1]
    rin = pair.agnesi_params[6, 1]
    req = pair.agnesi_params[7, 1]

    s = (r_vec .- rin) ./ (req .- rin .+ T(1e-10))
    x = one(T) ./ (one(T) .+ a .* s .^ 2 ./ T(2))
    y = b1 .* x .+ b0
    y = max.(-one(T), min.(one(T), y))

    # Chebyshev basis (no inner envelope for pair)
    n_polys = pair.n_polys
    P = chebyshev_basis_vectorized(y, n_polys, pair.poly_A, pair.poly_B, pair.poly_C)

    # Linear layer: P @ W_radial[:,:,1]^T -> [n_edges, n_basis]
    pair_features = P * transpose(pair.W_radial[:, :, 1])

    # Outer envelope: (s^(-p) - 1) * (1 - s)
    s_outer = r_vec ./ pair.rcut_outer
    s_safe = max.(T(1e-6), min.(s_outer, T(0.9999)))
    outer_env = (s_safe .^ (-pair.p_outer) .- one(T)) .* (one(T) .- s_safe)
    outer_env = outer_env .* (s_outer .< one(T))  # Zero beyond cutoff

    pair_features_env = pair_features .* reshape(outer_env, :, 1)

    # Readout: sum over edges and basis (single species: W_readout[:,1])
    return sum(pair_features_env * pair.W_readout[:, 1])
end

"""Vectorized ACE energy using selection matrices."""
function compute_ace_energy_vectorized(
    rij::AbstractMatrix{T},
    pool_matrix::AbstractMatrix{T},
    p::NamedTuple  # Pre-extracted parameters
) where T
    # Distance and direction
    r2 = sum(rij .^ 2, dims=2)
    r = sqrt.(r2)
    r_vec = dropdims(r, dims=2)
    eps_val = T(1e-6)
    rhat = rij ./ max.(r, eps_val)

    # Radial embedding with Agnesi + envelope + W_radial
    Rnl = radial_embedding_vectorized(r_vec, p.agnesi_a, p.agnesi_b0, p.agnesi_b1,
                                       p.agnesi_rin, p.agnesi_req, p.n_polys,
                                       p.poly_A, p.poly_B, p.poly_C, p.W_radial)

    # Solid harmonics
    Ylm = solid_harmonics_vectorized(r_vec, rhat, p.maxl)

    # Pooled sparse product via selection matrices
    Rnl_sel = Rnl * transpose(p.selector_R)
    Ylm_sel = Ylm * transpose(p.selector_Y)
    A_edge = Rnl_sel .* Ylm_sel
    A = pool_matrix * A_edge  # Pool to atoms

    # Sparse symmetric product
    AA1 = A * transpose(p.symm_sel1)
    AA2 = (A * transpose(p.symm_sel2_1)) .* (A * transpose(p.symm_sel2_2))
    AA = hcat(AA1, AA2)

    # Coupling and readout
    BB = AA * transpose(p.A2Bmap)
    return sum(BB * p.W_readout)
end

"""Vectorized Chebyshev basis evaluation."""
function chebyshev_basis_vectorized(y::AbstractVector{T}, n_polys::Int,
                                     poly_A, poly_B, poly_C) where T
    P0 = y .* zero(T) .+ T(poly_A[1])
    n_polys == 1 && return reshape(P0, :, 1)

    P1 = T(poly_A[2]) .* y .+ T(poly_B[2])
    result = hcat(reshape(P0, :, 1), reshape(P1, :, 1))
    Pnm2, Pnm1 = P0, P1

    for n in 3:n_polys
        Pn = (T(poly_A[n]) .* y .+ T(poly_B[n])) .* Pnm1 .+ T(poly_C[n]) .* Pnm2
        result = hcat(result, reshape(Pn, :, 1))
        Pnm2, Pnm1 = Pnm1, Pn
    end
    return result
end

"""Vectorized radial embedding: Agnesi + envelope + W_radial."""
function radial_embedding_vectorized(r_vec::AbstractVector{T},
    agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
    n_polys, poly_A, poly_B, poly_C, W_radial::AbstractMatrix{T}) where T

    # Agnesi transform
    s = (r_vec .- agnesi_rin) ./ (agnesi_req .- agnesi_rin .+ T(1e-10))
    x = one(T) ./ (one(T) .+ agnesi_a .* s .^ 2 ./ T(2))
    y = agnesi_b1 .* x .+ agnesi_b0
    y = max.(-one(T), min.(one(T), y))

    # Inner envelope (1 - y²)²
    env = (one(T) .- y .^ 2) .^ 2

    # Chebyshev basis
    P = chebyshev_basis_vectorized(y, n_polys, poly_A, poly_B, poly_C)
    P_env = P .* reshape(env, :, 1)

    # Linear layer: P_env @ W_radial^T
    return P_env * transpose(W_radial)
end

"""Vectorized solid harmonics (r^l * Y_lm) for maxl ≤ 2."""
function solid_harmonics_vectorized(r_vec::AbstractVector{T},
                                     rhat::AbstractMatrix{T}, maxl::Int) where T
    rx = r_vec .* rhat[:, 1]
    ry = r_vec .* rhat[:, 2]
    rz = r_vec .* rhat[:, 3]

    # l=0
    Y00 = r_vec .* zero(T) .+ T(0.28209479177387814)
    maxl == 0 && return reshape(Y00, :, 1)

    # l=1
    c1 = T(0.4886025119029199)
    Y1m1 = c1 .* ry
    Y10 = c1 .* rz
    Y1p1 = c1 .* rx
    maxl == 1 && return hcat(reshape(Y00, :, 1), reshape(Y1m1, :, 1),
                             reshape(Y10, :, 1), reshape(Y1p1, :, 1))

    # l=2
    c2_0, c2_1, c2_2 = T(0.31539156525252005), T(1.0925484305920792), T(0.5462742152960396)
    r2 = r_vec .^ 2
    Y2m2 = c2_1 .* rx .* ry
    Y2m1 = c2_1 .* ry .* rz
    Y20 = c2_0 .* (T(3) .* rz .^ 2 .- r2)
    Y2p1 = c2_1 .* rx .* rz
    Y2p2 = c2_2 .* (rx .^ 2 .- ry .^ 2)

    return hcat(reshape(Y00, :, 1), reshape(Y1m1, :, 1), reshape(Y10, :, 1), reshape(Y1p1, :, 1),
                reshape(Y2m2, :, 1), reshape(Y2m1, :, 1), reshape(Y20, :, 1), reshape(Y2p1, :, 1), reshape(Y2p2, :, 1))
end

"""Extract ACE parameters into a NamedTuple for vectorized computation."""
function extract_ace_params(ace::ReactantETACEState{T}) where T
    # Build selection matrices
    nRnl, nYlm = ace.n_rnl, ace.nYlm
    spec_R, spec_Y = Int.(ace.spec_R), Int.(ace.spec_Y)
    nA = length(spec_R)

    selector_R = zeros(T, nA, nRnl)
    selector_Y = zeros(T, nA, nYlm)
    for k in 1:nA
        selector_R[k, spec_R[k]] = one(T)
        selector_Y[k, spec_Y[k]] = one(T)
    end

    # Symmetric product selectors
    specs_mats = ace.specs_mats
    symm_sel1 = if length(specs_mats) >= 1 && size(specs_mats[1], 1) > 0
        sel = zeros(T, size(specs_mats[1], 1), nA)
        for (k, idx) in enumerate(specs_mats[1][:, 1]); sel[k, idx] = one(T); end
        sel
    else
        zeros(T, 1, nA)
    end

    symm_sel2_1, symm_sel2_2 = if length(specs_mats) >= 2 && size(specs_mats[2], 1) > 0
        n2 = size(specs_mats[2], 1)
        s1, s2 = zeros(T, n2, nA), zeros(T, n2, nA)
        for (k, row) in enumerate(eachrow(specs_mats[2]))
            s1[k, row[1]] = one(T)
            s2[k, row[2]] = one(T)
        end
        s1, s2
    else
        zeros(T, 1, nA), zeros(T, 1, nA)
    end

    # Embedding params (single species: pair_idx=1)
    pair_idx = 1
    return (
        selector_R = selector_R,
        selector_Y = selector_Y,
        symm_sel1 = symm_sel1,
        symm_sel2_1 = symm_sel2_1,
        symm_sel2_2 = symm_sel2_2,
        A2Bmap = T.(ace.A2Bmap),
        W_readout = T.(ace.W_readout[:, 1]),
        n_polys = ace.n_polys,
        maxl = ace.maxl,
        agnesi_a = T(ace.agnesi_params[3, pair_idx]),
        agnesi_b0 = T(ace.agnesi_params[4, pair_idx]),
        agnesi_b1 = T(ace.agnesi_params[5, pair_idx]),
        agnesi_rin = T(ace.agnesi_params[6, pair_idx]),
        agnesi_req = T(ace.agnesi_params[7, pair_idx]),
        poly_A = T.(ace.poly_A),
        poly_B = T.(ace.poly_B),
        poly_C = T.(ace.poly_C),
        W_radial = T.(ace.W_radial[:, :, pair_idx]),
    )
end

"""Build pool matrix from edge indices."""
function build_pool_matrix(edge_i::AbstractVector, n_atoms::Int, n_edges::Int, ::Type{T}) where T
    pool = zeros(T, n_atoms, n_edges)
    for e in 1:n_edges
        pool[edge_i[e], e] = one(T)
    end
    return pool
end

## ============================================================================
## Force and Virial Assembly (from edge gradients)
## ============================================================================

"""
    assemble_forces_from_edge_gradients!(forces, d_edge_rij, edge_i, edge_j, n_edges)

Assemble atomic forces from edge vector gradients.

Given ∂E/∂r_ij for each edge, compute forces on each atom:
    F_i = Σ_{e: edge_i[e]=i} ∂E/∂r_e - Σ_{e: edge_j[e]=i} ∂E/∂r_e

This follows from the chain rule: since r_ij = r_j - r_i,
∂r_ij/∂r_i = -I and ∂r_ij/∂r_j = +I.

# Arguments
- `forces`: (n_atoms, 3) output force array (modified in place)
- `d_edge_rij`: (n_edges, 3) gradients ∂E/∂r_ij
- `edge_i`, `edge_j`: (n_edges,) edge indices
- `n_edges`: number of edges
"""
function assemble_forces_from_edge_gradients!(forces, d_edge_rij, edge_i, edge_j, n_edges)
    fill!(forces, zero(eltype(forces)))
    for e in 1:n_edges
        i = edge_i[e]
        j = edge_j[e]
        for d in 1:3
            forces[i, d] += d_edge_rij[e, d]
            forces[j, d] -= d_edge_rij[e, d]
        end
    end
    return forces
end

"""
    compute_virial_from_edge_gradients(edge_rij, d_edge_rij, n_edges)

Compute virial tensor from edge vectors and their gradients.

The virial is:
    V_ab = -Σ_e r_e[a] * (∂E/∂r_e)[b]

This follows from the standard definition V_ab = -Σ_{i<j} r_ij[a] * f_ij[b]
where f_ij is the pair force.

# Arguments
- `edge_rij`: (n_edges, 3) edge vectors
- `d_edge_rij`: (n_edges, 3) gradients ∂E/∂r_ij
- `n_edges`: number of edges

# Returns
- (3, 3) virial tensor
"""
function compute_virial_from_edge_gradients(edge_rij, d_edge_rij, n_edges)
    T = eltype(edge_rij)
    virial = zeros(T, 3, 3)
    for e in 1:n_edges
        for a in 1:3
            for b in 1:3
                virial[a, b] -= edge_rij[e, a] * d_edge_rij[e, b]
            end
        end
    end
    return virial
end

## ============================================================================
## EFV (Energy, Forces, Virial) computation
## ============================================================================

"""
    stacked_efv_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                           n_atoms, n_edges, model::ReactantStackedModel)

Compute energy, forces, and virial from edge vectors.

**For Reactant compilation**: This function provides the forward energy computation.
Forces and virial are computed via Reactant's built-in StableHLO autodiff during
compilation. The compiled VMFB will include differentiated versions automatically.

**For Julia/testing**: Use finite differences in tests, or obtain edge gradients
from Zygote/Enzyme and call `assemble_forces_from_edge_gradients!` and
`compute_virial_from_edge_gradients`.

The force/virial assembly formulas are:
- Forces: F_i = Σ_{e: edge_i[e]=i} ∂E/∂r_e - Σ_{e: edge_j[e]=i} ∂E/∂r_e
- Virial: V_ab = -Σ_e r_e[a] * (∂E/∂r_e)[b]

# Returns
- (energy, forces, virial) tuple
  - energy: scalar total energy
  - forces: (n_atoms, 3) zeros (placeholder - computed by AD in compiled model)
  - virial: (3, 3) zeros (placeholder - computed by AD in compiled model)
"""
function stacked_efv_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                 n_atoms::Int32, n_edges::Int32,
                                 model::ReactantStackedModel)
    T = eltype(edge_rij)

    # Compute energy (this is what Reactant will differentiate)
    energy = stacked_energy_from_edges(edge_rij, atomic_numbers, edge_i, edge_j,
                                       n_atoms, n_edges, model)

    # Placeholders - actual forces/virial computed by AD in compiled model
    forces = zeros(T, n_atoms, 3)
    virial = zeros(T, 3, 3)

    return (energy, forces, virial)
end
