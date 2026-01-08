#=
ReactantETACEState - State conversion for Reactant compilation
=============================================================

Converts ETACE model state to Reactant-compatible format:
- Spec tuples → matrices (Reactant can't trace tuples)
- SparseMatCSX → dense matrices
- Extracts all parameters into flat arrays
=#

import EquivariantTensors as ET
import Polynomials4ML as P4ML
using SparseArrays: findnz, AbstractSparseMatrix
using LinearAlgebra: norm

## ============================================================================
## Data Structure Conversion Utilities
## ============================================================================

"""
    spec_to_matrix(spec::Vector{<:Tuple})

Convert a vector of tuples (spec for PooledSparseProduct or SparseSymmProd)
to a matrix for Reactant-compatible indexing.

Input:  Vector of N-tuples, e.g., [(1,2), (3,4), (5,6)]
Output: Matrix of size (length(spec), N), e.g., [1 2; 3 4; 5 6]
"""
function spec_to_matrix(spec::Vector{<:Tuple})
    isempty(spec) && return zeros(Int, 0, 0)
    N = length(spec[1])
    mat = zeros(Int, length(spec), N)
    for (i, ϕ) in enumerate(spec)
        for j in 1:N
            mat[i, j] = ϕ[j]
        end
    end
    return mat
end

"""
    sparse_to_dense(m::ET.SparseMatCSX)

Convert a SparseMatCSX to a dense matrix.
For typical ACE coupling matrices, this is acceptable since they are small.
"""
function sparse_to_dense(m::ET.SparseMatCSX)
    dense = zeros(eltype(m.nzval_csr), m.m, m.n)
    for row in 1:m.m
        for idx in m.rowptr[row]:(m.rowptr[row+1]-1)
            col = m.colval[idx]
            dense[row, col] = m.nzval_csr[idx]
        end
    end
    return dense
end

"""
    sparse_to_dense(m::AbstractSparseMatrix)

Convert Julia's SparseMatrixCSC to dense.
"""
function sparse_to_dense(m::AbstractSparseMatrix)
    return Matrix(m)
end

## ============================================================================
## ReactantETACEState - Complete state for Reactant compilation
## ============================================================================

"""
    ReactantETACEState{T}

Pre-processed ETACE state with all data converted to Reactant-compatible formats.

This structure contains everything needed to evaluate an ETACE model:
- ACE tensor specifications and coupling matrices
- Radial basis parameters (Agnesi transform + Chebyshev + linear layer)
- Spherical harmonics configuration
- Readout layer weights
- Reference energies (E0)

All tuple-based specs are converted to matrices, and sparse matrices to dense.
"""
struct ReactantETACEState{T}
    # Species configuration
    n_species::Int
    species_Z::Vector{Int}           # Atomic numbers for each species index
    rcut::T                          # Cutoff radius

    # ACE tensor specifications (from SparseACEbasis state)
    spec_R::Vector{Int}              # Radial indices for pooled product
    spec_Y::Vector{Int}              # Angular indices for pooled product
    specs_mats::Vector{Matrix{Int}}  # Per-order index matrices for symmetric product
    A2Bmap::Matrix{T}                # Dense coupling matrix (L=0 only for scalar models)

    # Radial basis parameters
    n_polys::Int                     # Number of polynomial basis functions
    n_rnl::Int                       # Number of (n,l) radial basis functions
    agnesi_params::Matrix{T}         # (7, n_pairs) - pin, pcut, a, b0, b1, rin, req per pair
    poly_A::Vector{T}                # Chebyshev recurrence coefficient A
    poly_B::Vector{T}                # Chebyshev recurrence coefficient B
    poly_C::Vector{T}                # Chebyshev recurrence coefficient C
    W_radial::Array{T,3}             # (n_rnl, n_polys, n_pairs) linear layer weights

    # Angular basis
    maxl::Int                        # Maximum angular momentum
    nYlm::Int                        # Number of spherical harmonics = (maxl+1)^2

    # Readout
    n_basis::Int                     # Number of ACE basis functions
    W_readout::Matrix{T}             # (n_basis, n_species) readout weights
    E0::Vector{T}                    # (n_species,) reference energies
end

## ============================================================================
## Extraction from pure EquivariantTensors basis state
## ============================================================================

"""
    prepare_reactant_state(st::NamedTuple)

Convert raw ACE basis state (from SparseACEbasis) to Reactant-compatible format.
This is for use with pure EquivariantTensors models (not full ACEpotentials).

Returns a NamedTuple with: spec_R, spec_Y, specs_mats, A2Bmaps_dense
"""
function prepare_reactant_state(st::NamedTuple)
    spec_R = [s[1] for s in st.aspec]
    spec_Y = [s[2] for s in st.aspec]
    specs_mats = [spec_to_matrix(Vector(s)) for s in st.aaspecs]
    A2Bmaps_dense = [sparse_to_dense(m) for m in st.A2Bmaps]

    return (
        spec_R = spec_R,
        spec_Y = spec_Y,
        specs_mats = specs_mats,
        A2Bmaps_dense = A2Bmaps_dense
    )
end

## ============================================================================
## Extraction from full ETACEPotential
## ============================================================================

# Import ETModels types for dispatch
import ACEpotentials.ETModels: ETACE, ETACEPotential, WrappedSiteCalculator

"""
    prepare_reactant_state(calc::ETACEPotential; T=Float32)

Convert an ETACEPotential to ReactantETACEState for Reactant compilation.

Extracts parameters from all Lux layers:
- rembed: radial embedding (Agnesi + Chebyshev + linear)
- yembed: angular embedding (spherical harmonics)
- basis: SparseACEbasis (coupling coefficients)
- readout: SelectLinL (species-aware linear layer)
"""
function prepare_reactant_state(calc::WrappedSiteCalculator{<:ETACE}; T::Type=Float32)
    model = calc.model
    ps = calc.ps
    st = calc.st
    rcut = T(calc.rcut)

    # -------------------------------------------------------------------------
    # Extract species info from readout layer and Agnesi transform state
    # -------------------------------------------------------------------------
    n_species = model.readout.ncat

    # Extract species_Z from Agnesi transform state
    # The zlist is stored in st.rembed.trans.zlist as a tuple of ChemicalSpecies
    species_Z = try
        rembed_st = st.rembed
        trans_st = rembed_st.trans
        if hasproperty(trans_st, :zlist)
            zlist = trans_st.zlist
            [Int(z.atomic_number) for z in zlist]
        else
            collect(1:n_species)
        end
    catch e
        @warn "Could not extract species_Z from state" exception=e
        collect(1:n_species)
    end

    # -------------------------------------------------------------------------
    # Extract ACE tensor state from basis layer
    # -------------------------------------------------------------------------
    basis_st = st.basis
    spec_R = [s[1] for s in basis_st.aspec]
    spec_Y = [s[2] for s in basis_st.aspec]
    specs_mats = [spec_to_matrix(Vector(s)) for s in basis_st.aaspecs]

    # For L=0 scalar models, we only have one A2Bmap
    A2Bmap = T.(sparse_to_dense(basis_st.A2Bmaps[1]))

    # -------------------------------------------------------------------------
    # Extract radial basis parameters
    # -------------------------------------------------------------------------
    # The rembed layer structure is: EdgeEmbed(EmbedDP(trans, polys, linl))
    # trans: NTtransformST with Agnesi parameters
    # polys: wrapped Chebyshev basis with envelope
    # linl: SelectLinL with weights W

    # Number of Chebyshev polynomials
    # Access through the rembed structure
    n_polys = _extract_n_polys(model.rembed)

    # Get n_rnl from the actual radial weight matrix shape (not from spec length!)
    # ps.rembed.post.W has shape (n_rnl, n_polys, n_pairs)
    n_rnl = _extract_n_rnl(ps.rembed)

    # Number of species pairs
    n_pairs = n_species * n_species

    # Extract Agnesi parameters from rembed.basis.trans
    agnesi_params = _extract_agnesi_params(model.rembed, st.rembed, n_pairs, T)

    # Extract Chebyshev recurrence coefficients from state
    poly_A, poly_B, poly_C = _extract_chebyshev_coeffs(st.rembed, n_polys, T)

    # Extract radial linear layer weights from ps.rembed.post.W
    # Shape: (n_rnl, n_polys, n_pairs)
    W_radial = _extract_radial_weights(ps.rembed, n_rnl, n_polys, n_pairs, T)

    # -------------------------------------------------------------------------
    # Extract angular basis parameters
    # -------------------------------------------------------------------------
    maxl = _extract_maxl(model.yembed)
    nYlm = (maxl + 1)^2

    # -------------------------------------------------------------------------
    # Extract readout weights
    # -------------------------------------------------------------------------
    n_basis = model.readout.in_dim
    # Readout W has shape (1, n_basis, n_species), we want (n_basis, n_species)
    W_readout = T.(dropdims(ps.readout.W, dims=1))

    # Reference energies - need to come from one-body model in StackedCalculator
    E0 = zeros(T, n_species)

    return ReactantETACEState{T}(
        n_species, species_Z, rcut,
        spec_R, spec_Y, specs_mats, A2Bmap,
        n_polys, n_rnl, agnesi_params, poly_A, poly_B, poly_C, W_radial,
        maxl, nYlm,
        n_basis, W_readout, E0
    )
end

## ============================================================================
## Parameter extraction helpers
## ============================================================================

"""
Extract n_rnl (number of radial basis outputs) from rembed parameters.
This comes from the shape of ps.rembed.post.W which is (n_rnl, n_polys, n_pairs).
"""
function _extract_n_rnl(ps_rembed)
    try
        if hasproperty(ps_rembed, :post) && hasproperty(ps_rembed.post, :W)
            return size(ps_rembed.post.W, 1)
        end
    catch
    end
    # Default fallback
    return 10
end

"""
Extract number of polynomials from rembed layer.
"""
function _extract_n_polys(rembed)
    # Navigate through EdgeEmbed -> EmbedDP -> basis structure
    try
        # EdgeEmbed wraps EmbedDP in a `layer` field
        inner = hasproperty(rembed, :layer) ? rembed.layer : rembed.basis
        if hasproperty(inner, :post)
            # SelectLinL structure - get from input dimension
            return inner.post.in_dim
        elseif hasproperty(inner, :basis) && hasproperty(inner.basis, :layers)
            # BranchLayer structure
            polys = inner.basis.layers[1]
            return length(polys)
        end
    catch
    end
    # Default fallback
    return 10
end

"""
Extract Agnesi transform parameters from rembed state.
Returns matrix of shape (7, n_pairs) with [pin, pcut, a, b0, b1, rin, req] per pair.
These are the full parameters needed for the generalized Agnesi transform.
"""
function _extract_agnesi_params(rembed, rembed_st, n_pairs, T)
    params = zeros(T, 7, n_pairs)

    try
        # State structure: rembed_st.trans.params is an SVector of NamedTuples
        # Each NamedTuple has (pin, pcut, a, b0, b1, rin, req)
        trans_st = rembed_st.trans
        if hasproperty(trans_st, :params)
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
        end
    catch e
        @warn "Could not extract Agnesi params" exception=e
        # Use reasonable defaults (computed for typical Si parameters)
        # pin=2, pcut=2, a computed for max slope at req, b0/b1 for [-1,1] normalization
        for idx in 1:n_pairs
            params[1, idx] = T(2)     # pin
            params[2, idx] = T(2)     # pcut
            params[3, idx] = T(0.5)   # a (typical value)
            params[4, idx] = T(-1.0)  # b0
            params[5, idx] = T(2.0)   # b1
            params[6, idx] = T(0.5)   # rin
            params[7, idx] = T(2.5)   # req
        end
    end

    return params
end

"""
Extract Chebyshev recurrence coefficients from rembed state.
The orthonormalized Chebyshev coefficients are stored in st.rembed.basis.layers.layer_1
"""
function _extract_chebyshev_coeffs(rembed_st, n_polys, T)
    # Standard Chebyshev fallback: T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
    A = fill(T(2), n_polys)
    A[1] = T(1)
    B = zeros(T, n_polys)
    C = fill(T(-1), n_polys)
    C[1] = T(0)

    # Try to extract actual coefficients from the polynomial state
    try
        # State structure: rembed_st.basis.layers.layer_1 contains (A, B, C)
        if hasproperty(rembed_st, :basis) && hasproperty(rembed_st.basis, :layers)
            layers = rembed_st.basis.layers
            # layer_1 contains the polynomial coefficients
            if hasproperty(layers, :layer_1) && hasproperty(layers.layer_1, :A)
                polys = layers.layer_1
                A = T.(polys.A[1:n_polys])
                B = T.(polys.B[1:n_polys])
                C = T.(polys.C[1:n_polys])
            end
        end
    catch e
        @warn "Could not extract Chebyshev coefficients, using standard" exception=e
    end

    return A, B, C
end

"""
Extract radial linear layer weights.
Shape: (n_rnl, n_polys, n_pairs)
"""
function _extract_radial_weights(ps_rembed, n_rnl, n_polys, n_pairs, T)
    W = zeros(T, n_rnl, n_polys, n_pairs)

    try
        # ps_rembed.post.W has shape (n_rnl, n_polys, n_pairs)
        if hasproperty(ps_rembed, :post) && hasproperty(ps_rembed.post, :W)
            W_raw = ps_rembed.post.W
            # Copy with type conversion
            for i in axes(W_raw, 1), j in axes(W_raw, 2), k in axes(W_raw, 3)
                if i <= n_rnl && j <= n_polys && k <= n_pairs
                    W[i, j, k] = T(W_raw[i, j, k])
                end
            end
        end
    catch e
        @warn "Could not extract radial weights" exception=e
    end

    return W
end

"""
Extract maximum angular momentum from yembed layer.
"""
function _extract_maxl(yembed)
    try
        # Navigate through EdgeEmbed -> EmbedDP -> basis
        inner = yembed.basis
        if hasproperty(inner, :basis)
            ylm_basis = inner.basis
            # Real spherical harmonics have (maxl+1)^2 functions
            nylm = length(ylm_basis)
            maxl = Int(sqrt(nylm)) - 1
            return maxl
        end
    catch
    end
    # Default fallback
    return 2
end

## ============================================================================
## Utility functions
## ============================================================================

"""
    zz_to_pair_index(iz, jz, n_species)

Convert species pair (iz, jz) to linear pair index.
"""
function zz_to_pair_index(iz::Int, jz::Int, n_species::Int)
    return (iz - 1) * n_species + jz
end

"""
    zz_to_pair_index_sym(iz, jz, n_species)

Convert species pair (iz, jz) to symmetric pair index.
Uses upper triangular indexing: pair(i,j) = pair(j,i).
"""
function zz_to_pair_index_sym(iz::Int, jz::Int, n_species::Int)
    i, j = minmax(iz, jz)
    return (i - 1) * n_species - (i - 1) * i ÷ 2 + j
end

"""
    z_to_species_index(Z, species_Z::AbstractVector)

Convert atomic number Z to species index (1-based).
"""
function z_to_species_index(Z::Integer, species_Z::AbstractVector{<:Integer})
    idx = findfirst(==(Int(Z)), species_Z)
    isnothing(idx) && error("Unknown atomic number: $Z")
    return idx
end
