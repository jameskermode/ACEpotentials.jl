#=
ReactantETACEState - State conversion for Reactant compilation
=============================================================

Converts ETACE model state to Reactant-compatible format:
- Spec tuples → matrices (Reactant can't trace tuples)
- SparseMatCSX → dense matrices
- Extracts all parameters into flat arrays
=#

import EquivariantTensors as ET

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
    agnesi_params::Matrix{T}         # (5, n_pairs) - pcut, pin, rin, req, rcut per species pair
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
## Extraction from ETACE models
## ============================================================================

"""
    prepare_reactant_state(calc::ETModels.ETACEPotential)

Convert an ETACEPotential to ReactantETACEState for Reactant compilation.

Extracts parameters from all Lux layers:
- rembed: radial embedding (Agnesi + Chebyshev + linear)
- yembed: angular embedding (spherical harmonics)
- basis: SparseACEbasis (coupling coefficients)
- readout: SelectLinL (species-aware linear layer)
"""
function prepare_reactant_state(calc)
    model = calc.model
    ps = calc.ps
    st = calc.st

    # Extract species info
    # TODO: Get from model metadata
    species_Z = collect(1:4)  # Placeholder - need to extract from model
    n_species = length(species_Z)
    rcut = calc.rcut

    # Extract ACE tensor state from basis layer
    basis_st = st.basis
    spec_R = [s[1] for s in basis_st.aspec]
    spec_Y = [s[2] for s in basis_st.aspec]
    specs_mats = [spec_to_matrix(Vector(s)) for s in basis_st.aaspecs]

    # For L=0 scalar models, we only have one A2Bmap
    # Convert to dense matrix
    A2Bmap = sparse_to_dense(basis_st.A2Bmaps[1])

    # Extract radial basis parameters from rembed layer
    # This depends on the specific structure of the ETACE rembed layer
    # TODO: Extract actual parameters from ps.rembed
    n_polys = 10  # Placeholder
    n_rnl = 5     # Placeholder
    n_pairs = n_species * n_species

    # Placeholder - need to extract from actual model
    agnesi_params = zeros(Float64, 5, n_pairs)
    poly_A = ones(Float64, n_polys)
    poly_B = zeros(Float64, n_polys)
    poly_C = zeros(Float64, n_polys)
    W_radial = zeros(Float64, n_rnl, n_polys, n_pairs)

    # Angular basis config
    maxl = 2  # Placeholder - extract from model
    nYlm = (maxl + 1)^2

    # Extract readout weights
    n_basis = size(A2Bmap, 1)
    W_readout = zeros(Float64, n_basis, n_species)  # Placeholder
    E0 = zeros(Float64, n_species)  # Placeholder

    return ReactantETACEState{Float64}(
        n_species, species_Z, rcut,
        spec_R, spec_Y, specs_mats, A2Bmap,
        n_polys, n_rnl, agnesi_params, poly_A, poly_B, poly_C, W_radial,
        maxl, nYlm,
        n_basis, W_readout, E0
    )
end

"""
    prepare_reactant_state(st::NamedTuple)

Convert raw ACE basis state (from SparseACEbasis) to Reactant-compatible format.
This is a lower-level function for direct basis state conversion.
"""
function prepare_reactant_state_from_basis(st)
    spec_R = [s[1] for s in st.aspec]
    spec_Y = [s[2] for s in st.aspec]
    specs_mats = [spec_to_matrix(Vector(s)) for s in st.aaspecs]
    A2Bmaps_dense = [sparse_to_dense(m) for m in st.A2Bmaps]

    return (spec_R, spec_Y, specs_mats, A2Bmaps_dense)
end

## ============================================================================
## Utility functions
## ============================================================================

"""
    zz_to_pair_index(iz, jz, n_species)

Convert species pair (iz, jz) to linear pair index.
Uses symmetric indexing: pair(i,j) = pair(j,i).
"""
function zz_to_pair_index(iz::Int, jz::Int, n_species::Int)
    # For symmetric species pairs, use upper triangular indexing
    i, j = minmax(iz, jz)
    return (i - 1) * n_species + j
end

"""
    z_to_species_index(Z::Int, species_Z::Vector{Int})

Convert atomic number Z to species index (1-based).
"""
function z_to_species_index(Z::Int, species_Z::Vector{Int})
    idx = findfirst(==(Z), species_Z)
    isnothing(idx) && error("Unknown atomic number: $Z")
    return idx
end
