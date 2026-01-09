#=
Reactant-Compatible ACE Kernel
==============================

Pure Julia implementations of the ACE evaluation pipeline that can be
traced by Reactant. Replaces KernelAbstractions-based implementations.

Pipeline:
1. PooledSparseProduct: (Rnl, Ylm) → A
2. SparseSymmProd: A → AA
3. Coupling: AA → BB (coupling coefficients)
=#

## ============================================================================
## PooledSparseProduct - Reactant-compatible implementation
## ============================================================================

"""
    pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)

Reactant-compatible pooled sparse product using gather + vectorized sum.

Computes: A[inode, iA] = Σⱼ Rnl[j, inode, spec_R[iA]] * Ylm[j, inode, spec_Y[iA]]

This replaces the KernelAbstractions kernel `_ka_evaluate_PooledSparseProduct_batched_v1!`

Uses vectorized gather operations instead of scalar indexing for Reactant compatibility.

# Arguments
- `Rnl_3`: (maxneigs, nnodes, nRnl) radial embeddings
- `Ylm_3`: (maxneigs, nnodes, nYlm) angular embeddings
- `spec_R`: Vector of radial indices
- `spec_Y`: Vector of angular indices

# Returns
- `A`: (nnodes, nA) pooled features
"""
function pooled_sparse_product_reactant(Rnl_3::AbstractArray{T,3},
                                         Ylm_3::AbstractArray{T,3},
                                         spec_R::AbstractVector{Int},
                                         spec_Y::AbstractVector{Int}) where T
    maxneigs, nnodes, nRnl = size(Rnl_3)
    nA = length(spec_R)

    # Gather: Rnl_gathered[j, inode, iA] = Rnl_3[j, inode, spec_R[iA]]
    # Using advanced indexing: Rnl_3[:, :, spec_R] gives [maxneigs, nnodes, nA]
    Rnl_gathered = Rnl_3[:, :, spec_R]
    Ylm_gathered = Ylm_3[:, :, spec_Y]

    # Elementwise product
    prod = Rnl_gathered .* Ylm_gathered

    # Sum over first dimension (neighbors)
    A = dropdims(sum(prod, dims=1), dims=1)  # [nnodes, nA]

    return A
end

## ============================================================================
## SparseSymmProd - Reactant-compatible implementation
## ============================================================================

"""
    sparse_symm_prod_order(A, spec_mat, order)

Compute symmetric product for a single correlation order using vectorized gather.

For order=0: AA[inode, i] = 1
For order=1: AA[inode, i] = A[inode, spec_mat[i, 1]]
For order=2: AA[inode, i] = A[inode, spec_mat[i, 1]] * A[inode, spec_mat[i, 2]]
etc.

# Arguments
- `A`: (nnodes, nA) pooled features
- `spec_mat`: (nspec, order) index matrix
- `order`: Correlation order

# Returns
- `AA_order`: (nnodes, nspec) symmetric products for this order
"""
function sparse_symm_prod_order(A::AbstractMatrix{T}, spec_mat::AbstractMatrix{Int}, order::Int) where T
    nnodes = size(A, 1)
    nspec = size(spec_mat, 1)

    if order == 0
        # Order 0: constant 1
        return ones(T, nnodes, nspec)
    end

    # First term: A[:, spec_mat[:, 1]] gives [nnodes, nspec]
    prod = A[:, spec_mat[:, 1]]

    # Multiply by remaining terms
    for t in 2:order
        prod = prod .* A[:, spec_mat[:, t]]
    end

    return prod
end

"""
    sparse_symm_prod_reactant(A, specs_mats)

Reactant-compatible sparse symmetric product using vectorized gather.

Computes: AA[inode, offset+i] = Πₜ A[inode, specs_mats[ord][i, t]]

This replaces the KernelAbstractions kernel `_ka_evaluate_SparseSymmProd_batched_v1!`

# Arguments
- `A`: (nnodes, nA) pooled features
- `specs_mats`: Vector of (nspec, order) index matrices, one per correlation order

# Returns
- `AA`: (nnodes, nAA) concatenated symmetric products
"""
function sparse_symm_prod_reactant(A::AbstractMatrix{T}, specs_mats::Vector{<:AbstractMatrix{Int}}) where T
    nnodes = size(A, 1)

    # Process each order and concatenate
    AA_parts = [sparse_symm_prod_order(A, spec_mat, size(spec_mat, 2)) for spec_mat in specs_mats]

    # Concatenate along second dimension
    AA = hcat(AA_parts...)

    return AA
end

## ============================================================================
## Full ACE Evaluation
## ============================================================================

"""
    ace_evaluate_reactant(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)

Full Reactant-compatible ACE basis evaluation.

Pipeline:
1. Pooled sparse product: (Rnl, Ylm) → A
2. Sparse symmetric product: A → AA
3. Coupling coefficient application: AA → BB

# Arguments
- `Rnl_3`: (maxneigs, nnodes, nRnl) radial embeddings
- `Ylm_3`: (maxneigs, nnodes, nYlm) angular embeddings
- `spec_R`, `spec_Y`: Indices for pooled product
- `specs_mats`: Index matrices for symmetric products
- `A2Bmap`: Dense coupling matrix

# Returns
- `BB`: (nnodes, nfeatures) ACE basis array
- `A`: (nnodes, nA) intermediate pooled features
- `AA`: (nnodes, nAA) intermediate symmetric products
"""
function ace_evaluate_reactant(Rnl_3::AbstractArray{T,3},
                                Ylm_3::AbstractArray{T,3},
                                spec_R::AbstractVector{Int},
                                spec_Y::AbstractVector{Int},
                                specs_mats::Vector{<:AbstractMatrix{Int}},
                                A2Bmap::AbstractMatrix{T}) where T
    # Step 1: Pooled sparse product
    A = pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)

    # Step 2: Sparse symmetric product
    AA = sparse_symm_prod_reactant(A, specs_mats)

    # Step 3: Apply coupling coefficients (dense matmul)
    # BB = AA * A2Bmap'  (transpose because A2Bmap is nfeatures x nAA)
    BB = AA * transpose(A2Bmap)

    return BB, A, AA
end
