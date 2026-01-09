/**
 * ACE Forces Library
 * ==================
 *
 * Shared library for computing ACE energy and forces.
 * Can be used from LAMMPS (C++) or Python (ctypes).
 *
 * Split Computation Architecture (IREE scatter-free):
 * ====================================================
 *
 * IREE has limitations compiling scatter operations in backward passes.
 * We solve this by splitting computation between host and IREE:
 *
 * ┌─────────────────────────────────────────────────────────────┐
 * │                        HOST CODE                            │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 1. Neighbor list: positions → rij (displacement vectors)   │
 * │ 2. Embeddings: rij → Rnl (Chebyshev), Ylm (spherical harm) │
 * │ 3. Pooled product: A = sum_j Rnl[j,spec_R] * Ylm[j,spec_Y] │
 * │    (gather operations happen here)                          │
 * │ 4. Symmetric product: AA = sparse_symm_prod(A, specs_mats)  │
 * │    (gather/selection operations happen here)                │
 * └─────────────────────────┬───────────────────────────────────┘
 *                           │ AA [n_atoms, nAA]
 *                           ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │              IREE MODULE (GPU-portable)                     │
 * ├─────────────────────────────────────────────────────────────┤
 * │ E, dE/dAA = ace_energy_and_grad(AA, A2Bmap, params)        │
 * │                                                             │
 * │ Operations: matmul, sum only                                │
 * │ NO scatter, NO gather → compiles cleanly in IREE!          │
 * └─────────────────────────┬───────────────────────────────────┘
 *                           │ dE/dAA [n_atoms, nAA]
 *                           ▼
 * ┌─────────────────────────────────────────────────────────────┐
 * │                     HOST CODE                               │
 * ├─────────────────────────────────────────────────────────────┤
 * │ 5. Chain rule: dE/dAA → dE/dA (via sparse_symm_prod adjoint)│
 * │ 6. Chain rule: dE/dA → dE/dRnl, dE/dYlm                    │
 * │ 7. Pair forces: F_ij = -(dE/dRnl * dRnl/dr + dE/dYlm * ...) │
 * │ 8. Accumulate: forces[i] -= F_ij, forces[j] += F_ij        │
 * └─────────────────────────────────────────────────────────────┘
 *
 * Key benefits:
 * - IREE module is GPU-portable (no scatter limitations)
 * - Host code handles all indexing/accumulation (natural loops)
 * - Same VMFB works on CPU and GPU backends
 */

#ifndef ACE_FORCES_H
#define ACE_FORCES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Types and Error Codes
 * ============================================================================ */

/* Opaque model handle */
typedef struct ACEForceModel ACEForceModel;

/* Error codes */
typedef enum {
    ACE_OK = 0,
    ACE_ERROR_NULL_ARG = 1,
    ACE_ERROR_LOAD_FAILED = 2,
    ACE_ERROR_INVOKE_FAILED = 3,
    ACE_ERROR_OUT_OF_MEMORY = 4,
    ACE_ERROR_DIMENSION_MISMATCH = 5,
} ACEError;

/* ============================================================================
 * Model Lifecycle
 * ============================================================================ */

/**
 * Create ACE model from VMFB file.
 *
 * The VMFB should contain a function that takes:
 *   AA [n_atoms, nAA], A2Bmap [nBB, nAA], params [nBB]
 * And returns:
 *   energy (scalar), dE/dAA [n_atoms, nAA]
 *
 * @param vmfb_path   Path to compiled .vmfb file
 * @param device      IREE device ("local-task", "local-sync", "cuda")
 * @return            Model handle or NULL on failure
 */
ACEForceModel* ace_create(const char* vmfb_path, const char* device);

/**
 * Destroy model and free resources.
 */
void ace_destroy(ACEForceModel* model);

/**
 * Get last error message.
 */
const char* ace_get_error(const ACEForceModel* model);

/* ============================================================================
 * IREE-Compiled Functions (GPU-portable, no scatter)
 * ============================================================================ */

/**
 * Compute energy from AA features via IREE.
 *
 * This is the core IREE function: E = sum((AA @ A2Bmap.T) @ params)
 * Only matrix multiplication and summation - no gather/scatter.
 *
 * @param model       Model handle
 * @param AA          Symmetric product features [n_atoms, nAA] (row-major)
 * @param n_atoms     Number of atoms
 * @param nAA         Number of AA features
 * @param A2Bmap      Coupling matrix [nBB, nAA] (row-major)
 * @param nBB         Number of basis functions
 * @param params      Linear parameters [nBB]
 * @return            Energy value (NaN on error)
 */
float ace_energy_from_AA(
    ACEForceModel* model,
    const float* AA, int n_atoms, int nAA,
    const float* A2Bmap, int nBB,
    const float* params
);

/**
 * Compute energy and gradient dE/dAA via IREE.
 *
 * Returns both energy and the gradient of energy w.r.t. AA features.
 * The gradient computation uses only transposed matmuls - no scatter.
 *
 * @param model       Model handle
 * @param AA          Symmetric product features [n_atoms, nAA] (row-major)
 * @param n_atoms     Number of atoms
 * @param nAA         Number of AA features
 * @param A2Bmap      Coupling matrix [nBB, nAA] (row-major)
 * @param nBB         Number of basis functions
 * @param params      Linear parameters [nBB]
 * @param dAA         Output: gradient dE/dAA [n_atoms, nAA] (row-major)
 * @return            Energy value (NaN on error)
 */
float ace_energy_grad_AA(
    ACEForceModel* model,
    const float* AA, int n_atoms, int nAA,
    const float* A2Bmap, int nBB,
    const float* params,
    float* dAA
);

/* ============================================================================
 * Host-Side Embedding Functions
 * ============================================================================ */

/**
 * Compute Chebyshev polynomial embeddings and derivatives.
 *
 * Evaluates Chebyshev polynomials T_n(y) where y = 2*r/rcut - 1.
 *
 * @param r           Distance
 * @param rcut        Cutoff radius
 * @param N_cheb      Number of Chebyshev polynomials
 * @param Rnl         Output: Chebyshev values [N_cheb]
 * @param dRnl_dr     Output: derivatives w.r.t. r [N_cheb] (NULL to skip)
 */
void ace_chebyshev(
    float r, float rcut, int N_cheb,
    float* Rnl, float* dRnl_dr
);

/**
 * Compute real spherical harmonics and derivatives.
 *
 * Uses the real spherical harmonics convention (Ylm with m >= 0 are cos,
 * m < 0 are sin combinations).
 *
 * @param x, y, z     Cartesian displacement vector (will be normalized)
 * @param maxl        Maximum angular momentum
 * @param Ylm         Output: spherical harmonics [(maxl+1)^2]
 * @param dYlm_dx     Output: derivatives w.r.t. x [(maxl+1)^2] (NULL to skip)
 * @param dYlm_dy     Output: derivatives w.r.t. y [(maxl+1)^2] (NULL to skip)
 * @param dYlm_dz     Output: derivatives w.r.t. z [(maxl+1)^2] (NULL to skip)
 */
void ace_ylm(
    float x, float y, float z, int maxl,
    float* Ylm,
    float* dYlm_dx, float* dYlm_dy, float* dYlm_dz
);

/* ============================================================================
 * Host-Side ACE Feature Functions
 * ============================================================================ */

/**
 * Compute pooled sparse product A from embeddings.
 *
 * A[i, k] = sum_j Rnl[j, i, spec_R[k]] * Ylm[j, i, spec_Y[k]]
 *
 * This is a gather + product + sum operation (runs on host).
 *
 * @param Rnl_3       Radial embeddings [maxneigs, n_atoms, nRnl] (row-major)
 * @param Ylm_3       Angular embeddings [maxneigs, n_atoms, nYlm] (row-major)
 * @param maxneigs    Maximum neighbors per atom
 * @param n_atoms     Number of atoms
 * @param nRnl        Number of radial features
 * @param nYlm        Number of angular features
 * @param spec_R      Radial spec indices [nA] (1-indexed Julia style)
 * @param spec_Y      Angular spec indices [nA] (1-indexed Julia style)
 * @param nA          Number of A features
 * @param A           Output: pooled features [n_atoms, nA] (row-major)
 */
void ace_pooled_sparse_product(
    const float* Rnl_3, const float* Ylm_3,
    int maxneigs, int n_atoms, int nRnl, int nYlm,
    const int* spec_R, const int* spec_Y, int nA,
    float* A
);

/**
 * Compute sparse symmetric product AA from A.
 *
 * For each order, computes products of A columns as specified by specs_mats.
 * Order 1: AA[i,k] = A[i, spec[k,0]]
 * Order 2: AA[i,k] = A[i, spec[k,0]] * A[i, spec[k,1]]
 * etc.
 *
 * @param A           Pooled features [n_atoms, nA] (row-major)
 * @param n_atoms     Number of atoms
 * @param nA          Number of A features
 * @param specs_mat1  Order-1 spec matrix [nspec1, 1] (row-major, 1-indexed)
 * @param nspec1      Number of order-1 specs
 * @param specs_mat2  Order-2 spec matrix [nspec2, 2] (row-major, 1-indexed)
 * @param nspec2      Number of order-2 specs
 * @param AA          Output: symmetric product [n_atoms, nspec1+nspec2] (row-major)
 */
void ace_sparse_symm_prod(
    const float* A, int n_atoms, int nA,
    const int* specs_mat1, int nspec1,
    const int* specs_mat2, int nspec2,
    float* AA
);

/* ============================================================================
 * Host-Side Chain Rule (Backward Pass)
 * ============================================================================ */

/**
 * Backward pass through sparse symmetric product.
 *
 * Given dE/dAA, compute dE/dA using the chain rule.
 *
 * @param dAA         Gradient w.r.t. AA [n_atoms, nAA] (row-major)
 * @param A           Pooled features [n_atoms, nA] (for order > 1)
 * @param n_atoms     Number of atoms
 * @param nA          Number of A features
 * @param nAA         Number of AA features (nspec1 + nspec2)
 * @param specs_mat1  Order-1 spec matrix [nspec1, 1]
 * @param nspec1      Number of order-1 specs
 * @param specs_mat2  Order-2 spec matrix [nspec2, 2]
 * @param nspec2      Number of order-2 specs
 * @param dA          Output: gradient w.r.t. A [n_atoms, nA] (row-major)
 */
void ace_sparse_symm_prod_backward(
    const float* dAA, const float* A,
    int n_atoms, int nA, int nAA,
    const int* specs_mat1, int nspec1,
    const int* specs_mat2, int nspec2,
    float* dA
);

/**
 * Backward pass through pooled sparse product.
 *
 * Given dE/dA, compute dE/dRnl and dE/dYlm.
 *
 * @param dA          Gradient w.r.t. A [n_atoms, nA] (row-major)
 * @param Rnl_3       Radial embeddings [maxneigs, n_atoms, nRnl]
 * @param Ylm_3       Angular embeddings [maxneigs, n_atoms, nYlm]
 * @param maxneigs    Maximum neighbors per atom
 * @param n_atoms     Number of atoms
 * @param nRnl        Number of radial features
 * @param nYlm        Number of angular features
 * @param nA          Number of A features
 * @param spec_R      Radial spec indices [nA]
 * @param spec_Y      Angular spec indices [nA]
 * @param dRnl_3      Output: gradient w.r.t. Rnl [maxneigs, n_atoms, nRnl]
 * @param dYlm_3      Output: gradient w.r.t. Ylm [maxneigs, n_atoms, nYlm]
 */
void ace_pooled_sparse_product_backward(
    const float* dA,
    const float* Rnl_3, const float* Ylm_3,
    int maxneigs, int n_atoms, int nRnl, int nYlm, int nA,
    const int* spec_R, const int* spec_Y,
    float* dRnl_3, float* dYlm_3
);

/**
 * Convert embedding gradients to pair forces.
 *
 * For each pair (i,j), computes:
 *   F_ij = -(dE/dRnl_ij * dRnl/dr * rhat_ij
 *          + dE/dYlm_ij * dYlm/drhat)
 *
 * @param dRnl_3      Gradient w.r.t. Rnl [maxneigs, n_atoms, nRnl]
 * @param dYlm_3      Gradient w.r.t. Ylm [maxneigs, n_atoms, nYlm]
 * @param rij         Displacement vectors [n_pairs, 3]
 * @param pair_i      Source atom indices [n_pairs]
 * @param pair_j      Target atom indices [n_pairs]
 * @param n_pairs     Number of pairs
 * @param n_atoms     Number of atoms
 * @param maxneigs    Maximum neighbors per atom
 * @param nRnl        Number of radial features
 * @param nYlm        Number of angular features
 * @param rcut        Cutoff radius
 * @param pair_forces Output: pair force contributions [n_pairs, 3]
 */
void ace_embedding_grad_to_pair_forces(
    const float* dRnl_3, const float* dYlm_3,
    const float* rij,
    const int* pair_i, const int* pair_j,
    int n_pairs, int n_atoms, int maxneigs,
    int nRnl, int nYlm, float rcut,
    float* pair_forces
);

/**
 * Accumulate pair forces to atomic forces.
 *
 * forces[pair_i[e]] -= pair_forces[e]
 * forces[pair_j[e]] += pair_forces[e]
 *
 * @param pair_forces Pair force contributions [n_pairs, 3]
 * @param pair_i      Source atom indices [n_pairs]
 * @param pair_j      Target atom indices [n_pairs]
 * @param n_pairs     Number of pairs
 * @param n_atoms     Number of atoms
 * @param forces      Output: atomic forces [n_atoms, 3] (accumulated, not zeroed)
 */
void ace_accumulate_forces(
    const float* pair_forces,
    const int* pair_i, const int* pair_j,
    int n_pairs, int n_atoms,
    float* forces
);

/* ============================================================================
 * High-Level Pipeline
 * ============================================================================ */

/**
 * Full pipeline: compute energy and forces from positions.
 *
 * Combines all steps:
 * 1. Compute embeddings: positions + neighbors → Rnl, Ylm
 * 2. Compute A: pooled sparse product (host)
 * 3. Compute AA: sparse symmetric product (host)
 * 4. Compute E, dE/dAA: via IREE (GPU-portable)
 * 5. Backward: dE/dAA → dE/dA → dE/dRnl, dE/dYlm (host)
 * 6. Pair forces: embedding gradients + embedding derivatives (host)
 * 7. Accumulate: pair forces → atomic forces (host)
 *
 * @param model       Model handle (IREE VMFB)
 * @param positions   Atomic positions [n_atoms, 3]
 * @param n_atoms     Number of atoms
 * @param pair_i      Source atom indices [n_pairs]
 * @param pair_j      Target atom indices [n_pairs]
 * @param n_pairs     Number of pairs (edges)
 * @param rcut        Cutoff radius
 * @param N_cheb      Number of Chebyshev polynomials
 * @param maxl        Maximum angular momentum
 * @param spec_R      Radial spec indices [nA]
 * @param spec_Y      Angular spec indices [nA]
 * @param nA          Number of A features
 * @param specs_mat1  Order-1 spec matrix [nspec1, 1]
 * @param nspec1      Number of order-1 specs
 * @param specs_mat2  Order-2 spec matrix [nspec2, 2]
 * @param nspec2      Number of order-2 specs
 * @param A2Bmap      Coupling matrix [nBB, nAA]
 * @param nBB         Number of basis functions
 * @param params      Linear parameters [nBB]
 * @param forces      Output: atomic forces [n_atoms, 3]
 * @return            Energy value
 */
float ace_energy_forces(
    ACEForceModel* model,
    const float* positions, int n_atoms,
    const int* pair_i, const int* pair_j, int n_pairs,
    float rcut, int N_cheb, int maxl,
    const int* spec_R, const int* spec_Y, int nA,
    const int* specs_mat1, int nspec1,
    const int* specs_mat2, int nspec2,
    const float* A2Bmap, int nBB,
    const float* params,
    float* forces
);

#ifdef __cplusplus
}
#endif

#endif /* ACE_FORCES_H */
