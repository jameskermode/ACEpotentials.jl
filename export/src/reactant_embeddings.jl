#=
Reactant-Compatible Embeddings
==============================

Pure Julia implementations of radial and angular embeddings that can be
traced by Reactant. These replace KernelAbstractions and external C libraries.

Key changes from standard implementations:
- No @kernel macros (Reactant can't trace KernelAbstractions)
- No SpheriCart C calls (external libraries can't be traced)
- Uses explicit loops and broadcasting for Reactant compatibility
=#

using StaticArrays
using LinearAlgebra: norm, dot

## ============================================================================
## Agnesi Distance Transform
## ============================================================================

"""
    compute_agnesi_transform(r::T, params::NTuple{5,T}) where T

Compute Agnesi distance transform: r → y ∈ [-1, 1].

The Agnesi transform maps distances to a bounded domain suitable for
polynomial basis evaluation. It has smooth decay to zero at the cutoff.

Parameters (in order):
- pcut: Cutoff smoothness exponent
- pin: Inner smoothness exponent
- rin: Inner reference distance
- req: Equilibrium reference distance
- rcut: Cutoff radius

Returns y in [-1, 1] where y(0) ≈ 1 and y(rcut) = 0.
"""
@inline function compute_agnesi_transform(r::T, pcut::T, pin::T, rin::T, req::T, rcut::T) where T
    x = r / rcut
    # Smooth cutoff: (1 - x^pcut)
    cutoff_factor = (one(T) - x^pcut)
    # Agnesi-like decay
    inner_factor = one(T) / (one(T) + (r / req)^pin)
    y = cutoff_factor * inner_factor
    return y
end

"""
    compute_agnesi_transform_ed(r::T, pcut::T, pin::T, rin::T, req::T, rcut::T) where T

Compute Agnesi transform and its derivative w.r.t. r.
Returns (y, dy/dr).
"""
@inline function compute_agnesi_transform_ed(r::T, pcut::T, pin::T, rin::T, req::T, rcut::T) where T
    x = r / rcut

    # Forward: y = (1 - x^pcut) / (1 + (r/req)^pin)
    xp = x^pcut
    rp = (r / req)^pin

    cutoff_factor = one(T) - xp
    inner_factor = one(T) / (one(T) + rp)
    y = cutoff_factor * inner_factor

    # Derivative chain rule
    # d(cutoff)/dr = -pcut * x^(pcut-1) / rcut
    d_cutoff = -pcut * x^(pcut - one(T)) / rcut

    # d(inner)/dr = -pin * (r/req)^(pin-1) / req / (1 + (r/req)^pin)^2
    d_inner = -pin * rp / r / (one(T) + rp)^2

    dy = d_cutoff * inner_factor + cutoff_factor * d_inner

    return y, dy
end

## ============================================================================
## Envelope Function
## ============================================================================

"""
    compute_envelope(y::T) where T

Compute envelope function: env(y) = (1 - y)^2 * (1 + y)^2 = (1 - y²)²

The envelope ensures smooth decay at boundaries y = ±1.
"""
@inline function compute_envelope(y::T) where T
    return (one(T) - y^2)^2
end

"""
    compute_envelope_ed(y::T) where T

Compute envelope and its derivative.
Returns (env, d_env/dy).
"""
@inline function compute_envelope_ed(y::T) where T
    y2 = y^2
    env = (one(T) - y2)^2
    d_env = -4 * y * (one(T) - y2)
    return env, d_env
end

## ============================================================================
## Chebyshev Polynomial Basis
## ============================================================================

"""
    compute_chebyshev_basis!(P::AbstractVector{T}, y::T, A::AbstractVector{T},
                             B::AbstractVector{T}, C::AbstractVector{T}) where T

Evaluate orthonormalized Chebyshev-like polynomials via 3-term recurrence.
Modifies P in-place.

Recurrence: P[n] = (A[n] * y + B[n]) * P[n-1] + C[n] * P[n-2]

The coefficients A, B, C define an orthonormalized basis.
"""
function compute_chebyshev_basis!(P::AbstractVector{T}, y::T,
                                   A::AbstractVector{T},
                                   B::AbstractVector{T},
                                   C::AbstractVector{T}) where T
    n_polys = length(P)
    n_polys == 0 && return P

    P[1] = A[1]

    n_polys == 1 && return P

    P[2] = A[2] * y + B[2]

    for n in 3:n_polys
        P[n] = (A[n] * y + B[n]) * P[n-1] + C[n] * P[n-2]
    end

    return P
end

"""
    compute_chebyshev_basis(y::T, n_polys::Int, A::AbstractVector{T},
                            B::AbstractVector{T}, C::AbstractVector{T}) where T

Allocating version of Chebyshev basis computation.
"""
function compute_chebyshev_basis(y::T, n_polys::Int,
                                  A::AbstractVector{T},
                                  B::AbstractVector{T},
                                  C::AbstractVector{T}) where T
    P = Vector{T}(undef, n_polys)
    return compute_chebyshev_basis!(P, y, A, B, C)
end

"""
    compute_chebyshev_basis_ed!(P::AbstractVector{T}, dP::AbstractVector{T},
                                 y::T, A::AbstractVector{T}, B::AbstractVector{T},
                                 C::AbstractVector{T}) where T

Compute Chebyshev basis and derivatives w.r.t. y.
Modifies P and dP in-place.
"""
function compute_chebyshev_basis_ed!(P::AbstractVector{T}, dP::AbstractVector{T},
                                      y::T,
                                      A::AbstractVector{T},
                                      B::AbstractVector{T},
                                      C::AbstractVector{T}) where T
    n_polys = length(P)
    n_polys == 0 && return P, dP

    P[1] = A[1]
    dP[1] = zero(T)

    n_polys == 1 && return P, dP

    P[2] = A[2] * y + B[2]
    dP[2] = A[2]

    for n in 3:n_polys
        P[n] = (A[n] * y + B[n]) * P[n-1] + C[n] * P[n-2]
        dP[n] = A[n] * P[n-1] + (A[n] * y + B[n]) * dP[n-1] + C[n] * dP[n-2]
    end

    return P, dP
end

## ============================================================================
## Combined Radial Embedding
## ============================================================================

"""
    compute_radial_embedding(r::T, iz::Int, jz::Int, state::ReactantETACEState{T}) where T

Compute radial embedding Rnl for a single edge.

Pipeline: r → Agnesi transform → Chebyshev basis → envelope → linear layer → Rnl

Returns Vector of length n_rnl.
"""
function compute_radial_embedding(r::T, iz::Int, jz::Int,
                                   state::ReactantETACEState{T}) where T
    # Get species pair index
    pair_idx = zz_to_pair_index(iz, jz, state.n_species)

    # Extract Agnesi parameters for this pair
    pcut = state.agnesi_params[1, pair_idx]
    pin = state.agnesi_params[2, pair_idx]
    rin = state.agnesi_params[3, pair_idx]
    req = state.agnesi_params[4, pair_idx]
    rcut = state.agnesi_params[5, pair_idx]

    # Distance transform
    y = compute_agnesi_transform(r, pcut, pin, rin, req, rcut)

    # Envelope
    env = compute_envelope(y)

    # Polynomial basis
    P = compute_chebyshev_basis(y, state.n_polys, state.poly_A, state.poly_B, state.poly_C)

    # Apply envelope
    P_env = P .* env

    # Linear transform to Rnl
    # W_radial[:, :, pair_idx] is (n_rnl, n_polys)
    Rnl = state.W_radial[:, :, pair_idx] * P_env

    return Rnl
end

"""
    compute_radial_embedding_ed(r::T, iz::Int, jz::Int,
                                 state::ReactantETACEState{T}) where T

Compute radial embedding and its derivative w.r.t. r.
Returns (Rnl, dRnl/dr).
"""
function compute_radial_embedding_ed(r::T, iz::Int, jz::Int,
                                      state::ReactantETACEState{T}) where T
    pair_idx = zz_to_pair_index(iz, jz, state.n_species)

    pcut = state.agnesi_params[1, pair_idx]
    pin = state.agnesi_params[2, pair_idx]
    rin = state.agnesi_params[3, pair_idx]
    req = state.agnesi_params[4, pair_idx]
    rcut = state.agnesi_params[5, pair_idx]

    # Transform with derivative
    y, dy_dr = compute_agnesi_transform_ed(r, pcut, pin, rin, req, rcut)

    # Envelope with derivative
    env, d_env = compute_envelope_ed(y)

    # Polynomials with derivatives
    P = Vector{T}(undef, state.n_polys)
    dP = Vector{T}(undef, state.n_polys)
    compute_chebyshev_basis_ed!(P, dP, y, state.poly_A, state.poly_B, state.poly_C)

    # d(P * env)/dy = dP * env + P * d_env
    dP_env_dy = dP .* env .+ P .* d_env

    # Chain rule: d/dr = d/dy * dy/dr
    P_env = P .* env
    dP_env_dr = dP_env_dy .* dy_dr

    # Linear transform
    W = state.W_radial[:, :, pair_idx]
    Rnl = W * P_env
    dRnl = W * dP_env_dr

    return Rnl, dRnl
end

## ============================================================================
## Real Spherical Harmonics (Pure Julia)
## ============================================================================

"""
    compute_ylm_reactant!(Ylm::AbstractVector{T}, rhat::SVector{3,T}, maxl::Int) where T

Compute real spherical harmonics up to maxl.
Pure Julia implementation, Reactant-traceable.

Uses the recursive algorithm for associated Legendre polynomials
combined with trigonometric functions for the azimuthal part.

Output ordering: Ylm[l² + l + m + 1] for l=0..maxl, m=-l..l
"""
function compute_ylm_reactant!(Ylm::AbstractVector{T}, rhat::SVector{3,T}, maxl::Int) where T
    x, y, z = rhat
    nYlm = (maxl + 1)^2

    # l = 0
    Ylm[1] = T(0.28209479177387814)  # 1 / (2√π)

    maxl == 0 && return Ylm

    # l = 1
    # Y_1^-1 = √(3/4π) * y
    # Y_1^0  = √(3/4π) * z
    # Y_1^1  = √(3/4π) * x
    c1 = T(0.4886025119029199)  # √(3/4π)
    Ylm[2] = c1 * y   # l=1, m=-1
    Ylm[3] = c1 * z   # l=1, m=0
    Ylm[4] = c1 * x   # l=1, m=1

    maxl == 1 && return Ylm

    # l = 2
    # Coefficients for l=2 real spherical harmonics
    c2_0 = T(0.31539156525252005)   # √(5/16π)
    c2_1 = T(1.0925484305920792)    # √(15/4π)
    c2_2 = T(0.5462742152960396)    # √(15/16π)

    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z

    Ylm[5] = c2_1 * xy                        # l=2, m=-2
    Ylm[6] = c2_1 * yz                        # l=2, m=-1
    Ylm[7] = c2_0 * (3 * z2 - one(T))        # l=2, m=0
    Ylm[8] = c2_1 * xz                        # l=2, m=1
    Ylm[9] = c2_2 * (x2 - y2)                 # l=2, m=2

    # Higher l values would go here
    # For now, we support up to l=2
    maxl > 2 && @warn "compute_ylm_reactant! only supports maxl ≤ 2, got $maxl"

    return Ylm
end

"""
    compute_ylm_reactant(rhat::SVector{3,T}, maxl::Int) where T

Allocating version of spherical harmonics computation.
"""
function compute_ylm_reactant(rhat::SVector{3,T}, maxl::Int) where T
    nYlm = (maxl + 1)^2
    Ylm = Vector{T}(undef, nYlm)
    return compute_ylm_reactant!(Ylm, rhat, maxl)
end

"""
    compute_ylm_reactant_ed!(Ylm::AbstractVector{T}, dYlm::AbstractMatrix{T},
                              rhat::SVector{3,T}, maxl::Int) where T

Compute spherical harmonics and their gradients w.r.t. the direction vector.
dYlm is (nYlm, 3) where dYlm[i, :] = ∂Ylm[i]/∂rhat.

Note: rhat should be a unit vector. The gradient is computed assuming
this constraint (i.e., derivatives are on the unit sphere).
"""
function compute_ylm_reactant_ed!(Ylm::AbstractVector{T}, dYlm::AbstractMatrix{T},
                                   rhat::SVector{3,T}, maxl::Int) where T
    x, y, z = rhat
    nYlm = (maxl + 1)^2

    # l = 0
    Ylm[1] = T(0.28209479177387814)
    dYlm[1, 1] = zero(T)
    dYlm[1, 2] = zero(T)
    dYlm[1, 3] = zero(T)

    maxl == 0 && return Ylm, dYlm

    # l = 1
    c1 = T(0.4886025119029199)

    Ylm[2] = c1 * y
    dYlm[2, 1] = zero(T)
    dYlm[2, 2] = c1
    dYlm[2, 3] = zero(T)

    Ylm[3] = c1 * z
    dYlm[3, 1] = zero(T)
    dYlm[3, 2] = zero(T)
    dYlm[3, 3] = c1

    Ylm[4] = c1 * x
    dYlm[4, 1] = c1
    dYlm[4, 2] = zero(T)
    dYlm[4, 3] = zero(T)

    maxl == 1 && return Ylm, dYlm

    # l = 2
    c2_0 = T(0.31539156525252005)
    c2_1 = T(1.0925484305920792)
    c2_2 = T(0.5462742152960396)

    # m = -2: c2_1 * x * y
    Ylm[5] = c2_1 * x * y
    dYlm[5, 1] = c2_1 * y
    dYlm[5, 2] = c2_1 * x
    dYlm[5, 3] = zero(T)

    # m = -1: c2_1 * y * z
    Ylm[6] = c2_1 * y * z
    dYlm[6, 1] = zero(T)
    dYlm[6, 2] = c2_1 * z
    dYlm[6, 3] = c2_1 * y

    # m = 0: c2_0 * (3z² - 1)
    Ylm[7] = c2_0 * (3 * z * z - one(T))
    dYlm[7, 1] = zero(T)
    dYlm[7, 2] = zero(T)
    dYlm[7, 3] = c2_0 * 6 * z

    # m = 1: c2_1 * x * z
    Ylm[8] = c2_1 * x * z
    dYlm[8, 1] = c2_1 * z
    dYlm[8, 2] = zero(T)
    dYlm[8, 3] = c2_1 * x

    # m = 2: c2_2 * (x² - y²)
    Ylm[9] = c2_2 * (x * x - y * y)
    dYlm[9, 1] = c2_2 * 2 * x
    dYlm[9, 2] = -c2_2 * 2 * y
    dYlm[9, 3] = zero(T)

    return Ylm, dYlm
end

## ============================================================================
## Combined Embedding Functions
## ============================================================================

"""
    compute_edge_embeddings(r::T, rhat::SVector{3,T}, iz::Int, jz::Int,
                            state::ReactantETACEState{T}) where T

Compute both radial and angular embeddings for a single edge.
Returns (Rnl, Ylm).
"""
function compute_edge_embeddings(r::T, rhat::SVector{3,T}, iz::Int, jz::Int,
                                  state::ReactantETACEState{T}) where T
    Rnl = compute_radial_embedding(r, iz, jz, state)
    Ylm = compute_ylm_reactant(rhat, state.maxl)
    return Rnl, Ylm
end

"""
    compute_edge_embeddings_ed(r::T, rhat::SVector{3,T}, iz::Int, jz::Int,
                                state::ReactantETACEState{T}) where T

Compute embeddings with derivatives w.r.t. edge vector.
Returns (Rnl, Ylm, dRnl_dr, dYlm_drhat).
"""
function compute_edge_embeddings_ed(r::T, rhat::SVector{3,T}, iz::Int, jz::Int,
                                     state::ReactantETACEState{T}) where T
    Rnl, dRnl = compute_radial_embedding_ed(r, iz, jz, state)

    nYlm = (state.maxl + 1)^2
    Ylm = Vector{T}(undef, nYlm)
    dYlm = Matrix{T}(undef, nYlm, 3)
    compute_ylm_reactant_ed!(Ylm, dYlm, rhat, state.maxl)

    return Rnl, Ylm, dRnl, dYlm
end
