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
    compute_agnesi_transform(r::T, pin::Int, pcut::Int, a::T, b0::T, b1::T, rin::T, req::T) where T

Compute Agnesi distance transform: r → y ∈ [-1, 1].

The generalized Agnesi transform is:
  s = (r - rin) / (req - rin)
  x = 1 / (1 + a * s^pin / (1 + s^(pin - pcut)))
  y = clamp(b1 * x + b0, -1, 1)

This matches EquivariantTensors.eval_agnesi.

Parameters (from agnesi_params):
- pin: Inner power exponent
- pcut: Cutoff power exponent
- a: Scaling parameter (computed to maximize slope at req)
- b0, b1: Linear normalization to [-1, 1]
- rin: Inner reference distance
- req: Equilibrium reference distance
"""
@inline function compute_agnesi_transform(r::T, pin::Integer, pcut::Integer,
                                           a::T, b0::T, b1::T, rin::T, req::T) where T
    # Compute scaled distance
    s = (r - rin) / (req - rin)

    # Agnesi function
    x = one(T) / (one(T) + a * s^pin / (one(T) + s^(pin - pcut)))

    # Linear map to [-1, 1]
    y = b1 * x + b0

    # Clamp to [-1, 1]
    y = max(-one(T), min(one(T), y))

    return y
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

    # Extract Agnesi parameters for this pair (7 parameters)
    # Order: [pin, pcut, a, b0, b1, rin, req]
    pin = Int(state.agnesi_params[1, pair_idx])
    pcut = Int(state.agnesi_params[2, pair_idx])
    a = state.agnesi_params[3, pair_idx]
    b0 = state.agnesi_params[4, pair_idx]
    b1 = state.agnesi_params[5, pair_idx]
    rin = state.agnesi_params[6, pair_idx]
    req = state.agnesi_params[7, pair_idx]

    # Distance transform (using 8-param version for full accuracy)
    y = compute_agnesi_transform(r, pin, pcut, a, b0, b1, rin, req)

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
    compute_solid_harmonics_reactant!(Ylm::AbstractVector{T}, r::T, rhat::SVector{3,T}, maxl::Int) where T

Compute SOLID spherical harmonics: r^l * Y_lm(rhat)
This is what ACE uses when Ytype=:solid.

The solid harmonics are homogeneous polynomials of degree l in (x, y, z).
"""
function compute_solid_harmonics_reactant!(Ylm::AbstractVector{T}, r::T, rhat::SVector{3,T}, maxl::Int) where T
    x, y, z = r * rhat[1], r * rhat[2], r * rhat[3]
    nYlm = (maxl + 1)^2

    # l = 0: r^0 * Y_00 = 1 / (2√π)
    Ylm[1] = T(0.28209479177387814)

    maxl == 0 && return Ylm

    # l = 1: r^1 * Y_1m = √(3/4π) * (x, y, z)
    c1 = T(0.4886025119029199)  # √(3/4π)
    Ylm[2] = c1 * y   # l=1, m=-1
    Ylm[3] = c1 * z   # l=1, m=0
    Ylm[4] = c1 * x   # l=1, m=1

    maxl == 1 && return Ylm

    # l = 2: r^2 * Y_2m - these are quadratic forms in (x, y, z)
    c2_0 = T(0.31539156525252005)   # √(5/16π)
    c2_1 = T(1.0925484305920792)    # √(15/4π)
    c2_2 = T(0.5462742152960396)    # √(15/16π)

    xy = x * y
    xz = x * z
    yz = y * z
    x2 = x * x
    y2 = y * y
    z2 = z * z
    r2 = x2 + y2 + z2

    Ylm[5] = c2_1 * xy                        # l=2, m=-2
    Ylm[6] = c2_1 * yz                        # l=2, m=-1
    Ylm[7] = c2_0 * (3 * z2 - r2)            # l=2, m=0
    Ylm[8] = c2_1 * xz                        # l=2, m=1
    Ylm[9] = c2_2 * (x2 - y2)                 # l=2, m=2

    maxl > 2 && @warn "compute_solid_harmonics_reactant! only supports maxl ≤ 2, got $maxl"

    return Ylm
end

"""
    compute_solid_harmonics_reactant(r::T, rhat::SVector{3,T}, maxl::Int) where T

Allocating version of solid harmonics computation.
"""
function compute_solid_harmonics_reactant(r::T, rhat::SVector{3,T}, maxl::Int) where T
    nYlm = (maxl + 1)^2
    Ylm = Vector{T}(undef, nYlm)
    return compute_solid_harmonics_reactant!(Ylm, r, rhat, maxl)
end
