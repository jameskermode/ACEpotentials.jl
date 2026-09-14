# Factor once, sweep lambda for free.
#
# ACEfit.QR(lambda) factorises the augmented [A; lambda*P] afresh for every
# lambda -- O(m n^2) per value, and for a tall-skinny A (m >> n) the lambda rows
# are a negligible fraction of the work being redone.  Instead:
#
#   1. qr(A) once                 O(m n^2)   -> R (n x n), c = Q' y
#   2. svd(R) once                O(n^3)     -> R = U S V'
#   3. per lambda                 O(n^2)     -> z = V * (S ./ (S.^2 .+ lambda^2)) .* (U' c)
#
# Steps 1-2 are orthogonal transformations of the same problem, so this is the
# augmented-QR solution to working precision, NOT the normal equations (which
# would square the condition number).  `A` must already carry the weights and
# the prior (A <- W * A / P), so the penalty is lambda^2 ||z||^2 with z = P x.
using LinearAlgebra, SharedArrays

struct TikhonovFactor
   U::Matrix{Float64}
   S::Vector{Float64}
   V::Matrix{Float64}
   d::Vector{Float64}      # U' * (Q' y)[1:n]
end

# `inplace = true` factorises A's own storage (LAPACK geqrf) and destroys it,
# so the peak memory is A + R instead of 2A.  This is what lets a 151 GB
# design matrix (categorical, degree 10, 3200 structures) factorise on a
# 376 GB node.  The caller must not use A afterwards.
function TikhonovFactor(A::AbstractMatrix, y::AbstractVector; inplace = false)
   F = inplace ? qr!(A isa SharedArray ? sdata(A) : A) : qr(A)
   m, n = size(A)
   # Underdetermined (m < n) is legal here -- a learning curve's small-N points
   # have fewer observations than parameters -- and R is then k x n with
   # k = min(m, n).  The thin SVD handles both cases; Tikhonov with lambda > 0
   # is well-posed regardless of m/n.
   k = min(m, n)
   c = (F.Q' * y)[1:k]
   R = Matrix(F.R)                               # k x n
   U, S, V = svd(R)                              # U k x k, S k, V n x k
   TikhonovFactor(U, S, V, U' * c)
end

# solution for one lambda, O(n^2)
tikhonov_solve(T::TikhonovFactor, lambda) = T.V * ((T.S ./ (T.S .^ 2 .+ lambda^2)) .* T.d)
