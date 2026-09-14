# Shared, rank-decomposable ill-conditioned tall-skinny test-matrix generator.
# Row block [i0:i1] can be generated identically on any process, so the same
# matrix is used by the single-node reference and every distributed solver.
module TestMat

using LinearAlgebra, Random, Printf

"n x n symmetric mixer with prescribed singular values, log-spaced over `cond`."
function mixer(n::Int, cnd::Float64; seed::Int = 20260912)
    rng = MersenneTwister(seed)
    V = Matrix(qr(randn(rng, n, n)).Q)
    s = exp10.(range(0.0, -log10(cnd); length = n))
    return V * Diagonal(s) * V'
end

"Deterministic Gaussian row i (length n), independent of how rows are partitioned."
function grow!(buf::AbstractVector{Float64}, i::Int, n::Int, seed::Int)
    rng = MersenneTwister(hash((seed, i)) % UInt32)
    randn!(rng, buf)
    return buf
end

"Generate rows i0:i1 of the m x n design matrix A = B*M."
function block(i0::Int, i1::Int, n::Int, M::Matrix{Float64}; seed::Int = 424242)
    nb = i1 - i0 + 1
    B = Matrix{Float64}(undef, nb, n)
    buf = Vector{Float64}(undef, n)
    @inbounds for (k, i) in enumerate(i0:i1)
        grow!(buf, i, n, seed)
        B[k, :] .= buf
    end
    return B * M
end

"x_true, deterministic."
xtrue(n::Int; seed::Int = 777) = randn(MersenneTwister(seed), n)

"Row-block boundaries for `m` rows over `p` parts (contiguous, near-equal)."
function partition(m::Int, p::Int)
    q, r = divrem(m, p)
    bnds = Tuple{Int,Int}[]
    s = 1
    for k in 1:p
        len = q + (k <= r ? 1 : 0)
        push!(bnds, (s, s + len - 1))
        s += len
    end
    return bnds
end

"""
Quality metrics for a candidate solution x. `nrmA` is ||A||_2 (or an estimate).
- res      : ||Ax-y|| / ||y||                     (fit quality)
- fwd      : ||x-xtrue|| / ||xtrue||              (parameter recovery)
- normality: ||A'(Ax-y)|| / (||A||*||Ax-y||)      (LS optimality; backward-stable
             algorithms give ~1e-15 here regardless of cond(A))
"""
function metrics(A, y, x, xt, nrmA)
    r = A * x - y
    nr = norm(r)
    (res = nr / norm(y),
     fwd = norm(x - xt) / norm(xt),
     normality = nr == 0 ? 0.0 : norm(A' * r) / (nrmA * nr))
end

end # module
