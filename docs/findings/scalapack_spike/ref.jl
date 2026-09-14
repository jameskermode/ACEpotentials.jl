include("testmat.jl")
using .TestMat, LinearAlgebra, Printf, IterativeSolvers, Random
BLAS.set_num_threads(4)

"TSQR: R and Q'y carried through a binary reduction tree over row blocks.
 One pass over A; per-block state is only R (n x n) and z (n)."
function tsqr_full(A::Matrix{Float64}, y::Vector{Float64}, nblk::Int)
    m, n = size(A)
    Rs = Matrix{Float64}[]; zs = Vector{Float64}[]
    for (i0, i1) in TestMat.partition(m, nblk)
        F = qr(A[i0:i1, :])
        push!(Rs, Matrix(F.R)); push!(zs, (F.Q' * y[i0:i1])[1:n])
    end
    while length(Rs) > 1
        nR = Matrix{Float64}[]; nz = Vector{Float64}[]
        for k in 1:2:length(Rs)
            if k == length(Rs); push!(nR, Rs[k]); push!(nz, zs[k])
            else
                F = qr(vcat(Rs[k], Rs[k+1]))
                push!(nR, Matrix(F.R)); push!(nz, (F.Q' * vcat(zs[k], zs[k+1]))[1:n])
            end
        end
        Rs = nR; zs = nz
    end
    return UpperTriangular(Rs[1]) \ zs[1], Rs[1]
end

"Seminormal: R from TSQR tree, x from R'R x = A'y. Cheaper (no Q'y) but this is
 the normal equations wearing a hat - included to show it is NOT acceptable."
function tsqr_semi(A, y, nblk; refine::Int = 1)
    m, n = size(A)
    Rs = [Matrix(qr(A[i0:i1, :]).R) for (i0,i1) in TestMat.partition(m, nblk)]
    while length(Rs) > 1
        nxt = Matrix{Float64}[]
        for k in 1:2:length(Rs)
            push!(nxt, k == length(Rs) ? Rs[k] : Matrix(qr(vcat(Rs[k], Rs[k+1])).R))
        end
        Rs = nxt
    end
    R = UpperTriangular(Rs[1])
    x = R \ (R' \ (A'y))
    for _ in 1:refine; x += R \ (R' \ (A' * (y - A*x))); end
    return x
end

function run(m, n, cnd, family; nblk = 8, noise = 1e-3)
    M = TestMat.mixer(n, cnd, family)
    A = TestMat.block(1, m, n, M)
    xt = TestMat.xtrue(n)
    sc = norm(A * xt) / sqrt(m)
    y = TestMat.rhs_block(A, 1, m, xt, noise, sc)
    sv = svdvals(A); kappa = sv[1]/sv[end]; nrmA = sv[1]
    @printf("\n### family=%s m=%d n=%d  requested cond=%.0e  MEASURED cond(A)=%.3e  ||A||=%.2e  noise=%.0e\n",
            family, m, n, cnd, kappa, nrmA, noise)

    # high-precision reference
    xref = setprecision(BigFloat, 400) do
        Float64.(qr(BigFloat.(A)) \ BigFloat.(y))
    end

    res = Any["[REF] BigFloat400 qr"        => xref,
              "qr (LAPACK, 1 node)"         => qr(A) \ y,
              "svd pinv (1 node)"           => svd(A) \ y,
              "TSQR full ($nblk blocks)"    => tsqr_full(A, y, nblk)[1],
              "TSQR seminormal + 1 refine"  => tsqr_semi(A, y, nblk),
              "normal eqns (cholesky)"      => (try cholesky(Symmetric(A'A))\(A'y) catch; fill(NaN,n) end),
              "normal eqns (lu)"            => (A'A) \ (A'y)]
    for damp in (0.0, 5e-3)
        x, ch = lsqr(A, y; damp=damp, atol=1e-14, btol=1e-14, conlim=1e20,
                     maxiter=20*n, log=true)
        push!(res, "LSQR damp=$damp (it=$(ch.iters))" => x)
    end

    @printf("%-30s  %-10s  %-10s  %-10s\n", "solver", "rel.resid", "fwd.err", "normality")
    for (k, x) in res
        mt = TestMat.metrics(A, y, x, xref, nrmA)
        @printf("%-30s  %-10.3e  %-10.3e  %-10.3e\n", k, mt.res, mt.fwd, mt.normality)
    end
    xq = res[2][2]
    for i in (4,5,3)
        @printf("   agreement  %-28s vs LAPACK qr : %.3e\n", res[i][1], norm(res[i][2]-xq)/norm(xq))
    end
    return nothing
end

for fam in (:scaled, :mixed), cnd in (1e8, 1e12, 1e16, 1e21)
    run(2000, 100, cnd, fam)
end
