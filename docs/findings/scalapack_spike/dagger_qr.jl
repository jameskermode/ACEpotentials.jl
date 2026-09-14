# Dagger.jl tiled/CA QR on a DArray over Distributed.jl workers (ACEfit's own
# parallel model), compared against single-node LAPACK qr on the same matrix.
using Distributed
@everywhere begin
    using Dagger, LinearAlgebra
    BLAS.set_num_threads(2)
end
include("testmat.jl")
using .TestMat, Printf
function run(m, n, cnd, family; bs = 500, p = 1)
    M = TestMat.mixer(n, cnd, family)
    A = TestMat.block(1, m, n, M)
    xt = TestMat.xtrue(n); sc = norm(A*xt)/sqrt(m)
    y = TestMat.rhs_block(A, 1, m, xt, 1e-3, sc)
    sv = svdvals(A); nrmA = sv[1]
    t0 = time(); xq = qr(A) \ y; tq = time() - t0
    DA = distribute(A, Blocks(bs, bs))
    t1 = time()
    F = qr!(DA; p = p)
    xd = collect(F \ distribute(y, Blocks(bs)))
    td = time() - t1
    mq = TestMat.metrics(A, y, xq, xq, nrmA); md = TestMat.metrics(A, y, xd, xq, nrmA)
    @printf("[DAGGER p=%d bs=%d] m=%d n=%d fam=%s MEASURED cond=%.3e workers=%d\n", p, bs, m, n, family, sv[1]/sv[end], nworkers())
    @printf("[DAGGER] LAPACK qr: resid=%.3e normality=%.3e (%.2fs) | Dagger qr: resid=%.3e normality=%.3e (%.2fs)\n",
            mq.res, mq.normality, tq, md.res, md.normality, td)
    @printf("[DAGGER] AGREEMENT ||x_dagger - x_qr||/||x_qr|| = %.3e  (cond*eps=%.1e)\n", md.fwd, sv[1]/sv[end]*eps())
end
for fam in (:scaled, :mixed), cnd in (1e12, 1e21)
    run(4000, 200, cnd, fam; bs = 200, p = 1)
end
run(20000, 500, 1e16, :scaled; bs = 500, p = 1)
try run(20000, 500, 1e16, :scaled; bs = 500, p = 4) catch e; println("[DAGGER p=4] FAILED: ", sprint(showerror, e)[1:min(end,300)]); end
