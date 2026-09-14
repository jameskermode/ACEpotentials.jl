# Is LSQR's `damp` the same regularisation as ACEfit.QR's `lambda`?
# Both target  min ||Ax-y||^2 + lam^2 ||x||^2 ; QR solves it as qr([A; lam*I]) \ [y; 0].
include("testmat.jl"); include("lsqr_impl.jl")
using .TestMat, LinearAlgebra, Printf
BLAS.set_num_threads(4)
for fam in (:scaled, :mixed), cnd in (1e12, 1e16, 1e21)
    m, n = 2000, 100
    M = TestMat.mixer(n, cnd, fam); A = TestMat.block(1, m, n, M)
    xt = TestMat.xtrue(n); sc = norm(A*xt)/sqrt(m); y = TestMat.rhs_block(A, 1, m, xt, 1e-3, sc)
    sv = svdvals(A)
    @printf("\nfamily=%-6s measured cond=%.2e\n", fam, sv[1]/sv[end])
    @printf("  %-8s %-12s %-12s %-10s %s\n", "lambda", "||xQR-xLSQR||", "/||xQR||", "iters", "(cond of augmented [A;lam*I])")
    for lam in (1e-6, 1e-3, 1e-1)
        xq = qr([A; lam*I(n)]) \ [y; zeros(n)]
        xl, st = lsqr_cb((u,x)->mul!(u,A,x), (v,u)->mul!(v,A',u), norm, y, n;
                         damp=lam, atol=1e-15, btol=1e-15, maxiter=50n)
        sva = svdvals([A; lam*I(n)])
        @printf("  %-8.0e %-12.3e %-12.3e %-10d cond=%.1e\n", lam, norm(xq-xl), norm(xq-xl)/norm(xq), st.iters, sva[1]/sva[end])
    end
end
