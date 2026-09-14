include("testmat.jl"); using .TestMat, LinearAlgebra, Printf
BLAS.set_num_threads(32)
m, n = 120000, 2000
M = TestMat.mixer(n, 1e16, :scaled); A = TestMat.block(1, m, n, M)
xt = TestMat.xtrue(n); y = TestMat.rhs_block(A, 1, m, xt, 1e-3, norm(A*xt)/sqrt(m))
qr(A[1:5000, :]) \ y[1:5000]   # warm up
for k in 1:2
    GC.gc(); t = @elapsed x = qr(A) \ y
    @printf("CPU LAPACK qr(A)\\y  m=%d n=%d  BLAS threads=%d : %.2fs\n", m, n, BLAS.get_num_threads(), t)
end
