# Distributed LSQR: row-block operator, A*x local, A'*r local + Allreduce.
# Optional right preconditioning by column norms (ACEfit's `P`), which needs
# one extra Allreduce of length n up front.
include("mpi_common.jl")
using IterativeSolvers
P = parse_args()
i0, i1, Ablk, yblk, xt, M = local_block(P)
precond = any(a -> a == "precond=1", ARGS)
damp = 0.0
for a in ARGS; startswith(a, "damp=") && (global damp = parse(Float64, split(a,"=")[2])); end
n = P.n
include("lsqr_impl.jl")
cn = sqrt.(MPI.Allreduce(vec(sum(abs2, Ablk; dims=1)), +, comm))   # global column norms
D = precond ? cn : ones(n)
AD = Ablk ./ D'                                                     # right-preconditioned local block
Ax!(u, x)  = mul!(u, AD, x)
Atu!(v, u) = (v .= MPI.Allreduce(AD' * u, +, comm); v)
unorm(u)   = sqrt(MPI.Allreduce(sum(abs2, u), +, comm))
MPI.Barrier(comm); t0 = MPI.Wtime()
maxit = 20 * n
xs, st = lsqr_cb(Ax!, Atu!, unorm, yblk, n; damp = damp, atol = 1e-14, btol = 1e-14, maxiter = maxit)
x = xs ./ D
t1 = MPI.Wtime()
rank == 0 && @printf("[LSQR] distributed: precond=%s damp=%g iters=%d/%d  est.normality=%.2e  time %.2fs on %d ranks\n",
                     precond, damp, st.iters, maxit, st.test2, t1-t0, nprocs)
reference_on_root(P, M, xt, x; label = "LSQR")
MPI.Finalize()
