# Elemental.jl leastSquares on a DistMatrix filled from each rank's row block.
include("mpi_common.jl")
using Elemental
const El = Elemental
P = parse_args()
m, n = P.m, P.n
i0, i1, Ablk, yblk, xt, M = local_block(P)
MPI.Barrier(comm); t0 = MPI.Wtime()
A = El.DistMatrix(Float64); El.zeros!(A, m, n)
b = El.DistMatrix(Float64); El.zeros!(b, m, 1)
# Elemental's default [MC,MR] layout is element-cyclic in 2-D: every entry of our
# contiguous row block has to be shipped to its owner via queueUpdate.
El.reserve(A, length(Ablk)); El.reserve(b, length(yblk))
for j in 1:n, (k, i) in enumerate(i0:i1)
    El.queueUpdate(A, i, j, Ablk[k, j])
end
for (k, i) in enumerate(i0:i1); El.queueUpdate(b, i, 1, yblk[k]); end
El.processQueues(A); El.processQueues(b)
t1 = MPI.Wtime()
X = El.leastSquares(A, b)
t2 = MPI.Wtime()
# Array(::DistMatrix) is broken in Elemental.jl v0.6.1 (undefined copyto!); use collective El.get
x = [El.get(X, i, 1) for i in 1:n]
rank == 0 && @printf("[ELEMENTAL] fill+redistribute %.2fs, leastSquares %.2fs on %d ranks (Elemental.jl %s)\n",
                     t1-t0, t2-t1, nprocs, string(pkgversion(Elemental)))
reference_on_root(P, M, xt, x; label = "ELEMENTAL")
El.Finalize(); MPI.Finalize()
