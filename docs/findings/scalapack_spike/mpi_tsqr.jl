# Distributed TSQR over MPI: local Householder QR per rank, then a binary
# reduction tree on (R, Q'y). One pass over A. State per rank: n x n + n.
include("mpi_common.jl")
P = parse_args()
i0, i1, Ablk, yblk, xt, M = local_block(P)
MPI.Barrier(comm)
t0 = MPI.Wtime()
n = P.n
F = qr(Ablk)
R = Matrix(F.R)                      # (min(mb,n)) x n
z = (F.Q' * yblk)[1:size(R,1)]
# binary tree: at level with stride s, rank r with r % 2s == 0 receives from r+s
s = 1
while s < nprocs
    global s, R, z
    if rank % (2s) == 0
        src = rank + s
        if src < nprocs
            nr = MPI.Recv(Int, comm; source = src, tag = 0)
            Rr = Array{Float64}(undef, nr, n); zr = Array{Float64}(undef, nr)
            MPI.Recv!(Rr, comm; source = src, tag = 1)
            MPI.Recv!(zr, comm; source = src, tag = 2)
            G = qr(vcat(R, Rr)); k = min(size(R,1)+nr, n)
            R = Matrix(G.R)[1:k, :]; z = (G.Q' * vcat(z, zr))[1:k]
        end
    elseif rank % (2s) == s
        dst = rank - s
        MPI.Send(size(R,1), comm; dest = dst, tag = 0)
        MPI.Send(R, comm; dest = dst, tag = 1)
        MPI.Send(z, comm; dest = dst, tag = 2)
    end
    s *= 2
end
x = rank == 0 ? UpperTriangular(R) \ z : zeros(n)
MPI.Bcast!(x, comm; root = 0)
t1 = MPI.Wtime()
rank == 0 && @printf("[TSQR] distributed solve time: %.2fs on %d ranks (local block %d x %d)\n", t1-t0, nprocs, i1-i0+1, n)
reference_on_root(P, M, xt, x; label = "TSQR")
MPI.Finalize()
