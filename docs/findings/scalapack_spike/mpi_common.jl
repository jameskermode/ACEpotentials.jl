# Common setup for the distributed drivers. Each rank generates ONLY its own
# contiguous row block of A and y - exactly ACEfit's pmap-over-packets layout.
include("testmat.jl")
using .TestMat, MPI, LinearAlgebra, Printf
MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
BLAS.set_num_threads(max(1, 32 ÷ nprocs ÷ 2))

function parse_args()
    d = Dict("m"=>20000, "n"=>500, "cond"=>1e15, "family"=>"scaled", "noise"=>1e-3, "grid"=>"1d")
    for a in ARGS
        k, v = split(a, "="); d[k] = v
    end
    (m=parse(Int, string(d["m"])), n=parse(Int, string(d["n"])), cnd=parse(Float64, string(d["cond"])),
     family=Symbol(d["family"]), noise=parse(Float64, string(d["noise"])), grid=String(d["grid"]))
end

"Generate this rank's row block. Returns (i0, i1, Ablk, yblk, xt, M)."
function local_block(P)
    M = TestMat.mixer(P.n, P.cnd, P.family)
    xt = TestMat.xtrue(P.n)
    i0, i1 = TestMat.partition(P.m, nprocs)[rank+1]
    Ablk = TestMat.block(i0, i1, P.n, M)
    # scale for noise must be partition-independent: use ||A xt||/sqrt(m) via Allreduce
    loc = sum(abs2, Ablk * xt)
    sc = sqrt(MPI.Allreduce(loc, +, comm) / P.m)
    yblk = TestMat.rhs_block(Ablk, i0, i1, xt, P.noise, sc)
    return i0, i1, Ablk, yblk, xt, M
end

"Rank 0 builds the full A,y and the single-node reference qr(A)\\y (only if small)."
function reference_on_root(P, M, xt, x_dist; label = "")
    rank == 0 || return
    bytes = P.m * P.n * 8
    if bytes > 2_000_000_000
        @printf("[%s] full matrix would be %.1f GB - skipping single-node reference\n", label, bytes/1e9)
        return
    end
    A = TestMat.block(1, P.m, P.n, M)
    sc = norm(A * xt) / sqrt(P.m)
    y = TestMat.rhs_block(A, 1, P.m, xt, P.noise, sc)
    t = @elapsed xq = qr(A) \ y
    sv = svdvals(A); nrmA = sv[1]
    mq = TestMat.metrics(A, y, xq, xq, nrmA)
    md = TestMat.metrics(A, y, x_dist, xq, nrmA)
    @printf("[%s] MEASURED cond(A)=%.3e  (m=%d n=%d family=%s ranks=%d)\n", label, sv[1]/sv[end], P.m, P.n, P.family, nprocs)
    @printf("[%s] single-node LAPACK qr:  rel.resid=%.3e  normality=%.3e  (%.2fs)\n", label, mq.res, mq.normality, t)
    @printf("[%s] distributed solution :  rel.resid=%.3e  normality=%.3e\n", label, md.res, md.normality)
    @printf("[%s] AGREEMENT ||x_dist - x_qr||/||x_qr|| = %.3e   (cond*eps = %.1e)\n", label, md.fwd, sv[1]/sv[end]*eps())
end
