# ScaLAPACK pdgels through SCALAPACK32_jll, called directly with ccall.
# grid=1d : nprow = nranks, npcol = 1, MB = rows-per-rank  ->  each rank's local
#           array IS its contiguous row block. This is ACEfit's layout verbatim;
#           no redistribution at all.
# grid=2d : build A in the 1d layout, then pdgemr2d it to a Pr x Pc block-cyclic
#           layout (NB=64) before pdgels. Measures the redistribution cost.
include("mpi_common.jl")
using SCALAPACK32_jll, OpenBLAS32_jll
# SCALAPACK32_jll calls LP64 BLAS/LAPACK through libblastrampoline. Julia only
# forwards its ILP64 OpenBLAS by default, so an LP64 library must be forwarded
# explicitly or every lsame_/dgemm_ call reports "no BLAS/LAPACK library loaded".
BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path; clear = false)
const lib = SCALAPACK32_jll.libscalapack32
const SI = Int32

blacs_pinfo() = (r=Ref{SI}(); p=Ref{SI}(); ccall((:Cblacs_pinfo, lib), Cvoid, (Ref{SI},Ref{SI}), r, p); (r[], p[]))
function blacs_get(ctx, what) ; v=Ref{SI}(); ccall((:Cblacs_get, lib), Cvoid, (SI,SI,Ref{SI}), ctx, what, v); v[]; end
function blacs_gridinit(ctx, nprow, npcol)
    c = Ref{SI}(ctx); ccall((:Cblacs_gridinit, lib), Cvoid, (Ref{SI},Cstring,SI,SI), c, "Row", nprow, npcol); c[]
end
function blacs_gridinfo(ctx)
    a=Ref{SI}();b=Ref{SI}();c=Ref{SI}();d=Ref{SI}()
    ccall((:Cblacs_gridinfo, lib), Cvoid, (SI,Ref{SI},Ref{SI},Ref{SI},Ref{SI}), ctx,a,b,c,d); (a[],b[],c[],d[])
end
blacs_gridexit(ctx) = ccall((:Cblacs_gridexit, lib), Cvoid, (SI,), ctx)
numroc(n, nb, iproc, isrc, np) = ccall((:numroc_, lib), SI, (Ref{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI}), n, nb, iproc, isrc, np)
function descinit(m, n, mb, nb, irsrc, icsrc, ctx, lld)
    desc = zeros(SI, 9); info = Ref{SI}()
    ccall((:descinit_, lib), Cvoid, (Ptr{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI},Ref{SI}),
          desc, m, n, mb, nb, irsrc, icsrc, ctx, max(lld,1), info)
    info[] == 0 || error("descinit info=$(info[])"); desc
end
function pdgemr2d!(m, n, A, desca, B, descb, ctx)
    ccall((:pdgemr2d_, lib), Cvoid, (Ref{SI},Ref{SI},Ptr{Float64},Ref{SI},Ref{SI},Ptr{SI},Ptr{Float64},Ref{SI},Ref{SI},Ptr{SI},Ref{SI}),
          m, n, A, 1, 1, desca, B, 1, 1, descb, ctx)
end
function pdgels!(m, n, nrhs, A, desca, B, descb)
    info = Ref{SI}(); work = [0.0]; lwork = Ref{SI}(-1)
    for pass in 1:2
        ccall((:pdgels_, lib), Cvoid,
              (Ref{UInt8},Ref{SI},Ref{SI},Ref{SI},Ptr{Float64},Ref{SI},Ref{SI},Ptr{SI},
               Ptr{Float64},Ref{SI},Ref{SI},Ptr{SI},Ptr{Float64},Ref{SI},Ref{SI},Csize_t),
              UInt8('N'), m, n, nrhs, A, 1, 1, desca, B, 1, 1, descb, work, lwork, info, 1)
        info[] == 0 || error("pdgels info=$(info[]) pass=$pass")
        pass == 1 && (lwork[] = SI(ceil(work[1])); work = zeros(Int(lwork[])))
    end
    return Int(lwork[])
end

P = parse_args()
m, n = P.m, P.n
i0, i1, Ablk, yblk, xt, M = local_block(P)
mb1 = cld(m, nprocs)                      # 1-D: MB = rows per rank, NB = n
# sanity: ScaLAPACK's block-cyclic 1-D layout with MB=cld(m,p) must equal our partition
@assert i0 == rank*mb1 + 1 || i0 > m
@assert i1 - i0 + 1 == numroc(m, mb1, rank, 0, nprocs)  "row partition mismatch on rank $rank"

myrank, np = blacs_pinfo()
ctx0 = blacs_get(0, 0)
ctx1 = blacs_gridinit(ctx0, nprocs, 1)   # 1-D process column
loc_rows = Int(numroc(m, mb1, rank, 0, nprocs))
descA1 = descinit(m, n, mb1, n, 0, 0, ctx1, loc_rows)
descB1 = descinit(m, 1, mb1, 1, 0, 0, ctx1, loc_rows)
A1 = Ablk; B1 = reshape(copy(yblk), :, 1)
MPI.Barrier(comm); t0 = MPI.Wtime()
if P.grid == "1d"
    lwork = pdgels!(m, n, 1, A1, descA1, B1, descB1)
    t1 = MPI.Wtime()
    xloc = vec(B1)                          # rows 1:n of the distributed B hold x
    rank == 0 && @printf("[SCALAPACK 1d] pdgels on %d x 1 grid, MB=%d NB=%d, lwork=%d doubles/rank: %.2fs\n", nprocs, mb1, n, lwork, t1-t0)
else
    # 2-D grid, block-cyclic NB=64
    local gpr = Int(floor(sqrt(nprocs))); while nprocs % gpr != 0; gpr -= 1; end; local gpc = nprocs ÷ gpr; local pr = gpr; local pc = gpc
    nb = 64
    ctx2 = blacs_gridinit(ctx0, pr, pc)
    _, _, myrow, mycol = blacs_gridinfo(ctx2)
    lr = Int(numroc(m, nb, myrow, 0, pr)); lc = Int(numroc(n, nb, mycol, 0, pc)); lcb = Int(numroc(1, nb, mycol, 0, pc))
    descA2 = descinit(m, n, nb, nb, 0, 0, ctx2, lr); descB2 = descinit(m, 1, nb, nb, 0, 0, ctx2, lr)
    A2 = zeros(lr, lc); B2 = zeros(lr, max(lcb,1))
    tr0 = MPI.Wtime()
    pdgemr2d!(m, n, A1, descA1, A2, descA2, ctx2)   # redistribute 1d -> 2d
    pdgemr2d!(m, 1, B1, descB1, B2, descB2, ctx2)
    tr1 = MPI.Wtime()
    lwork = pdgels!(m, n, 1, A2, descA2, B2, descB2)
    t1 = MPI.Wtime()
    pdgemr2d!(m, 1, B2, descB2, B1, descB1, ctx2)   # solution back to 1d layout
    xloc = vec(B1)
    rank == 0 && @printf("[SCALAPACK 2d] grid %dx%d NB=%d: redistribute 1d->2d %.2fs, pdgels %.2fs, lwork=%d\n", pr, pc, nb, tr1-tr0, t1-tr1, lwork)
    blacs_gridexit(ctx2)
end
# gather solution (first n entries of the distributed B, in row order)
counts = Int32[numroc(m, mb1, r, 0, nprocs) for r in 0:nprocs-1]
full = MPI.Allgatherv!(xloc, MPI.VBuffer(zeros(m), counts), comm)
x = full[1:n]
blacs_gridexit(ctx1)
reference_on_root(P, M, xt, x; label = "SCALAPACK $(P.grid)")
MPI.Finalize()
