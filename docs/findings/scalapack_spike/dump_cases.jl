# Dump the SAME test matrices + single-node LAPACK references to raw binary for
# the JAX/lineax arm. Row-major so numpy fromfile().reshape(m,n) gives A.
include("testmat.jl")
using .TestMat, LinearAlgebra, Printf
BLAS.set_num_threads(32)
outdir = length(ARGS) >= 1 ? ARGS[1] : "cases"
mkpath(outdir)
function dump(name, m, n, cnd, fam; lambdas = (0.0, 1e-3), do_cond = true)
    M = TestMat.mixer(n, cnd, fam); A = TestMat.block(1, m, n, M)
    xt = TestMat.xtrue(n); sc = norm(A*xt)/sqrt(m); y = TestMat.rhs_block(A, 1, m, xt, 1e-3, sc)
    d = joinpath(outdir, name); mkpath(d)
    if do_cond; write(joinpath(d, "A.bin"), permutedims(A)); else; write(joinpath(d, "A.bin"), A); end   # big cases: column-major, no transpose copy
    write(joinpath(d, "y.bin"), y)
    meta = String["\"order\": \"$(do_cond ? "C" : "F")\"", "\"m\": $m", "\"n\": $n", "\"family\": \"$fam\"", "\"cond_requested\": $cnd"]
    if do_cond
        sv = svdvals(A); push!(meta, "\"cond_measured\": $(sv[1]/sv[end])", "\"normA\": $(sv[1])")
    else
        push!(meta, "\"normA\": $(opnorm(A, 1))")   # not used for accuracy; timing case only
    end
    for lam in lambdas
        Aa = lam == 0 ? A : [A; lam*I(n)]; ya = lam == 0 ? y : [y; zeros(n)]
        GC.gc(); t = @elapsed xq = qr(Aa) \ ya
        write(joinpath(d, "xqr_lam$(lam).bin"), xq)
        push!(meta, "\"t_cpu_qr_lam$(lam)\": $t")
        @printf("%s lam=%g  CPU LAPACK qr (%d threads): %.2fs\n", name, lam, BLAS.get_num_threads(), t)
    end
    write(joinpath(d, "meta.json"), "{" * join(meta, ", ") * "}\n")
end
mode = length(ARGS) >= 2 ? ARGS[2] : "small"
if mode == "small"
    for fam in (:scaled, :mixed), cnd in (1e12, 1e16, 1e21)
        dump("acc_$(fam)_$(cnd)", 20000, 500, cnd, fam)
    end
else
    dump("time_50k_4k", 50000, 4000, 1e16, :scaled; lambdas = (1e-3,), do_cond = false)
    dump("time_101k_8k", 101000, 8000, 1e16, :scaled; lambdas = (1e-3,), do_cond = false)
end
