using ACEpotentials, LinearAlgebra, SparseArrays, Printf
include(joinpath(@__DIR__, "ttmap.jl"))
M = ACEpotentials.Models
D = parse(Int, get(ENV, "DEG", "4"))
m = ace1_model(elements = [:Cr,:Mn,:Fe,:Co,:Ni], order = 3, totaldegree = D)
t = @elapsed tm = build_ttmap(m.model; D = D)
@printf("D=%d built in %.1f s: nB=%d  N_c=%d  #β=%d  maxres=%.2e  nnz(Φ)=%d  rank(Φ)=%d\n",
        D, t, tm.nB, length(tm.idx), length(tm.betas), tm.maxres, nnz(tm.Φ), rank(Matrix(tm.Φ)))
for (ib, β) in enumerate(tm.betas)
   n = count(x -> x.ibeta == ib, tm.idx)
   @printf("  β=%-40s ν=%d nη=%d emb=%d  ncoef=%d\n", string(β), length(β), n_eta(tm, ib), tm.beta_emb[ib], n)
end
# surjectivity: every categorical column reachable
rows_hit = unique(rowvals(tm.Φ)); println("rows covered: ", length(rows_hit), " / ", tm.nB)
miss = setdiff(1:tm.nB, rows_hit)
for r in miss
   bb = tm.block_nnll[r]
   println("  missing row $r: ", bb, "  unfolded: ", [(unfold_n(b.n, tm.S), b.l) for b in bb])
end
