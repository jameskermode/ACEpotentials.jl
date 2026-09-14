using ACEpotentials, LinearAlgebra, SparseArrays, Random, Printf
include("./ttmap.jl"); include("./tt.jl")
m = ace1_model(elements = [:Cr,:Mn,:Fe,:Co,:Ni], order = 3, totaldegree = 4)
tm = build_ttmap(m.model; D = 4)
for share in (true, false)
sp = TTSpec(5, 5, [[1,3],[1,4,6],[1,5,7,8]], share)
mdl = TTModel(sp, tm, 100; init = :random)
for z0 in 1:5, ib in 1:length(tm.betas); mdl.v[z0][ib] .= randn(size(mdl.v[z0][ib])); end
c = coefficients(mdl)
for t in 1:3
   Mt, off = core_map(mdl, t)
   g = zeros(size(Mt, 2))
   for ν in t:3; n = length(mdl.G[ν][t]); g[off[ν]+1:off[ν]+n] .= vec(mdl.G[ν][t]); end
   println("share=$share t=$t: |M_t g − c| = ", norm(Mt * g - c), "  ncols ", size(Mt,2))
end
Mv, offv = readout_map(mdl)
w = zeros(size(Mv, 2)); for z0 in 1:5, ib in 1:length(tm.betas); n = length(mdl.v[z0][ib]); w[offv[z0,ib]+1:offv[z0,ib]+n] .= vec(mdl.v[z0][ib]); end
println("share=$share v: |M_v w − c| = ", norm(Mv * w - c))
end
sp = TTSpec(5, 5, [[1,3],[1,4,6],[1,5,7,8]], true)
mdl = TTModel(sp, tm, 100; init = :random)
for z0 in 1:5, ib in 1:length(tm.betas); mdl.v[z0][ib] .= randn(size(mdl.v[z0][ib])); end
c = coefficients(mdl)
Mt, off = core_map(mdl, 2)
g = zeros(size(Mt, 2)); for ν in 2:3; n = length(mdl.G[ν][2]); g[off[ν]+1:off[ν]+n] .= vec(mdl.G[ν][2]); end
d = Mt * g - c
bad = findall(x -> abs(x) > 1e-10, d)
println("bad rows: ", length(bad), " of ", length(d), " first: ", bad[1:5])
for r in bad[1:3]
   j = mod1(r, length(tm.idx)); ix = tm.idx[j]
   println("  row $r j=$j ν=$(length(ix.ζ)) ζ=$(ix.ζ) β=$(ix.β): Mt*g=$(dot(Mt[r,:], g)) c=$(c[r])")
end
ok = findall(x -> abs(x) <= 1e-10, d)
println("ok example: ", [ (tm.idx[mod1(r, length(tm.idx))].ζ) for r in ok[1:3] ])
println("--- block-step consistency: touched rows via M_t, untouched constant")
Mt2, _ = core_map(mdl, 2); touched = vec(any(!=(0.0), Mt2; dims = 2))
println("touched rows: ", count(touched), " untouched rows all order-1? ", all(length(tm.idx[mod1(r, length(tm.idx))].ζ) == 1 for r in findall(.!touched)))
