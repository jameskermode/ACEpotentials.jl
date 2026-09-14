# FD verification of the general embedding columns (powers, Chebyshev, cross terms)
using ACEpotentials, AtomsBase, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "fs_embed.jl"))
DATA = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad/distil/cantor1k_b_mh1.xyz"
data = ACEpotentials.ExtXYZ.load(DATA)
sys = data[1]
ELS = [24, 25, 26, 27, 28]; K = 3
base = EmbedSpec(ELS, [], [])
# ranges from the first 20 structures (any range is fine for an FD check)
lo_s, hi_s = density_ranges(base, data[1:20], :sqrt)
lo_l, hi_l = density_ranges(base, data[1:20], :log)
println("sqrt(rho) ranges per width: ", round.(lo_s, digits=3), " .. ", round.(hi_s, digits=3))
println("log(rho)  ranges per width: ", round.(lo_l, digits=3), " .. ", round.(hi_l, digits=3))
specs = [
   ("powers m in {1/8,1/4,1/2,3/4,1,2}", EmbedSpec(ELS, [power_fun(K, k, m) for m in (1/8, 1/4, 1/2, 3/4, 1, 2) for k in 1:K], ["p" for _ in 1:6K])),
   ("cheb sqrt Kc=8",  EmbedSpec(ELS, vcat([cheb_funs(K, k, 8, :sqrt, lo_s[k], hi_s[k]) for k in 1:K]...), ["c" for _ in 1:8K])),
   ("cheb log Kc=8",   EmbedSpec(ELS, vcat([cheb_funs(K, k, 8, :log, lo_l[k], hi_l[k]) for k in 1:K]...), ["c" for _ in 1:8K])),
   ("cross sqrt(rho_k1 rho_k2)", EmbedSpec(ELS, [cross_fun(K, 1, 2), cross_fun(K, 1, 3), cross_fun(K, 2, 3)], ["x", "x", "x"])),
   ("weighted sqrt",   EmbedSpec(ELS, [wsqrt_fun(K, SVector(1.0, 1.0, 1.0)), wsqrt_fun(K, SVector(1.0, 0.5, 0.25))], ["w", "w"])),
]
for (name, es) in specs
   r = fd_check(es, sys; h = 1e-5, natoms_check = 6)
   @printf("%-32s ncol=%3d  |F|max=%.3e |V|max=%.3e   FD err F=%.3e  V=%.3e\n",
           name, ncolumns(es), r.maxabs_F, r.maxabs_V, r.maxerr_F, r.maxerr_V)
   X = fs_feature_matrix(es, sys)
   @assert size(X, 1) == 1 + 3length(sys) + 6
end
# consistency with fs_columns.jl: sqrt(rho_tot) via EmbedSpec == FSSpec(total=true)
es = EmbedSpec(ELS, [power_fun(K, k, 0.5) for k in 1:K], ["s", "s", "s"])
fs = FSSpec(ELS; phi = :sqrt, total = true)
Xa = fs_feature_matrix(es, sys); Xb = fs_feature_matrix(fs, sys)
println("max |EmbedSpec sqrt - FSSpec(total) sqrt| = ", maximum(abs, Xa - Xb))
# step-size scaling for the Chebyshev virial (largest FD error above)
es = specs[2][2]
for h in (1e-3, 1e-4, 1e-5, 1e-6)
   r = fd_check(es, sys; h = h, natoms_check = 2)
   @printf("cheb sqrt Kc=8  h=%.0e  FD err F=%.3e  V=%.3e\n", h, r.maxerr_F, r.maxerr_V)
end
# and Kc=4 at h=1e-5 (lower-order polynomial, smaller third derivative)
es4 = EmbedSpec(ELS, vcat([cheb_funs(K, k, 4, :sqrt, lo_s[k], hi_s[k]) for k in 1:K]...), ["c" for _ in 1:4K])
r = fd_check(es4, sys; h = 1e-5, natoms_check = 6)
@printf("cheb sqrt Kc=4  h=1e-05  |V|max=%.3e  FD err F=%.3e  V=%.3e\n", r.maxabs_V, r.maxerr_F, r.maxerr_V)
