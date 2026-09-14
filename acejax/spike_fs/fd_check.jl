# Finite-difference verification of the FS column forces and virials.
using ACEpotentials, AtomsBase, StaticArrays, LinearAlgebra, Printf
include(joinpath(@__DIR__, "fs_columns.jl"))
DATA = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad/distil/cantor1k_b_mh1.xyz"
data = ACEpotentials.ExtXYZ.load(DATA)
sys = data[1]
println("structure 1: natoms=", length(sys), " species=", unique(atomic_number(sys, :)))
ELS = [24, 25, 26, 27, 28]
for phi in (:sqrt, :linear), shared in (false, true)
   fs = FSSpec(ELS; phi = phi, shared = shared)
   E, F, V = fs_efv(fs, sys)
   r = fd_check(fs, sys; h = 1e-5, natoms_check = 6)
   @printf("phi=%-7s shared=%-5s ncol=%3d  |F|max=%.3e  |V|max=%.3e   FD err F=%.3e  V=%.3e\n",
           phi, shared, ncolumns(fs), r.maxabs_F, r.maxabs_V, r.maxerr_F, r.maxerr_V)
   # also check the row layout of the feature matrix against fs_efv
   X = fs_feature_matrix(fs, sys)
   @assert size(X, 1) == 1 + 3length(sys) + 6
   @assert X[1, :] == E
   @assert X[2:4, 1] == F[1, 1] && X[5:7, 1] == F[2, 1]
   @assert X[end-5:end, 1] == V[1][SVector(1,5,9,6,3,2)]
end
# Step-size sweep for the sqrt variant to show the FD error is O(h^2), not a bug
fs = FSSpec(ELS)
for h in (1e-2, 1e-3, 1e-4, 1e-5)
   r = fd_check(fs, sys; h = h, natoms_check = 3)
   @printf("h=%.0e  FD err F=%.3e  V=%.3e\n", h, r.maxerr_F, r.maxerr_V)
end
# sanity: force sum is zero (translation invariance) per column
E, F, V = fs_efv(fs, sys)
println("max |sum_j F_j| over columns = ", maximum(norm(sum(F[:, c])) for c in 1:ncolumns(fs)))
println("E (first 6 cols) = ", E[1:6])
