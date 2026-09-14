# Fixed Finnis-Sinclair density columns for a linear ACE design matrix.
#
# For site i with central species z_i, neighbour species s and radial index k:
#     rho_i^{s,k} = sum_{j : z_j = s, r_ij < rcut} g_k(r_ij)
#     g_k(r)      = exp(-alpha_k (r/r0 - 1)) * fcut(r)
#     fcut(r)     = (1 - r/rcut)^2 (1 + 2 r/rcut)       (r < rcut, else 0)
# and the column ("basis function") for (z_i = a, s, k) is
#     B_{a,s,k}(R) = sum_{i : z_i = a} phi(rho_i^{s,k})
# with phi(rho) = sqrt(rho + eps) - sqrt(eps)   (:sqrt, the FS term), or
#      phi(rho) = rho                            (:linear, the control -- this
#                                                 is a plain pair potential and
#                                                 lies in the span of the 2-body
#                                                 ACE basis).
# Nothing here is fitted: the columns multiply linear coefficients c_{a,s,k}
# exactly like ACE basis functions, so the fit stays linear least squares.
#
# Row layout per structure matches ACEfit.feature_matrix(::AtomsData) in
# src/atoms_data.jl: [energy (1); forces (3*natoms, atom-major x,y,z);
# virial (6) = vec(V)[[1,5,9,6,3,2]] i.e. xx, yy, zz, yz, xz, xy].
# Virial convention (calculators.jl `_site_virial`): V = - sum_i dV_i (x) R_i
# = -dE/d(strain) at zero strain; checked by finite differences in fd_check().

using StaticArrays, LinearAlgebra, AtomsBase, Unitful
using ACEpotentials
const NL = ACEpotentials.Models.NeighbourLists

struct FSSpec
   species::Vector{Int}          # atomic numbers, in column order
   alphas::Vector{Float64}       # K values
   r0::Float64
   rcut::Float64
   eps::Float64
   phi::Symbol                   # :sqrt or :linear
   shared::Bool                  # true: one column set for all central species
   total::Bool                   # true: phi(sum_s rho^{s,k}) -- one density per k
end

FSSpec(species; alphas = [2.0, 4.0, 6.0], r0 = 2.5, rcut = 6.25, eps = 1e-8,
                phi = :sqrt, shared = false, total = false) =
   FSSpec(Int.(species), Float64.(alphas), r0, rcut, eps, phi, shared, total)

nspecies(fs::FSSpec) = length(fs.species)
nK(fs::FSSpec) = length(fs.alphas)
# number of distinct neighbour-density channels per central species
nS(fs::FSSpec) = fs.total ? 1 : nspecies(fs)
ncolumns(fs::FSSpec) = (fs.shared ? 1 : nspecies(fs)) * nS(fs) * nK(fs)

# column index for (central species index a, neighbour species index s, k)
colidx(fs::FSSpec, a, s, k) =
   ((fs.shared ? 0 : (a - 1)) * nS(fs) + (fs.total ? 0 : (s - 1))) * nK(fs) + k

function fcut(r, rcut)
   r >= rcut && return 0.0
   x = r / rcut
   return (1 - x)^2 * (1 + 2x)
end
function dfcut(r, rcut)
   r >= rcut && return 0.0
   x = r / rcut
   # d/dx [(1-x)^2 (1+2x)] = -2(1-x)(1+2x) + 2(1-x)^2 = -6x(1-x)
   return -6 * x * (1 - x) / rcut
end

g(r, alpha, r0, rcut) = exp(-alpha * (r / r0 - 1)) * fcut(r, rcut)
function dg(r, alpha, r0, rcut)
   e = exp(-alpha * (r / r0 - 1))
   return e * (-alpha / r0 * fcut(r, rcut) + dfcut(r, rcut))
end

phi(fs::FSSpec, rho) = fs.phi == :sqrt ? sqrt(rho + fs.eps) - sqrt(fs.eps) : rho
dphi(fs::FSSpec, rho) = fs.phi == :sqrt ? 1 / (2 * sqrt(rho + fs.eps)) : 1.0

"""
    fs_efv(fs, sys) -> (E, F, V)

E :: Vector (ncol), F :: Matrix{SVector{3}} (natoms x ncol), V :: Vector{SMatrix{3,3}} (ncol)
Energy, forces and virial of every FS column on one structure.
"""
function fs_efv(fs::FSSpec, sys)
   nat = length(sys)
   ncol = ncolumns(fs)
   S, K = nS(fs), nK(fs)
   Z = [atomic_number(sys, i) for i in 1:nat]
   zidx = [findfirst(==(z), fs.species) for z in Z]
   any(isnothing, zidx) && error("species not in FSSpec")
   nlist = NL.PairList(sys, fs.rcut * u"Å")
   # pass 1: densities rho[i, s, k]
   rho = zeros(nat, S, K)
   for (i, j, R) in NL.pairs(nlist)
      r = norm(R)
      r >= fs.rcut && continue
      s = fs.total ? 1 : zidx[j]
      for k in 1:K
         rho[i, s, k] += g(r, fs.alphas[k], fs.r0, fs.rcut)
      end
   end
   # energies
   E = zeros(ncol)
   for i in 1:nat, s in 1:S, k in 1:K
      E[colidx(fs, zidx[i], s, k)] += phi(fs, rho[i, s, k])
   end
   # pass 2: forces and virials.  E_col = sum_i phi(rho_i); R = r_j - r_i (+shift)
   # dE/dr_j += phi'(rho_i) g'(r) R/r ;  dE/dr_i -= same ;  F = -dE/dr
   z3 = zero(SVector{3, Float64})
   F = fill(z3, nat, ncol)
   V = fill(zero(SMatrix{3, 3, Float64}), ncol)
   for (i, j, R) in NL.pairs(nlist)
      r = norm(R)
      r >= fs.rcut && continue
      s = fs.total ? 1 : zidx[j]
      Rhat = R / r
      for k in 1:K
         c = colidx(fs, zidx[i], s, k)
         dV = dphi(fs, rho[i, s, k]) * dg(r, fs.alphas[k], fs.r0, fs.rcut) * Rhat  # dE_i/dR_ij
         F[j, c] -= dV
         F[i, c] += dV
         V[c] -= dV * R'
      end
   end
   return E, F, V
end

"""
    fs_feature_matrix(fs, sys; energy=true, force=true, virial=true) -> Matrix

Rows in the ACEfit.feature_matrix layout: [1 energy; 3*natoms forces; 6 virial].
"""
function fs_feature_matrix(fs::FSSpec, sys; energy = true, force = true, virial = true)
   nat = length(sys)
   E, F, V = fs_efv(fs, sys)
   nrow = energy + 3 * nat * force + 6 * virial
   X = zeros(nrow, ncolumns(fs))
   i = 1
   if energy
      X[i, :] .= E; i += 1
   end
   if force
      X[i:i+3nat-1, :] .= reinterpret(Float64, F)   # atom-major x,y,z: same as _f_mat
      i += 3nat
   end
   if virial
      for c in 1:ncolumns(fs)
         X[i:i+5, c] .= V[c][SVector(1, 5, 9, 6, 3, 2)]
      end
      i += 6
   end
   return X
end

fs_feature_matrix(fs::FSSpec, data::AbstractVector; kw...) =
   reduce(vcat, [fs_feature_matrix(fs, sys; kw...) for sys in data])

# ---------------------------------------------------------------------------
# finite-difference verification of forces and virial on one structure
# ---------------------------------------------------------------------------
_sv(x) = SVector{3}(x)

function _displaced(sys, X)
   # rebuild the system with new (unitful) positions
   return FastSystem(cell_vectors(sys), periodicity(sys),
                     [_sv(x) for x in X], species(sys, :), mass(sys, :))
end

function _strained(sys, eps)
   Fm = SMatrix{3, 3}(I + eps)
   cv = cell_vectors(sys)
   newcell = ntuple(a -> Fm * _sv(cv[a]), 3)
   X = [Fm * _sv(position(sys, i)) for i in 1:length(sys)]
   return FastSystem(newcell, periodicity(sys), X, species(sys, :), mass(sys, :))
end

function fd_check(fs::FSSpec, sys; h = 1e-4, natoms_check = 4)
   E0, F0, V0 = fs_efv(fs, sys)
   nat = length(sys)
   ncol = ncolumns(fs)
   # forces: F_j = -dE/dr_j, central differences on the first few atoms
   maxerr_F = 0.0
   X0 = [_sv(position(sys, i)) for i in 1:nat]
   for j in 1:min(nat, natoms_check), a in 1:3
      Xp = copy(X0); Xp[j] = Xp[j] + SVector{3}(a == 1, a == 2, a == 3) * h * u"Å"
      Xm = copy(X0); Xm[j] = Xm[j] - SVector{3}(a == 1, a == 2, a == 3) * h * u"Å"
      Ep, _, _ = fs_efv(fs, _displaced(sys, Xp))
      Em, _, _ = fs_efv(fs, _displaced(sys, Xm))
      dE = (Ep - Em) / (2h)
      for c in 1:ncol
         maxerr_F = max(maxerr_F, abs(-dE[c] - F0[j, c][a]))
      end
   end
   # virial: V_ab = -dE/d eps_ab at zero strain
   maxerr_V = 0.0
   for a in 1:3, b in 1:3
      eps = zeros(3, 3); eps[a, b] = h
      Ep, _, _ = fs_efv(fs, _strained(sys, eps))
      Em, _, _ = fs_efv(fs, _strained(sys, -eps))
      dE = (Ep - Em) / (2h)
      for c in 1:ncol
         maxerr_V = max(maxerr_V, abs(-dE[c] - V0[c][a, b]))
      end
   end
   return (maxerr_F = maxerr_F, maxerr_V = maxerr_V,
           maxabs_F = maximum(norm, F0), maxabs_V = maximum(x -> maximum(abs, x), V0))
end
