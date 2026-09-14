# RMSEs from the design matrix instead of re-evaluating the model.
#
# Given (A, Y) from `ACEpotentials.assemble(structs, model; energy_key, ...)`
# (rows in structure order: [E; F (3 natoms); V (6)] per structure) and a
# coefficient vector c in the basis ordering of `set_linear_parameters!`,
# `rmse_from_design(A, Y, c, layout)` returns Dict("E", "F", "V") of RMSEs that
# match `ACEpotentials.compute_errors(structs, model)["rmse"]["set"]`:
#   E : per-atom energy error, RMS over structures
#   F : per-component force error, RMS over all components
#   V : per-atom virial error (6 Voigt components), RMS over 6*nstruct
using LinearAlgebra

struct RowLayout
   natoms::Vector{Int}
   has_E::Vector{Bool}
   has_F::Vector{Bool}
   has_V::Vector{Bool}
end

# the layout is what AtomsData / count_observations produce, in data order
function RowLayout(structs; energy_key, force_key, virial_key)
   n = length(structs)
   nat = [length(s) for s in structs]
   hE = [ACEpotentials._find_similar_key(s, energy_key) !== nothing for s in structs]
   hF = [ACEpotentials._find_similar_key(s, force_key) !== nothing for s in structs]
   hV = [ACEpotentials._find_similar_key(s, virial_key) !== nothing for s in structs]
   return RowLayout(nat, hE, hF, hV)
end

nrows(L::RowLayout) = sum(L.has_E) + 3 * sum(L.natoms .* L.has_F) + 6 * sum(L.has_V)

# residual vector r = A c - Y, then RMSEs per observation type
function rmse_from_residual(r::AbstractVector, L::RowLayout)
   @assert length(r) == nrows(L)
   sE = 0.0; nE = 0; sF = 0.0; nF = 0; sV = 0.0; nV = 0
   i = 1
   for k = 1:length(L.natoms)
      n = L.natoms[k]
      if L.has_E[k]
         sE += (r[i] / n)^2; nE += 1; i += 1
      end
      if L.has_F[k]
         @inbounds for j = i:i+3n-1
            sF += r[j]^2
         end
         nF += 3n; i += 3n
      end
      if L.has_V[k]
         @inbounds for j = i:i+5
            sV += (r[j] / n)^2
         end
         nV += 6; i += 6
      end
   end
   return Dict("E" => nE == 0 ? 0.0 : sqrt(sE / nE),
               "F" => nF == 0 ? 0.0 : sqrt(sF / nF),
               "V" => nV == 0 ? 0.0 : sqrt(sV / nV))
end

rmse_from_design(A::AbstractMatrix, Y::AbstractVector, c::AbstractVector, L::RowLayout) =
   rmse_from_residual(A * c .- Y, L)

# for a lambda sweep: precompute nothing, each lambda costs one matvec (m n)
