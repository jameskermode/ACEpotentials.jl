# The map from a species-ordered (TT / CP) coefficient space onto the categorical
# ACE basis, derived through the 𝔸 (product) layer rather than guessed.
#
# Categorical basis (per central species z0): B_{b,q} = Σ_{M} A2B[(b,q), (b,M)] 𝔸_{b,M},
# where b is a block = sorted tuple of (n, l) with FOLDED n = (n'-1)*S + z
# (z = neighbour species), q = 1..num_b indexes ET's PI-symmetrised invariant
# couplings of the block, and 𝔸_{b,M} = Π_t A_{n_t l_t m_t} (sorted (n,l,m) tuple).
#
# TT coefficient space: index (β, η, ζ) with
#   β = sorted tuple of (n', l) pairs        (the "coupling block", radial+angular)
#   η = 1..num_η(β): OUR invariant coupling basis C^η_{L,M} for the ORDERED
#       l-tuple L of β in β's canonical slot order (O3.coupling_coeffs PI=false,
#       orthonormalised); shared by every species assignment of β.
#   ζ ∈ [S]^ν: ORDERED species tuple, slot t carries (n'_t, l_t, ζ_t).
# The function f_{β,η,ζ} = Σ_M C^η_{L,M} Π_t A_{(n'_t,ζ_t), l_t, m_t} is rotation
# invariant and lies in span{B_{b,·}} for the block b = sort((fold(n'_t,ζ_t), l_t)_t);
# Φ[(b,q), (β,η,ζ)] is its expansion, obtained per block by solving
# S_bᵀ x = d (S_b = A2B block, d = 𝔸-coefficients of f).  The residual of that
# solve is checked (it is the completeness check of ET's coupling basis, and it
# would expose any permutation / multiplicity convention mismatch).
#
# The symmetric-basis rule of the spec ("the categorical column for τ absorbs
# all slot orderings") is therefore not assumed but computed: distinct orderings
# ζ, ζ' of the same species multiset (only possible when β has repeated (n',l)
# pairs) give separate columns of Φ, which coincide iff the coupling is
# symmetric under that slot swap.

using ACEpotentials, LinearAlgebra, SparseArrays
const ET_ = ACEpotentials.Models.EquivariantTensors

struct TTIndex
   β::Vector{Tuple{Int,Int}}      # canonical slot order: sorted (n', l)
   η::Int
   ζ::Vector{Int}                 # ordered species per slot
   block::Int                     # categorical block id
   ibeta::Int                     # id of β in `betas`
end

struct TTMap
   S::Int
   nB::Int                        # categorical MB columns per z0
   betas::Vector{Vector{Tuple{Int,Int}}}
   beta_L::Vector{Vector{Int}}    # l-tuple in slot order
   beta_C::Vector{Matrix{Float64}}    # num_η × |Mgrid| coupling basis
   beta_M::Vector{Vector{Vector{Int}}}
   beta_emb::Vector{Bool}         # in the channel-free-degree (embedded) spec?
   idx::Vector{TTIndex}           # the TT coefficient index set (length N_c)
   Φ::SparseMatrixCSC{Float64,Int}    # nB × N_c
   block_nnll::Vector{Vector{@NamedTuple{n::Int,l::Int}}}
   maxres::Float64
end

unfold_n(n, S) = (div(n - 1, S) + 1, mod1(n, S))
fold_n(np, z, S) = (np - 1) * S + z

function build_ttmap(mm; wL = 1.5, D = nothing)
   S = ACEpotentials.Models._get_nz(mm)
   tensor = mm.tensor
   A2B = tensor.A2Bmaps[1]
   AAspec = tensor.meta["𝔸spec"]
   nnll = ACEpotentials.Models.get_nnll_spec(tensor)
   nB = length(nnll)
   # --- categorical blocks
   # ET's nnll tuples are not always sorted ([(2,0),(1,1),(1,1)] occurs); key
   # the blocks by the SORTED (n,l) tuple, which is what the 𝔸spec order gives.
   blocks = unique(sort.(nnll))
   binv = Dict(b => i for (i, b) in enumerate(blocks))
   brows = [Int[] for _ in blocks]
   for (i, bb) in enumerate(nnll); push!(brows[binv[sort(bb)]], i); end
   bcols = [Int[] for _ in blocks]
   AA_nl = [sort([(n = b.n, l = b.l) for b in bb]) for bb in AAspec]
   for (j, bb) in enumerate(AA_nl)
      haskey(binv, bb) && push!(bcols[binv[bb]], j)
   end
   # 𝔸spec tuples are in the block's slot order, not sorted: key by the sorted tuple
   colinv = [Dict(sort(AAspec[j]) => k for (k, j) in enumerate(bcols[i])) for i in 1:length(blocks)]
   @assert all(length(colinv[i]) == length(bcols[i]) for i in 1:length(blocks))
   Sb = [Matrix(A2B[brows[i], bcols[i]]) for i in 1:length(blocks)]
   # --- β's
   βof(bb) = sort([(unfold_n(b.n, S)[1], b.l) for b in bb])
   betas = unique(βof.(blocks))
   sort!(betas, by = β -> (length(β), β))
   βinv = Dict(β => i for (i, β) in enumerate(betas))
   beta_L = [[p[2] for p in β] for β in betas]
   beta_C = Matrix{Float64}[]; beta_M = Vector{Vector{Int}}[]
   for L in beta_L
      U, MM = ET_.O3.coupling_coeffs(0, Tuple(L), collect(1:length(L)); PI = false, basis = real)
      # orthonormalise rows (rank-revealing), so η is a clean basis
      if size(U, 1) > 0
         F = svd(Matrix(U)); r = count(>(1e-10 * F.S[1]), F.S)
         C = Matrix(F.Vt[1:r, :])
      else
         C = zeros(0, length(MM))
      end
      push!(beta_C, C); push!(beta_M, [Vector{Int}(m) for m in MM])
   end
   lev(β) = sum(p[1] + wL * p[2] for p in β)
   Dmax = D === nothing ? maximum(lev.(betas[[b for b in 1:length(betas) if true]])) : D
   beta_emb = [lev(β) <= Dmax + 1e-9 for β in betas]
   # --- enumerate (β, η, ζ), build Φ
   idx = TTIndex[]
   I_ = Int[]; J_ = Int[]; V_ = Float64[]
   maxres = 0.0
   for (ib, β) in enumerate(betas)
      ν = length(β); L = beta_L[ib]; C = beta_C[ib]; MM = beta_M[ib]
      nη = size(C, 1)
      nη == 0 && continue
      for ζ in Iterators.product(ntuple(_ -> 1:S, ν)...)
         ζv = collect(ζ)
         bb = sort([(n = fold_n(β[t][1], ζv[t], S), l = β[t][2]) for t in 1:ν])
         haskey(binv, bb) || continue
         b = binv[bb]
         # 𝔸 coefficients of every η at once: d (|cols_b| × nη)
         d = zeros(length(bcols[b]), nη)
         for (jm, m) in enumerate(MM)
            tup = sort([(n = fold_n(β[t][1], ζv[t], S), l = β[t][2], m = m[t]) for t in 1:ν])
            k = get(colinv[b], tup, 0)
            if k == 0
               # pruned 𝔸 element: the coefficient must vanish
               maxres = max(maxres, maximum(abs, C[:, jm]))
            else
               d[k, :] .+= C[:, jm]
            end
         end
         X = Sb[b]' \ d                     # least squares, num_b × nη
         res = norm(Sb[b]' * X - d) / max(norm(d), 1e-300)
         maxres = max(maxres, res)
         for η in 1:nη
            push!(idx, TTIndex(β, η, ζv, b, ib))
            col = length(idx)
            for (r, row) in enumerate(brows[b])
               abs(X[r, η]) > 1e-14 || continue
               push!(I_, row); push!(J_, col); push!(V_, X[r, η])
            end
         end
      end
   end
   Φ = sparse(I_, J_, V_, nB, length(idx))
   return TTMap(S, nB, betas, beta_L, beta_C, beta_M, beta_emb, idx, Φ, blocks, maxres)
end

# number of η per β, orders, etc.
n_eta(tm::TTMap, ib) = size(tm.beta_C[ib], 1)
order(tm::TTMap, ib) = length(tm.betas[ib])
