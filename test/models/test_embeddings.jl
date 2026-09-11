# Frozen element embeddings: the radial basis must come out as
#     R(n'k)l(r, Z1, Z2) = P_n'(r) * emb[Z2, k]
# which is `set_onehot_weights!`'s construction with the one-hot δ replaced by a
# frozen embedding row.  See src/models/embeddings.jl.

using ACEpotentials, Test, LinearAlgebra, Random, JSON, StaticArrays, ACEfit
M = ACEpotentials.Models

# local, so the file runs in any environment that can load ACEpotentials
print_tf(r::Test.Pass) = printstyled("+", bold=true, color=:green)
print_tf(r::Test.Fail) = printstyled("-", bold=true, color=:red)
print_tf(r) = printstyled("x", bold=true, color=:magenta)
println_slim(r::Test.Pass) = printstyled("Test Passed\n", bold=true, color=:green)
println_slim(r) = printstyled("Test Failed\n", bold=true, color=:red)

##

@info("ElementEmbedding: widths, row lookup, artefact round-trip")

# d_ν = min(d_max, C(S+ν-1, ν)); with no cap every order gets its full dimension
println_slim(@test M.embedding_widths(3, 4) == [3, 6, 10, 15])
println_slim(@test M.embedding_widths(10, 3) == [10, 55, 220])
println_slim(@test M.embedding_widths(10, 3; d_max = 32) == [10, 32, 32])
println_slim(@test M.embedding_widths(1, 3) == [1, 1, 1])     # single species

# a synthetic artefact, so the test needs no downloaded checkpoint
tmp = tempname() * ".json"
Zs = [14, 6, 8]
emb = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]
open(tmp, "w") do io
   JSON.print(io, Dict("Z" => Zs, "emb" => [emb[i, :] for i = 1:size(emb, 1)],
                       "checkpoint" => "synthetic-for-tests"))
end
e = M.read_mace_embedding(tmp)
println_slim(@test e.Z == Zs)
println_slim(@test e.emb == emb)
println_slim(@test e.meta["checkpoint"] == "synthetic-for-tests")

# rows come back in the order asked for, truncated to d
println_slim(@test M.embedding_rows(e, [6, 14]; d = 2, normalise = false) ==
                   [5.0 6.0; 1.0 2.0])

# and normalised AFTER truncation by default: each returned row is a unit vector
# in the d channels actually used, not in the full table width.  Normalising the
# full row and then truncating leaves ~1/sqrt(d_full) per channel, which shows up
# as a badly regularised fit rather than as an error.
rn = M.embedding_rows(e, [6, 14]; d = 2)
println_slim(@test all(isapprox(norm(rn[i, :]), 1.0; atol = 1e-14) for i = 1:2))
println_slim(@test rn[1, :] ≈ [5.0, 6.0] ./ norm([5.0, 6.0]))
println_slim(@test_throws Exception M.embedding_rows(e, [79]))      # Au absent
println_slim(@test_throws Exception M.embedding_rows(e, [14]; d = 99))

##

@info("set_embedding_weights!: Rnl == P_n'(r) * emb[Z2, k]")

# NOTE: `ace1_model` SPLINES its radial basis (SplineRnlrzzBasis), and a spline
# has no Wnlq to set -- so the embedding must be applied to the LEARNABLE basis,
# before splining.  `ace_model` keeps the analytic/learnable branch, which is the
# one this operates on.
using Random: MersenneTwister
import Lux
d = size(emb, 2)
ri = M._default_rin0cuts((:Si, :C, :O))
ri = (x -> (rin = x.rin, r0 = x.r0, rcut = 5.5)).(ri)
raw = M.ace_model(; elements = (:Si, :C, :O), order = 2, Ytype = :solid,
                  level = M.TotalDegree(), max_level = 8, maxl = 4,
                  pair_maxn = 8, rin0cuts = ri,
                  init_WB = :glorot_normal, init_Wpair = :glorot_normal)
ps0, st0 = Lux.setup(MersenneTwister(1234), raw)
model = M.ACEPotential(raw, ps0, st0)
rbasis = model.model.rbasis
println_slim(@test rbasis isa M.LearnableRnlrzzBasis)
ps = deepcopy(model.ps)
M.set_embedding_weights!(rbasis, ps.rbasis, emb)

NZ = length(Zs)

# CHECK 1 -- against trusted existing code.  With emb = I the embedding row for
# species iz2 IS the one-hot δ_{k, iz2}, so `set_embedding_weights!` must
# reproduce `set_onehot_weights!` bit for bit.  This is a far stronger check than
# re-deriving P_n'(r) here would be, and it cannot drift from the convention.
ps_hot = deepcopy(model.ps)
M.set_onehot_weights!(rbasis, ps_hot.rbasis)
ps_eye = deepcopy(model.ps)
M.set_embedding_weights!(rbasis, ps_eye.rbasis, Matrix{Float64}(I, NZ, NZ))
println_slim(@test ps_eye.rbasis.Wnlq == ps_hot.rbasis.Wnlq)

# CHECK 2 -- the defining property, without touching any internal basis API.
# R depends on the neighbour species ONLY through its embedding row, linearly.
# Scaling one row by c must scale that species' Rnl by exactly c and leave every
# other species untouched.
rng = MersenneTwister(7)
c = 3.7
for iz2_scaled = 1:NZ
   emb2 = copy(emb); emb2[iz2_scaled, :] .*= c
   psA = deepcopy(model.ps); M.set_embedding_weights!(rbasis, psA.rbasis, emb)
   psB = deepcopy(model.ps); M.set_embedding_weights!(rbasis, psB.rbasis, emb2)
   ok = true
   for iz1 = 1:NZ, iz2 = 1:NZ
      r = 1.5 + 2.5 * rand(rng)
      RA = M.evaluate(rbasis, r, Zs[iz1], Zs[iz2], psA.rbasis, model.st.rbasis)
      RB = M.evaluate(rbasis, r, Zs[iz1], Zs[iz2], psB.rbasis, model.st.rbasis)
      want = (iz2 == iz2_scaled) ? c .* RA : RA
      ok &= isapprox(collect(RB), collect(want); rtol = 1e-12, atol = 1e-14)
   end
   print_tf(@test ok)
end
println()

# and the weights must not depend on the CENTRE species -- only on the neighbour
psC = deepcopy(model.ps); M.set_embedding_weights!(rbasis, psC.rbasis, emb)
println_slim(@test all(psC.rbasis.Wnlq[:, :, 1, j] == psC.rbasis.Wnlq[:, :, 2, j]
                       for j = 1:NZ))

##

@info("ace_embedding_model: construction, equivariance, rank, and a fit")

const EMB_JSON = get(ENV, "ACE_EMBEDDING_JSON", "")
if isempty(EMB_JSON) || !isfile(EMB_JSON)
   @warn("set ACE_EMBEDDING_JSON to a frozen embedding artefact to run these; " *
         "see scripts/extract_mace_embedding.py")
else
   emb_real = M.read_mace_embedding(EMB_JSON)

   # n_B must match the standalone per-order calculation (scripts/spike_perorder_widths.jl)
   mdl = M.ace_embedding_model(elements = (:Si, :C, :O), order = 3,
                               totaldegree = 8, embedding = emb_real)
   println_slim(@test size(mdl.ps.WB, 1) == 392)
   println_slim(@test mdl.model.meta["embedding"]["widths"] == [3, 6, 10])

   # --- equivariance: the site energy is invariant under rotation and permutation
   rng = MersenneTwister(11)
   Zs0 = [14, 6, 8]
   Nnb = 14
   Rs = [ (rand(rng) * 3.0 + 1.2) * normalize(randn(rng, SVector{3, Float64}))
          for _ = 1:Nnb ]
   Zs = [ Zs0[mod1(i, 3)] for i = 1:Nnb ]
   z0 = 14
   E = M.evaluate(mdl.model, Rs, Zs, z0, mdl.ps, mdl.st)

   A = randn(rng, 3, 3); Q = Matrix(qr(A).Q); Q = Q * det(Q)     # proper rotation
   EQ = M.evaluate(mdl.model, [SVector{3}(Q * r) for r in Rs], Zs, z0, mdl.ps, mdl.st)
   println_slim(@test abs(E - EQ) < 1e-12 * max(abs(E), 1.0))

   p = shuffle(rng, 1:Nnb)
   EP = M.evaluate(mdl.model, Rs[p], Zs[p], z0, mdl.ps, mdl.st)
   println_slim(@test abs(E - EP) < 1e-12 * max(abs(E), 1.0))

   # --- losslessness: with d_nu = dim_nu the design matrix has full column rank
   # the basis is (n_B per centre species) + pair terms, so sample generously:
   # fewer columns than rows can only ever report rank = #columns
   sample() = M.evaluate_basis(mdl.model,
                 [ (rand(rng)*3.0 + 1.2) * normalize(randn(rng, SVector{3,Float64}))
                   for _ = 1:Nnb ],
                 [ Zs0[mod1(i,3)] for i = 1:Nnb ], Zs0[rand(rng, 1:3)],
                 mdl.ps, mdl.st)
   len_basis = length(sample())
   B = reduce(hcat, [ sample() for _ = 1:2*len_basis ])
   r = rank(B; rtol = 1e-12)
   @info("  design matrix $(size(B)), numerical rank $r of $(size(B,1))")
   println_slim(@test r == size(B, 1))

   # --- it fits: acefit! on a single-element embedding model
   m1 = M.ace_embedding_model(elements = (:Si,), order = 3, totaldegree = 8,
                              embedding = emb_real)
   data = ACEpotentials.example_dataset("Si_tiny").train[1:20]
   # NOTE: compute_errors must be given the SAME keys as acefit!.  With the
   # defaults it finds no reference data in Si_tiny (whose keys are dft_*) and
   # reports 0.0 for every observable -- a gate that passes while measuring
   # nothing.  Assert the errors are positive as well as finite.
   kw = (energy_key = "dft_energy", force_key = "dft_force",
         virial_key = "dft_virial")
   acefit!(data, m1; kw..., solver = ACEfit.BLR(), verbose = false)
   rmse = ACEpotentials.compute_errors(data, m1; kw..., verbose = false)["rmse"]["set"]
   @info("  single-element embedding: n_B = $(size(m1.ps.WB,1)), " *
         "E = $(rmse["E"]), F = $(rmse["F"])")
   println_slim(@test all(isfinite(v) && v > 0 for v in values(rmse)))

   # --- like-for-like: at S = 1 the embedding contributes a single scalar, so an
   # embedded model must fit essentially as well as `ace1_model` at the same n_B.
   # It does NOT unless the embedding is normalised after truncation: the raw
   # MACE entries are O(0.1), so at order ν the basis is scaled by ~1e-3, and
   # BLR's prior on coefficient magnitude is not scale-invariant.  With the raw
   # table this gate reads F = 24.0 against ace1's 1.47, from bases that are
   # exactly proportional -- a pure regularisation artefact.
   mref = ace1_model(elements = [:Si], order = 3, totaldegree = 8)
   acefit!(data, mref; kw..., solver = ACEfit.BLR(), verbose = false)
   rref = ACEpotentials.compute_errors(data, mref; kw..., verbose = false)["rmse"]["set"]
   @info("  vs ace1_model: E $(rmse["E"]) / $(rref["E"]), F $(rmse["F"]) / $(rref["F"])")
   println_slim(@test size(m1.ps.WB, 1) == size(mref.ps.WB, 1))
   println_slim(@test isapprox(rmse["E"], rref["E"]; rtol = 0.02))
   println_slim(@test isapprox(rmse["F"], rref["F"]; rtol = 0.05))
end
