# Oracle O2 (function level): the CP special case of the TT space -- diagonal
# cores diag(E[z,k]) shared by all slots, r = K, only v free -- must reproduce the
# site basis of `ace_embedding_model(d_max = K)` on random environments, for
# every central species and every column, to 1e-10.  This checks the whole map
# (the 𝔸-level projection, the permutation/multiplicity bookkeeping, and the
# per-order widths) against ET's own construction of the embedded basis.
#
#   julia --project=acejax/julia -t 4 acejax/spike_tt/oracle_o2.jl   (DEG=4|5|6, DMAX=16)
using ACEpotentials, LinearAlgebra, SparseArrays, Random, Printf, StaticArrays, AtomsBase
include(joinpath(@__DIR__, "ttmap.jl"))
M = ACEpotentials.Models
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
D = parse(Int, get(ENV, "DEG", "4"))
DMAX = parse(Int, get(ENV, "DMAX", "16"))
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]
emb = M.read_mace_embedding(joinpath(SCRATCH, "distil", "mace_mh1_embedding.json"))

mc = ace1_model(elements = ELS, order = 3, totaldegree = D)
YT = Symbol(get(ENV, "YTYPE", "spherical"))   # ace1_model uses :spherical; ace_embedding_model defaults to :solid
me = M.ace_embedding_model(elements = Tuple(ELS), order = 3, totaldegree = D, embedding = emb, d_max = DMAX, Ytype = YT)
tm = build_ttmap(mc.model; D = D)
S = tm.S
widths = me.model.meta["embedding"]["widths"]; d = maximum(widths)
E = M.embedding_rows(emb, ZS; d = d)
@printf("D=%d d_max=%d widths=%s  E %s\n", D, DMAX, string(widths), string(size(E)))

# --- CP-in-TT columns, as categorical coefficient vectors (per z0 identical)
# column (ib, η, k): x = Σ_ζ Φ[:, (ib,η,ζ)] Π_t E[ζ_t, k]
cpcols = Dict{Tuple{Int,Int,Int}, Vector{Float64}}()
for (j, ix) in enumerate(tm.idx)
   ν = length(ix.ζ)
   for k in 1:widths[ν]
      key = (ix.ibeta, ix.η, k)
      x = get!(cpcols, key, zeros(tm.nB))
      x .+= tm.Φ[:, j] .* prod(E[ix.ζ[t], k] for t in 1:ν)
   end
end
# restrict to embedded-spec β
keys_emb = sort([k for k in keys(cpcols) if tm.beta_emb[k[1]]])
Xcp = hcat([cpcols[k] for k in keys_emb]...)            # nB_cat × ncp
@printf("CP-in-TT columns on embedded-spec β: %d;  embedded model nB per z0: %d\n",
        length(keys_emb), length(me.model.tensor))

# --- embedded model columns -> (β, k)
nnll_e = M.get_nnll_spec(me.model.tensor)
unf_e(n) = (div(n - 1, d) + 1, mod1(n, d))
βk_e = [(sort([(unf_e(b.n)[1], b.l) for b in bb]), unique([unf_e(b.n)[2] for b in bb])) for bb in nnll_e]
@assert all(length(x[2]) == 1 for x in βk_e) "embedded spec is not channel-diagonal?"
βinv = Dict(β => i for (i, β) in enumerate(tm.betas))

# --- random environments
rng = MersenneTwister(7)
rcut = maximum(x.rcut for x in mc.model.rbasis.rin0cuts)
function randenv(rng, n)
   Rs = SVector{3,Float64}[]; Zs = Int[]
   while length(Rs) < n
      r = 1.8 + (rcut - 0.2 - 1.8) * rand(rng)
      u = normalize(randn(rng, SVector{3,Float64}))
      push!(Rs, r * u); push!(Zs, ZS[rand(rng, 1:S)])
   end
   return Rs, Zs
end
NENV = 300
maxrel = 0.0; worst = nothing
for (iz0, z0) in enumerate(ZS)
   Bc = zeros(NENV, tm.nB); Be = zeros(NENV, length(me.model.tensor))
   for e in 1:NENV
      Rs, Zs = randenv(rng, rand(rng, 8:40))
      bc = M.evaluate_basis(mc.model, Rs, Zs, z0, mc.ps, mc.st)
      be = M.evaluate_basis(me.model, Rs, Zs, z0, me.ps, me.st)
      Bc[e, :] .= bc[M.get_basis_inds(mc.model, z0)]
      Be[e, :] .= be[M.get_basis_inds(me.model, z0)]
   end
   Ycp = Bc * Xcp                          # NENV × ncp : the CP-in-TT site functions
   # every embedded column must be a combination of the CP-in-TT columns of the
   # same (β, k) (over η); with nη = 1 that is a single scalar
   for (ie, (β, ks)) in enumerate(βk_e)
      k = ks[1]; ib = get(βinv, β, 0)
      ib == 0 && (println("  embedded β $β not in categorical spec!"); continue)
      cols = [ic for (ic, key) in enumerate(keys_emb) if key[1] == ib && key[3] == k]
      isempty(cols) && (println("  no CP-in-TT column for β=$β k=$k"); continue)
      T = Ycp[:, cols] \ Be[:, ie]
      res = norm(Ycp[:, cols] * T - Be[:, ie]) / max(norm(Be[:, ie]), 1e-300)
      if res > maxrel; global maxrel = res; global worst = (z0, β, k, T); end
   end
   # and the other way: every CP-in-TT column of the embedded spec lies in the embedded span
   T2 = Be \ Ycp
   res2 = norm(Be * T2 - Ycp) / norm(Ycp)
   @printf("  z0=%d: max rel residual embedded -> CP-in-TT so far %.2e; CP-in-TT -> embedded %.2e  (rank Ycp %d, rank Be %d)\n",
           z0, maxrel, res2, rank(Ycp), rank(Be))
end
@printf("O2 RESULT D=%d d_max=%d: max relative residual over all z0 and all %d embedded columns = %.3e  (worst: %s)\n",
        D, DMAX, length(nnll_e), maxrel, string(worst))
