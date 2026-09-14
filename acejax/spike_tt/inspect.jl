# Inspect the categorical and embedded model specs at degree D (default 4).
using ACEpotentials, LinearAlgebra, Printf, SparseArrays
M = ACEpotentials.Models
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
D = parse(Int, get(ENV, "DEG", "4"))
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
m = ace1_model(elements = ELS, order = 3, totaldegree = D)
mm = m.model
S = M._get_nz(mm)
println("categorical: length_basis = ", M.length_basis(mm), "  nB per z0 = ", length(mm.tensor), "  npair per z0 = ", length(mm.pairbasis))
println("rbasis spec (folded n): ", length(mm.rbasis.spec), " first: ", mm.rbasis.spec[1:min(12,end)])
mb = mm.tensor.meta["mb_spec"]; AAs = mm.tensor.meta["𝔸spec"]
println("mb_spec: ", length(mb), "  𝔸spec: ", length(AAs), "  A2B size ", size(mm.tensor.A2Bmaps[1]))
nnll = M.get_nnll_spec(mm.tensor)
println("nnll per B: ", length(nnll), " orders: ", [count(b -> length(b) == k, nnll) for k = 1:3])
# unfold
unf(n) = (div(n - 1, S) + 1, mod1(n, S))
println("first 10 nnll: ", nnll[1:10])
println("mb_spec sorted? ", all(issorted(bb) for bb in mb))
# count β = sorted (n', l) tuples, channel-free level
wL = 1.5
betas = Dict{Any, Set}()
for bb in nnll
   β = sort([(unf(b.n)[1], b.l) for b in bb])
   ζ = [unf(b.n)[2] for b in bb]
   push!(get!(betas, β, Set()), ζ)
end
lev(β) = sum(b[1] + wL * b[2] for b in β)
nemb = count(β -> lev(β) <= D, keys(betas)); ntail = length(betas) - nemb
println("distinct β: ", length(betas), "  embedded-spec β: ", nemb, "  tail β: ", ntail)
ncols_tail = sum(length(nnll[i]) >= 0 && lev(sort([(unf(b.n)[1], b.l) for b in nnll[i]])) > D for i in 1:length(nnll))
println("categorical B functions in tail blocks per z0: ", ncols_tail, " of ", length(nnll))
for ν = 1:3
   bs = [β for β in keys(betas) if length(β) == ν]
   println(" order $ν: #β = ", length(bs), "  emb: ", count(β -> lev(β) <= D, bs))
end
# embedded model
emb = M.read_mace_embedding(joinpath(SCRATCH, "distil", "mace_mh1_embedding.json"))
me = M.ace_embedding_model(elements = Tuple(ELS), order = 3, totaldegree = D, embedding = emb, d_max = 16)
println("embedded d16: length_basis = ", M.length_basis(me.model), " nB per z0 = ", length(me.model.tensor), " npair = ", length(me.model.pairbasis), " widths ", me.model.meta["embedding"]["widths"])
nnll_e = M.get_nnll_spec(me.model.tensor)
println(" orders: ", [count(b -> length(b) == k, nnll_e) for k = 1:3])
# radial bases: same transforms / rin0cuts?
println("cat rin0cuts: ", unique(mm.rbasis.rin0cuts), " emb: ", unique(me.model.rbasis.rin0cuts))
println("cat rbasis type: ", typeof(mm.rbasis).name.name, " emb: ", typeof(me.model.rbasis).name.name)
println("pair spec cat: ", mm.pairbasis.spec[1:min(6,end)], " n=", length(mm.pairbasis.spec), " emb: ", me.model.pairbasis.spec[1:min(6,end)], " n=", length(me.model.pairbasis.spec))
