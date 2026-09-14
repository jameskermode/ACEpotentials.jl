# TT-compressed ACE coefficients fitted by ALS on the cached categorical design
# matrix.  Oracles O1 / O2(fit), CP vs TT at matched ranks, transfer, cost.
#
#   julia --project=acejax/julia -t 4 acejax/spike_tt/run_tt.jl      (DEG=4, LAM=1e-7)
using ACEpotentials, LinearAlgebra, SparseArrays, Random, Printf, Serialization, AtomsBase
include(joinpath(@__DIR__, "ttmap.jl"))
include(joinpath(@__DIR__, "tt.jl"))
include(joinpath(@__DIR__, "..", "bench", "distil", "tikhonov.jl"))
BLAS.set_num_threads(4)
M = ACEpotentials.Models
const SCRATCH = "/private/tmp/claude-502/-Users-u1470235--julia-dev-ACEpotentials/e8fb3bd6-77a9-4730-a1ac-f7afc57a3f6b/scratchpad"
const CACHEDIR = joinpath(SCRATCH, "fs_spike_cache")
const D = parse(Int, get(ENV, "DEG", "4"))
const LAM = parse(Float64, get(ENV, "LAM", "1e-7"))
const NSWEEPS = parse(Int, get(ENV, "NSWEEPS", "8"))
const STAGES = split(get(ENV, "STAGES", "base,o1,cp,tt,rand,transfer,cost"), ",")
const OUT = joinpath(@__DIR__, "results_D$(D).jls")
ELS = [:Cr, :Mn, :Fe, :Co, :Ni]
ZS = [AtomsBase.atomic_number(ChemicalSpecies(el)) for el in ELS]
@info "degree $D, lambda $LAM, stages $STAGES"

# ---------------------------------------------------------------- model, map, data
m = ace1_model(elements = ELS, order = 3, totaldegree = D)
mm = m.model
tm = build_ttmap(mm; D = D)
S = tm.S; nB = tm.nB; Nc = length(tm.idx); NZ = 5
npair = length(mm.pairbasis)
NPAIR = 5 * npair            # pair columns are per central species too
@printf("map: nB=%d per z0, N_c=%d per z0, #β=%d (embedded-spec %d), maxres=%.1e\n",
        nB, Nc, length(tm.betas), count(tm.beta_emb), tm.maxres)
Φfull = blockdiag(ntuple(_ -> tm.Φ, NZ)...)
MBcols = 1:NZ*nB; paircols = NZ*nB+1:NZ*(nB+npair)
@assert M.length_basis(mm) == NZ*(nB+npair)
P = M.algebraic_smoothness_prior(mm; p = 4).diag

function load_asm(tag)
   f = joinpath(CACHEDIR, "asm_cantor_D$(D)_$(tag).jls")
   isfile(f) || error("missing cache $f")
   A, Y, W = deserialize(f); return Matrix(A), Y, W
end
function row_layout_from(A, natoms)
   # rows per structure: 1 E, 3n F, 6 V
   kind = Int[]; nat = Int[]
   for n in natoms
      push!(kind, 1); push!(nat, n); append!(kind, fill(2, 3n)); append!(nat, fill(n, 3n))
      append!(kind, fill(3, 6)); append!(nat, fill(n, 6))
   end
   @assert length(kind) == size(A, 1)
   return kind, nat
end
data_all = ACEpotentials.ExtXYZ.load(joinpath(SCRATCH, "distil", "cantor1k_b_mh1.xyz"))
rng = MersenneTwister(0); perm = shuffle(rng, 1:length(data_all))
tr = data_all[perm[1:200]]; te = data_all[perm[201:300]]
Atr, Ytr, Wtr = load_asm("train200"); Ate, Yte, Wte = load_asm("test100")
kind_tr, nat_tr = row_layout_from(Atr, length.(tr)); kind_te, nat_te = row_layout_from(Ate, length.(te))
function rmse_efv(resid, kind, nat)
   r = copy(resid); r[kind .!= 2] ./= nat[kind .!= 2]
   f(k) = sqrt(sum(abs2, r[kind .== k]) / count(==(k), kind))
   (F = f(2), E = f(1), V = f(3))
end
function make_data(A, Y, W)
   AΦ = (W .* A[:, MBcols]) * Φfull
   TTData(AΦ, W .* A[:, paircols], W .* Y, sparse(Diagonal(P[MBcols])) * Φfull, P[paircols])
end
t = @elapsed dtr = make_data(Atr, Ytr, Wtr)
@printf("train data: AΦ %s built in %.1f s\n", string(size(dtr.AΦ)), t)
AΦte = Ate[:, MBcols] * Φfull; Apte = Ate[:, paircols]
function test_errors(mdl::TTModel; AΦ = AΦte, Ap = Apte, Y = Yte, kind = kind_te, nat = nat_te)
   c = coefficients(mdl)
   rmse_efv(AΦ * c + Ap * mdl.p - Y, kind, nat)
end
function train_errors(mdl::TTModel)
   c = coefficients(mdl)
   rmse_efv(Atr[:, MBcols] * (Φfull * c) + Atr[:, paircols] * mdl.p - Ytr, kind_tr, nat_tr)
end

results = Dict{String, Any}()
save() = serialize(OUT, results)

# ---------------------------------------------------------------- flop model
# n_A1: distinct single-channel (n', l, m); DAG nodes per (order, depth)
function dag_counts(tm)
   νmax = maximum(length.(tm.betas))
   nodes = [Set{Vector{Tuple{Int,Int,Int}}}() for ν in 1:νmax, t in 1:νmax]
   a1 = Set{Tuple{Int,Int,Int}}()
   for (ib, β) in enumerate(tm.betas)
      ν = length(β)
      for mvec in tm.beta_M[ib]
         tup = [(β[t][1], β[t][2], mvec[t]) for t in 1:ν]
         for t in 1:ν; push!(nodes[ν, t], tup[1:t]); push!(a1, tup[t]); end
      end
   end
   n_nodes = [[length(nodes[ν, t]) for t in 1:ν] for ν in 1:νmax]
   n_blocks = [sum(n_eta(tm, ib) for ib in 1:length(tm.betas) if length(tm.betas[ib]) == ν; init = 0) for ν in 1:νmax]
   return length(a1), n_nodes, n_blocks
end
n_A1, n_nodes, n_blocks = dag_counts(tm)
@printf("flop model: n_A1=%d, DAG nodes per order/depth %s, blocks per order %s\n", n_A1, string(n_nodes), string(n_blocks))
cp_flops(K) = n_A1 * S * K + sum(sum(n_nodes[ν]) * K for ν in 1:3) + sum(n_blocks[ν] * min(K, [5,15,35][ν]) for ν in 1:3)
cat_flops() = sum(sum(n_nodes[ν]) * S^ν for ν in 1:3)  # categorical: every species-resolved 𝔸 node, 1 op each
# (the categorical count is per-site products of species-resolved A's: S^t nodes at depth t)
cat_flops_exact() = sum(sum(n_nodes[ν][t] * S^t for t in 1:ν) for ν in 1:3)

# ---------------------------------------------------------------- baseline (categorical)
if "base" in STAGES
   # exactly the FS-spike solver (factor once, sweep lambda)
   Aw = (Atr[:, :] ./ reshape(P, 1, :)) .* Wtr
   T = TikhonovFactor(Aw, Wtr .* Ytr)
   best = nothing
   for lam in 10.0 .^ (0:-1:-8)
      x = tikhonov_solve(T, lam) ./ P
      e = rmse_efv(Ate * x - Yte, kind_te, nat_te)
      etr = rmse_efv(Atr * x - Ytr, kind_tr, nat_tr)
      @printf("  categorical lam=%-6g test F=%.4f E=%.5f V=%.4f | train F=%.4f\n", lam, e.F, e.E, e.V, etr.F)
      (best === nothing || e.F < best.te.F) && (global best = (te = e, tr = etr, lam = lam, x = x))
   end
   results["categorical"] = (te = best.te, tr = best.tr, lam = best.lam, nparams = length(best.x),
                             flops = cat_flops_exact())
   @printf("RES categorical   nparams=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lambda=%g\n",
           length(best.x), best.te.F, best.te.E, best.te.V, best.tr.F, best.lam)
   # the same fit through my generalised-Tikhonov solver at LAM (harness check)
   x2 = tikh_solve(Wtr .* Atr, Matrix(Diagonal(P)), Wtr .* Ytr, LAM)
   e2 = rmse_efv(Ate * x2 - Yte, kind_te, nat_te)
   xl = tikhonov_solve(T, LAM) ./ P
   @printf("CHECK tikh_solve vs TikhonovFactor at lam=%g: test F %.6f vs %.6f, |dx|/|x| = %.2e\n",
           LAM, e2.F, rmse_efv(Ate * xl - Yte, kind_te, nat_te).F, norm(x2 - xl) / norm(xl))
   Jcat = sum(abs2, Wtr .* (Atr * xl - Ytr)) + LAM^2 * sum(abs2, P .* xl)
   results["categorical_lam"] = (x = xl, J = Jcat, te = rmse_efv(Ate * xl - Yte, kind_te, nat_te))
   save()
end

# ---------------------------------------------------------------- O1: full-rank TT ⊇ categorical
if "o1" in STAGES
   xl = results["categorical_lam"].x; Jcat = results["categorical_lam"].J
   # least-norm c with Φfull c = x_MB (each Φ column has one entry)
   c = zeros(NZ * Nc)
   colnorm = vec(sum(abs2, Φfull; dims = 1))
   rows = rowvals(Φfull); vals = nonzeros(Φfull)
   rowsum = zeros(size(Φfull, 1)); for j in 1:size(Φfull, 2), k in nzrange(Φfull, j); rowsum[rows[k]] += vals[k]^2; end
   for j in 1:size(Φfull, 2), k in nzrange(Φfull, j)
      c[j] = xl[rows[k]] * vals[k] / rowsum[rows[k]]
   end
   @printf("O1: |Φc − x| = %.2e\n", norm(Φfull * c - xl[MBcols]))
   sp = TTSpec(S, NZ, [[1, 5], [1, 5, 25], [1, 5, 25, 125]], true)
   mo = TTModel(sp, tm, NPAIR)
   # unit cores: state after t slots = index of (ζ_1..ζ_t)
   for ν in 1:3, t in 1:ν, z in 1:S
      G = mo.G[ν][t]
      for a in 1:size(G, 1)
         G[a, a + size(G, 1) * (z - 1), z, 1] = 1.0
      end
   end
   for z0 in 1:NZ, (j, ix) in enumerate(tm.idx)
      ν = length(ix.ζ); idx = ix.ζ[1] + (ν >= 2 ? 5 * (ix.ζ[2] - 1) : 0) + (ν >= 3 ? 25 * (ix.ζ[3] - 1) : 0)
      mo.v[z0][ix.ibeta][idx, ix.η] = c[(z0 - 1) * Nc + j]
   end
   mo.p .= xl[paircols]
   @printf("O1: |c_TT − c| = %.2e\n", norm(coefficients(mo) - c))
   J0 = objective(mo, dtr, LAM)
   @printf("O1: J_TT(θ_cat) = %.12e  J_cat = %.12e  rel diff %.2e\n", J0, Jcat, abs(J0 - Jcat) / Jcat)
   e0 = test_errors(mo)
   ts = @elapsed readout_step!(mo, dtr, LAM)
   J1 = objective(mo, dtr, LAM); e1 = test_errors(mo)
   @printf("O1: after one v-step (%.1f s, %d readout params): J = %.12e  rel diff vs J_cat %.2e; test F %.6f vs %.6f\n",
           ts, n_v_params(mo), J1, abs(J1 - Jcat) / Jcat, e1.F, e0.F)
   # a core step from there must not move either (already optimal; G3 at full
   # rank has 15625 columns, so only G1 and G2 are exercised)
   ts = @elapsed core_step!(mo, dtr, LAM, 1); J2 = objective(mo, dtr, LAM)
   ts2 = @elapsed core_step!(mo, dtr, LAM, 2); J3 = objective(mo, dtr, LAM)
   @printf("O1: after a G1 step (%.0f s): J = %.12e rel diff %.2e; after a G2 step (%.0f s): J = %.12e rel diff %.2e; test F %.6f\n",
           ts, J2, abs(J2 - Jcat) / Jcat, ts2, J3, abs(J3 - Jcat) / Jcat, test_errors(mo).F)
   results["O1"] = (J0 = J0, J1 = J1, J2 = J2, J3 = J3, Jcat = Jcat)
   save()
end

# ---------------------------------------------------------------- CP-K and TT schedules
ranks_cp(K) = [[1, min(K, 5)], [1, min(K, 5), min(K, 15)], [1, min(K, 5), min(K, 15), K]]
ranks_wide(K) = [[1, min(K, 5)], [1, 5, min(K, 25)], [1, 5, 25, K]]
ranks_uni(r) = [[1, min(r, 5)], [1, min(r, 5), min(r, 25)], [1, min(r, 5), min(r, 25), r]]

function cp_fit(K; lam = LAM, emb_only = false)
   E = M.embedding_rows(emb, ZS; d = K)
   sp = TTSpec(S, NZ, ranks_cp(K), true)
   mdl = TTModel(sp, tm, NPAIR)
   vcp = [[zeros(K, n_eta(tm, ib)) for ib in 1:length(tm.betas)] for _ in 1:NZ]
   init_from_tensor!(mdl, [cp_tensor(E, ν) for ν in 1:3], vcp)
   d = dtr
   if emb_only
      # zero the readout columns of tail β (they stay zero: not in the v-step)
      d = restrict_to_emb(dtr)
   end
   ts = @elapsed readout_step!(mdl, d, lam)
   return mdl, ts
end
# restrict: drop the tail-β coefficients by masking AΦ columns (kept zero)
function restrict_to_emb(d::TTData)
   mask = [tm.beta_emb[ix.ibeta] for z0 in 1:NZ for ix in tm.idx]
   AΦ = copy(d.AΦ); AΦ[:, .!mask] .= 0
   PΦ = copy(d.PΦ); PΦ[:, .!mask] .= 0; dropzeros!(PΦ)
   TTData(AΦ, d.Apair, d.y, PΦ, d.Ppair)
end
emb = M.read_mace_embedding(joinpath(SCRATCH, "distil", "mace_mh1_embedding.json"))

function report(name, mdl, extra...; cpK = nothing)
   e = test_errors(mdl); etr = train_errors(mdl)
   sp = mdl.sp
   # CP: the cores are the FROZEN embedding, not parameters, and the evaluation
   # is the per-channel product (cp_flops), not the TT chain
   fl = cpK === nothing ? flops_per_site(sp, n_nodes, n_A1, n_blocks) : cp_flops(cpK)
   ncore = cpK === nothing ? n_core_params(mdl) : 0
   np = ncore + n_v_params(mdl) + length(mdl.p)
   @printf("RES %-34s nparams=%5d (cores %4d, readout %4d, pair %3d) flops/site=%8.0f  test F=%.4f E=%.5f V=%.4f  train F=%.4f %s\n",
           name, np, ncore, n_v_params(mdl), length(mdl.p), fl, e.F, e.E, e.V, etr.F, join(string.(extra), " "))
   flush(stdout)
   results[name] = (te = e, tr = etr, nparams = np, ncore = ncore, nv = n_v_params(mdl),
                    flops = fl, ranks = sp.ranks, extra = extra)
   save()
   return e
end

if "cp" in STAGES
   # lambda check for CP-16 (v-only, convex)
   for lam in (1e-5, 1e-6, 1e-7, 1e-8)
      mdl, _ = cp_fit(16; lam = lam); e = test_errors(mdl)
      @printf("  CP-16 lam=%-6g test F=%.4f E=%.5f V=%.4f\n", lam, e.F, e.E, e.V)
   end
   for K in (4, 8, 16, 35)
      mdl, ts = cp_fit(K); report("CP-$K", mdl, @sprintf("vstep=%.1fs", ts); cpK = K)
      if K == 16; (mdl2, _) = cp_fit(K; emb_only = true); report("CP-16 (embedded-spec β only)", mdl2; cpK = K); end
   end
   @printf("categorical flops/site (species-resolved 𝔸 nodes) = %d ; CP-4/8/16/35 = %d/%d/%d/%d\n", cat_flops_exact(),
           cp_flops(4), cp_flops(8), cp_flops(16), cp_flops(35))
   for (nm, rk) in (("TT(1,2,2,2)", ranks_uni(2)), ("TT(1,4,4,4)", ranks_uni(4)), ("TT(1,5,8,8)", ranks_cp(8)),
                    ("TT(1,5,15,16)", ranks_cp(16)), ("TT(1,5,25,16)", ranks_wide(16)), ("TT(1,5,25,35)", ranks_wide(35)))
      @printf("  flops/site %-14s = %d\n", nm, flops_per_site(TTSpec(S, NZ, rk, true), n_nodes, n_A1, n_blocks))
   end
end

# ---------------------------------------------------------------- TT by ALS
function tt_run(name, ranks, K_init; noise = 0.0, nsweeps = NSWEEPS, init = :cp, seed = 1, share = true, rtol = 1e-4)
   sp = TTSpec(S, NZ, ranks, share)
   mdl = TTModel(sp, tm, NPAIR; rng = MersenneTwister(seed), init = init == :cp ? :zeros : :random)
   if init == :cp
      mcp, _ = cp_fit(K_init)
      init_from_tensor!(mdl, [tt_tensor(mcp, ν) for ν in 1:3], mcp.v; noise = noise, rng = MersenneTwister(seed))
      mdl.p .= mcp.p
   end
   readout_step!(mdl, dtr, LAM)                 # v optimal for the initial cores
   e0 = test_errors(mdl)
   @printf("TT %s init (%s, K=%d, noise=%g): test F=%.4f\n", name, init, K_init, noise, e0.F)
   trace_te = [e0]
   cb(sweep, J) = (push!(trace_te, test_errors(mdl)); @printf("   sweep %d test F=%.4f E=%.5f V=%.4f\n", sweep, trace_te[end].F, trace_te[end].E, trace_te[end].V))
   t0 = time()
   trace, nonmono = als!(mdl, dtr, LAM; nsweeps = nsweeps, callback = cb, tag = name, rtol = rtol)
   wall = time() - t0
   nsw = length(trace_te) - 1
   report(name, mdl, @sprintf("sweeps=%d wall/sweep=%.0fs nonmono=%d initF=%.4f", nsw, wall / max(nsw, 1), nonmono, e0.F))
   results[name * "_trace"] = (J = trace, te = trace_te, wall = wall)
   save()
   return mdl
end

tt_models = Dict{String, TTModel}()
if "tt" in STAGES
   tt_models["TT(1,5,15,16) from CP-16"] = tt_run("TT(1,5,15,16) from CP-16", ranks_cp(16), 16)
   tt_models["TT(1,5,8,8) from CP-8"] = tt_run("TT(1,5,8,8) from CP-8", ranks_cp(8), 8)
   tt_models["TT(1,4,4,4) from CP-4"] = tt_run("TT(1,4,4,4) from CP-4", ranks_uni(4), 4)
   tt_models["TT(1,5,25,16) wide from CP-16"] = tt_run("TT(1,5,25,16) wide from CP-16", ranks_wide(16), 16; noise = 1e-2)
   tt_models["TT(1,5,25,35) wide from CP-35"] = tt_run("TT(1,5,25,35) wide from CP-35", ranks_wide(35), 35; noise = 1e-2)
   tt_models["TT(1,2,2,2) from CP-2"] = tt_run("TT(1,2,2,2) from CP-2", ranks_uni(2), 2)
end
if "extra" in STAGES
   # longer runs of the small schedules (they were still improving at 8 sweeps),
   # and cores per central species instead of shared
   tt_models["TT(1,4,4,4) 24 sweeps"] = tt_run("TT(1,4,4,4) 24 sweeps", ranks_uni(4), 4; nsweeps = 24, rtol = 1e-5)
   tt_models["TT(1,5,8,8) 24 sweeps"] = tt_run("TT(1,5,8,8) 24 sweeps", ranks_cp(8), 8; nsweeps = 24, rtol = 1e-5)
   tt_models["TT(1,5,8,8) per-z0 cores"] = tt_run("TT(1,5,8,8) per-z0 cores", ranks_cp(8), 8; share = false, nsweeps = 12)
   tt_models["TT(1,3,3,3) from CP-3"] = tt_run("TT(1,3,3,3) from CP-3", ranks_uni(3), 3; nsweeps = 24, rtol = 1e-5)
end
if "rand" in STAGES
   tt_models["TT(1,5,15,16) random init s1"] = tt_run("TT(1,5,15,16) random init s1", ranks_cp(16), 16; init = :random, seed = 1)
   tt_models["TT(1,5,15,16) random init s2"] = tt_run("TT(1,5,15,16) random init s2", ranks_cp(16), 16; init = :random, seed = 2)
   tt_models["TT(1,5,8,8) random init s1"] = tt_run("TT(1,5,8,8) random init s1", ranks_cp(8), 8; init = :random, seed = 1)
end

# ---------------------------------------------------------------- transfer to split 1
if "transfer" in STAGES && isfile(joinpath(CACHEDIR, "asm_cantor_D$(D)_split1_train200.jls"))
   if isempty(tt_models)
      tt_models["TT(1,5,15,16) from CP-16"] = tt_run("TT(1,5,15,16) from CP-16", ranks_cp(16), 16)
      tt_models["TT(1,5,8,8) from CP-8"] = tt_run("TT(1,5,8,8) from CP-8", ranks_cp(8), 8; nsweeps = 16)
      tt_models["TT(1,4,4,4) from CP-4"] = tt_run("TT(1,4,4,4) from CP-4", ranks_uni(4), 4; nsweeps = 24)
      tt_models["TT(1,2,2,2) from CP-2"] = tt_run("TT(1,2,2,2) from CP-2", ranks_uni(2), 2; nsweeps = 24)
   end
   # split 1: MersenneTwister(1) 200/100 from the 700 structures unused by split 0
   rest = sort(perm[301:end]); p1 = shuffle(MersenneTwister(1), rest)   # as varpro_assemble_split1.jl
   tr1 = data_all[p1[1:200]]; te1 = data_all[p1[201:300]]
   A1, Y1, W1 = load_asm("split1_train200"); A1t, Y1t, W1t = load_asm("split1_test100")
   k1, n1 = row_layout_from(A1, length.(tr1)); k1t, n1t = row_layout_from(A1t, length.(te1))
   d1 = make_data(A1, Y1, W1); AΦ1t = A1t[:, MBcols] * Φfull; Ap1t = A1t[:, paircols]
   te1err(mdl) = test_errors(mdl; AΦ = AΦ1t, Ap = Ap1t, Y = Y1t, kind = k1t, nat = n1t)
   # categorical on split 1
   x1 = tikh_solve(W1 .* A1, Matrix(Diagonal(P)), W1 .* Y1, LAM)
   e = rmse_efv(A1t * x1 - Y1t, k1t, n1t)
   @printf("TRANSFER categorical (split1 fit)          test F=%.4f E=%.5f V=%.4f\n", e.F, e.E, e.V)
   results["transfer categorical"] = e
   for K in (8, 16)
      mdl, _ = cp_fit(K); readout_step!(mdl, d1, LAM); e1 = te1err(mdl)
      @printf("TRANSFER CP-%-2d (v refit on split1)          test F=%.4f E=%.5f V=%.4f\n", K, e1.F, e1.E, e1.V)
      results["transfer CP-$K"] = e1
   end
   for (name, mdl) in tt_models
      m2 = copy(mdl); readout_step!(m2, d1, LAM); e1 = te1err(m2)
      @printf("TRANSFER %-36s (cores frozen, v refit on split1) test F=%.4f E=%.5f V=%.4f\n", name, e1.F, e1.E, e1.V)
      results["transfer " * name] = e1
   end
   # and TT learned ON split 1 (in-sample there), for the comparison
   if haskey(tt_models, "TT(1,5,15,16) from CP-16")
      global dtr_saved = dtr
      global dtr = d1
      m3 = tt_run("TT(1,5,15,16) learned on split1", ranks_cp(16), 16; nsweeps = 5)
      e1 = te1err(m3)
      @printf("TRANSFER TT(1,5,15,16) learned on split1 (in-sample) test F=%.4f E=%.5f V=%.4f\n", e1.F, e1.E, e1.V)
      results["transfer TT learned on split1"] = e1
      global dtr = dtr_saved
   end
   save()
end

# ---------------------------------------------------------------- cost
if "cost" in STAGES
   mdl = haskey(tt_models, "TT(1,5,15,16) from CP-16") ? tt_models["TT(1,5,15,16) from CP-16"] : first(cp_fit(16))
   for t in 1:3
      tmap = @elapsed (Mt, off) = core_map(mdl, t)
      tdes = @elapsed A = hcat(dtr.AΦ * Mt, dtr.Apair)
      tsol = @elapsed core_step!(mdl, dtr, LAM, t)
      @printf("COST core step t=%d: map %.2f s, design gemm %.2f s (%s), full step %.2f s, %d columns\n",
              t, tmap, tdes, string(size(A)), tsol, size(Mt, 2))
   end
   tv = @elapsed readout_step!(mdl, dtr, LAM)
   (Mv, _) = readout_map(mdl)
   @printf("COST readout step: %.2f s, %d columns\n", tv, size(Mv, 2))
   tc = @elapsed tikh_solve(Wtr .* Atr, Matrix(Diagonal(P)), Wtr .* Ytr, LAM)
   tf = @elapsed TikhonovFactor((Atr ./ reshape(P, 1, :)) .* Wtr, Wtr .* Ytr)
   @printf("COST categorical solve: augmented QR %.2f s (%d columns); TikhonovFactor %.2f s\n", tc, size(Atr, 2), tf)
   results["cost"] = (tv = tv, tc = tc)
   save()
end
println("done.")
