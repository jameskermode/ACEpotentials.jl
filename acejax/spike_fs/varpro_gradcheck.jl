# Exp. 1: (a) the projected inner solve agrees with a full factorisation of
# [A X]; (b) the column builder agrees with fs_embed.jl's hand-tilt columns;
# (c) the Kaufman/ForwardDiff gradient of the reduced objective agrees with
# central finite differences on random theta components.  Also times the
# pieces.   julia --project=acejax/julia acejax/spike_fs/varpro_gradcheck.jl
include(joinpath(@__DIR__, "varpro_core.jl"))
const D = 4
const LAM = 1e-8
data_all = load_all()
tr, te = split0(data_all)
lay_tr = row_layout(tr)
asm = load_asm("asm_cantor_D$(D)_train200.jls")
Pdiag = prior_diag(D)
Aw = (asm.A ./ reshape(Pdiag, 1, :)) .* asm.W; Yw = asm.W .* asm.Y
t = @elapsed sps = [StructPairs(sys) for sys in tr]
@printf("pair data: %d structures, %d pairs, %.2f s\n", length(sps), sum(length(sp.pi) for sp in sps), t)
@assert sum(nrows, sps) == size(Aw, 1)

# (b) builder vs fs_embed.jl: rho_tot (equal weights) and a hand tilt
P = 2
W0 = ones(KS, P); W0[:, 2] .= vec(repeat([2.0, 1.0, 1.0, 1.0, 1.0]', K, 1))   # p=2: rho_tot + rho^Cr
t = @elapsed X = extra_X(sps, W0, ALPHAS0)
@printf("extra_X (P=2, 200 structures): %.2f s\n", t)
wvec(k, ws) = SVector{KS}(ntuple(q -> ((q - 1) % K + 1 == k) ? ws[(q - 1) ÷ K + 1] : 0.0, KS))
es = EmbedSpec(ZS, vcat([wsqrt_fun(KS, wvec(k, ones(S))) for k in 1:K],
                        [wsqrt_fun(KS, wvec(k, [2.0, 1, 1, 1, 1])) for k in 1:K]), ["" for _ in 1:2K]; per_species = true)
Xref = fs_feature_matrix(es, tr)
# fs_embed column order: (a-1)*nfun + f, f = (variant-1)*K + k ; ours: per width the density is
# summed over k with weight 1 -> compare sum over k of Xref columns
Xref_sum = zeros(size(X))
for a in 1:S, p in 1:P
   Xref_sum[:, (a - 1) * P + p] = sum(Xref[:, (a - 1) * 2K + (p - 1) * K + k] for k in 1:K)
end
# not the same function: sqrt(sum_k rho_k) != sum_k sqrt(rho_k).  Compare instead with K=1-like
# single-width columns: set W to select one width only.
maxdiff = 0.0
for k in 1:K
   Wk = zeros(KS, 1); Wk[k:K:end, 1] .= 1.0
   Xk = extra_X(sps, Wk, ALPHAS0)
   global maxdiff = max(maxdiff, maximum(abs, Xk - Xref[:, [(a - 1) * 2K + k for a in 1:S]]))
   Wk[k:K:end, 1] .= [2.0, 1, 1, 1, 1]
   Xk = extra_X(sps, Wk, ALPHAS0)
   global maxdiff = max(maxdiff, maximum(abs, Xk - Xref[:, [(a - 1) * 2K + K + k for a in 1:S]]))
end
@printf("builder vs fs_embed.jl (single-width sqrt(rho_tot) and sqrt(rho_tot + rho^Cr)): max |diff| = %.2e\n", maxdiff)

# (a) inner solve: projection vs full factorisation
t = @elapsed af = ACEFactor(Aw, Yw, LAM)
@printf("ACE block QR (%d x %d + lam I): %.2f s\n", size(Aw)..., t)
pspec = ParamSpec(:mixed, P, false)
theta0 = vec(log.(W0))
vp = VarPro(af, Aw, asm.W, sps, pspec, theta0, lay_tr, asm.Y, asm.A, Pdiag)
Xw = weighted_X(vp, X)
t = @elapsed (L, r, cA, cX) = reduced(af, Aw, Xw)
@printf("projected inner solve (nX=%d): %.2f s   L=%.8e\n", size(Xw, 2), t, L)
t = @elapsed T = TikhonovFactor(hcat(Aw, Xw), Yw)
@printf("smallest / largest singular value of [A X] (weighted, prior-scaled): %.2e / %.2e\n", minimum(T.S), maximum(T.S))
z = tikhonov_solve(T, LAM)
Lfull = sum(abs2, hcat(Aw, Xw) * z - Yw) + LAM^2 * sum(abs2, z)
@printf("full TikhonovFactor of [A X]: %.2f s   L=%.8e   rel diff L=%.2e   max|c diff|/max|c|=%.2e\n",
        t, Lfull, abs(L - Lfull) / Lfull, maximum(abs, vcat(cA, cX) - z) / maximum(abs, z))
e = errors(vp, cA, cX, X)
@printf("train F at init (rho_tot + tilt Cr, lam=%g): %.5f\n", LAM, e.tr.F)

# (c) gradient check
t = @elapsed grad = gradient(vp, theta0)
@printf("ForwardDiff gradient (%d params): %.2f s   |g|=%.3e\n", length(theta0), t, norm(grad))
rng = MersenneTwister(7)
idx = sort(shuffle(rng, 1:length(theta0))[1:3])
println("component   analytic        FD(h=1e-4)      FD(h=1e-5)      rel err(1e-5)")
for i in idx
   fds = Float64[]
   for h in (1e-4, 1e-5)
      tp = copy(theta0); tp[i] += h; tm = copy(theta0); tm[i] -= h
      push!(fds, (objective(vp, tp) - objective(vp, tm)) / (2h))
   end
   @printf("theta[%2d]   %+.8e  %+.8e  %+.8e  %.2e\n", i, grad[i], fds[1], fds[2], abs(fds[2] - grad[i]) / abs(grad[i]))
end
# full-vector check at h=1e-5 for completeness (15*2 = 30 evaluations, ~0.3 s each)
gfd = similar(grad)
for i in eachindex(theta0)
   h = 1e-5
   tp = copy(theta0); tp[i] += h; tm = copy(theta0); tm[i] -= h
   gfd[i] = (objective(vp, tp) - objective(vp, tm)) / (2h)
end
@printf("full gradient: max rel err = %.2e   |g - gfd|/|g| = %.2e\n",
        maximum(abs.(grad - gfd) ./ max.(abs.(grad), 1e-12)), norm(grad - gfd) / norm(grad))
# gradient along the scale direction of density 1 (should vanish: sqrt(c rho) = sqrt(c) sqrt(rho))
u = zeros(length(theta0)); u[1:KS] .= 1
@printf("gradient along the overall-scale direction of density 1: %.2e (|g| = %.2e)\n", dot(grad, u) / norm(u), norm(grad))
