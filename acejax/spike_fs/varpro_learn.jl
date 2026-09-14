# Exps. 2-4: learn the species weights inside the FS density by VarPro on the
# degree-4 split-0 cache; P = 1, 2, 3; init sensitivity; 150/50 early stopping.
#   julia --project=acejax/julia acejax/spike_fs/varpro_learn.jl
include(joinpath(@__DIR__, "varpro_core.jl"))
const D = parse(Int, get(ENV, "DEG", "4"))
const LAM = 1e-8            # best lambda of the sqrt(rho_tot) fit at degree 4 (FINDINGS_fs_spike)
const ITERS = parse(Int, get(ENV, "ITERS", "100"))
const OUT = joinpath(@__DIR__, get(ENV, "OUT", "varpro_learned_D$(D).jls"))
data_all = load_all()
tr, te = split0(data_all)
lay_tr = row_layout(tr); lay_te = row_layout(te)
asm = load_asm("asm_cantor_D$(D)_train200.jls"); asmte = load_asm("asm_cantor_D$(D)_test100.jls")
Pdiag = prior_diag(D)
Aw = (asm.A ./ reshape(Pdiag, 1, :)) .* asm.W; Yw = asm.W .* asm.Y
nace = size(Aw, 2)
sps_tr = [StructPairs(sys) for sys in tr]; sps_te = [StructPairs(sys) for sys in te]
@assert sum(nrows, sps_tr) == size(Aw, 1) && sum(nrows, sps_te) == size(asmte.A, 1)
t = @elapsed af = ACEFactor(Aw, Yw, LAM)
@printf("D=%d nace=%d rows=%d  ACE factor %.1f s  lambda=%g\n", D, nace, size(Aw, 1), t, LAM)

# paired bootstrap over test structures (as in species_tilt.jl)
nte = length(te); fmask = lay_te.kind .== 2
sse(resid) = [sum(abs2, resid[fmask .& (lay_te.sid .== s)]) for s in 1:nte]
nF = [3 * length(sys) for sys in te]
function boot_ci(s0, s1)
   brng = MersenneTwister(1); diffs = Float64[]
   for b in 1:2000
      idx = rand(brng, 1:nte, nte)
      push!(diffs, 100 * (sqrt(sum(s1[idx]) / sum(nF[idx])) / sqrt(sum(s0[idx]) / sum(nF[idx])) - 1))
   end
   return quantile(diffs, 0.025), quantile(diffs, 0.975)
end

results = []      # (name, ncol, ntheta, iters, wall, fixed-lam errs, sweep-best errs, resid_te, theta, ps)
function report!(name, Xtr, Xte; ntheta = 0, iters = 0, wall = 0.0, theta = nothing, ps = nothing)
   sw = sweep_fit(Aw, Yw, asm.W, Pdiag, asm.A, asm.Y, lay_tr, asmte.A, asmte.Y, lay_te, Xtr, Xte)
   fixed = sw.all[findfirst(o -> o.lam == LAM, sw.all)]
   push!(results, (name = name, ncol = nace + size(Xtr, 2), ntheta = ntheta, iters = iters, wall = wall,
                   fixed = fixed, best = sw.best, theta = theta, ps = ps))
   @printf("RES %-44s ncol=%5d  [lam=%g] test F=%.4f E=%.5f V=%.4f train F=%.4f | [best lam=%g] test F=%.4f E=%.5f V=%.4f train F=%.4f | %d it %.0f s\n",
           name, nace + size(Xtr, 2), LAM, fixed.te.F, fixed.te.E, fixed.te.V, fixed.tr.F,
           sw.best.lam, sw.best.te.F, sw.best.te.E, sw.best.te.V, sw.best.tr.F, iters, wall)
   flush(stdout)
   return results[end]
end
columns(theta, ps) = (extra_X(sps_tr, unpack(theta, ps)...), extra_X(sps_te, unpack(theta, ps)...))

# ---------------------------------------------------------------- baselines
println("\n==== baselines (fixed columns) ====")
E0 = zeros(size(Aw, 1), 0); E0te = zeros(size(asmte.A, 1), 0)
report!("linear ACE", E0, E0te)
pw1 = ParamSpec(:perwidth, 1, false)
th_ref = zeros(ntheta(pw1))
report!("sqrt(rho_tot) K=3 [ref, = perwidth P=1 init]", columns(th_ref, pw1)...; theta = th_ref, ps = pw1)
# hand tilts: ref + sqrt(rho_tot + rho^s), and which single-species tilt is best
pw2 = ParamSpec(:perwidth, 2, false)
tilt_res = []
for s in 1:S
   ws = ones(S); ws[s] = 2.0
   r = report!("ref + tilt +$(ELS[s]) (perwidth P=2 fixed)", columns(theta_perwidth([ones(S), ws]), pw2)...)
   push!(tilt_res, (s, r.fixed.te.F))
end
pw6 = ParamSpec(:perwidth, 6, false)
th_tilts = theta_perwidth(vcat([ones(S)], [(ws = ones(S); ws[s] = 2.0; ws) for s in 1:S]))
report!("ref + all 5 tilts sqrt(rho_tot + rho^s) [hand]", columns(th_tilts, pw6)...)
best_s = sort(tilt_res, by = x -> x[2])
@printf("single tilts ranked by test F: %s\n", join(["$(ELS[s]) $(round(f, digits = 4))" for (s, f) in best_s], ", "))
s1, s2 = best_s[1][1], best_s[2][1]

# ---------------------------------------------------------------- VarPro runs
function learn(name, ps, theta0; iters = ITERS, af = af, Aw = Aw, sps = sps_tr, lay = lay_tr, Y = asm.Y,
               Araw = asm.A, Wrow = asm.W, callback = nothing, show = true)
   println("\n---- $name  (mode=$(ps.mode) P=$(ps.P) ntheta=$(length(theta0)))")
   vp = VarPro(af, Aw, Wrow, sps, ps, theta0, lay, Y, Araw, Pdiag)
   L0 = objective(vp, theta0); e0 = errors(vp, vp.last.cA, vp.last.cX, vp.last.X)
   @printf("  init  L=%.6e  train F=%.5f\n", L0, e0.tr.F)
   wall = @elapsed (theta, res) = optimise!(vp, theta0; iters, callback, show)
   Ls = [t.L for t in vp.trace]
   nonmono = count(i -> Ls[i] > Ls[i-1] * (1 + 1e-12), 2:length(Ls))
   @printf("  done: %d iterations, %.0f s, L %.6e -> %.6e (%.2f%%), non-monotone steps: %d, |g|=%.2e\n",
           length(vp.trace), wall, L0, Ls[end], 100 * (Ls[end] / L0 - 1), nonmono, res.g_residual)
   return theta, vp, wall
end

learned = Dict{String, Any}()
println("\n==== Exp 2: P=1 ====")
theta, vp, wall = learn("perwidth P=1 from equal", pw1, th_ref)
report!("VarPro perwidth P=1 (init equal)", columns(theta, pw1)...; ntheta = length(theta), iters = length(vp.trace), wall, theta, ps = pw1)
learned["pw1"] = theta
print(weight_table(theta, pw1))
const FULL = get(ENV, "FULL", "0") == "1"     # init-sensitivity, mixed and full variants (see varpro_learn_D4_fdgrad_partial.log)
for seed in (FULL ? (1, 2) : ())
   th0 = 0.5 .* randn(MersenneTwister(seed), ntheta(pw1))
   theta_r, vp_r, wall_r = learn("perwidth P=1 from random seed $seed (sd 0.5)", pw1, th0; show = false)
   report!("VarPro perwidth P=1 (init random $seed)", columns(theta_r, pw1)...; ntheta = length(theta_r), iters = length(vp_r.trace), wall = wall_r, theta = theta_r, ps = pw1)
   sh = species_shares(theta_r, pw1); sh0 = species_shares(theta, pw1)
   @printf("  max |species share - equal-init run| over widths = %.3f\n", maximum(maximum(abs, sh[c] - sh0[c]) for c in 1:K))
end
mx1 = ParamSpec(:mixed, 1, false)
if FULL
theta_m, vp_m, wall_m = learn("mixed P=1 from equal (literal spec: one density, widths mixed)", mx1, zeros(ntheta(mx1)); show = false)
report!("VarPro mixed P=1 (init equal)", columns(theta_m, mx1)...; ntheta = length(theta_m), iters = length(vp_m.trace), wall = wall_m, theta = theta_m, ps = mx1)
report!("mixed P=1 at init (5 cols, fixed)", columns(zeros(ntheta(mx1)), mx1)...)
print(weight_table(theta_m, mx1))
learned["mx1"] = theta_m
end

println("\n==== Exp 3: P=2, 3 ====")
pw3 = ParamSpec(:perwidth, 3, false)
tilt(s) = (ws = ones(S); ws[s] = 2.0; ws)
inits2 = [("init equal + tilt $(ELS[s1])", theta_perwidth([ones(S), tilt(s1)]))]
FULL && append!(inits2, [("init equal + random 1", vcat(zeros(KS), 0.5 .* randn(MersenneTwister(11), KS))),
                         ("init equal + random 2", vcat(zeros(KS), 0.5 .* randn(MersenneTwister(12), KS)))])
for (lab, th0) in inits2
   theta2, vp2, wall2 = learn("perwidth P=2 $lab", pw2, th0; show = false)
   r = report!("VarPro perwidth P=2 ($lab)", columns(theta2, pw2)...; ntheta = length(theta2), iters = length(vp2.trace), wall = wall2, theta = theta2, ps = pw2)
   print(weight_table(theta2, pw2))
   haskey(learned, "pw2") && learned["pw2_F"] <= r.fixed.te.F || (learned["pw2"] = theta2; learned["pw2_F"] = r.fixed.te.F)
end
inits3 = [("init equal + tilts $(ELS[s1]),$(ELS[s2])", theta_perwidth([ones(S), tilt(s1), tilt(s2)]))]
FULL && push!(inits3, ("init equal + random", vcat(zeros(KS), 0.5 .* randn(MersenneTwister(21), 2KS))))
for (lab, th0) in inits3
   theta3, vp3, wall3 = learn("perwidth P=3 $lab", pw3, th0; show = false)
   r = report!("VarPro perwidth P=3 ($lab)", columns(theta3, pw3)...; ntheta = length(theta3), iters = length(vp3.trace), wall = wall3, theta = theta3, ps = pw3)
   print(weight_table(theta3, pw3))
   haskey(learned, "pw3") && learned["pw3_F"] <= r.fixed.te.F || (learned["pw3"] = theta3; learned["pw3_F"] = r.fixed.te.F)
end
# full mixing (45 params per density x width): does freeing the cross-width mixing add anything?
fl1 = ParamSpec(:full, 1, false)
if FULL
th0 = log.(vec([Float64(k == kk) + 1e-3 for s in 1:S, k in 1:K, kk in 1:K]))  # start at per-width identity (+small floor)
theta_f, vp_f, wall_f = learn("full P=1 from perwidth-identity", fl1, th0; show = false)
report!("VarPro full P=1 (45 params, init identity)", columns(theta_f, fl1)...; ntheta = length(theta_f), iters = length(vp_f.trace), wall = wall_f, theta = theta_f, ps = fl1)
end

println("\n==== Exp 4: overfitting control, learn on 150 with early stopping on 50, refit c on 200 ====")
n150 = 150
rows150 = findall(lay_tr.sid .<= n150); rows50 = findall(lay_tr.sid .> n150)
Aw150 = Aw[rows150, :]; Yw150 = Yw[rows150]
lay150 = row_layout(tr[1:n150]); lay50 = row_layout(tr[n150+1:end])
af150 = ACEFactor(Aw150, Yw150, LAM)
A50 = asm.A[rows50, :]; Y50 = asm.Y[rows50]
sps150 = sps_tr[1:n150]; sps50 = sps_tr[n150+1:end]
function early_stop_cb(ps; patience = 10)
   best = Ref((Inf, 0, Float64[])); X50 = Ref{Any}(nothing); last_theta = Ref{Any}(nothing)
   cb = (vp, theta, it) -> begin
      W, al = unpack(theta, ps)
      X50v = extra_X(sps50, W, al)
      c_ace = vp.last.cA ./ Pdiag; cx = vp.xscale .* vp.last.cX
      e50 = rmse_efv(A50 * c_ace + X50v * cx - Y50, lay50)
      if e50.F < best[][1]
         best[] = (e50.F, it, copy(theta))
      end
      @printf("  it %3d  train150 F=%.5f  val50 F=%.5f  (best %.5f @ it %d)\n", it, vp.trace[end].trainF, e50.F, best[][1], best[][2])
      return it - best[][2] >= patience
   end
   return cb, best
end
for (lab, ps, th0) in [("P=1", pw1, th_ref), ("P=2", pw2, inits2[1][2]), ("P=3", pw3, inits3[1][2])]
   cb, best = early_stop_cb(ps)
   theta_es, vp_es, wall_es = learn("perwidth $lab on 150, early stop on 50", ps, th0;
                                    af = af150, Aw = Aw150, sps = sps150, lay = lay150, Y = asm.Y[rows150],
                                    Araw = asm.A[rows150, :], Wrow = asm.W[rows150], callback = cb, show = false)
   th_best = best[][3]
   @printf("  early stop: best val50 F=%.5f at iteration %d of %d\n", best[][1], best[][2], length(vp_es.trace))
   report!("VarPro perwidth $lab (learned on 150, early-stopped, c refit on 200)", columns(th_best, ps)...;
           ntheta = length(th_best), iters = best[][2], wall = wall_es, theta = th_best, ps = ps)
   # also: the final (non-early-stopped) 150 solution
   report!("VarPro perwidth $lab (learned on 150, final iterate, c refit on 200)", columns(theta_es, ps)...;
           ntheta = length(theta_es), iters = length(vp_es.trace), wall = wall_es, theta = theta_es, ps = ps)
   learned["es_" * lab] = th_best
end

# ---------------------------------------------------------------- summary
println("\n==== SUMMARY  D=$D  split0 (train 200 / test 100)  lambda fixed = $LAM ====")
lin = results[1]; ref = results[2]
tilts = results[findfirst(r -> startswith(r.name, "ref + all 5 tilts"), results)]
s_lin = sse(lin.fixed.resid_te); s_ref = sse(ref.fixed.resid_te)
@printf("%-62s %5s %4s %8s %8s %8s %8s | %8s %8s | %7s %7s %7s | %6s %5s %6s\n", "variant", "ncol", "nth", "train F", "test F", "test E", "test V",
        "bestlam", "best F", "vs lin", "vs sqrt", "vs tilt", "CI(sqrt)", "it", "wall")
for r in results
   lo, hi = boot_ci(s_ref, sse(r.fixed.resid_te))
   @printf("%-62s %5d %4d %8.4f %8.4f %8.5f %8.4f | %8g %8.4f | %+6.1f%% %+6.1f%% %+6.1f%% | [%+.1f,%+.1f] %5d %5.0fs\n",
           r.name, r.ncol, r.ntheta, r.fixed.tr.F, r.fixed.te.F, r.fixed.te.E, r.fixed.te.V, r.best.lam, r.best.te.F,
           100 * (r.fixed.te.F / lin.fixed.te.F - 1), 100 * (r.fixed.te.F / ref.fixed.te.F - 1),
           100 * (r.fixed.te.F / tilts.fixed.te.F - 1), lo, hi, r.iters, r.wall)
end
serialize(OUT, (learned = learned, results = [(name = r.name, ncol = r.ncol, ntheta = r.ntheta, iters = r.iters, wall = r.wall,
                                                fixed = (tr = r.fixed.tr, te = r.fixed.te), best = (lam = r.best.lam, te = r.best.te, tr = r.best.tr),
                                                theta = r.theta, ps = r.ps) for r in results]))
println("saved $OUT")
