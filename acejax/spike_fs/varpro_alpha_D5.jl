# Exp. 6: (a) learn the radial exponents alpha_k too (theta includes log alpha),
# degree 4; (b) degree 5: VarPro P=1,2 vs the D5 reference and hand tilts, and
# the split0/D4-learned densities frozen and re-used at degree 5.
#   julia --project=acejax/julia acejax/spike_fs/varpro_alpha_D5.jl
include(joinpath(@__DIR__, "varpro_core.jl"))
const LAM = 1e-8
const ITERS = parse(Int, get(ENV, "ITERS", "100"))
data_all = load_all()
tr, te = split0(data_all)
lay_tr = row_layout(tr); lay_te = row_layout(te)
sps_tr = [StructPairs(sys) for sys in tr]; sps_te = [StructPairs(sys) for sys in te]
learned = deserialize(joinpath(@__DIR__, "varpro_learned_D4.jls")).learned
pw1 = ParamSpec(:perwidth, 1, false); pw2 = ParamSpec(:perwidth, 2, false); pw3 = ParamSpec(:perwidth, 3, false)
pw1a = ParamSpec(:perwidth, 1, true); pw2a = ParamSpec(:perwidth, 2, true)
pw6 = ParamSpec(:perwidth, 6, false)
tilt(s) = (ws = ones(S); ws[s] = 2.0; ws)
th_tilts = theta_perwidth(vcat([ones(S)], [tilt(s) for s in 1:S]))
columns(theta, ps) = (extra_X(sps_tr, unpack(theta, ps)...), extra_X(sps_te, unpack(theta, ps)...))

function setup(D)
   asm = load_asm("asm_cantor_D$(D)_train200.jls"); asmte = load_asm("asm_cantor_D$(D)_test100.jls")
   Pdiag = prior_diag(D)
   Aw = (asm.A ./ reshape(Pdiag, 1, :)) .* asm.W; Yw = asm.W .* asm.Y
   t = @elapsed af = ACEFactor(Aw, Yw, LAM)
   @printf("D=%d nace=%d  ACE factor %.1f s\n", D, size(Aw, 2), t)
   return (asm = asm, asmte = asmte, Pdiag = Pdiag, Aw = Aw, Yw = Yw, af = af)
end
function evaluate(st, name, Xtr, Xte)
   sw = sweep_fit(st.Aw, st.Yw, st.asm.W, st.Pdiag, st.asm.A, st.asm.Y, lay_tr, st.asmte.A, st.asmte.Y, lay_te, Xtr, Xte)
   fixed = sw.all[findfirst(o -> o.lam == LAM, sw.all)]
   @printf("RES %-58s ncol=%5d  [lam=%g] test F=%.4f E=%.5f V=%.4f train F=%.4f | [best lam=%g] test F=%.4f\n",
           name, size(st.Aw, 2) + size(Xtr, 2), LAM, fixed.te.F, fixed.te.E, fixed.te.V, fixed.tr.F, sw.best.lam, sw.best.te.F)
   flush(stdout)
   return fixed
end
function learn(st, name, ps, theta0)
   vp = VarPro(st.af, st.Aw, st.asm.W, sps_tr, ps, theta0, lay_tr, st.asm.Y, st.asm.A, st.Pdiag)
   L0 = objective(vp, theta0)
   wall = @elapsed (theta, res) = optimise!(vp, theta0; iters = ITERS, show = false)
   Ls = [t.L for t in vp.trace]
   @printf("---- %s: %d it, %.0f s, L %.4e -> %.4e (%.2f%%), non-monotone %d\n", name, length(vp.trace), wall, L0, Ls[end],
           100 * (Ls[end] / L0 - 1), count(i -> Ls[i] > Ls[i-1] * (1 + 1e-12), 2:length(Ls)))
   return theta
end

println("\n==== Exp 6a: degree 4, learn alpha too ====")
st4 = setup(4)
E0 = zeros(size(st4.Aw, 1), 0); E0te = zeros(size(st4.asmte.A, 1), 0)
evaluate(st4, "D4 linear ACE", E0, E0te)
evaluate(st4, "D4 sqrt(rho_tot) [ref]", columns(zeros(ntheta(pw1)), pw1)...)
evaluate(st4, "D4 VarPro perwidth P=1 (alphas fixed, from varpro_learn)", columns(learned["pw1"], pw1)...)
th = learn(st4, "D4 perwidth P=1 + alpha, init equal / (2,4,6)", pw1a, theta_perwidth([ones(S)]; learn_alpha = true))
evaluate(st4, "D4 VarPro perwidth P=1 + learned alpha", columns(th, pw1a)...)
print(weight_table(th, pw1a))
th = learn(st4, "D4 perwidth P=1 + alpha, init learned-w / (2,4,6)", pw1a, vcat(learned["pw1"], log.(ALPHAS0)))
evaluate(st4, "D4 VarPro perwidth P=1 + learned alpha (init from learned w)", columns(th, pw1a)...)
print(weight_table(th, pw1a))
th2 = learn(st4, "D4 perwidth P=2 + alpha, init learned P=2 / (2,4,6)", pw2a, vcat(learned["pw2"], log.(ALPHAS0)))
evaluate(st4, "D4 VarPro perwidth P=2 (alphas fixed, from varpro_learn)", columns(learned["pw2"], pw2)...)
evaluate(st4, "D4 VarPro perwidth P=2 + learned alpha", columns(th2, pw2a)...)
print(weight_table(th2, pw2a))
st4 = nothing; GC.gc()

println("\n==== Exp 6b: degree 5 ====")
st5 = setup(5)
E0 = zeros(size(st5.Aw, 1), 0); E0te = zeros(size(st5.asmte.A, 1), 0)
evaluate(st5, "D5 linear ACE", E0, E0te)
evaluate(st5, "D5 sqrt(rho_tot) [ref]", columns(zeros(ntheta(pw1)), pw1)...)
evaluate(st5, "D5 ref + 5 hand tilts", columns(th_tilts, pw6)...)
evaluate(st5, "D5 + D4/split0-learned P=1 (frozen)", columns(learned["pw1"], pw1)...)
evaluate(st5, "D5 + D4/split0-learned P=2 (frozen)", columns(learned["pw2"], pw2)...)
evaluate(st5, "D5 + D4/split0-learned P=3 (frozen)", columns(learned["pw3"], pw3)...)
th51 = learn(st5, "D5 perwidth P=1 from equal", pw1, zeros(ntheta(pw1)))
evaluate(st5, "D5 VarPro perwidth P=1 (learned at D5)", columns(th51, pw1)...)
print(weight_table(th51, pw1))
th52 = learn(st5, "D5 perwidth P=2 from equal + tilt Mn", pw2, theta_perwidth([ones(S), tilt(2)]))
evaluate(st5, "D5 VarPro perwidth P=2 (learned at D5)", columns(th52, pw2)...)
print(weight_table(th52, pw2))
sh4 = species_shares(learned["pw1"], pw1); sh5 = species_shares(th51, pw1)
@printf("P=1 species-share max |D4-learned - D5-learned| over widths: %.3f\n", maximum(maximum(abs, sh4[c] - sh5[c]) for c in 1:K))
serialize(joinpath(@__DIR__, "varpro_learned_D5.jls"), (pw1 = th51, pw2 = th52))
