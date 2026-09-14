# analytic gradient vs ForwardDiff gradient, all parametrisations, incl. alpha
include(joinpath(@__DIR__, "varpro_core.jl"))
data_all = load_all(); tr, te = split0(data_all); lay_tr = row_layout(tr)
asm = load_asm("asm_cantor_D4_train200.jls"); Pdiag = prior_diag(4)
Aw = (asm.A ./ reshape(Pdiag, 1, :)) .* asm.W; Yw = asm.W .* asm.Y
sps = [StructPairs(sys) for sys in tr]
af = ACEFactor(Aw, Yw, 1e-8)
rng = MersenneTwister(3)
for ps in (ParamSpec(:perwidth, 1, false), ParamSpec(:perwidth, 2, true), ParamSpec(:mixed, 2, true), ParamSpec(:full, 1, false), ParamSpec(:perwidth, 3, false))
   th0 = 0.3 .* randn(rng, ntheta(ps)); ps.learn_alpha && (th0[end-K+1:end] .= log.(ALPHAS0) .+ 0.1 .* randn(rng, K))
   vp = VarPro(af, Aw, asm.W, sps, ps, th0, lay_tr, asm.Y, asm.A, Pdiag)
   t1 = @elapsed g1 = gradient(vp, th0)
   t2 = @elapsed g2 = gradient_analytic(vp, th0)
   t2 = @elapsed g2 = gradient_analytic(vp, th0)
   @printf("%-8s P=%d alpha=%-5s ntheta=%2d  ForwardDiff %.2f s  analytic %.3f s  |g|=%.3e  rel diff %.2e  (alpha part %.2e)\n",
           ps.mode, ps.P, ps.learn_alpha, length(th0), t1, t2, norm(g1), norm(g1 - g2) / norm(g1),
           ps.learn_alpha ? norm(g1[end-K+1:end] - g2[end-K+1:end]) / norm(g1[end-K+1:end]) : 0.0)
end
