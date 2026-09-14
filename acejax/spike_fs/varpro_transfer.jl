# Exp. 5: TRANSFER.  Freeze the densities learned on split 0 (varpro_learn.jl),
# append the frozen columns to the degree-4 assembly of a disjoint split 1
# (varpro_assemble_split1.jl), convex fit, evaluate.  Compare with the linear
# baseline, sqrt(rho_tot), the hand tilts, and with densities learned on split 1
# itself (in-split upper reference); also the reverse direction (split1-learned
# densities on split 0).
#   julia --project=acejax/julia acejax/spike_fs/varpro_transfer.jl
include(joinpath(@__DIR__, "varpro_core.jl"))
const D = 4
const LAM = 1e-8
const ITERS = parse(Int, get(ENV, "ITERS", "100"))
data_all = load_all()
tr0, te0 = split0(data_all); tr1, te1 = split1(data_all)
saved = deserialize(joinpath(@__DIR__, "varpro_learned_D$(D).jls"))
learned = saved.learned
Pdiag = prior_diag(D)
pw1 = ParamSpec(:perwidth, 1, false); pw2 = ParamSpec(:perwidth, 2, false); pw3 = ParamSpec(:perwidth, 3, false)
mx1 = ParamSpec(:mixed, 1, false); pw6 = ParamSpec(:perwidth, 6, false)
tilt(s) = (ws = ones(S); ws[s] = 2.0; ws)
th_tilts = theta_perwidth(vcat([ones(S)], [tilt(s) for s in 1:S]))

struct Split
   name::String; tr; te; lay_tr; lay_te; asm; asmte; Aw; Yw; sps_tr; sps_te; af
end
function Split(name, tr, te, tag)
   lay_tr = row_layout(tr); lay_te = row_layout(te)
   asm = load_asm("asm_cantor_D$(D)_$(tag)train200.jls"); asmte = load_asm("asm_cantor_D$(D)_$(tag)test100.jls")
   Aw = (asm.A ./ reshape(Pdiag, 1, :)) .* asm.W; Yw = asm.W .* asm.Y
   sps_tr = [StructPairs(sys) for sys in tr]; sps_te = [StructPairs(sys) for sys in te]
   @assert sum(nrows, sps_tr) == size(Aw, 1) && sum(nrows, sps_te) == size(asmte.A, 1)
   af = ACEFactor(Aw, Yw, LAM)
   Split(name, tr, te, lay_tr, lay_te, asm, asmte, Aw, Yw, sps_tr, sps_te, af)
end
sp0 = Split("split0", tr0, te0, ""); sp1 = Split("split1", tr1, te1, "split1_")
@printf("split1: train rows %d, test rows %d; overlap with split0 structures: %d (by construction 0)\n",
        size(sp1.Aw, 1), size(sp1.asmte.A, 1), 0)

columns(sp, theta, ps) = (extra_X(sp.sps_tr, unpack(theta, ps)...), extra_X(sp.sps_te, unpack(theta, ps)...))
function boot_ci(sp, r0, r1)
   nte = length(sp.te); fmask = sp.lay_te.kind .== 2
   sse(resid) = [sum(abs2, resid[fmask .& (sp.lay_te.sid .== s)]) for s in 1:nte]
   nF = [3 * length(sys) for sys in sp.te]
   s0 = sse(r0); s1 = sse(r1)
   brng = MersenneTwister(1); diffs = Float64[]
   for b in 1:2000
      idx = rand(brng, 1:nte, nte)
      push!(diffs, 100 * (sqrt(sum(s1[idx]) / sum(nF[idx])) / sqrt(sum(s0[idx]) / sum(nF[idx])) - 1))
   end
   return quantile(diffs, 0.025), quantile(diffs, 0.975)
end
function evaluate(sp, name, theta, ps)
   Xtr, Xte = theta === nothing ? (zeros(size(sp.Aw, 1), 0), zeros(size(sp.asmte.A, 1), 0)) : columns(sp, theta, ps)
   sw = sweep_fit(sp.Aw, sp.Yw, sp.asm.W, Pdiag, sp.asm.A, sp.asm.Y, sp.lay_tr, sp.asmte.A, sp.asmte.Y, sp.lay_te, Xtr, Xte)
   fixed = sw.all[findfirst(o -> o.lam == LAM, sw.all)]
   return (name = name, ncol = size(sp.Aw, 2) + size(Xtr, 2), fixed = fixed, best = sw.best)
end
function learn_on(sp, name, ps, theta0)
   vp = VarPro(sp.af, sp.Aw, sp.asm.W, sp.sps_tr, ps, theta0, sp.lay_tr, sp.asm.Y, sp.asm.A, Pdiag)
   L0 = objective(vp, theta0)
   wall = @elapsed (theta, res) = optimise!(vp, theta0; iters = ITERS, show = false)
   Ls = [t.L for t in vp.trace]
   @printf("learned %s on %s: %d it, %.0f s, L %.4e -> %.4e, non-monotone %d\n", name, sp.name, length(vp.trace), wall, L0, Ls[end],
           count(i -> Ls[i] > Ls[i-1] * (1 + 1e-12), 2:length(Ls)))
   return theta
end

# densities learned on split 1 itself (in-split reference)
best_s1 = argmax(species_shares(learned["pw2"], pw2)[K+1])   # the species density 2 up-weighted most on split 0 (just for an init)
th1_pw1 = learn_on(sp1, "perwidth P=1", pw1, zeros(ntheta(pw1)))
th1_pw2 = learn_on(sp1, "perwidth P=2", pw2, theta_perwidth([ones(S), tilt(best_s1)]))
th1_pw3 = learn_on(sp1, "perwidth P=3", pw3, learned["pw3"] .* 0 .+ vcat(zeros(KS), 0.5 .* randn(MersenneTwister(21), 2KS)))

variants = [
   ("linear ACE", nothing, nothing),
   ("sqrt(rho_tot) K=3 [ref]", zeros(ntheta(pw1)), pw1),
   ("ref + 5 hand tilts", th_tilts, pw6),
   ("split0-learned perwidth P=1 (frozen)", learned["pw1"], pw1),
   ("split0-learned perwidth P=2 (frozen)", learned["pw2"], pw2),
   ("split0-learned perwidth P=3 (frozen)", learned["pw3"], pw3),
   ("split0-learned P=1 early-stopped 150/50 (frozen)", learned["es_P=1"], pw1),
   ("split0-learned P=2 early-stopped 150/50 (frozen)", learned["es_P=2"], pw2),
   ("split0-learned P=3 early-stopped 150/50 (frozen)", learned["es_P=3"], pw3),
   ("split1-learned perwidth P=1 (in-split)", th1_pw1, pw1),
   ("split1-learned perwidth P=2 (in-split)", th1_pw2, pw2),
   ("split1-learned perwidth P=3 (in-split)", th1_pw3, pw3),
]
haskey(learned, "mx1") && insert!(variants, 7, ("split0-learned mixed P=1 (frozen)", learned["mx1"], mx1))
for sp in (sp1, sp0)
   println("\n==== evaluation on $(sp.name)  (lambda fixed = $LAM; 'best' = best of the lambda sweep) ====")
   rs = [evaluate(sp, v...) for v in variants]
   lin = rs[1]; ref = rs[2]; tl = rs[3]
   @printf("%-52s %5s %8s %8s %8s %8s | %8s | %7s %7s %7s  %s\n", "variant", "ncol", "train F", "test F", "test E", "test V", "best F", "vs lin", "vs sqrt", "vs tilt", "CI vs lin")
   for r in rs
      lo, hi = boot_ci(sp, lin.fixed.resid_te, r.fixed.resid_te)
      @printf("%-52s %5d %8.4f %8.4f %8.5f %8.4f | %8.4f | %+6.1f%% %+6.1f%% %+6.1f%%  [%+.1f,%+.1f]\n",
              r.name, r.ncol, r.fixed.tr.F, r.fixed.te.F, r.fixed.te.E, r.fixed.te.V, r.best.te.F,
              100 * (r.fixed.te.F / lin.fixed.te.F - 1), 100 * (r.fixed.te.F / ref.fixed.te.F - 1),
              100 * (r.fixed.te.F / tl.fixed.te.F - 1), lo, hi)
   end
end
println("\nsplit1-learned weights:")
print(weight_table(th1_pw1, pw1)); print(weight_table(th1_pw2, pw2))
println("split0-learned weights (for comparison):")
print(weight_table(learned["pw1"], pw1)); print(weight_table(learned["pw2"], pw2))
# similarity of the learned species shares across splits
sh0 = species_shares(learned["pw1"], pw1); sh1 = species_shares(th1_pw1, pw1)
@printf("P=1 species-share max |split0 - split1| over widths: %.3f\n", maximum(maximum(abs, sh0[c] - sh1[c]) for c in 1:K))
serialize(joinpath(@__DIR__, "varpro_learned_split1_D$(D).jls"), (pw1 = th1_pw1, pw2 = th1_pw2, pw3 = th1_pw3))
