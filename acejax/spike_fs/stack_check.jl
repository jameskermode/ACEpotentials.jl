# Do the species tilts (shape of rho) and the Chebyshev embedding (shape of F)
# stack?  D=4,5 solve-only.  Reuses the definitions in species_tilt.jl by
# including it with DEGS empty, then adds the combined variants.
ENV["DEGS"] = ""
include(joinpath(@__DIR__, "species_tilt.jl"))
base0 = EmbedSpec(ZS, [], [])
lo_s, hi_s = density_ranges(base0, tr, :sqrt)
cheb4 = EmbedSpec(ZS, vcat([cheb_funs(K, k, 4, :sqrt, lo_s[k], hi_s[k]) for k in 1:K]...), ["" for _ in 1:4K])
Xcheb = (fs_feature_matrix(cheb4, tr), fs_feature_matrix(cheb4, te))
combos = [
   ("sqrt(rho_tot) [ref]",        Xc["sqrt(rho_tot) [ref]"]),
   ("+ cheb[sqrt] Kc=4",          Xcheb),
   ("+ both tilts",               Xc["+ both tilts"]),
   ("+ both tilts + cheb Kc=4",   (hcat(Xc["+ both tilts"][1], Xcheb[1]), hcat(Xc["+ both tilts"][2], Xcheb[2]))),
]
for D in (4, 5)
   Atr, Ytr, Wtr = deserialize(joinpath(CACHEDIR, "asm_cantor_D$(D)_train200.jls"))
   Ate, Yte, Wte = deserialize(joinpath(CACHEDIR, "asm_cantor_D$(D)_test100.jls"))
   m = ace1_model(elements = ELS, order = 3, totaldegree = D)
   Pdiag = M.algebraic_smoothness_prior(m.model; p = 4).diag
   Aw = (Atr ./ reshape(Pdiag, 1, :)) .* Wtr; Yw = Wtr .* Ytr
   nace = size(Aw, 2)
   med = sort([norm(c) for c in eachcol(Aw)])[div(end, 2)]
   println("\n==== D=$D  (nace=$nace) ====")
   ref = nothing; s_ref = nothing
   for (name, (Xtr, Xte)) in combos
      xn = [norm(Wtr .* c) for c in eachcol(Xtr)]; xscale = med ./ max.(xn, 1e-12)
      r = solve_sweep(hcat(Aw, (Wtr .* Xtr) .* reshape(xscale, 1, :)), Yw, nace, Pdiag, xscale, Atr, Ate, Xtr, Xte, Ytr, Yte)
      s1 = sse(r.resid_te)
      if ref === nothing
         ref = r; s_ref = s1
         @printf("%-30s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g\n",
                 name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam)
      else
         lo, hi = boot_ci(s_ref, s1)
         @printf("%-30s ncol=%5d  test F=%.4f E=%.5f V=%.4f  train F=%.4f  lam=%-5g  dF(vs sqrt)=%+.1f%% CI[%+.1f,%+.1f]\n",
                 name, nace + size(Xtr, 2), r.te.F, r.te.E, r.te.V, r.tr.F, r.lam, 100 * (r.te.F / ref.te.F - 1), lo, hi)
      end
   end
end
