# stage breakdown of the scratch evaluator's site_ed! (what is left after the fixes)
include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "fast_efv.jl"))
function tmin(f; tmax = 0.3)
   f(); f(); best = Inf; t0 = time_ns(); n = 0
   while (time_ns() - t0) < tmax * 1e9 || n < 20
      t = @elapsed f(); best = min(best, t); n += 1
   end
   return best * 1e6
end
frames = load_frames(1); big = supercell(frames[1], 2); si_big = supercell(si_frame(), 2)
models = make_models((6, 8))
for (name, m) in models
   sys = name == "Si_D10" ? si_big : big
   fe = FastEFV(m)
   res, wss = fast_efv(sys, fe; ntasks = 1)
   nlist = PairList(sys, fe.rcut * u"Å")
   izs = [ findfirst(==(atomic_number(sys, i)), fe.i2z)::Int for i = 1:length(sys) ]
   ws = wss[1]; nX = _gather!(ws, fe, nlist, izs, 1); iz = izs[1]
   Rs = view(ws.Rs, 1:nX); jzs = view(ws.jzs, 1:nX); rs = view(ws.rs, 1:nX)
   Rnl = view(ws.Rnl, 1:nX, :); dRnl = view(ws.dRnl, 1:nX, :)
   Ylm = view(ws.Ylm, 1:nX, :); dYlm = view(ws.dYlm, 1:nX, :)
   ∂Rnl = view(ws.∂Rnl, 1:nX, :); ∂Ylm = view(ws.∂Ylm, 1:nX, :)
   Rp = view(ws.Rp, 1:nX, :); dRp = view(ws.dRp, 1:nX, :)
   model = fe.model; wAA = fe.wAA[iz]
   site_ed!(fe, ws, nX, iz)
   T = [
     ("gather (nlist -> Rs, jzs)", tmin(() -> _gather!(ws, fe, nlist, izs, 1))),
     ("radial_ed! (Rnl, LEN=$(fe.rad.LEN), NU=$(fe.rad.NU))", tmin(() -> radial_ed!(Rnl, dRnl, fe.rad, rs, iz, jzs, ws.es, ws.Pv, ws.Pg, ws.pr))),
     ("  of which per-edge scalars (transform/envelope/weights)", tmin(() -> (for j = 1:nX; ws.es[j] = _edge_spl(fe.rad, rs[j], iz, jzs[j]); end))),
     ("Ylm ed", tmin(() -> P4ML.evaluate_ed!(Ylm, dYlm, model.ybasis, Rs))),
     ("A evaluate!", tmin(() -> ET.evaluate!(ws.A, model.tensor.abasis, (Rnl, Ylm)))),
     ("AA evaluate!", tmin(() -> ET.evaluate!(ws.AA, model.tensor.aabasis, ws.A))),
     ("Ei = wAA ⋅ AA", tmin(() -> dot(wAA, ws.AA))),
     ("∂A pullback!(AA)", tmin(() -> ET.pullback!(ws.∂A, wAA, model.tensor.aabasis, ws.A))),
     ("∂Rnl,∂Ylm pullback!(A)", tmin(() -> ET.pullback!((∂Rnl, ∂Ylm), ws.∂A, model.tensor.abasis, (Rnl, Ylm)))),
     ("assemble ∇Ei (production loop)", tmin(() -> M._assemble_grad_ed!(ws.∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, view(ws.∇rs, 1:nX)))),
     ("assemble ∇Ei (reduce-first)", tmin(() -> _assemble_grad_fast!(ws.∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, view(ws.∇rs, 1:nX), ws.rs_scratch, nX))),
     ("pair radial_ed! (LEN=$(fe.pair.LEN), NU=$(fe.pair.NU))", tmin(() -> radial_ed!(Rp, dRp, fe.pair, rs, iz, jzs, ws.es, ws.Ppv, ws.Ppg, ws.pr))),
     ("site_ed! TOTAL", tmin(() -> site_ed!(fe, ws, nX, iz))),
   ]
   tot = T[end][2]
   println("\n=== $name (fast evaluator), site 1 with $nX neighbours ===")
   for (k, t) in T
      @printf("  %-58s %8.2f µs  %5.1f%%\n", k, t, 100 * t / tot)
   end
end
