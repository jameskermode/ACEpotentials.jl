# Component breakdown + sampling profiles of the forward path.
#   julia -t 1 --project=acejax/julia acejax/bench/profile_forward/02_profile.jl
include(joinpath(@__DIR__, "common.jl"))
using Profile
const ET = ACEpotentials.Models.EquivariantTensors
using ACEpotentials.Models: radii_ed!, evaluate_ed_batched!, evaluate_ed_batched,
      evaluate_ed, _assemble_grad_ed!, _z2i, get_neighbours, PairList
const P4ML = ACEpotentials.Models.P4ML
using ACEpotentials.Models: @no_escape, @alloc, @withalloc

frames = load_frames(1)
big = supercell(frames[1], 2)
si_big = supercell(si_frame(), 2)
models = make_models((6, 8))

# ---------------------------------------------------------------------------
# 1. stage-by-stage timing of ONE site (min over BenchmarkTools samples)
# ---------------------------------------------------------------------------
# min time (µs) of f over >= 0.3 s of repeats (f is a closure over globals-free locals)
function tmin(f; tmax = 0.3)
   f(); f()
   best = Inf; t0 = time_ns(); n = 0
   while (time_ns() - t0) < tmax * 1e9 || n < 20
      t = @elapsed f()
      best = min(best, t); n += 1
   end
   return best * 1e6
end

function stage_times(m, sys)
   model = m.model; ps = m.ps; st = m.st
   nlist = PairList(sys, M.cutoff_radius(m))
   Js, Rs, Zs, z0 = get_neighbours(sys, m, nlist, 1)
   i_z0 = _z2i(model.rbasis, z0)
   rs, ∇rs = radii_ed!(zeros(length(Rs)), zeros(SVector{3,Float64}, length(Rs)), Rs)
   Rnl, dRnl = evaluate_ed_batched(model.rbasis, rs, z0, Zs, ps.rbasis, st.rbasis)
   Ylm, dYlm = P4ML.evaluate_ed(model.ybasis, Rs)
   A = zeros(length(model.tensor.abasis))
   ET.evaluate!(A, model.tensor.abasis, (Rnl, Ylm))
   AA = ET.ka_evaluate(model.tensor.aabasis, A)
   A2B = model.tensor.A2Bmaps[1]
   B = A2B * AA
   ∂B = ps.WB[:, i_z0]
   ∂AA = A2B' * ∂B
   ∂A = zeros(length(A))
   ∂Rnl = zeros(size(Rnl)); ∂Ylm = zeros(size(Ylm))
   ∇Ei = zeros(SVector{3,Float64}, length(Rs))
   Rnl2 = similar(Rnl); dRnl2 = similar(dRnl)
   Ylm2 = similar(Ylm); dYlm2 = similar(dYlm)
   rs2 = similar(rs); ∇rs2 = similar(∇rs)
   AA2 = similar(AA); B2 = similar(B); ∂AA2 = similar(∂AA)

   T = Dict{String, Float64}()
   T["get_neighbours"]   = tmin(() -> get_neighbours(sys, m, nlist, 1))
   T["radii_ed!"]        = tmin(() -> radii_ed!(rs2, ∇rs2, Rs))
   T["Rnl ed (spline)"]  = tmin(() -> evaluate_ed_batched!(Rnl2, dRnl2, (model.rbasis), rs, z0, Zs, (ps.rbasis), (st.rbasis)))
   T["Ylm ed"]           = tmin(() -> P4ML.evaluate_ed!(Ylm2, dYlm2, (model.ybasis), Rs))
   T["A  evaluate!"]     = tmin(() -> ET.evaluate!(A, (model.tensor.abasis), (Rnl, Ylm)))
   T["A  ka_evaluate (alloc)"] = tmin(() -> ET.ka_evaluate((model.tensor.abasis), (Rnl, Ylm)))
   T["AA evaluate!"]     = tmin(() -> ET.evaluate!(AA2, (model.tensor.aabasis), A))
   T["AA ka_evaluate (alloc)"] = tmin(() -> ET.ka_evaluate((model.tensor.aabasis), A))
   T["B = A2B*AA (mul!)"] = tmin(() -> mul!(B2, A2B, AA))
   T["B = A2B*AA (alloc)"] = tmin(() -> A2B * AA)
   T["ET.evaluate (fwd, as used)"] = tmin(() -> ET.evaluate((model.tensor), Rnl, Ylm, NamedTuple(), NamedTuple()))
   T["∂AA = A2B'*∂B (mul!)"] = tmin(() -> mul!(∂AA2, (A2B'), ∂B))
   T["∂AA = A2B'*∂B (alloc, as used)"] = tmin(() -> (A2B') * ∂B)
   T["∂A  pullback!(AA)"] = tmin(() -> ET.pullback!(∂A, ∂AA, (model.tensor.aabasis), A))
   T["∂Rnl,∂Ylm pullback!(A)"] = tmin(() -> ET.pullback!((∂Rnl, ∂Ylm), ∂A, (model.tensor.abasis), (Rnl, Ylm)))
   T["ET.pullback (as used)"] = tmin(() -> ET.pullback([∂B], (model.tensor), Rnl, Ylm, A))
   T["assemble ∇Ei"]     = tmin(() -> _assemble_grad_ed!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs))
   T["pair ed (alloc)"]  = tmin(() -> evaluate_ed_batched((model.pairbasis), rs, z0, Zs, (ps.pairbasis), (st.pairbasis)))
   T["evaluate_ed TOTAL"] = tmin(() -> evaluate_ed(model, Rs, Zs, z0, ps, st))
   T["eval_site (energy only)"] = tmin(() -> M.evaluate(model, Rs, Zs, z0, ps, st))
   return T, length(Rs)
end

order = ["get_neighbours", "radii_ed!", "Rnl ed (spline)", "Ylm ed",
         "A  evaluate!", "A  ka_evaluate (alloc)", "AA evaluate!", "AA ka_evaluate (alloc)",
         "B = A2B*AA (mul!)", "B = A2B*AA (alloc)", "ET.evaluate (fwd, as used)",
         "∂AA = A2B'*∂B (mul!)", "∂AA = A2B'*∂B (alloc, as used)",
         "∂A  pullback!(AA)", "∂Rnl,∂Ylm pullback!(A)", "ET.pullback (as used)",
         "assemble ∇Ei", "pair ed (alloc)", "evaluate_ed TOTAL", "eval_site (energy only)"]

for (name, m) in models
   sys = name == "Si_D10" ? si_big : big
   T, nn = stage_times(m, sys)
   println("\n=== $name : per-site stage timings (µs), site 1 with $nn neighbours ===")
   tot = T["evaluate_ed TOTAL"]
   for k in order
      @printf("  %-34s %9.2f µs  %5.1f%% of evaluate_ed\n", k, T[k], 100*T[k]/tot)
   end
end

# ---------------------------------------------------------------------------
# 2. sampling profile of the full energy_forces_virial call (256 atoms)
# ---------------------------------------------------------------------------
function run_profile(name, m, sys; nrep = 20)
   efv(sys, m)
   Profile.clear()
   Profile.init(n = 10^7, delay = 0.0002)
   Profile.@profile for _ = 1:nrep; efv(sys, m); end
   println("\n\n######## FLAT PROFILE: $name ($(length(sys)) atoms) ########")
   Profile.print(format = :flat, sortedby = :count, noisefloor = 2, mincount = 20, C = false)
   println("\n\n######## TREE PROFILE: $name ($(length(sys)) atoms) ########")
   Profile.print(format = :tree, maxdepth = 22, noisefloor = 2, mincount = 30, C = false)
end

for name in ["cat_D6", "emb16_D6", "Si_D10"]
   m = models[findfirst(p -> p.first == name, models)].second
   sys = name == "Si_D10" ? si_big : big
   run_profile(name, m, sys)
end
