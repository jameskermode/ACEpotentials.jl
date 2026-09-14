
# Tests for the fast forward evaluator (spline radial tables, the
# allocation-free site kernel `evaluate_ed!` and the `energy_forces_virial`
# driver) against the previous implementation, which is kept here verbatim:
# `_evaluate_ed_ref` is the old `evaluate_ed` (scalar Interpolations splines
# per edge, `EquivariantTensors.evaluate` / `pullback`, `_assemble_grad_ed!`)
# and `_efv_ref` the old generic SitePotential driver loop.

using Test, ACEbase
using Polynomials4ML.Testing: print_tf, println_slim
using ACEpotentials
M = ACEpotentials.Models
ET = M.EquivariantTensors
P4ML = M.P4ML

using ForwardDiff, Random, LuxCore, StaticArrays, LinearAlgebra, Unitful
using Unitful: ustrip
import AtomsBase, AtomsBuilder, AtomsCalculators
using AtomsCalculators: energy_unit, force_unit
using AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius

rng = Random.MersenneTwister(1234)
Random.seed!(11)

##
# ---- reference implementations (pre-fast-path code, verbatim) --------------

function _assemble_grad_ed_ref!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs)
   @inbounds for t = 1:size(∂Rnl, 2)
      for j = 1:size(∂Rnl, 1)
         ∇Ei[j] += (∂Rnl[j, t] * dRnl[j, t]) * ∇rs[j]
      end
   end
   @inbounds for t = 1:size(∂Ylm, 2)
      for j = 1:size(∂Ylm, 1)
         ∇Ei[j] += ∂Ylm[j, t] * dYlm[j, t]
      end
   end
   return ∇Ei
end

# the previous per-edge spline radial evaluation (Interpolations, Dual numbers)
function _rad_ed_ref(basis, rs, Z0, Zs, ps, st)
   Rnl = zeros(length(rs), length(basis)); dRnl = zeros(length(rs), length(basis))
   for j = 1:length(rs)
      v, d = M.evaluate_ed(basis, rs[j], Z0, Zs[j], ps, st)
      Rnl[j, :] .= v; dRnl[j, :] .= d
   end
   return Rnl, dRnl
end

function _evaluate_ed_ref(model, Rs::AbstractVector{SVector{3, T}}, Zs, Z0, ps, st) where {T}
   i_z0 = M._z2i(model.rbasis, Z0)
   length(Rs) == 0 && return model.Vref.E0[Z0], SVector{3, T}[]
   rs = norm.(Rs); ∇rs = Rs ./ rs
   Rnl, dRnl = _rad_ed_ref(model.rbasis, rs, Z0, Zs, ps.rbasis, st.rbasis)
   Ylm, dYlm = P4ML.evaluate_ed(model.ybasis, Rs)
   A = zeros(T, length(model.tensor.abasis))
   ET.evaluate!(A, model.tensor.abasis, (Rnl, Ylm))
   BB = ET.evaluate(model.tensor, Rnl, Ylm, NamedTuple(), NamedTuple())
   B = BB[1]
   Ei = dot(B, (@view ps.WB[:, i_z0]))
   ∂B = @view ps.WB[:, i_z0]
   ∂Rnl, ∂Ylm = ET.pullback([∂B], model.tensor, Rnl, Ylm, A)
   ∇Ei = zeros(SVector{3, T}, length(Rs))
   _assemble_grad_ed_ref!(∇Ei, ∂Rnl, dRnl, ∂Ylm, dYlm, ∇rs)
   Rpair, dRpair = _rad_ed_ref(model.pairbasis, rs, Z0, Zs, ps.pairbasis, st.pairbasis)
   Apair = sum(Rpair, dims = 1)[:]
   Wp_i = @view ps.Wpair[:, i_z0]
   Ei += dot(Apair, Wp_i)
   for j = 1:length(Rs)
      ∇Ei[j] += dot(Wp_i, (@view dRpair[j, :])) * (Rs[j] / rs[j])
   end
   Ei += model.Vref.E0[Z0]
   return Ei, ∇Ei
end

# the generic SitePotential driver (serial), with the reference kernel
function _efv_ref(at, calc, ps, st; nlist = PairList(at, cutoff_radius(calc)))
   E = AtomsCalculators.zero_energy(at, calc)
   frc = AtomsCalculators.zero_forces(at, calc)
   vir = AtomsCalculators.zero_virial(at, calc)
   for i in 1:length(at)
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      Ei, ∇Ei = _evaluate_ed_ref(calc.model, Rs, Zs, z0, ps, st)
      E += Ei * energy_unit(calc)
      vir += M._site_virial(∇Ei, Rs) * energy_unit(calc)
      for α in 1:length(Js)
         frc[Js[α]] -= ∇Ei[α] * force_unit(calc)
      end
      frc[i] += sum(∇Ei) * force_unit(calc)
   end
   return (energy = E, forces = frc, virial = vir)
end

_fl(x) = reinterpret(Float64, ustrip.(x))
_maxabs(x) = maximum(abs, x)
_relerr(a, b) = _maxabs(a .- b) / max(_maxabs(b), 1.0)

function _rand_sys(elements; rep = (2, 1, 1), rattle = 0.2)
   at = AtomsBuilder.rattle!(AtomsBuilder.bulk(:Si, cubic = true) * rep, rattle)
   return AtomsBuilder.randz!(at, [ s => 1.0 for s in elements ])
end

function _check_efv(at, calc, ps, st)
   ref = _efv_ref(at, calc, ps, st)
   efv = M.energy_forces_virial(at, calc, ps, st)
   print_tf(@test typeof(efv.energy) == typeof(ref.energy))
   print_tf(@test typeof(efv.forces) == typeof(ref.forces))
   print_tf(@test typeof(efv.virial) == typeof(ref.virial))
   print_tf(@test abs(ustrip(efv.energy - ref.energy)) <= 1e-12 * max(1.0, abs(ustrip(ref.energy))))
   print_tf(@test _relerr(_fl(efv.forces), _fl(ref.forces)) <= 1e-10)
   print_tf(@test _relerr(_fl(efv.virial), _fl(ref.virial)) <= 1e-10)
   # the AtomsCalculators entry point routes to the same path
   efv2 = AtomsCalculators.energy_forces_virial(at, calc)
   print_tf(@test efv2.energy == efv.energy && efv2.forces == efv.forces)
   # serial vs chunked
   efv_s = M.energy_forces_virial(at, calc, ps, st; ntasks = 1)
   efv_t = M.energy_forces_virial(at, calc, ps, st; ntasks = 3)
   print_tf(@test abs(ustrip(efv_t.energy - efv_s.energy)) <= 1e-12 * max(1.0, abs(ustrip(efv_s.energy))))
   print_tf(@test _relerr(_fl(efv_t.forces), _fl(efv_s.forces)) <= 1e-12)
   print_tf(@test _relerr(_fl(efv_t.virial), _fl(efv_s.virial)) <= 1e-12)
   # the generic (Folds) energy and virial drivers use the value kernel
   E1 = AtomsCalculators.potential_energy(at, calc)
   print_tf(@test abs(ustrip(E1 - ref.energy)) <= 1e-12 * max(1.0, abs(ustrip(ref.energy))))
   return nothing
end

##

models = Pair{String, Any}[]
for D in (6, 8)
   local m = ace1_model(elements = [:Si, :O, :C], order = 3, totaldegree = D)
   M.set_linear_parameters!(m, randn(rng, M.length_basis(m)))
   push!(models, "ace1_model D=$D (Si,O,C)" => (m, (:Si, :O, :C)))
end
let m = ace1_model(elements = [:Si], order = 3, totaldegree = 10)
   M.set_linear_parameters!(m, randn(rng, M.length_basis(m)))
   push!(models, "ace1_model D=10 (Si)" => (m, (:Si,)))
end
let model = M.ace_model(; elements = (:Si, :O, :C), order = 3, Ytype = :solid,
                          level = M.TotalDegree(), max_level = 8, maxl = 4, pair_maxn = 8,
                          init_WB = :glorot_normal, init_Wpair = :glorot_normal)
   ps, st = LuxCore.setup(rng, model)
   push!(models, "ace_model learnable (Si,O,C)" => (M.ACEPotential(model, ps, st), (:Si, :O, :C)))
   lin = M.splinify(model, ps; nnodes = 500)
   ps_lin, st_lin = LuxCore.setup(rng, lin)
   ps_lin.WB[:] .= ps.WB[:]; ps_lin.Wpair[:] .= ps.Wpair[:]
   push!(models, "ace_model splinified (Si,O,C)" => (M.ACEPotential(lin, ps_lin, st_lin), (:Si, :O, :C)))
end

##

@info("Spline tables: factorisation and exactness against the per-edge splines")
for (name, (calc, elements)) in models
   local model = calc.model
   model.rbasis isa M.SplineRnlrzzBasis || continue
   for b in (model.rbasis, model.pairbasis)
      fac = b.meta["radial_factorisation"]
      println("   $name: LEN = $(fac.LEN), NU = $(fac.NU)")
      Rs, Zs, z0 = M.rand_atenv(model, 20)
      rs = norm.(Rs)
      R_ref, dR_ref = _rad_ed_ref(b, rs, z0, Zs, NamedTuple(), NamedTuple())
      R, dR = M.evaluate_ed_batched(b, rs, z0, Zs, NamedTuple(), NamedTuple())
      R2 = M.evaluate_batched(b, rs, z0, Zs, NamedTuple(), NamedTuple())
      print_tf(@test _relerr(R, R_ref) <= 1e-13)
      print_tf(@test _relerr(dR, dR_ref) <= 1e-12)
      print_tf(@test _relerr(R2, R_ref) <= 1e-13)
      # the unfactorised tables give the same result through the same kernel
      tab0 = M.RnlSplineTables(b._i2z, b.transforms, b.envelopes, b.splines; factorise = false)
      print_tf(@test tab0.NU == tab0.LEN && !tab0.factorised)
      R0 = zeros(20, length(b)); dR0 = zeros(20, length(b))
      M._spline_tables_batched!(R0, dR0, tab0, M._z2i(b, z0), rs, Zs, b)
      print_tf(@test _relerr(R0, R_ref) <= 1e-13 && _relerr(dR0, dR_ref) <= 1e-12)
      # Dual numbers through the value kernel reproduce the derivative
      f(x) = vec(sum(M.evaluate_batched(b, x, z0, Zs, NamedTuple(), NamedTuple()), dims = 2))
      J = ForwardDiff.jacobian(f, rs)
      print_tf(@test _relerr(diag(J), vec(sum(dR, dims = 2))) <= 1e-12)
   end
end
println()
# the one-hot bases are factorised, the random-weight one is not
println_slim(@test models[1].second[1].model.rbasis.meta["radial_factorisation"].NU <
                   models[1].second[1].model.rbasis.meta["radial_factorisation"].LEN)
println_slim(@test models[end].second[1].model.rbasis.meta["radial_factorisation"].NU ==
                   models[end].second[1].model.rbasis.meta["radial_factorisation"].LEN)

##

for (name, (calc, elements)) in models
   @info("=== forward evaluator: $name ===")
   local model, ps, st
   model, ps, st = calc.model, calc.ps, calc.st

   @info("  evaluate_ed against the reference kernel, @inferred, allocations")
   for ntest = 1:5
      local Rs, Zs, z0
      Rs, Zs, z0 = M.rand_atenv(model, rand(rng, 6:14))
      E_ref, dE_ref = _evaluate_ed_ref(model, Rs, Zs, z0, ps, st)
      E, dE = M.evaluate_ed(model, Rs, Zs, z0, ps, st)
      print_tf(@test abs(E - E_ref) <= 1e-12 * max(1.0, abs(E_ref)))
      print_tf(@test _relerr(reinterpret(Float64, dE), reinterpret(Float64, dE_ref)) <= 1e-10)
   end
   println()
   Rs, Zs, z0 = M.rand_atenv(model, 10)
   println_slim(@test (@inferred M.evaluate_ed(model, Rs, Zs, z0, ps, st)) isa
                      Tuple{Float64, Vector{SVector{3, Float64}}})
   ws = M.SiteEDWorkspace(model, 4)     # too small: must grow
   wAA = M.fold_readout_weights(model, ps, M._z2i(model, z0))
   E1 = M.evaluate_ed!(ws, model, Rs, Zs, z0, ps, st, wAA)
   E2, dE2 = M.evaluate_ed(model, Rs, Zs, z0, ps, st)
   println_slim(@test E1 == E2 && ws.∇Ei[1:10] == dE2)
   # allocation-free site kernel (spline radial bases; the learnable basis
   # still allocates per edge in its own evaluate_ed_batched!)
   is_spline = model.rbasis isa M.SplineRnlrzzBasis
   Zs_int = Int.(Zs)
   _kernel() = M.evaluate_ed!(ws, model, Rs, Zs_int, z0, ps, st, wAA)
   _kernel()
   nb = @allocated _kernel()
   println_slim(@test nb <= (is_spline ? 64 : 100_000))
   # empty neighbourhood
   println_slim(@test M.evaluate_ed(model, SVector{3, Float64}[], Int[], z0, ps, st)[1] == model.Vref.E0[z0])
   # Dual numbers through the kernel (hessians): symmetric second derivatives
   g(x) = collect(M.__vec(M.evaluate_ed(model, M.__svecs(x), Zs, z0, ps, st)[2]))
   J = ForwardDiff.jacobian(g, collect(M.__vec(Rs)))
   println_slim(@test _maxabs(J - J') <= 1e-10 * max(1.0, _maxabs(J)))

   @info("  energy_forces_virial against the reference driver")
   for ntest = 1:3
      local at = _rand_sys(elements)
      _check_efv(at, calc, ps, st)
   end
   println()

   @info("  domain, nlist and workspace reuse, allocations")
   local at = _rand_sys(elements; rep = (2, 2, 1))
   nlist = PairList(at, cutoff_radius(calc))
   efv = M.energy_forces_virial(at, calc, ps, st; ntasks = 1)
   efv_a = M.energy_forces_virial(at, calc, ps, st; domain = 3:5, ntasks = 1)
   efv_b = M.energy_forces_virial(at, calc, ps, st; domain = [3, 4, 5], ntasks = 2, nlist = nlist)
   println_slim(@test efv_a.energy ≈ efv_b.energy && all(efv_a.forces .≈ efv_b.forces))
   E_dom = AtomsCalculators.potential_energy(at, calc; domain = 3:5)
   println_slim(@test efv_a.energy ≈ E_dom)
   wss = [ M.SiteEDWorkspace(model, 8) for _ = 1:2 ]
   efv_w = M.energy_forces_virial(at, calc, ps, st; ntasks = 2, nlist = nlist, ws = wss)
   println_slim(@test efv_w.energy ≈ efv.energy && all(efv_w.forces .≈ efv.forces))
   # with the neighbour list and workspaces reused, the call allocates only
   # the outputs and a few small arrays (the old path: ~1150 per site)
   if model.rbasis isa M.SplineRnlrzzBasis
      _efv() = M.energy_forces_virial(at, calc, ps, st; ntasks = 1, nlist = nlist, ws = wss)
      _efv()
      nalloc = @allocated _efv()
      println_slim(@test nalloc < 200_000 + 100 * length(at))
      stats = @timed _efv()
      println_slim(@test Base.gc_alloc_count(stats.gcstats) < 100 + 10 * length(at))
   end
end
