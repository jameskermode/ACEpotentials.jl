
# Tests for the pushforward implementation of `evaluate_basis_ed` and the
# type-stable `energy_forces_virial_basis` (design-matrix assembly).
#
# The reference is the previous implementation: a ForwardDiff Jacobian of
# `evaluate_basis` over the neighbour positions (kept here verbatim, as
# `_evaluate_basis_ed_fd`) and the unit-ful accumulation loop over all basis
# functions (`_efv_basis_ref`).

using Test, ACEbase
using Polynomials4ML.Testing: print_tf, println_slim
using ACEpotentials
M = ACEpotentials.Models

using ForwardDiff, Random, LuxCore, StaticArrays, LinearAlgebra, Unitful
using Unitful: ustrip
import AtomsBase, AtomsBuilder, AtomsCalculators
using AtomsCalculators: energy_unit, force_unit
using AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius

rng = Random.MersenneTwister(1234)
Random.seed!(11)

##
# ---- reference implementations (the pre-pushforward code, verbatim) --------

function _evaluate_basis_ed_fd(model, Rs::AbstractVector{SVector{3, T}}, Zs, Z0, ps, st) where {T}
   if length(Rs) == 0
      return zeros(T, M.length_basis(model)), zeros(SVector{3, T}, (M.length_basis(model), 0))
   end
   B = M.evaluate_basis(model, Rs, Zs, Z0, ps, st)
   dB_vec = ForwardDiff.jacobian(
               _Rs -> M.evaluate_basis(model, M.__svecs(_Rs), Zs, Z0, ps, st),
               M.__vec(Rs))
   dB1 = M.__svecs(collect(dB_vec')[:])
   dB = collect(permutedims(reshape(dB1, length(Rs), length(B)), (2, 1)))
   return B, dB
end

function _efv_basis_ref(at, calc, ps, st)
   nlist = PairList(at, cutoff_radius(calc))
   N_basis = M.length_basis(calc)
   T = Float64
   E = fill(zero(T) * energy_unit(calc), N_basis)
   F = fill(zero(SVector{3, T}) * force_unit(calc), length(at), N_basis)
   V = fill(zero(SMatrix{3, 3, T}) * energy_unit(calc), N_basis)
   for i in 1:length(at)
      Js, Rs, Zs, z0 = get_neighbours(at, calc, nlist, i)
      v, dv = _evaluate_basis_ed_fd(calc.model, Rs, Zs, z0, ps, st)
      for k = 1:N_basis
         E[k] += v[k] * energy_unit(calc)
         for α = 1:length(Js)
            F[Js[α], k] -= dv[k, α] * force_unit(calc)
            F[i, k]     += dv[k, α] * force_unit(calc)
         end
         V[k] += M._site_virial(dv[k, :], Rs) * energy_unit(calc)
      end
   end
   return (energy = E, forces = F, virial = V)
end

_relerr(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())
_fl(x) = reinterpret(Float64, ustrip.(x))

_E0s(at, calc) = sum(calc.model.Vref.E0[z] for z in AtomsBase.atomic_number(at, :))

function _rand_sys(elements)
   at = AtomsBuilder.rattle!(AtomsBuilder.bulk(:Si, cubic = true) * (2, 1, 1), 0.2)
   return AtomsBuilder.randz!(at, [ s => 1.0 for s in elements ])
end

##

elements = (:Si, :O, :C)

models = Pair{String, Any}[]

# ace1_model: spline radial basis
push!(models, "ace1_model (splines)" =>
      ace1_model(elements = collect(elements), order = 3, totaldegree = 8))

# ace_model: learnable radial basis
push!(models, "ace_model (learnable)" =>
      M.ace_model(; elements = elements, order = 3, Ytype = :solid,
                    level = M.TotalDegree(), max_level = 8, maxl = 4, pair_maxn = 8,
                    init_WB = :glorot_normal, init_Wpair = :glorot_normal))

for (name, m) in models
   @info("=== evaluate_basis_ed: $name ===")
   local model, ps, st, calc
   if m isa M.ACEPotential
      calc = m
      M.set_linear_parameters!(calc, randn(rng, M.length_basis(calc)))
      model, ps, st = calc.model, calc.ps, calc.st
   else
      model = m
      ps, st = LuxCore.setup(rng, model)
      calc = M.ACEPotential(model, ps, st)
   end

   @info("  against the ForwardDiff Jacobian and finite differences")
   for ntest = 1:5
      local Rs, Zs, z0, B, dB, B0, dB0, Bfd, nB
      Nat = rand(rng, 6:14)
      Rs, Zs, z0 = M.rand_atenv(model, Nat)
      B, dB = M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st)
      B0, dB0 = _evaluate_basis_ed_fd(model, Rs, Zs, z0, ps, st)
      print_tf(@test B ≈ B0)
      print_tf(@test size(dB) == size(dB0) == (M.length_basis(model), Nat))
      print_tf(@test dB isa Matrix{SVector{3, Float64}})
      print_tf(@test _relerr(reinterpret(Float64, dB), reinterpret(Float64, dB0)) < 1e-12)

      # finite differences of evaluate_basis along a random direction
      Us = randn(rng, SVector{3, Float64}, Nat)
      f(t) = M.evaluate_basis(model, Rs + t * Us, Zs, z0, ps, st)
      dfd = ForwardDiff.derivative(f, 0.0)                # exact directional derivative
      dpf = [ sum(dot(dB[k, j], Us[j]) for j = 1:Nat) for k = 1:size(dB, 1) ]
      print_tf(@test _relerr(dpf, dfd) < 1e-12)
      h = 1e-5
      dfd2 = (f(h) - f(-h)) / (2h)
      print_tf(@test _relerr(dpf, dfd2) < 1e-6)
   end
   println()

   @info("  type stability and the empty neighbourhood")
   Rs, Zs, z0 = M.rand_atenv(model, 10)
   println_slim(@test (@inferred M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st)) isa
                      Tuple{Vector{Float64}, Matrix{SVector{3, Float64}}})
   ws = M.BasisEDWorkspace(model, 4)     # smaller than needed: must grow
   B1, dB1 = M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st; ws = ws)
   B2, dB2 = M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st)
   println_slim(@test B1 == B2 && dB1 == dB2)
   B3, dB3 = M.evaluate_basis_ed(model, SVector{3, Float64}[], Int[], z0, ps, st)
   println_slim(@test all(iszero, B3) && size(dB3) == (M.length_basis(model), 0))

   @info("  allocations per call are bounded (workspace reuse)")
   M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st; ws = ws)
   nB = M.length_basis(model)
   # the returned (B, dB) alone cost 8 nB + 24 nB Nat bytes; allow 4x that plus
   # 2 MB of slack for the radial basis internals.  The old implementation
   # allocated ~100 MB per call here.
   bytes = @allocated M.evaluate_basis_ed(model, Rs, Zs, z0, ps, st; ws = ws)
   println_slim(@test bytes < 4 * (8 * nB + 24 * nB * length(Rs)) + 2_000_000)

   @info("  energy_forces_virial_basis against the reference and energy_forces_virial")
   for ntest = 1:3
      local at, efv, efv_ref, efv0, θ
      at = _rand_sys(elements)
      efv_ref = _efv_basis_ref(at, calc, ps, st)
      efv = M.energy_forces_virial_basis(at, calc, ps, st)
      print_tf(@test typeof(efv.energy) == typeof(efv_ref.energy))
      print_tf(@test typeof(efv.forces) == typeof(efv_ref.forces))
      print_tf(@test typeof(efv.virial) == typeof(efv_ref.virial))
      print_tf(@test size(efv.forces) == (length(at), M.length_basis(calc)))
      print_tf(@test _relerr(ustrip.(efv.energy), ustrip.(efv_ref.energy)) < 1e-12)
      print_tf(@test _relerr(_fl(efv.forces), _fl(efv_ref.forces)) < 1e-12)
      print_tf(@test _relerr(_fl(efv.virial), _fl(efv_ref.virial)) < 1e-12)

      # serial and chunked (threaded) paths agree
      efv_s = M.energy_forces_virial_basis(at, calc, ps, st; ntasks = 1)
      efv_t = M.energy_forces_virial_basis(at, calc, ps, st; ntasks = 3)
      print_tf(@test _relerr(ustrip.(efv_t.energy), ustrip.(efv_s.energy)) < 1e-12)
      print_tf(@test _relerr(_fl(efv_t.forces), _fl(efv_s.forces)) < 1e-12)
      print_tf(@test _relerr(_fl(efv_t.virial), _fl(efv_s.virial)) < 1e-12)

      # contraction with the linear parameters reproduces energy_forces_virial
      θ = M.get_basis_params(model, ps)
      efv0 = M.energy_forces_virial(at, calc, ps, st)
      print_tf(@test dot(efv.energy, θ) + _E0s(at, calc) * u"eV" ≈ efv0.energy)
      print_tf(@test all(efv.forces * θ .≈ efv0.forces))
      print_tf(@test sum(θ .* efv.virial) ≈ efv0.virial)
   end
   println()

   @info("  domain restriction")
   at = _rand_sys(elements)
   efv_a = M.energy_forces_virial_basis(at, calc, ps, st; domain = 3:5, ntasks = 1)
   efv_b = M.energy_forces_virial_basis(at, calc, ps, st; domain = [3, 4, 5], ntasks = 2)
   println_slim(@test ustrip.(efv_a.energy) ≈ ustrip.(efv_b.energy))
   println_slim(@test _fl(efv_a.forces) ≈ _fl(efv_b.forces))
   e_dom = M.potential_energy_basis(at, calc, ps, st; domain = 3:5)
   println_slim(@test ustrip.(efv_a.energy) ≈ ustrip.(e_dom))
end

##

@info("evaluate_basis_ed after splinification of a learnable model")
model = models[2].second
ps, st = LuxCore.setup(rng, model)
lin_ace = M.splinify(model, ps; nnodes = 500)
ps_lin, st_lin = LuxCore.setup(rng, lin_ace)
for ntest = 1:3
   local Rs, Zs, z0, B, dB, B0, dB0
   Rs, Zs, z0 = M.rand_atenv(lin_ace, rand(rng, 6:12))
   B, dB = M.evaluate_basis_ed(lin_ace, Rs, Zs, z0, ps_lin, st_lin)
   B0, dB0 = _evaluate_basis_ed_fd(lin_ace, Rs, Zs, z0, ps_lin, st_lin)
   print_tf(@test B ≈ B0)
   print_tf(@test _relerr(reinterpret(Float64, dB), reinterpret(Float64, dB0)) < 1e-12)
end
println()
