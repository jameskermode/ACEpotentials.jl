# Phase 0 spike: export a Si ETACE model + reference values for the JAX port.
#
# SPIKE CODE. Throwaway quality, but the schema here is the draft of the
# Stage 1 exporter described in docs/plans/jax_ace_port_plan.md.
#
# Run:  ~/.juliaup/bin/julialauncher --project=. spike/jax_phase0/export_model.jl
#
# NOTE: uses M.ace_model (LearnableRnlrzzBasis), NOT ace1_model. ace1_model
# splinifies the radial basis and convert2et rejects SplineRnlrzzBasis.

using ACEpotentials, StaticArrays, Lux, Random, LuxCore, AtomsBuilder, Unitful, JSON
using SparseArrays: findnz
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels
import EquivariantTensors as ET
import Polynomials4ML as P4ML

const OUT = joinpath(@__DIR__, "si_model.json")

# ---------------------------------------------------------------- model
rcut = 5.5
rin0cuts = M._default_rin0cuts((:Si,))
rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

model = M.ace_model(; elements = (:Si,), order = 3, Ytype = :solid,
                      level = M.TotalDegree(), max_level = 10, maxl = 6,
                      pair_maxn = 10, rin0cuts = rin0cuts,
                      init_WB = :glorot_normal, init_Wpair = :glorot_normal)
ps, st = Lux.setup(MersenneTwister(1234), model)

et = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(MersenneTwister(1234), et)

# copy the learnable radial weights across (NZ = 1 so a single category)
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]

# ---------------------------------------------------------------- specs
polys      = model.rbasis.polys
rnl_spec   = model.rbasis.spec
ylm_spec   = P4ML.natural_indices(et.yembed.layer.basis)
aspec      = et.basis.abasis.spec            # Vector{Tuple{Int,Int}} (1-based)
aaspecs    = et.basis.aabasis.specs          # tuple of Vector{NTuple{N,Int}}
aaranges   = et.basis.aabasis.ranges
agnesi     = et_st.rembed.trans.params[1]
rows, cols, vals = findnz(et.basis.A2Bmaps[1])

@assert !et.basis.aabasis.hasconst "hasconst not handled by the spike"

# ---------------------------------------------------------------- test structure
Random.seed!(20260909)   # reproducible rattle
sys = AtomsBuilder.bulk(:Si, cubic = true) * 2      # 64 atoms
rattle!(sys, 0.1u"Å")
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

B_ref = ETM.site_basis(et, G, et_ps, et_st)                  # (nnodes, nB)
phi_ref, _ = et(G, et_ps, et_st)                             # site energies

edge_r = [Vector(e.𝐫) for e in G.edge_data]

# probe vectors for intermediate checks (bisecting convention mismatches)
using LinearAlgebra: norm
z_si = model.rbasis._i2z[1]   # raw form expected by _z2i
Random.seed!(7)
probe_r = [randn(SVector{3,Float64}) for _ in 1:6]
probe_r = [r * (1.0 + 3.5*rand()) / norm(r) for r in probe_r]

D = Dict(
  "meta" => Dict("elements" => ["Si"], "rcut" => rcut, "order" => 3,
                 "n_polys" => length(polys), "n_rnl" => length(rnl_spec),
                 "n_ylm" => length(ylm_spec), "n_A" => length(aspec),
                 "n_AA" => sum(length, aaspecs), "n_B" => size(B_ref, 2),
                 "julia_version" => string(VERSION),
                 "acepotentials_version" => string(pkgversion(ACEpotentials))),
  # radial: y = agnesi(r); P = polys(y); Pe = P * (1-y^2)^2; Rnl = W_rnl * Pe
  "agnesi" => Dict(string(k) => getfield(agnesi, k) for k in keys(agnesi)),
  "polys_A" => collect(polys.refstate.A),
  "polys_B" => collect(polys.refstate.B),
  "polys_C" => collect(polys.refstate.C),
  "W_rnl"   => [et_ps.rembed.post.W[i, j, 1]
                for i in 1:size(et_ps.rembed.post.W, 1),
                    j in 1:size(et_ps.rembed.post.W, 2)],   # (n_rnl, n_polys)
  "rnl_spec" => [[b.n, b.l] for b in rnl_spec],
  "ylm_spec" => [[b.l, b.m] for b in ylm_spec],
  # A[i,k] = sum_j Rnl[j,i,aspec_r[k]] * Ylm[j,i,aspec_y[k]]   (0-based below)
  "aspec_r" => [t[1] - 1 for t in aspec],
  "aspec_y" => [t[2] - 1 for t in aspec],
  # AA grouped by correlation order; 0-based indices into A
  "aaspec_by_order" => [[collect(t) .- 1 for t in aaspecs[N]] for N in 1:length(aaspecs)],
  "aaranges" => [[first(r) - 1, last(r) - 1] for r in aaranges],
  # B = A2B * AA   (0-based COO)
  "A2B_rows" => rows .- 1, "A2B_cols" => cols .- 1, "A2B_vals" => vals,
  "A2B_shape" => [size(et.basis.A2Bmaps[1])...],
  "W_readout" => [et_ps.readout.W[1, i, 1] for i in 1:size(et_ps.readout.W, 2)],
  # ---- test structure ----
  "test" => Dict(
     "n_atoms" => length(sys), "n_edges" => ET.nedges(G),
     "edge_i" => G.ii .- 1, "edge_j" => G.jj .- 1,
     "edge_rij" => edge_r,
     "B_ref" => [B_ref[i, j] for i in 1:size(B_ref, 1), j in 1:size(B_ref, 2)],
     "phi_ref" => vec(collect(phi_ref)),
  ),
  "probe" => Dict(
     "rij"      => [Vector(r) for r in probe_r],
     "y_agnesi" => [ET.eval_agnesi(norm(r), agnesi) for r in probe_r],
     "Ylm"      => [collect(P4ML.evaluate(et.yembed.layer.basis, r)) for r in probe_r],
     "Rnl"      => [collect(M.evaluate(model.rbasis, norm(r), z_si, z_si,
                                       ps.rbasis, st.rbasis)) for r in probe_r],
  ),
)

open(OUT, "w") do io; JSON.print(io, D); end
println("wrote $OUT  ($(round(filesize(OUT)/1e6, digits=2)) MB)")
println("n_rnl=$(length(rnl_spec)) n_ylm=$(length(ylm_spec)) n_A=$(length(aspec)) ",
        "n_AA=$(sum(length, aaspecs)) n_B=$(size(B_ref,2)) nedges=$(ET.nedges(G))")
