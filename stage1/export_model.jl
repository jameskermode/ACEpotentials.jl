# Stage 1 exporter: fit an ace1_model in Julia, export it to npz for the JAX
# evaluator.  See docs/plans/jax_ace_port_plan.md.
#
#   julia --project=stage1 stage1/export_model.jl [output.npz]
#
# Exports the SPLINED radial branch, which is what ace1_model actually builds
# (src/ace1_compat.jl:283) and therefore what a fitted production model is.
# Reproducing Julia's own splines means JAX and Julia agree bit-for-bit rather
# than differing by a re-tabulation error.
#
# npz, not JSON: Julia writes matrices column-major, so 2-D arrays round-trip
# through JSON transposed.  Every array below is written in the orientation the
# Python loader expects and checked by stage1/tests/test_roundtrip.py.

using ACEpotentials, NPZ, JSON, StaticArrays, LinearAlgebra, Random, Printf
using AtomsCalculators, ACEfit
using LazyArtifacts, ExtXYZ, AtomsBase, Unitful
M = ACEpotentials.Models

const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "si_fitted.npz")

# ---------------------------------------------------------------- fit
elements = [:Si]; order = 3; totaldegree = 10
@info "building ace1_model(elements=$elements, order=$order, totaldegree=$totaldegree)"
model = ace1_model(elements = elements, order = order, totaldegree = totaldegree)

@info "fitting on Si_tiny with acefit!"
data = ACEpotentials.example_dataset("Si_tiny").train
acefit!(data, model;
        energy_key = "dft_energy", force_key = "dft_force",
        virial_key = "dft_virial", solver = ACEfit.BLR(), verbose = false)
@info "fit done"

m = model.model; ps = model.ps; st = model.st
NZ = length(m.rbasis._i2z)
i2z = collect(m.rbasis._i2z)

# ---------------------------------------------------------------- radial spline
# SplineRnlrzzBasis evaluates  Rnl(r) = spline(x) * envelope(r, x),  x = T(r).
# The spline is Interpolations.cubic_spline_interpolation over a uniform grid on
# [-1,1]: 100 nodes, 102 B-spline coefficients (one pad each side).
function spline_arrays(basis)
    NZ = length(basis._i2z)
    s11 = basis.splines[1, 1]
    ncoef = length(s11.itp.itp.coefs)
    LEN = length(basis.spec)
    rng = s11.itp.ranges[1]
    coefs = zeros(NZ, NZ, ncoef, LEN)
    for iz = 1:NZ, jz = 1:NZ
        co = basis.splines[iz, jz].itp.itp.coefs
        for (p, c) in enumerate(co)          # p = 1..ncoef  <->  offset index 0..ncoef-1
            coefs[iz, jz, p, :] .= c
        end
        r = basis.splines[iz, jz].itp.ranges[1]
        @assert first(r) ≈ first(rng) && step(r) ≈ step(rng) && length(r) == length(rng)
    end
    return coefs, Float64(first(rng)), Float64(step(rng)), length(rng)
end

rnl_coefs, rnl_x0, rnl_h, rnl_n = spline_arrays(m.rbasis)
pair_coefs, pair_x0, pair_h, pair_n = spline_arrays(m.pairbasis)

# ---------------------------------------------------------------- transforms
# NormalizedTransform(GeneralizedAgnesiTransform):
#   y = r <= rin ? 1 : 1/(1 + a s^q/(1 + s^(q-p))),  s = (r-rin)/(r0-rin)
#   x = clamp(-1 + 2 (y - yin)/(ycut - yin), -1, 1)
function transform_params(basis)
    NZ = length(basis._i2z)
    P = zeros(NZ, NZ, 7)     # p q a rin r0 yin ycut
    for iz = 1:NZ, jz = 1:NZ
        t = basis.transforms[iz, jz]
        g = t.trans
        @assert g isa M.GeneralizedAgnesiTransform "unsupported transform $(typeof(g))"
        P[iz, jz, :] = [g.p, g.q, g.a, g.rin, g.r0, t.yin, t.ycut]
    end
    P
end

# envelopes
# many-body : PolyEnvelope2sX(x1,x2,p1,p2,s)  env = s (x-x1)^p1 (x2-x)^p2 on (x1,x2)
# pair      : ACE1_PolyEnvelope1sR(rcut,r0,p) env = s^-p - sc^-p + p sc^(-p-1)(s-sc)
function env2sx_params(basis)
    NZ = length(basis._i2z); P = zeros(NZ, NZ, 5)
    for iz = 1:NZ, jz = 1:NZ
        e = basis.envelopes[iz, jz]
        @assert e isa M.PolyEnvelope2sX "unsupported envelope $(typeof(e))"
        P[iz, jz, :] = [e.x1, e.x2, e.p1, e.p2, e.s]
    end
    P
end
function env1sr_params(basis)
    NZ = length(basis._i2z); P = zeros(NZ, NZ, 3)
    for iz = 1:NZ, jz = 1:NZ
        e = basis.envelopes[iz, jz]
        @assert e isa M.ACE1_PolyEnvelope1sR "unsupported pair envelope $(typeof(e))"
        P[iz, jz, :] = [e.rcut, e.r0, e.p]
    end
    P
end

# ---------------------------------------------------------------- tensor
aspec = m.tensor.abasis.spec                       # Vector{Tuple{Int,Int}} (Rnl idx, Ylm idx)
aspec_r = Int32[t[1] - 1 for t in aspec]           # 0-based for python
aspec_y = Int32[t[2] - 1 for t in aspec]
aa_specs = m.tensor.aabasis.specs                  # tuple of per-order specs
A2B = Matrix(m.tensor.A2Bmaps[1])

# rin0cuts (rcut per species pair)
rcuts = [m.rbasis.rin0cuts[iz, jz].rcut for iz = 1:NZ, jz = 1:NZ]
pair_rcuts = [m.pairbasis.rin0cuts[iz, jz].rcut for iz = 1:NZ, jz = 1:NZ]

# WB / Wpair, and E0
WB = Matrix(ps.WB)                                  # (n_B, NZ)
Wpair = Matrix(ps.Wpair)                            # (n_pair, NZ)
E0 = [Float64(m.Vref.E0[z]) for z in i2z]

# ---------------------------------------------------------------- probe values
# Per-stage reference values so a mismatch localises to a stage rather than
# only showing up at the end (this idea earned its keep in the Phase 0 spike).
Random.seed!(20260909)
n_probe = 24
probe_r = collect(range(0.9, maximum(rcuts) - 1e-6, length = n_probe))
probe_dirs = [normalize(SVector{3}(randn(3))) for _ in 1:n_probe]
probe_rij = reduce(hcat, [collect(probe_r[i] * probe_dirs[i]) for i in 1:n_probe])  # (3, n)
z = i2z[1]
probe_x   = [m.rbasis.transforms[1,1](probe_r[i]) for i in 1:n_probe]
probe_env = [M.evaluate(m.rbasis.envelopes[1,1], probe_r[i], probe_x[i]) for i in 1:n_probe]
probe_Rnl = Matrix(M.evaluate_batched(m.rbasis, probe_r, z, fill(z, n_probe), ps.rbasis, st.rbasis))
probe_Rpair = Matrix(M.evaluate_batched(m.pairbasis, probe_r, z, fill(z, n_probe), ps.pairbasis, st.pairbasis))
import Polynomials4ML as P4ML
probe_Ylm = Matrix(P4ML.evaluate(m.ybasis, [SVector{3}(probe_rij[:, i]) for i in 1:n_probe]))

# ---------------------------------------------------------------- test system
using AtomsBuilder
sys = AtomsBuilder.bulk(:Si, cubic = true) * 2
rattle!(sys, 0.15u"Å")
nat = length(sys)
test_pos = reduce(hcat, [ustrip.(u"Å", p) for p in AtomsBase.position(sys, :)])   # (3, nat)
test_cell = reduce(hcat, [ustrip.(u"Å", v) for v in AtomsBase.cell_vectors(sys)]) # (3, 3) columns
test_Z = Int32[AtomsBase.atomic_number(sys, i) for i in 1:nat]

# per-site energies + total, via the model's own site evaluation on a full nlist
import AtomsCalculatorsUtilities.SitePotentials as SP
calc = model
nlist = SP.PairList(sys, SP.cutoff_radius(calc))
site_E = zeros(nat)
edge_i = Int32[]; edge_j = Int32[]; edge_rij = Vector{Float64}[]
for i = 1:nat
    Js, Rs, Zs, z0 = SP.get_neighbours(sys, calc, nlist, i)
    site_E[i] = M.evaluate(m, Rs, Zs, z0, ps, st)
    for (jj, R) in zip(Js, Rs)
        push!(edge_i, Int32(i - 1)); push!(edge_j, Int32(jj - 1))
        push!(edge_rij, collect(ustrip.(R)))
    end
end
test_edge_rij = reduce(hcat, edge_rij)     # (3, n_edges)
efv = AtomsCalculators.energy_forces_virial(sys, calc)
test_E = ustrip(u"eV", efv.energy)
test_F = reduce(hcat, [ustrip.(u"eV/Å", f) for f in efv.forces])   # (3, nat)
# Julia convention (AtomsCalculatorsUtilities sitepotentials/assembly.jl:6):
#   site_virial(dV, Rs) = - sum(dv_i * r_i')   i.e.  V = -sum_e dE/dr_e (x) r_e
test_V = Matrix(ustrip.(u"eV", efv.virial))                       # (3,3)
test_pbc = Int32[Bool(b) for b in AtomsBase.periodicity(sys)]
@printf("test system: %d atoms, %d edges, E = %.10f eV\n", nat, length(edge_i), test_E)
@printf("  sum(site_E) = %.10f   |diff| = %.2e\n", sum(site_E), abs(sum(site_E) - test_E))
@printf("  virial trace = %.6f eV,  |V - V'| = %.2e (symmetry check)\n",
        test_V[1,1]+test_V[2,2]+test_V[3,3], maximum(abs.(test_V .- test_V')))

# ---------------------------------------------------------------- meta
meta = Dict(
  "schema_version" => 1,
  "source" => "ACEpotentials.jl ace1_model + acefit!(Si_tiny, BLR)",
  "acepotentials_version" => string(pkgversion(ACEpotentials)),
  "julia_version" => string(VERSION),
  "elements" => i2z,
  "order" => order, "totaldegree" => totaldegree,
  "radial_kind" => "spline",              # Stage 2 may emit "analytic"
  "transform_kind" => "agnesi_normalized",
  "envelope_kind" => "poly2sx",
  "pair_envelope_kind" => "ace1_poly1sr",
  "ybasis_kind" => (occursin("Solid", string(typeof(m.ybasis.scbasis))) ?
                    "real_solidharmonics" : "real_sphericalharmonics"),
  "lmax" => Int(isqrt(length(m.ybasis)) - 1),
  "n_rnl" => length(m.rbasis.spec), "n_pair" => length(m.pairbasis.spec),
  "n_ylm" => length(m.ybasis), "n_A" => length(aspec),
  "n_AA" => sum(length, aa_specs), "n_B" => size(A2B, 1),
  "aa_orders" => [length(s[1]) for s in aa_specs],
  "aa_lens" => [length(s) for s in aa_specs],
  "rnl_spline" => Dict("x0"=>rnl_x0, "h"=>rnl_h, "n"=>rnl_n, "ncoef"=>size(rnl_coefs,3)),
  "pair_spline" => Dict("x0"=>pair_x0, "h"=>pair_h, "n"=>pair_n, "ncoef"=>size(pair_coefs,3)),
  "rcut" => maximum(rcuts),
  "nnll" => [[ [b.n, b.l] for b in bb ] for bb in M.get_nnll_spec(m.tensor)],
)

D = Dict{String, Any}(
  "meta_json" => Vector{UInt8}(JSON.json(meta)),
  "rnl_spline_coefs" => rnl_coefs,          # (NZ,NZ,ncoef,n_rnl)
  "pair_spline_coefs" => pair_coefs,        # (NZ,NZ,ncoef,n_pair)
  "rnl_transform" => transform_params(m.rbasis),      # (NZ,NZ,7)
  "pair_transform" => transform_params(m.pairbasis),
  "rnl_envelope" => env2sx_params(m.rbasis),          # (NZ,NZ,5)
  "pair_envelope" => env1sr_params(m.pairbasis),      # (NZ,NZ,3)
  "rcuts" => rcuts, "pair_rcuts" => pair_rcuts,
  "aspec_r" => aspec_r, "aspec_y" => aspec_y,
  "A2B" => A2B,                                        # (n_B, n_AA) dense
  "WB" => WB, "Wpair" => Wpair, "E0" => E0,
  "elements" => Int32.(i2z),
  # probes
  "probe_r" => probe_r, "probe_rij" => probe_rij, "probe_x" => probe_x,
  "probe_env" => probe_env, "probe_Rnl" => probe_Rnl,
  "probe_Rpair" => probe_Rpair, "probe_Ylm" => probe_Ylm,
  # test system
  "test_pos" => test_pos, "test_cell" => test_cell, "test_Z" => test_Z,
  "test_V" => test_V, "test_pbc" => test_pbc,
  "test_edge_i" => edge_i, "test_edge_j" => edge_j, "test_edge_rij" => test_edge_rij,
  "test_site_E" => site_E, "test_E" => [test_E], "test_F" => test_F,
)
for (k, s) in enumerate(aa_specs)
    D["aa_spec_$(k)"] = Int32.(reduce(hcat, [collect(t) for t in s])' .- Int32(1))  # (n_v, order) 0-based
end
npzwrite(OUT, D)
@printf("wrote %s  (%.1f MB)\n", OUT, filesize(OUT)/2^20)
