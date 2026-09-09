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
using Lux
using LazyArtifacts, ExtXYZ, AtomsBase, Unitful
M = ACEpotentials.Models

# usage: export_model.jl [out.npz] [ace1|ace]
const OUT  = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "si_fitted.npz")
const KIND = length(ARGS) >= 2 ? ARGS[2] : "ace1"
@assert KIND in ("ace1", "ace") "model kind must be ace1 or ace"

# ---------------------------------------------------------------- fit
elements = [:Si]; order = 3; totaldegree = 10
if KIND == "ace1"
    @info "building ace1_model(elements=$elements, order=$order, totaldegree=$totaldegree)"
    model = ace1_model(elements = elements, order = order, totaldegree = totaldegree)
else
    # ace_model: LEARNABLE (analytic) rbasis, solid harmonics.  Its pair basis is
    # still splined (ace_heuristics.jl:213), so the branches are per-basis.
    @info "building ace_model(elements=$elements, order=$order, max_level=$totaldegree, Ytype=:solid)"
    rcut0 = 5.5
    ri = M._default_rin0cuts(tuple(elements...))
    ri = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut0)).(ri)
    raw = M.ace_model(; elements = tuple(elements...), order = order, Ytype = :solid,
                      level = M.TotalDegree(), max_level = totaldegree, maxl = 6,
                      pair_maxn = totaldegree, rin0cuts = ri,
                      init_WB = :glorot_normal, init_Wpair = :glorot_normal)
    ps0, st0 = Lux.setup(MersenneTwister(1234), raw)
    model = M.ACEPotential(raw, ps0, st0)
end

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

_is_spline(b) = b isa M.SplineRnlrzzBasis
rkind  = _is_spline(m.rbasis)    ? "spline" : "analytic"
pkind  = _is_spline(m.pairbasis) ? "spline" : "analytic"
@info "radial branches: rbasis=$rkind pairbasis=$pkind"

# `Wnlq` stays a live parameter for the analytic branch -- Stage 2 needs it
# trainable, and splines are not differentiable w.r.t. what generated them.
function analytic_arrays(basis, ps_b)
    NZ = length(basis._i2z)
    W = zeros(NZ, NZ, length(basis.spec), length(basis.polys))
    for iz = 1:NZ, jz = 1:NZ
        W[iz, jz, :, :] = ps_b.Wnlq[:, :, iz, jz]
    end
    rs = basis.polys.refstate
    return W, collect(rs.A), collect(rs.B), collect(rs.C)
end

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
# ace1_model's pair basis uses ACE1_PolyEnvelope1sR (rcut, r0, p);
# ace_model's uses PolyEnvelope1sR (rcut, p) -- a different formula, not just
# different parameters.
function env1sr_params(basis)
    NZ = length(basis._i2z)
    e1 = basis.envelopes[1, 1]
    if e1 isa M.ACE1_PolyEnvelope1sR
        P = zeros(NZ, NZ, 3)
        for iz = 1:NZ, jz = 1:NZ
            e = basis.envelopes[iz, jz]
            P[iz, jz, :] = [e.rcut, e.r0, e.p]
        end
        return P, "ace1_poly1sr"
    elseif e1 isa M.PolyEnvelope1sR
        P = zeros(NZ, NZ, 2)
        for iz = 1:NZ, jz = 1:NZ
            e = basis.envelopes[iz, jz]
            P[iz, jz, :] = [e.rcut, e.p]
        end
        return P, "poly1sr"
    end
    error("unsupported pair envelope $(typeof(e1))")
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
# ace_model may carry no Vref; treat that as zero one-body energies
E0 = (m.Vref === nothing) ? zeros(length(i2z)) :
     [Float64(get(m.Vref.E0, z, 0.0)) for z in i2z]

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

# ---------------------------------------------------------------- branch data
rnl_coefs = rnl_W = rnl_pA = rnl_pB = rnl_pC = nothing
rnl_spl_meta = nothing
if rkind == "spline"
    rnl_coefs, rx0, rh, rn = spline_arrays(m.rbasis)
    rnl_spl_meta = Dict("x0"=>rx0, "h"=>rh, "n"=>rn, "ncoef"=>size(rnl_coefs,3))
else
    rnl_W, rnl_pA, rnl_pB, rnl_pC = analytic_arrays(m.rbasis, ps.rbasis)
end
pair_coefs = pair_W = pair_pA = pair_pB = pair_pC = nothing
pair_spl_meta = nothing
if pkind == "spline"
    pair_coefs, px0, ph, pn = spline_arrays(m.pairbasis)
    pair_spl_meta = Dict("x0"=>px0, "h"=>ph, "n"=>pn, "ncoef"=>size(pair_coefs,3))
else
    pair_W, pair_pA, pair_pB, pair_pC = analytic_arrays(m.pairbasis, ps.pairbasis)
end
pair_env, pair_env_kind = env1sr_params(m.pairbasis)

# ---------------------------------------------------------------- meta
meta = Dict(
  "schema_version" => 1,
  "source" => "ACEpotentials.jl ace1_model + acefit!(Si_tiny, BLR)",
  "acepotentials_version" => string(pkgversion(ACEpotentials)),
  "julia_version" => string(VERSION),
  "elements" => i2z,
  "order" => order, "totaldegree" => totaldegree,
  "radial_kind" => rkind, "pair_radial_kind" => pkind,
  "transform_kind" => "agnesi_normalized",
  "envelope_kind" => "poly2sx",
  "pair_envelope_kind" => pair_env_kind,
  "ybasis_kind" => (occursin("Solid", string(typeof(m.ybasis.scbasis))) ?
                    "real_solidharmonics" : "real_sphericalharmonics"),
  "lmax" => Int(isqrt(length(m.ybasis)) - 1),
  "n_rnl" => length(m.rbasis.spec), "n_pair" => length(m.pairbasis.spec),
  "n_ylm" => length(m.ybasis), "n_A" => length(aspec),
  "n_AA" => sum(length, aa_specs), "n_B" => size(A2B, 1),
  "aa_orders" => [length(s[1]) for s in aa_specs],
  "aa_lens" => [length(s) for s in aa_specs],
  "rnl_spline" => rnl_spl_meta, "pair_spline" => pair_spl_meta,
  "rcut" => maximum(rcuts),
  "nnll" => [[ [b.n, b.l] for b in bb ] for bb in M.get_nnll_spec(m.tensor)],
)

D = Dict{String, Any}(
  "meta_json" => Vector{UInt8}(JSON.json(meta)),
  "rnl_transform" => transform_params(m.rbasis),      # (NZ,NZ,7)
  "pair_transform" => transform_params(m.pairbasis),
  "rnl_envelope" => env2sx_params(m.rbasis),          # (NZ,NZ,5)
  "pair_envelope" => pair_env,
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
# only the populated radial branch is written; the loader defaults the other
if rkind == "spline"
    D["rnl_spline_coefs"] = rnl_coefs
else
    D["rnl_Wnlq"] = rnl_W; D["polys_A"] = rnl_pA; D["polys_B"] = rnl_pB; D["polys_C"] = rnl_pC
end
if pkind == "spline"
    D["pair_spline_coefs"] = pair_coefs
else
    D["pair_Wnlq"] = pair_W
    D["pair_polys_A"] = pair_pA; D["pair_polys_B"] = pair_pB; D["pair_polys_C"] = pair_pC
end

for (k, s) in enumerate(aa_specs)
    D["aa_spec_$(k)"] = Int32.(reduce(hcat, [collect(t) for t in s])' .- Int32(1))  # (n_v, order) 0-based
end
npzwrite(OUT, D)
@printf("wrote %s  (%.1f MB)\n", OUT, filesize(OUT)/2^20)
