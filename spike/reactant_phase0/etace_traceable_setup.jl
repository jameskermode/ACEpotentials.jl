# THE question the earlier spike should have asked: is the STANDARD ETACE code
# path traceable by Reactant? The ETACE rewrite was designed with that in mind;
# lammps-jax's ace_export.jl hand-writes arrays instead, but that is its choice,
# not a demonstration that hand-writing is required.
#
# Escalating attempts, each reporting exactly where it fails. SPIKE CODE.

using ACEpotentials, StaticArrays, Lux, LuxCore, AtomsBuilder, AtomsBase,
      Unitful, Random, LinearAlgebra, Printf
import EquivariantTensors as ET
import DecoratedParticles as DP
using Reactant
Reactant.set_default_backend("cpu")
M = ACEpotentials.Models; ETM = ACEpotentials.ETModels
println("Reactant ", pkgversion(Reactant), "  ACEpotentials ", pkgversion(ACEpotentials))

rng = MersenneTwister(1234); Random.seed!(1234)
elements = (:Si,); rcut = 5.5
rin0cuts = M._default_rin0cuts(elements)
rin0cuts = (x -> (rin=x.rin, r0=x.r0, rcut=rcut)).(rin0cuts)
model = M.ace_model(; elements=elements, order=3, Ytype=:solid,
                    level=M.TotalDegree(), max_level=10, maxl=6, pair_maxn=10,
                    rin0cuts=rin0cuts, init_WB=:glorot_normal,
                    init_Wpair=:glorot_normal)
ps, st = Lux.setup(rng, model)
et_model = ETM.convert2et(model)
et_ps, et_st = LuxCore.setup(rng, et_model)
et_ps.rembed.post.W[:, :, 1] = ps.rbasis.Wnlq[:, :, 1, 1]
et_ps.readout.W[1, :, 1] .= ps.WB[:, 1]

sys = AtomsBuilder.bulk(:Si, cubic=true) * 2
rattle!(sys, 0.1u"Å")
G = ET.Atoms.interaction_graph(sys, rcut * u"Å")
println("system: $(length(sys)) atoms, $(ET.nedges(G)) edges, maxneigs=$(G.maxneigs)")

B_ref = ETM.site_basis(et_model, G, et_ps, et_st)
