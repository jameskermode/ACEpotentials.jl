# Fit an ACEpotentials v0.6.12 model on the SAME data as our v0.10 model and
# export it to .yace for `pair_style pace`.
#
# v0.10's export2lammps is dead (src/outdated/export.jl is built on ACE1.jl +
# JuLIP, neither of which is a v0.10 dependency), so the fair comparator is a
# v0.6 fit of the same model space on the same data.
#
#   julia +1.11 --project=. fit_v06.jl
using ACEpotentials, JuLIP, Printf

elements = [:Si]; order = 3; totaldegree = 10; rcut = 6.0
@info "acemodel(elements=$elements, order=$order, totaldegree=$totaldegree, rcut=$rcut)"
# Eref is required: export2lammps wants a 3-component SumIP
#   (PolyPairPot, PIPotential, OneBody); without it there are only 2.
#   E0 = 0 matches our v0.10 model, whose Vref.E0[Si] is also 0.
model = acemodel(elements = elements, order = order, totaldegree = totaldegree,
                 rcut = rcut, Eref = [:Si => 0.0])
@printf("v0.6.12 basis length = %d\n", length(model.basis))

data = ACEpotentials.example_dataset("Si_tiny").train
@info "fitting $(length(data)) configs"
acefit!(model, data;
        energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial",
        solver = ACEfit.BLR())

out = joinpath(@__DIR__, "si_v06.yace")
export2lammps(out, model.potential)
@printf("wrote %s (%.2f MB)\n", out, filesize(out)/2^20)

# sanity: energy of a small bulk cell, so we can tell later whether pace agrees
at = bulk(:Si, cubic=true) * 2
rattle!(at, 0.05)
@printf("reference: %d atoms, E = %.10f eV\n", length(at), energy(model.potential, at))
JuLIP.write_extxyz(joinpath(@__DIR__, "si_v06_check.xyz"), at)
