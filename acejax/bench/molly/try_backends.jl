# Does ACEpotentials offer a faster evaluation path for THIS model than the
# classic one the benchmark uses?  Two candidates, both tried so that the Julia
# side is not reported at less than its best:
#
#   ETModels.convert2et_full  -- the EquivariantTensors-backed StackedCalculator
#   Models.fast_evaluator     -- the experimental fused evaluator
#
# Prints whether each constructs at all for an `ace1_model`, and if it does,
# whether energies match the classic calculator.  Timing, if any, is left to
# md_molly.jl so that it is a whole-run measurement.
#
#   julia --project=. try_backends.jl work/si_model.json work/state_rep3.npz

using ACEpotentials, AtomsCalculators, Unitful, StaticArrays, NPZ, Molly,
      LinearAlgebra, Printf
const ETM = ACEpotentials.ETModels
const M   = ACEpotentials.Models

model, _ = ACEpotentials.load_model(ARGS[1])
S = npzread(ARGS[2])
pos = S["positions"]; cellm = S["cell"]; n = size(pos, 1); L = cellm[1,1]
sys = Molly.System(
    atoms = [Molly.Atom(index=i, mass=S["mass_amu"]*u"u", charge=0.0,
                        σ=0.0u"Å", ϵ=0.0u"eV") for i in 1:n],
    atoms_data = [Molly.AtomData(element="Si") for i in 1:n],
    coords = [SVector{3,Float64}(pos[i,1],pos[i,2],pos[i,3])*u"Å" for i in 1:n],
    velocities = [zeros(SVector{3,Float64})*u"Å/fs" for i in 1:n],
    boundary = Molly.CubicBoundary(L*u"Å"), general_inters = (model,),
    neighbor_finder = Molly.NoNeighborFinder(),
    energy_units = u"eV", force_units = u"eV/Å")

E_ref = AtomsCalculators.potential_energy(sys, model)
println("classic ACEPotential   E = $E_ref")

for (name, build) in (("convert2et_full", () -> ETM.convert2et_full(model.model, model.ps, model.st)),
                      ("fast_evaluator",  () -> M.fast_evaluator(model)))
    try
        calc = build()
        E = AtomsCalculators.potential_energy(sys, calc)
        println(@sprintf("%-18s constructed;  E = %s   |dE| = %.3e",
                         name, E, abs(ustrip(u"eV", E - E_ref))))
    catch err
        msg = sprint(showerror, err)
        println("$name  NOT AVAILABLE for this model: ",
                first(split(msg, '\n')))
    end
end
