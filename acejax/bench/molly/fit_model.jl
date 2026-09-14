# Fit the SAME model that produced acejax/fixtures/si_fitted.npz and save it as
# JSON so the Molly benchmark loads exactly one model definition.
#
# The fit here is character-for-character the fit in acejax/julia/export_model.jl
# with its default environment (ACE_ELEMENTS=Si, ACE_ORDER=3,
# ACE_TOTALDEGREE=10, no ACE_RCUT, ACE_NOFIT unset):
#
#     model = ace1_model(elements = [:Si], order = 3, totaldegree = 10)
#     acefit!(Si_tiny.train, model; energy_key="dft_energy",
#             force_key="dft_force", virial_key="dft_virial",
#             solver = ACEfit.BLR())
#
# Equality with the npz is not assumed: check_model_match.jl re-runs the
# exporter and diffs every array against the shipped fixture.
#
#   julia --project=. fit_model.jl out.json

using ACEpotentials, ACEfit, LazyArtifacts

const OUT = length(ARGS) >= 1 ? ARGS[1] : "si_model.json"

model = ace1_model(elements = [:Si], order = 3, totaldegree = 10)
@info "model built" length_basis = ACEpotentials.length_basis(model)

data = ACEpotentials.example_dataset("Si_tiny").train
acefit!(data, model;
        energy_key = "dft_energy", force_key = "dft_force",
        virial_key = "dft_virial", solver = ACEfit.BLR(), verbose = false)
@info "fit done"

ACEpotentials.save_model(model, OUT; save_project = false)
@info "saved" OUT
