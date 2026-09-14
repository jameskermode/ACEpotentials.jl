# `acejax/bench/molly` — JAX MD vs ACEpotentials.jl + Molly.jl, CPU

The first comparison in this project between the JAX port and the Julia
original **as MD engines** rather than as descriptor evaluators.  Results and
method are in `RESULTS.md`.

## Files

| file | what it is |
|---|---|
| `dump_initial_state.py` | writes the exact initial state `spike_distmd/ace_md.py` builds (same RNG stream), so both engines start from identical positions *and* velocities |
| `fit_model.jl` | reproduces the fit behind `fixtures/si_fitted.npz` and saves it as ACEpotentials JSON |
| `check_model_match.py` | re-runs `acejax/julia/export_model.jl` and diffs every array against the shipped fixture |
| `md_molly.jl` | NVE velocity Verlet through Molly, three integration modes (`naive`, `cached`, `bare`) |
| `run_julia.sh`, `run_jax.sh`, `run_master.sh` | the two-pass experiment, Linux host |
| `run_tuned.sh`, `run_after.sh` | the 0.5 A-skin configuration + skin sweeps |
| `run_mac.sh`, `run_mac_tuned.sh` | the same experiment on macOS (no `taskset` there) |
| `try_backends.jl` | probes `convert2et_full` / `fast_evaluator` for this model |
| `make_plot.py` | the two-host figure |
| `collect.py` | parses the logs into the tables in `RESULTS.md` |

Nothing here modifies `acejax/acejax/` or `acejax/spike_distmd/`; `ace_md.py`
is run verbatim.

## Reproducing

On a host with the Julia project (`ACEpotentials` dev checkout + `Molly`) and
the `acejax` venv:

```
python dump_initial_state.py --rep 3 --out state_rep3.npz    # 216 atoms
julia --project=. fit_model.jl si_model.json
bash run_master.sh                                            # ~2 h, two passes
bash run_tuned.sh t1 && bash run_tuned.sh t2
python collect.py results/                                    # writes collected.json
python make_plot.py xeon/collected.json mac/collected.json molly_vs_jax.png
```

On macOS use `run_mac.sh` / `run_mac_tuned.sh` instead; they take the scratch
directory as an argument and set thread counts at the engine, since macOS has
no `taskset`.
