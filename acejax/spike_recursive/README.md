# THROWAWAY SPIKE — recursive/DAG AA products

**This directory is a one-day spike, not product code.** Nothing here is
imported by `acejax/`, nothing here is covered by the test suite, and none of it
should be shipped. It exists to answer one question and to leave the evidence
behind:

> Does a recursive/DAG formulation of the AA products give a worthwhile speedup
> in JAX on GPU?

- `dag.py` — DAG construction from an exported AA spec: the Julia greedy from
  `EquivariantTensors/src/ace/symmprod_dag.jl`, plus a depth-minimising
  `balanced` variant and a `chain` variant used as an exact correctness gate.
- `dagjax.py` — three JAX evaluators for the levelled DAG (`dus`, `concat`,
  `cvjp`); `cvjp` adds a hand-written reverse-level backward.
- `verify_dag.py` — correctness gate. CPU only, no GPU needed.
- `bench_dag.py` — end-to-end timing, `site_basis` and `energy+forces`.
- `run_gpu.sh` — the GPU sweep, recording GPU occupancy around every run.

Findings are in the spike report, not here.

## Evidence in this directory

- `results_verify.txt` — correctness gate output (CPU, all fixtures, all modes)
- `results_gpu_pass1.txt`, `results_gpu_pass2.txt` — the two GPU passes
- `results_compare.txt` — pass-to-pass agreement
- `results_cpu.txt` — CPU directional run

The GPU passes were taken with the device verified empty beforehand and a
sampler alongside; see the `###` lines.
