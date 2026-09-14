# spike_distmd — THROWAWAY

Half-day spike for Phase 11 of `docs/plans/jax_ace_port_plan.md`: can `acejax`
run distributed MD through `lammps_jax/dist/parallel/`, and what does it cost
relative to the LAMMPS route?

**This is throwaway code.** It is an existence proof, not a product. It is
untested, unpackaged, and not intended to be maintained. Nothing in
`acejax/acejax/` imports it and nothing should.

## Files

- `ace_dist.py` — the whole adapter is `ace_node_energies` (ACE in the shape
  `feature_exchange.node_energies` expects) plus a 12-line reimplementation of
  `feature_exchange.ghost_energy`. The rest is a driver that runs one
  distributed force evaluation and checks it against `acejax.ACECalculator`.
- `ace_md.py` — NVE velocity Verlet, `--inner` steps inside one jitted
  `shard_map` call, rebuild between legs. Validates the final configuration
  against the ASE calculator.
- `bench_dist.py` — same-process cost of the distributed step against a raw
  `energy_forces_virial` call on the exact edge list. `--cap-owned` /
  `--cap-sub` vary the static capacities, which is how the padding was
  attributed (by difference over whole computations, per `bench/results.md`).

## Environment

`lammps-jax` core only — the `[dist]` extra (`jax-md`, `e3nn-jax`, `nequix`) is
**not** required, because `ghost_exchange` and `local_neighbor_list` import
nothing but jax and numpy.

```
uv venv --python 3.12 spikeenv
uv pip install -p spikeenv/bin/python -e path/to/acejax 'ase>=3.22'
uv pip install -p spikeenv/bin/python -e path/to/lammps-jax
```

Run from this directory with `PYTHONPATH=.`, e.g.

```
PYTHONPATH=. python ace_dist.py --model ../fixtures/si_fitted.npz --rep 4 --ranks 1 2 4 8
PYTHONPATH=. python ace_md.py   --model ../fixtures/si_fitted.npz --rep 4 --ranks 4 --dt 0.25
PYTHONPATH=. python bench_dist.py --model ../fixtures/si_s69.npz --reps 4 --a2b-sparse
```

Multi-rank on a CPU-only host uses `--xla_force_host_platform_device_count`,
set by default in these scripts. That is a correctness harness, **not** a
scaling measurement: the "ranks" are threads on one CPU.
