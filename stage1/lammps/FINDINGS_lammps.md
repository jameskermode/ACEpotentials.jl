# Phase 6 (LAMMPS export): blocked on the host, not on the bundle

## Status

The bundle exports cleanly and validates through lammps-jax's own ABI wrappers.
**LAMMPS itself cannot start on lestrade**, so the pair-style gate is not met.
This is a host/build mismatch and is independent of anything in the bundle.

## The blocker

`lmp -h`, with no input deck and no bundle, dies with SIGILL (exit 132) before
printing anything:

```
Program received signal SIGILL, Illegal instruction.
#0  __static_initialization_and_destruction_0() from .../lib/liblammps.so.0
=> 0x...: vmovdqu8 %ymm0,0x10(%rsp)
```

`vmovdqu8` is AVX-512 (AVX512BW/VL). `liblammps.so.0` contains 1796 `vmovdqu8`,
169 `vpermt2`, 38 `vmovdqu16`, 25 `vpternlog`, 1 `kmovd`.

lestrade is an **Intel i9-14900K** (Raptor Lake): `/proc/cpuinfo` reports **no
`avx512*` flags at all** -- Intel fused AVX-512 off in that generation. The
build directory is named `build-SKX-AMPERE86`; SKX is Skylake-X, which has
AVX-512. So the binary targets a CPU this host does not have.

Confirmations that it is not the bundle, the deck, or Kokkos:

| test | result |
|---|---|
| our Si bundle | SIGILL |
| the existing, known-good `examples/lj.lammps-jax.json` | SIGILL |
| `lmp -h`, no deck at all | SIGILL |
| with and without `-k on g 1 -sf kk` | SIGILL |
| via `$V/bin/lmp` wrapper and via `lmp.real` directly | SIGILL |
| pinned to cores 0, 2, 8, 16, 24 (`taskset`) | SIGILL on all |

The per-core test rules out the hybrid P-core/E-core explanation: this CPU has
no AVX-512 on any core.

**No contract field was rejected.** The bundle was never read, because the
binary never reached `pair_coeff`.

## What was validated instead

`validate_abi.py` drives the exported `energy_fn` through lammps-jax's real
`wrap_energy_fn` wrappers with LAMMPS-shaped inputs -- fixed capacities, padded
edge list, out-of-range padded indices, edge mask, forces by autodiff -- and
compares against the model core called directly:

```
cluster: 64 atoms, non-periodic, 1382 edges (capacity 163840)
  core E = -8233.6166404368 eV
  ABI  E = -8233.6166404368 eV   |dE| = 1.819e-12  (2.21e-16 rel)
  max|dF| on real atoms = 2.096e-13 eV/A  (|F| scale 49.6775)
  max|F| on padded rows = 0.000e+00 eV/A
  non-finite in F: 0
```

So the ABI plumbing, padding conventions and autodiff forces are correct. What
remains uncovered is the VHLO round-trip through PJRT, the C++ pair style
binding, and ghost bookkeeping (the cluster is non-periodic, so nghost = 0).

## Exported contract

```
format               lammps-jax-json
programs             energy_mlir_b64, force_mlir_b64, energy_and_forces_mlir_b64
cutoff               6.0            unit_style   metal
precision            float64        newton       on
force_output         atom-force     n_hops       1
max_atoms            2560           max_edges    163840
n_species            1              uses_box     False
input_layout         sparse-edge    edge_pairing full
custom_call_targets  []             comm_widths  []
```

`custom_call_targets` is empty because the harmonics are a pure-JAX recursion
(see `acejax/harmonics.py`): no FFI handler needs registering via
`LAMMPS_JAX_FFI_HANDLERS`, and `contrib/ffi-replay` is not needed.

## To finish this

Either run on a host whose CPU matches `build-SKX-AMPERE86`, or rebuild LAMMPS
for the run host. Then `test_si_bundle.sh` should run as-is; it checks 1 rank
against 2 ranks and against the Python calculator, in the shape of
`test_eam_bundle.sh`.

`cmpdump.py` is a rewrite -- the original lived in a `/tmp` scratchpad that has
since been cleared.
