# Phase 6: the Si bundle runs under `pair_style jax/kk`

## Gate: passed

Fitted `ace1_model` (Si, order 3, totaldegree 10; `acefit!` on Si_tiny, BLR),
exported to a lammps-jax bundle and run on **moriarty** under
`pair_style jax/kk`. Periodic diamond Si, 216 atoms, 16.29 A cell, f64.

| comparison | energy | forces |
|---|---|---|
| 1 rank vs acejax Python | **0.000e+00** | 1.368e-13 eV/A |
| 2 ranks vs acejax Python | **0.000e+00** | 1.483e-13 eV/A |
| 1 rank vs 2 ranks (dumps) | positions 0.000e+00 | 9.193e-14 eV/A |

`|F|` scale 1.21 eV/A, so forces agree to ~1.2e-13 relative. LAMMPS PotEng is
-35238.76853368402 eV on both rank counts, bit-identical.

**Ghost bookkeeping is covered.** The cell is periodic and the runs carry 1314
ghosts on 1 rank and 1024/1025 on 2 -- decomposition-dependent, which is
precisely what the earlier non-periodic ABI check (`nghost = 0`) could not
reach.

No contract field was rejected. The bundle was exported with **jax 0.11.1**,
matching the PJRT plugin the venv ships; Part A's removal of the
`sphericart-jax` 0.10.1 pin is what made that possible.

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
(`acejax/harmonics.py`), so nothing needs registering via
`LAMMPS_JAX_FFI_HANDLERS` and `contrib/ffi-replay` is not involved. Actual
usage: 216 local + 1314 ghost = 1530 atoms against a 2560 capacity, and 15466
LAMMPS neighbours (rcut + 1.0 skin) filtered by the pair style to the model's
9880 edges against a 163840 capacity.

## Two environment gotchas, both host-side

**1. Wrong host is fatal and silent.** On lestrade every LAMMPS invocation --
including `lmp -h` with no deck -- dies with SIGILL in
`__static_initialization_and_destruction_0` of `liblammps.so.0`, at
`vmovdqu8 %ymm0`. That is AVX-512; the library holds 1796 `vmovdqu8`, 169
`vpermt2`, 38 `vmovdqu16`, 25 `vpternlog`. lestrade is an i9-14900K with no
`avx512*` flags. The build is `build-SKX-AMPERE86` -- Skylake-X -- and moriarty
is a Xeon Silver 4216 (AVX-512) with an RTX A4500 (compute 8.6). Nothing to fix
in the build; just use the right host.

**2. `gpu/aware off` is required for multi-rank.** With GPU-aware MPI the
2-rank run aborts inside `CommKokkos::borders_device<Kokkos::Cuda>` with
`Assertion failure at prov/psm3/psm3/ptl_am/ptl.c:196`. The stock
`examples/lj.lammps-jax.json` fails the same way, so this is the host's
MPI/CUDA interaction rather than anything in our bundle.

## One deck gotcha, ours

`displace_atoms ... random` is domain-decomposition dependent: generating the
geometry inside each run gave the 1-rank and 2-rank runs *different systems*
(max position difference 8.14 A under minimum image -- not a wrapping
artefact). Each run was internally consistent and matched the Python calculator
on its own geometry, but the rank comparison was meaningless. `in.si_setup`
now builds the configuration once into `si.data`, which both runs `read_data`.

The reference `examples/in.mlip_al` displaces inside the run the same way, so
any rank comparison built on it has the same flaw.

## Files

| file | purpose |
|---|---|
| `export_bundle.py` | fitted model -> lammps-jax bundle |
| `in.si_setup` | build the rattled geometry once into `si.data` |
| `in.mlip_si` | the deck; reads `si.data` |
| `test_si_bundle.sh` | the gate: 1 rank, 2 ranks, dump comparison, vs Python |
| `check_vs_python.py` | rebuilds the config from the dump, evaluates acejax |
| `cmpdump.py` | dump comparison (rewrite; the original scratchpad was cleared) |
| `validate_abi.py` | ABI wrappers without LAMMPS, kept as a fast pre-check |

## Reproducing

```bash
# on moriarty (NOT lestrade -- see the AVX-512 note above)
cd stage1/lammps
python export_bundle.py --npz ../si_fitted.npz --out si_ace.lammps-jax.json
./test_si_bundle.sh                      # PYTHON=... to pick the interpreter
```

`run_artifacts/` holds the evidence for the numbers in this file: both logs,
both dumps, the `si.data` geometry, and the 1.1 MB bundle itself.
