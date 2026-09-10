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

## Static analysis of the `cudaErrorIllegalAddress` crash (no GPU available)

Done by reading `lammps-jax` `cpp/pair_jax_kokkos.cpp` while the Warwick network
was down. **The bug was not found**, but two hypotheses are refuted and one
premise we had both been working from is wrong. Recorded so the next attempt
starts here rather than repeating it.

**Wrong premise: "the plugin does not validate capacity at run time."** It does,
in three places:

- the edge-pack functor does `atomic_fetch_add`, then `if (edge + edges_to_add >
  max_edges) { edge_overflow() = 1; return; }` — it returns **before** writing,
  so an edge overflow cannot itself corrupt memory
- `pack_atoms` clamps with `span = std::min(nall, max_atoms)`, and the pack
  functor guards both `i >= max_atoms` and `j >= max_atoms`
- before every launch, an `MPI_Allreduce` on `nall` and `nlocal` raises
  `"LAMMPS-JAX atom capacity exceeded"` / `"owned-row capacity exceeded"`

So a genuine capacity overflow aborts cleanly with a clear message. It does not
present as an illegal address, and it is not what we are seeing.

**Refuted: stale edge indices after reneighbouring.** `rebuild_edges` is
hardcoded `true`, so the edge list is repacked every step, not cached across
rebuilds. Atom re-sorting or migration cannot leave stale indices behind.

**Refuted: the force scatter running past the model output.**
`add_model_forces` uses `limit = newton_pair ? nall : nlocal` against a
`max_atoms`-row view, which would be out of bounds if `nall > max_atoms` — but
the `MPI_Allreduce` check above fires first.

**Also checked and safe:** padded edges carry `senders = receivers = max_atoms`,
one past the end of a `(max_atoms, 3)` array, but our exporter masks the index
with `where(mask, idx, 0)` before gathering, per the nequip template.

**Where that leaves it.** The pack and scatter paths are bounded and guarded, so
suspicion moves to the exported program itself or to the PJRT/stream interaction
rather than the C++ packing. The decisive experiment is still the staged one:
does the abort track reneighbour *count* or step count
(`nevery`/`ncheck` in `bench/in.si_bench`, script at `/tmp/nbr_test.sh` on
moriarty)? Run that first; `compute-sanitizer` on a short run would localise the
access directly.


## Reneighbouring confirmed as the trigger (2026-09-10)

The staged experiment ran. 216 atoms, 300 steps, both plugin builds:

| plugin | `nevery` | `check` | result |
|---|---|---|---|
| old (`build-plugin-shared-cudart`) | 1 | yes | abort, `cudaErrorIllegalAddress` |
| old | 1 | no | abort |
| **old** | **1000000** | **no** | **rc=0, completed 300 steps** |
| new (upstream `main`, `a4304a2`) | 1 | yes | abort |
| new | 1 | no | abort |
| **new** | **1000000** | **no** | **rc=0, completed 300 steps** |

Symmetric across both builds. **With reneighbouring disabled the run completes;
with it enabled — forced or checked — it aborts.** The correlation with step
count is now a cause: the abort tracks neighbour-list rebuilds.

**Updating to latest upstream does not fix it.** Expected once the `fp64 support`
commit's C++ changes turned out to be confined to the comm path
(`model_comm.cpp/h`, the `PackComm*` functors), which a single-rank `n_hops = 1`
bundle never touches — but worth establishing rather than assuming.

### A third hypothesis weakened

A missing fence between the async pack and LAMMPS's neighbour rebuild looked
likely, but the packs and the model all run on the **same** stream
(`stream = exec.cuda_stream()`, and `pack_atoms` / `pack_edges` launch on `exec`),
so they are ordered with respect to each other. `exec.fence()` at
`pair_jax_kokkos.cpp:946` comes after execution, only to read the edge count.

That is three static hypotheses refuted (stale edges, capacity, missing fence).
Static reading has reached its limit here; `compute-sanitizer --tool memcheck`
is the next step, to name the faulting kernel rather than infer it.


## RESOLVED: the crash is the test potential, not any of our code (2026-09-10)

`compute-sanitizer` named the faulting kernel, and it is not ours:

```
Invalid __global__ atomic of size 4 bytes
  at NPairKokkosBinAtomsFunctor<Kokkos::Cuda>
  Access ... 93 bytes after the nearest allocation of size 176 bytes
  LAMMPS_NS::NBinKokkos<Kokkos::Cuda>::bin_atoms()
  LAMMPS_NS::NeighborKokkos::build_kokkos<Kokkos::Cuda>(int)
```

No `pair_jax_kokkos` frame. That is LAMMPS's own neighbour binning, which is why
it only fires on reneighbour steps and why our bundle capacities were irrelevant.

**But LAMMPS is not at fault either.** LAMMPS `develop` (2026-09-09) carries
`49dc8dc687` "KOKKOS: stop binning an atom that has left the bins", which adds

```cpp
if ((ibin < 0) || (ibin >= mbins))
  Kokkos::abort("Atom outside of neighbor bin range - simulation unstable");
```

That converts the cryptic illegal address into a clear message. It is a better
diagnostic, **not a cure** — the underlying condition is atoms leaving the box.
(We are on `patch_4Jul2026`, two months behind develop.)

### The potential has no repulsive core

Reproduced with **no LAMMPS at all**, in pure ASE NVE with the `acejax`
calculator, 216 atoms, 1 fs:

| step | energy drift (meV/atom) | min interatomic distance |
|---|---|---|
| 20 | 0.21 | 1.88 Å |
| 25 | 0.77 | 1.49 Å |
| 30 | 88.8 | 0.56 Å |
| 35 | 5.0e7 | — |

Atoms collapse into each other while the energy **falls**. The dimer curve shows
why:

| r (Å) | 3.00 | 2.35 | 2.00 | **1.50** | 1.00 | 0.70 | 0.50 |
|---|---|---|---|---|---|---|---|
| E (eV) | -52.25 | -43.42 | -30.33 | **-17.13** | -26.24 | -54.23 | -107.87 |

Repulsive down to ~1.5 Å, then it turns over and diverges attractively. This is
the textbook ACE extrapolation failure below the shortest distance present in
the training data: `Si_tiny` is a small near-equilibrium dataset and
`acefit!` defaults to `repulsion_restraint = false`.

**So the pipeline is vindicated end to end** — `acejax`, the bundle, the pair
style and LAMMPS all faithfully evaluated a potential that is simply not
MD-stable. It was fitted to pass a numerical-agreement gate, not to run dynamics.

### What this changes

- The **High risk** "jax/kk aborts beyond ~50 MD steps" is not a code defect.
- Energy conservation should be re-tested with a potential refitted using
  `acefit!(..., repulsion_restraint = true)`. That is a fitting option, not a
  code change.
- Worth updating LAMMPS to develop regardless: the clear abort message would
  have saved this entire investigation.

**Not verified:** whether the port matches Julia *below* ~2 Å specifically. All
prior agreement (1e-13 on energies, forces, virial, descriptors) was measured on
near-equilibrium structures with minimum separations around 2.2 Å. A Julia dimer
comparison was attempted and blocked by an unrelated expired TLS certificate
breaking Julia's HTTP precompilation on the dev machine. The turnover itself is
a property of the fitted basis, so this does not affect the conclusion, but the
short-range fidelity check remains open.
