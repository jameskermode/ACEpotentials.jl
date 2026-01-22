#!/usr/bin/env python3
"""
Benchmark LAMMPS IREE Kokkos Plugin with Bucket VMFBs

Tests the pair_iree_kokkos LAMMPS plugin with CUDA backend
across different system sizes using bucket VMFBs.

Bucket sizes:
  - 2000 edges (~77 atoms)
  - 10000 edges (~385 atoms)
  - 50000 edges (~1923 atoms)
  - 100000 edges (~3846 atoms)
"""

import subprocess
import os
import re
import json
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
LAMMPS_EXE = Path.home() / "lammps/lammps-22Jul2025/build/lmp"
BUCKET_DIR = SCRIPT_DIR.parent.parent / "benchmark/bucket_energy_gradient"

# Bucket configurations: (max_edges, max_atoms, vmfb_path)
BUCKETS = [
    (2000, 77, "bucket_2000"),
    (10000, 385, "bucket_10000"),
    (50000, 1923, "bucket_50000"),
    (100000, 3846, "bucket_100000"),
]

def select_bucket(n_atoms, rcut=5.5):
    """Select appropriate bucket based on atom count.

    Estimates edges as ~26 * n_atoms (based on Si diamond lattice).
    Returns (max_edges, bucket_dir) or (None, None) if too large.
    """
    # Estimate edges: Si diamond has ~26 neighbors per atom within 5.5 A
    estimated_edges = int(n_atoms * 26)

    for max_edges, max_atoms, bucket_name in BUCKETS:
        if estimated_edges <= max_edges:
            bucket_path = BUCKET_DIR / bucket_name
            return max_edges, bucket_path

    return None, None


def generate_si_diamond(nx, ny, nz, a=5.43, displacement=0.0):
    """Generate Si diamond lattice with optional random displacement."""
    basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]

    positions = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for b in basis:
                    x = (ix + b[0]) * a
                    y = (iy + b[1]) * a
                    z = (iz + b[2]) * a
                    if displacement > 0:
                        x += np.random.randn() * displacement
                        y += np.random.randn() * displacement
                        z += np.random.randn() * displacement
                    positions.append([x, y, z])

    box = [nx * a, ny * a, nz * a]
    return np.array(positions), box


def write_lammps_data(positions, box, filename):
    """Write LAMMPS data file."""
    n_atoms = len(positions)
    with open(filename, 'w') as f:
        f.write("Si benchmark system\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write("1 atom types\n\n")
        f.write(f"0.0 {box[0]} xlo xhi\n")
        f.write(f"0.0 {box[1]} ylo yhi\n")
        f.write(f"0.0 {box[2]} zlo zhi\n\n")
        f.write("Masses\n\n")
        f.write("1 28.0855\n\n")
        f.write("Atoms # atomic\n\n")
        for i, pos in enumerate(positions):
            # Wrap positions into box
            x = pos[0] % box[0]
            y = pos[1] % box[1]
            z = pos[2] % box[2]
            f.write(f"{i+1} 1 {x} {y} {z}\n")


def write_lammps_input(backend, vmfb_path, data_file, n_steps, output_file):
    """Write LAMMPS input script for benchmarking."""
    plugin_path = SCRIPT_DIR / "libpair_iree_kokkos.so"

    input_content = f"""# LAMMPS Benchmark: IREE {backend.upper()} Backend
# ============================================

plugin load {plugin_path}

package kokkos neigh half newton on

units metal
atom_style atomic
boundary p p p
newton on

read_data {data_file}

pair_style iree/kk {vmfb_path} {backend}
pair_coeff * * Si

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

# Warmup
run 10

# Reset timer
reset_timestep 0
timer timeout off

# Timed run
run {n_steps}

# Print final energy
variable pe equal pe
print "BENCHMARK_ENERGY: ${{pe}}"
print "BENCHMARK_COMPLETE"
"""
    with open(output_file, 'w') as f:
        f.write(input_content)


def run_lammps_benchmark(backend, n_atoms, n_steps=100):
    """Run LAMMPS benchmark and return timing."""
    # Determine system size (cubic root approximation)
    atoms_per_cell = 8
    n_cells = max(1, int(np.ceil((n_atoms / atoms_per_cell) ** (1/3))))
    actual_atoms = n_cells ** 3 * atoms_per_cell

    # Select bucket
    max_edges, bucket_path = select_bucket(actual_atoms)
    if bucket_path is None:
        return {
            "backend": backend,
            "atoms": actual_atoms,
            "success": False,
            "error": f"No bucket large enough for {actual_atoms} atoms"
        }

    vmfb_path = bucket_path / f"energy_gradient_f64_{backend}.vmfb"
    if not vmfb_path.exists():
        return {
            "backend": backend,
            "atoms": actual_atoms,
            "success": False,
            "error": f"VMFB not found: {vmfb_path}"
        }

    # Generate data
    np.random.seed(42)  # Reproducible
    positions, box = generate_si_diamond(n_cells, n_cells, n_cells, displacement=0.05)

    # Write files
    data_file = SCRIPT_DIR / f"bench_{backend}_{actual_atoms}.data"
    input_file = SCRIPT_DIR / f"bench_{backend}_{actual_atoms}.in"

    write_lammps_data(positions, box, data_file)
    write_lammps_input(backend, vmfb_path, data_file, n_steps, input_file)

    # Build LAMMPS command
    if backend == "cuda":
        cmd = f"{LAMMPS_EXE} -k on g 1 -sf kk -in {input_file}"
    else:
        cmd = f"{LAMMPS_EXE} -k on t 4 -sf kk -in {input_file}"

    # Run with module loads
    full_cmd = f"module load GCC/13.3.0 OpenMPI/5.0.3 Python/3.12.3 CUDA/12.5.0 2>/dev/null; {cmd}"

    try:
        start = time.time()
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=SCRIPT_DIR
        )
        wall_time = time.time() - start

        output = result.stdout + result.stderr

        # Parse timing from LAMMPS output
        loop_match = re.search(r'Loop time of ([\d.]+) on', output)
        loop_time = float(loop_match.group(1)) if loop_match else None

        # Parse pair timing
        pair_match = re.search(r'Pair\s+\|\s+([\d.e+-]+)', output)
        pair_time = float(pair_match.group(1)) if pair_match else None

        # Parse energy
        energy_match = re.search(r'BENCHMARK_ENERGY:\s*([-\d.e+]+)', output)
        energy = float(energy_match.group(1)) if energy_match else None

        # Cleanup temp files
        data_file.unlink(missing_ok=True)
        input_file.unlink(missing_ok=True)

        return {
            "backend": backend,
            "atoms": actual_atoms,
            "bucket_edges": max_edges,
            "steps": n_steps,
            "loop_time_s": loop_time,
            "pair_time_s": pair_time,
            "wall_time_s": wall_time,
            "energy_eV": energy,
            "time_per_step_ms": loop_time * 1000 / n_steps if loop_time else None,
            "time_per_atom_us": loop_time * 1e6 / (n_steps * actual_atoms) if loop_time else None,
            "success": True,
            "error": None
        }

    except subprocess.TimeoutExpired:
        return {
            "backend": backend,
            "atoms": actual_atoms,
            "success": False,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "backend": backend,
            "atoms": actual_atoms,
            "success": False,
            "error": str(e)
        }


def main():
    print("=" * 70)
    print("LAMMPS IREE Kokkos Plugin Benchmark with Bucket VMFBs")
    print("=" * 70)

    # Check prerequisites
    if not LAMMPS_EXE.exists():
        print(f"ERROR: LAMMPS not found at {LAMMPS_EXE}")
        return

    if not BUCKET_DIR.exists():
        print(f"ERROR: Bucket directory not found at {BUCKET_DIR}")
        return

    # Print available buckets
    print("\nAvailable buckets:")
    for max_edges, max_atoms, bucket_name in BUCKETS:
        bucket_path = BUCKET_DIR / bucket_name
        cpu_vmfb = bucket_path / "energy_gradient_f64_cpu.vmfb"
        cuda_vmfb = bucket_path / "energy_gradient_f64_cuda.vmfb"
        cpu_ok = "✓" if cpu_vmfb.exists() else "✗"
        cuda_ok = "✓" if cuda_vmfb.exists() else "✗"
        print(f"  {bucket_name}: {max_edges} edges, ≤{max_atoms} atoms [CPU:{cpu_ok} CUDA:{cuda_ok}]")

    # Test configurations
    # Test across all bucket sizes
    atom_counts = [8, 64, 256, 1024, 2744]  # From tiny to xlarge
    backends = ["cuda"]  # Kokkos compiled with GPU support only
    n_steps = 500

    results = []

    print(f"\nRunning {n_steps} MD steps for each configuration...\n")

    for backend in backends:
        print(f"\n{'='*40}")
        print(f"Backend: {backend.upper()}")
        print(f"{'='*40}")

        for n_atoms in atom_counts:
            max_edges, bucket_path = select_bucket(n_atoms)
            if bucket_path is None:
                print(f"\n  ~{n_atoms} atoms: SKIPPED (no bucket large enough)")
                continue

            print(f"\n  Testing ~{n_atoms} atoms (bucket: {max_edges} edges)...")
            result = run_lammps_benchmark(backend, n_atoms, n_steps)
            results.append(result)

            if result["success"]:
                print(f"    Actual atoms: {result['atoms']}")
                if result['loop_time_s'] is not None:
                    print(f"    Loop time: {result['loop_time_s']:.4f} s")
                    print(f"    Per step: {result['time_per_step_ms']:.3f} ms")
                    print(f"    Per atom: {result['time_per_atom_us']:.2f} μs")
                else:
                    print(f"    Wall time: {result['wall_time_s']:.4f} s")
                if result['energy_eV'] is not None:
                    print(f"    Energy: {result['energy_eV']:.6f} eV")
            else:
                print(f"    FAILED: {result['error']}")

    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    cuda_results = {r['atoms']: r for r in results if r['backend'] == 'cuda' and r['success']}

    if cuda_results:
        print("\nLAMMPS IREE Kokkos CUDA Performance:")
        print(f"{'Atoms':<10} {'Bucket':<10} {'Time/Step (ms)':<15} {'Time/Atom (μs)':<15} {'Energy (eV)':<15}")
        print("-" * 65)

        for n_atoms in sorted(cuda_results.keys()):
            r = cuda_results[n_atoms]
            bucket = r.get('bucket_edges', 'N/A')
            step_time = r.get('time_per_step_ms')
            atom_time = r.get('time_per_atom_us')
            energy = r.get('energy_eV')

            step_str = f"{step_time:.3f}" if step_time else "N/A"
            atom_str = f"{atom_time:.2f}" if atom_time else "N/A"
            energy_str = f"{energy:.4f}" if energy else "N/A"

            print(f"{n_atoms:<10} {bucket:<10} {step_str:<15} {atom_str:<15} {energy_str:<15}")

        # Scaling analysis
        atoms_list = sorted(cuda_results.keys())
        if len(atoms_list) >= 2:
            print("\nScaling Analysis:")
            small = atoms_list[0]
            large = atoms_list[-1]
            small_time = cuda_results[small].get('time_per_atom_us')
            large_time = cuda_results[large].get('time_per_atom_us')
            if small_time and large_time:
                scaling = small_time / large_time
                print(f"  {small} → {large} atoms: {scaling:.1f}x better per-atom efficiency")

    # Save results
    output_file = SCRIPT_DIR / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
