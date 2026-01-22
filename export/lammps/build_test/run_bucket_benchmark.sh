#!/bin/bash
#
# Benchmark LAMMPS IREE Kokkos Plugin with Bucket VMFBs
#
# Tests different system sizes using appropriate bucket VMFBs.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMMPS_EXE="$HOME/lammps/lammps-22Jul2025/build/lmp"
BUCKET_DIR="$SCRIPT_DIR/../../benchmark/bucket_energy_gradient"
PLUGIN="$SCRIPT_DIR/libpair_iree_kokkos.so"

# Bucket sizes: edges, approx atoms
BUCKETS=("2000:77" "10000:385" "50000:1923" "100000:3846")

# System sizes to test (number of unit cells in each direction)
# n_atoms = 8 * n_cells^3
CELL_SIZES=(1 2 3 4 5 6 7)  # 8, 64, 216, 512, 1000, 1728, 2744 atoms

N_STEPS=500
A_SI=5.43  # Si lattice constant

echo "========================================================================"
echo "LAMMPS IREE Kokkos Benchmark with Bucket VMFBs"
echo "========================================================================"
echo ""
echo "LAMMPS: $LAMMPS_EXE"
echo "Plugin: $PLUGIN"
echo "Buckets: $BUCKET_DIR"
echo ""

# Check prerequisites
if [ ! -f "$LAMMPS_EXE" ]; then
    echo "ERROR: LAMMPS not found at $LAMMPS_EXE"
    exit 1
fi

if [ ! -f "$PLUGIN" ]; then
    echo "ERROR: Plugin not found at $PLUGIN"
    exit 1
fi

# Function to select bucket for given atom count
select_bucket() {
    local n_atoms=$1
    local estimated_edges=$((n_atoms * 26))

    for bucket in "${BUCKETS[@]}"; do
        local max_edges="${bucket%%:*}"
        if [ $estimated_edges -le $max_edges ]; then
            echo "$max_edges"
            return
        fi
    done
    echo ""
}

# Function to run benchmark for given cell size
run_benchmark() {
    local n_cells=$1
    local n_atoms=$((8 * n_cells * n_cells * n_cells))
    local box_size=$(echo "$n_cells * $A_SI" | bc -l)

    local max_edges=$(select_bucket $n_atoms)
    if [ -z "$max_edges" ]; then
        echo "  $n_atoms atoms: SKIPPED (no bucket large enough)"
        return
    fi

    local vmfb="$BUCKET_DIR/bucket_$max_edges/energy_gradient_f64_cuda.vmfb"
    if [ ! -f "$vmfb" ]; then
        echo "  $n_atoms atoms: SKIPPED (VMFB not found: $vmfb)"
        return
    fi

    echo ""
    echo "  Testing $n_atoms atoms (${n_cells}x${n_cells}x${n_cells} cells, bucket: $max_edges edges)..."

    # Create LAMMPS input
    local input_file="$SCRIPT_DIR/bench_${n_atoms}.in"
    local log_file="$SCRIPT_DIR/bench_${n_atoms}.log"

    cat > "$input_file" << EOF
# LAMMPS Benchmark: $n_atoms atoms with bucket $max_edges
plugin load $PLUGIN
package kokkos neigh half newton on
units metal
atom_style atomic
boundary p p p
newton on

lattice diamond $A_SI
region box block 0 $n_cells 0 $n_cells 0 $n_cells
create_box 1 box
create_atoms 1 box

# Add small random displacement
displace_atoms all random 0.05 0.05 0.05 42

mass 1 28.0855

pair_style iree/kk $vmfb cuda
pair_coeff * * Si

neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes

# Warmup
run 10

# Reset timer
reset_timestep 0
timer timeout off

# Timed run
run $N_STEPS

# Print results
variable pe equal pe
print "BENCHMARK_ENERGY: \${pe}"
print "BENCHMARK_ATOMS: $n_atoms"
print "BENCHMARK_BUCKET: $max_edges"
print "BENCHMARK_COMPLETE"
EOF

    # Run LAMMPS
    $LAMMPS_EXE -k on g 1 -sf kk -in "$input_file" -log "$log_file" 2>&1 | grep -E "(Loop time|BENCHMARK_|Error|ERROR)" || true

    # Parse results from log
    if [ -f "$log_file" ]; then
        local loop_time=$(grep "Loop time of" "$log_file" | tail -1 | awk '{print $4}')
        local energy=$(grep "BENCHMARK_ENERGY:" "$log_file" | awk '{print $2}')

        if [ -n "$loop_time" ] && [ "$loop_time" != "0" ]; then
            local time_per_step=$(echo "scale=4; $loop_time * 1000 / $N_STEPS" | bc -l)
            local time_per_atom=$(echo "scale=4; $loop_time * 1000000 / ($N_STEPS * $n_atoms)" | bc -l)

            echo "    Loop time: ${loop_time}s"
            echo "    Per step: ${time_per_step}ms"
            echo "    Per atom: ${time_per_atom}μs"
            echo "    Energy: ${energy} eV"

            # Save to results file
            echo "$n_atoms,$max_edges,$loop_time,$time_per_step,$time_per_atom,$energy" >> "$SCRIPT_DIR/benchmark_results.csv"
        fi
    fi

    # Cleanup
    rm -f "$input_file" "$log_file"
}

# Initialize results file
echo "atoms,bucket,loop_time_s,time_per_step_ms,time_per_atom_us,energy_eV" > "$SCRIPT_DIR/benchmark_results.csv"

echo ""
echo "Available buckets:"
for bucket in "${BUCKETS[@]}"; do
    max_edges="${bucket%%:*}"
    max_atoms="${bucket##*:}"
    vmfb="$BUCKET_DIR/bucket_$max_edges/energy_gradient_f64_cuda.vmfb"
    if [ -f "$vmfb" ]; then
        echo "  bucket_$max_edges: ≤$max_atoms atoms [CUDA: ✓]"
    else
        echo "  bucket_$max_edges: ≤$max_atoms atoms [CUDA: ✗]"
    fi
done

echo ""
echo "Running benchmarks ($N_STEPS MD steps each)..."
echo "========================================================================"

for n_cells in "${CELL_SIZES[@]}"; do
    run_benchmark $n_cells
done

echo ""
echo "========================================================================"
echo "BENCHMARK COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $SCRIPT_DIR/benchmark_results.csv"
cat "$SCRIPT_DIR/benchmark_results.csv"
