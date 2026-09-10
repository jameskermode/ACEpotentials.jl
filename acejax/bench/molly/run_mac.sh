#!/usr/bin/env bash
# Same experiment as run_julia.sh + run_jax.sh, on the macOS host.
#
# Difference from the Linux runner, stated rather than hidden: macOS has no
# `taskset`, so CPUs cannot be pinned.  Thread counts are set at the engine
# (julia -t N, XLA flags) and the OS places them.  The M3 Pro is also
# heterogeneous (6 performance + 6 efficiency cores), so "12 threads" is not 12
# equal cores; a 6-thread point is included for that reason.
#
#   ./run_mac.sh <pass-label> <scratch-dir> <results-dir>
set -u
PASS=${1:-m1}
SP=${2:?scratch dir}
OUT=${3:-$SP/results}
REPO=/Users/u1470235/.julia/dev/ACEpotentials
JL=$HOME/.julia/juliaup/julia-1.11.9+0.aarch64.apple.darwin14/Julia-1.11.app/Contents/Resources/julia/bin/julia
PY=$REPO/acejax/.venv/bin/python
LJ=$SP/lammps-jax-main/python
mkdir -p "$OUT"

cd "$REPO/acejax/bench/molly"
for rep in 3 4 5 6; do
  for cfg in "naive 1" "naive 12" "cached 1" "cached 12" "bare 1"; do
    set -- $cfg; MODE=$1; THR=$2
    TAG="${PASS}_rep${rep}_${MODE}_t${THR}"
    echo "=== $TAG"
    "$JL" --project="$SP/mollyenv" -t "$THR" md_molly.jl \
        --model "$SP/si_model.json" --state "$SP/state_rep$rep.npz" \
        --mode "$MODE" --inner 10 --dt 0.25 --skin 1.0 --steps 60 --reps 4 \
        --tag "$TAG" > "$OUT/$TAG.log" 2>&1
    grep -h "^RESULT" "$OUT/$TAG.log" || echo "FAILED $TAG"
  done
done

cd "$REPO/acejax/spike_distmd"
for rep in 3 4 5 6; do
  for cfg in "default --xla_force_host_platform_device_count=1" \
             "1thread --xla_force_host_platform_device_count=1|--xla_cpu_multi_thread_eigen=false|intra_op_parallelism_threads=1"; do
    set -- $cfg; NAME=$1; FLAGS=$(echo "$2" | tr '|' ' ')
    for i in 1 2 3; do
      TAG="${PASS}_rep${rep}_jax_${NAME}_run${i}"
      echo "=== $TAG"
      XLA_FLAGS="$FLAGS" PYTHONPATH=".:$LJ" "$PY" ace_md.py \
          --model ../fixtures/si_fitted.npz --rep "$rep" --ranks 1 --dt 0.25 \
          --outer 7 --inner 10 --quiet > "$OUT/$TAG.log" 2>&1
      grep -hE "inner step|rebuild " "$OUT/$TAG.log" || echo "FAILED $TAG"
    done
  done
done
echo "DONE $(date -Is)"
