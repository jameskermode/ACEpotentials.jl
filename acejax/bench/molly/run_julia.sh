#!/usr/bin/env bash
# ACEpotentials + Molly NVE series.  One pass over the size series.
#   ./run_julia.sh <pass-label> [outdir]
# Run from ~/si-ace/molly (the Julia project).  Pins CPUs explicitly so the
# thread count in the report is a fact about the run, not a default.
set -u
PASS=${1:-p1}
OUT=${2:-work/results}
W=work
mkdir -p "$OUT"
JL="julia +1.11 --project=."

for rep in 3 4 5 6; do
  ST=$W/state_rep$rep.npz
  for cfg in "cached 1 0" "cached 16 0-15" "naive 1 0" "naive 16 0-15" "bare 1 0"; do
    set -- $cfg; MODE=$1; THR=$2; CPUS=$3
    TAG="${PASS}_rep${rep}_${MODE}_t${THR}"
    echo "=== $TAG"
    taskset -c "$CPUS" $JL -t "$THR" $W/md_molly.jl \
        --model $W/si_model.json --state "$ST" --mode "$MODE" \
        --inner 10 --dt 0.25 --skin 1.0 --steps 60 --reps 4 --tag "$TAG" \
        > "$OUT/$TAG.log" 2>&1
    grep -h "^RESULT" "$OUT/$TAG.log" || echo "FAILED $TAG"
  done
done
