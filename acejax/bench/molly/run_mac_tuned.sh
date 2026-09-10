#!/usr/bin/env bash
# macOS: the 0.5 A-skin configuration the Linux sweep found to be Julia's best,
# plus a repeat pass of the main series for pass-to-pass repeatability.
set -u
PASS=${1:-mt1}
SP=${2:?scratch dir}
OUT=${3:-$SP/results}
REPO=/Users/u1470235/.julia/dev/ACEpotentials
JL=$HOME/.julia/juliaup/julia-1.11.9+0.aarch64.apple.darwin14/Julia-1.11.app/Contents/Resources/julia/bin/julia
mkdir -p "$OUT"
cd "$REPO/acejax/bench/molly"
for rep in 3 4 5 6; do
  for THR in 1 12; do
    TAG="${PASS}_rep${rep}_tuned_t${THR}"
    echo "=== $TAG"
    "$JL" --project="$SP/mollyenv" -t "$THR" md_molly.jl \
        --model "$SP/si_model.json" --state "$SP/state_rep$rep.npz" \
        --mode cached --inner 10 --dt 0.25 --skin 0.5 --steps 60 --reps 4 \
        --tag "$TAG" > "$OUT/$TAG.log" 2>&1
    grep -h "^RESULT" "$OUT/$TAG.log" || echo "FAILED $TAG"
  done
done
echo "TUNED DONE $(date -Is)"
