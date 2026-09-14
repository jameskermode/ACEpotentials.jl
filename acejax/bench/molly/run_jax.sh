#!/usr/bin/env bash
# spike_distmd/ace_md.py NVE series, UNMODIFIED script.  One pass.
#   ./run_jax.sh <pass-label> [outdir]
# Run from acejax/spike_distmd.
#
# --outer 7 --inner 10 discards leg 0 (compilation) and times legs 1-6, i.e.
# steps 10-70 -- the same window of the same trajectory the Julia runner times.
# Three invocations per point; the report takes the min of the three (each is
# itself a 6-leg mean).
#
# Two configurations, both single-rank, differing only in how many CPUs the
# process may use.  That is the thread statement for this engine: XLA's own
# threading is the only thing that could use the extra cores.
set -u
PASS=${1:-p1}
OUT=${2:-$HOME/si-ace/molly/work/results}
PY=$HOME/si-ace/spikeenv/bin/python
MODEL=../fixtures/si_fitted.npz
mkdir -p "$OUT"

for rep in 3 4 5 6; do
  for cfg in "1core 0" "16core 0-15"; do
    set -- $cfg; NAME=$1; CPUS=$2
    for i in 1 2 3; do
      TAG="${PASS}_rep${rep}_jax_${NAME}_run${i}"
      echo "=== $TAG"
      XLA_FLAGS="--xla_force_host_platform_device_count=1" \
      taskset -c "$CPUS" env PYTHONPATH=. "$PY" ace_md.py \
          --model "$MODEL" --rep "$rep" --ranks 1 --dt 0.25 \
          --outer 7 --inner 10 --quiet > "$OUT/$TAG.log" 2>&1
      grep -hE "inner step|rebuild " "$OUT/$TAG.log" || echo "FAILED $TAG"
    done
  done
done
