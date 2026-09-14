#!/bin/bash
# THROWAWAY SPIKE runner.
#
# moriarty is shared.  Every timing run therefore (a) refuses to start unless
# nvidia-smi shows the device empty, and (b) runs a 5 s sampler alongside, so
# the log can state that only this PID was ever on the device.  The whole series
# is run twice (PASS=1,2) and the two passes are compared: the effect being
# looked for may be small, and a difference smaller than the run-to-run spread
# is "no measurable difference", not a speedup.
set -u
cd "$(dirname "$0")/.."
PY=~/si-ace/.venv/bin/python
PASS=${PASS:-1}
LOG=${LOG:-/tmp/dag_gpu_pass${PASS}.log}
SAMP=/tmp/dag_gpu_sampler_pass${PASS}.log
: > "$LOG"; : > "$SAMP"

occupancy() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }

run() {
  local occ; occ=$(occupancy)
  if [ -n "$occ" ]; then
    echo "ABORT: device not empty before '$*':" >>"$LOG"; echo "$occ" >>"$LOG"; return 1
  fi
  echo "### device empty before: $* ($(date +%T))" >>"$LOG"
  ( while true; do echo "$(date +%T) $(occupancy | tr '\n' ';')" >>"$SAMP"; sleep 5; done ) &
  local sp=$!
  $PY spike_recursive/bench_dag.py "$@" >>"$LOG" 2>&1
  kill $sp 2>/dev/null
  echo "### after: $* ($(date +%T)) occupancy: [$(occupancy | tr '\n' ';')]" >>"$LOG"
}

run --npz fixtures/si_l2849.npz --reps 6 --repeats 20 --modes julia balanced
run --npz fixtures/si_l2849.npz --reps 6 --repeats 20 --modes julia balanced --f32
run --npz fixtures/si_m710.npz  --reps 6 --repeats 20 --modes julia balanced
run --npz fixtures/si_l2849.npz --reps 3 --repeats 20 --modes julia balanced
run --npz fixtures/si_s69.npz   --reps 6 --repeats 20 --modes julia balanced
echo "DONE pass $PASS" >>"$LOG"
