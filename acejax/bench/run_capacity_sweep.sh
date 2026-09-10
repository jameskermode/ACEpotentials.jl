#!/usr/bin/env bash
# Time `pair jax/kk` at 1728 atoms against a ladder of max_atoms capacities,
# max_edges held exactly fixed.  See sweep_capacity.py for the design.
#
# GPU EXCLUSIVITY: refuses to start if another process holds the device, and
# samples compute-apps throughout, so a contaminated run is visible afterwards.
set -uo pipefail
V=/storage/eng/essswb/venvs/lammps-jax
export PJRT=$V/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export LAMMPS_PLUGIN_PATH=/storage/eng/essswb/lammps-jax-build/build-plugin-shared-cudart
LMP=${LMP:-/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-mlpace/lmp}
# The build directory MUST precede $V/lib; see run_bench.sh for why.
export LD_LIBRARY_PATH=$(dirname $LMP):$V/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:${LD_LIBRARY_PATH:-}
cd "$(dirname "$0")"
KK="-k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware off"
DIR=${DIR:?set DIR to the bundle directory (absolute path)}
OUT=${OUT:?set OUT to the results file}
WARM=${WARM:-5}
STEPS=${STEPS:-50}
REPS=6
N=1728

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then
  echo "REFUSING: $busy compute app(s) already on the GPU" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  exit 1
fi

: > "$OUT"
CON="${OUT%.txt}_contention.txt"
: > "$CON"
( while true; do
    echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' '|')"
    sleep 3
  done ) >> "$CON" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

printf "# dir=%s warm=%s steps=%s atoms=%s\n" "$DIR" "$WARM" "$STEPS" "$N" >> "$OUT"
printf "%-10s %10s %12s %14s %s\n" max_atoms loop_s ms_per_step atom-steps/s note >> "$OUT"
for ma in $(ls "$DIR"/cap_a*.lammps-jax.json | sed 's|.*/cap_a||; s|\.lammps-jax\.json||' | sort -n); do
  f="$DIR/cap_a${ma}.lammps-jax.json"
  log=$(mktemp /tmp/capsweep_XXXX.log)
  $LMP $KK -var reps $REPS -var warm $WARM -var steps $STEPS \
       -var bundle "$f" -var pjrt $PJRT -in in.si_capacity > $log 2>&1
  rc=$?
  # the TIMED run is the second "Loop time of" line
  t=$(grep "Loop time of" $log | tail -1 | awk '{print $4}')
  if [ $rc -ne 0 ] || [ -z "$t" ]; then
    note=$(grep -m1 -iE "error|ERROR" $log | cut -c1-90)
    printf "%-10s %10s %12s %14s %s\n" "$ma" - - - "FAILED rc=$rc ${note:-no timing}" >> "$OUT"
    cp $log ${OUT%.txt}_fail_a${ma}.log
    continue
  fi
  printf "%-10s %10s %12s %14s %s\n" "$ma" "$t" \
    "$(python3 -c "print(f'{$t/$STEPS*1e3:.3f}')")" \
    "$(python3 -c "print(f'{$N*$STEPS/$t:.4g}')")" "" >> "$OUT"
  rm -f $log
done
echo "ALLDONE" >> "$OUT"
