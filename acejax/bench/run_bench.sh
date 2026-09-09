#!/usr/bin/env bash
# Phase 8 throughput benchmark on moriarty.  See FINDINGS_benchmark.md.
set -uo pipefail
V=/storage/eng/essswb/venvs/lammps-jax
export PJRT=$V/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export LAMMPS_PLUGIN_PATH=/storage/eng/essswb/lammps-jax-build/build-plugin-shared-cudart
LMP=${LMP:-/storage/eng/essswb/lammps-jax-build/lammps/build-SKX-AMPERE86-mlpace/lmp}
export LD_LIBRARY_PATH=$V/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:${LD_LIBRARY_PATH:-}
cd "$(dirname "$0")"
KK="-k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware off"
STEPS=${STEPS:-20}
REPS=${REPS:-"2 3 4 5 6 8"}

printf "%-6s %-8s %-7s %10s %14s %s\n" style atoms steps "loop_s" "atom-steps/s" note
for reps in $REPS; do
  n=$((8*reps*reps*reps))
  for style in "$@"; do
    case $style in
      jax)  extra="-var bundle $PWD/bundles/si_r${reps}_n${n}.lammps-jax.json -var pjrt $PJRT" ;;
      pace) extra="-var yace $PWD/${YACE:-si_v06.yace}" ;;
      eam)  extra="-var eamfile $PWD/${EAMFILE:-Si.eam.alloy}" ;;
    esac
    log=/tmp/bench_${style}_${reps}.log
    $LMP $KK -var reps $reps -var steps $STEPS -var style $style $extra \
        -in in.si_bench > $log 2>&1
    if [ $? -ne 0 ]; then
      printf "%-6s %-8s %-7s %10s %14s %s\n" $style $n $STEPS - - "FAILED ($(grep -m1 -iE 'error' $log | cut -c1-60))"
      continue
    fi
    t=$(grep -m1 "Loop time of" $log | awk '{print $4}')
    [ -z "$t" ] && { printf "%-6s %-8s %-7s %10s %14s %s\n" $style $n $STEPS - - "no timing"; continue; }
    printf "%-6s %-8s %-7s %10s %14.4g %s\n" $style $n $STEPS "$t" \
      "$(python3 -c "print($n*$STEPS/$t)" 2>/dev/null || echo -)" ""
  done
done
