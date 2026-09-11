#!/bin/bash
# runone.sh <reps> <style> <arg> [extra lammps vars...]
#   style jax          -> arg is the bundle path
#   style sym64|sym32  -> arg is the symmetrix .json
# Prints: "<ms_per_step> <atom_steps_per_s> <poteng>" or "FAIL <reason>".
# Keys on the SECOND "Loop time" line, never on the exit code: this LAMMPS
# aborts in __cxa_finalize at process teardown (a static-destruction double free
# between liblammps and libkokkoskernels), AFTER the run has finished and the
# timing has been printed.  Treating rc as the signal would discard every good
# measurement.
cd /storage/eng/essswb/phase13; . scripts/env.sh
reps=$1; style=$2; arg=$3; shift 3
n=$((8*reps*reps*reps))
case $style in
  jax)  extra="-var style jax -var bundle $arg -var pjrt $PJRT" ;;
  sym*) extra="-var style $style -var macejson $arg -var symmode ${SYMMODE:-no_domain_decomposition}" ;;
esac
log=$(mktemp /tmp/p13_XXXXXX.log)
$LMP $KK -var reps $reps -var warm ${WARM:-3} -var steps ${STEPS:-20} $extra "$@" \
     -in scripts/in.si_phase13 > $log 2>&1
t=$(grep "Loop time of" $log | tail -1 | awk "{print \$4}")
pe=$(grep -E "^ *[0-9]+ +$n +-" $log | tail -1 | awk "{print \$3}")
if [ -z "$t" ]; then
  echo "FAIL $(grep -m1 -iE "error|out of memory|exceeded" $log | cut -c1-110)"
  cp $log /storage/eng/essswb/phase13/logs/fail_${style}_r${reps}_$$.log
else
  python3 -c "print(f\"{$t/${STEPS:-20}*1e3:.4f} {$n*${STEPS:-20}/$t:.4g} $pe\")"
fi
rm -f $log
