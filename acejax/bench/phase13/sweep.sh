#!/bin/bash
# Phase 13 three-way sweep.  GPU exclusivity enforced; contention sampled.
cd /storage/eng/essswb/phase13
OUT=${OUT:-results/phase13_sweep.txt}; mkdir -p results
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -ne 0 ]; then echo "REFUSING: $busy compute app(s) on the GPU" >&2; exit 1; fi
CON="${OUT%.txt}_contention.txt"
( while true; do echo "$(date +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" "|")"; sleep 3; done ) > "$CON" 2>&1 &
trap "kill $! 2>/dev/null" EXIT

B=$HOME/si-ace/acejax/bench
P=/storage/eng/essswb/phase13
MATRIX="
ace69_f64|jax|$B/r_si_s69_float64/si
ace69_f32|jax|$B/r_si_s69_float32/si
ace710_f64|jax|$B/r_si_m710_float64/si
ace710_f32|jax|$B/r_si_m710_float32/si
ace2849_f64|jax|$B/r_si_l2849_float64/si
ace2849_f32|jax|$B/r_si_l2849_float32/si
mp0sml_jax32|jax|$P/bundles/mp0sml_comm
m0b2sml_jax32|jax|$P/bundles/m0b2sml_comm
m0b3med_jax32|jax|$P/bundles/m0b3med_comm
m0b2sml_sym64|sym64|$P/checkpoints/mace-0b2-small-Si.json
m0b2sml_sym32|sym32|$P/checkpoints/mace-0b2-small-Si.json
m0b3med_sym64|sym64|$P/checkpoints/mace-mp-0b3-medium-Si.json
m0b3med_sym32|sym32|$P/checkpoints/mace-mp-0b3-medium-Si.json
"
: > "$OUT"
printf "# steps=%s warm=%s symmode=%s\n" "${STEPS:-50}" "${WARM:-3}" "${SYMMODE:-no_domain_decomposition}" >> "$OUT"
printf "%-14s %-7s %12s %14s %20s\n" model atoms ms_per_step atom-steps/s poteng >> "$OUT"
for reps in ${REPS:-2 3 4 5 6}; do
  n=$((8*reps*reps*reps))
  for row in $MATRIX; do
    tag=${row%%|*}; rest=${row#*|}; eng=${rest%%|*}; arg=${rest#*|}
    if [ "$eng" = jax ]; then
      f="${arg}_r${reps}_n${n}.lammps-jax.json"
      [ -f "$f" ] || { printf "%-14s %-7s %12s %14s %20s\n" "$tag" "$n" - - "no-bundle" >> "$OUT"; continue; }
    else f="$arg"; fi
    r=$(STEPS=${STEPS:-50} WARM=${WARM:-3} ./scripts/runone.sh $reps $eng "$f" 2>/dev/null)
    if [ "${r:0:4}" = FAIL ]; then
      printf "%-14s %-7s %12s %14s %20s\n" "$tag" "$n" - - "${r}" >> "$OUT"
    else
      set -- $r
      printf "%-14s %-7s %12s %14s %20s\n" "$tag" "$n" "$1" "$2" "$3" >> "$OUT"
    fi
  done
done
echo SWEEPDONE >> "$OUT"
