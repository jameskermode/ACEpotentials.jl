#!/usr/bin/env bash
# The skin sweep at 216 atoms found the Julia optimum is NOT the default:
# a 0.5 A Verlet skin rebuilt every 10 steps beats rebuilding the exact list
# every step, because in Si diamond the next neighbour shell after 5.92 A sits
# at 6.65 A -- so a skin up to ~0.6 A adds no pairs at all, and its rebuilds are
# amortised for free.  This runs that configuration over the whole series so the
# Julia side is reported at its best, and re-checks that the optimum does not
# move with system size.
set -u
PASS=${1:-t1}
R=$HOME/si-ace/molly/work/results
W=$HOME/si-ace/molly/work
JL="julia +1.11 --project=."
cd "$HOME/si-ace/molly"

for rep in 3 4 5 6; do
  for cfg in "1 0" "16 0-15"; do
    set -- $cfg; THR=$1; CPUS=$2
    TAG="${PASS}_rep${rep}_tuned_t${THR}"
    echo "=== $TAG"
    taskset -c "$CPUS" $JL -t "$THR" $W/md_molly.jl --model $W/si_model.json \
        --state $W/state_rep$rep.npz --mode cached --inner 10 --dt 0.25 \
        --skin 0.5 --steps 60 --reps 4 --tag "$TAG" > "$R/$TAG.log" 2>&1
    grep -h "^RESULT" "$R/$TAG.log" || echo "FAILED $TAG"
  done
done

# does the skin optimum move with size?  same sweep at 1728 atoms
if [ "$PASS" = "t1" ]; then
  for s in 0.25 0.5 0.75 1.0; do
    TAG="skin6_s${s}_i10"
    taskset -c 0 $JL -t 1 $W/md_molly.jl --model $W/si_model.json \
        --state $W/state_rep6.npz --mode cached --inner 10 --dt 0.25 \
        --skin $s --steps 60 --reps 2 --tag "$TAG" > "$R/$TAG.log" 2>&1
    grep -h "^RESULT" "$R/$TAG.log" || echo "FAILED $TAG"
  done
fi
echo "TUNED $PASS DONE $(date -Is)"
