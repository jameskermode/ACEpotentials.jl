#!/usr/bin/env bash
# Follow-ups, run AFTER run_master.sh so they do not contend for CPU.
#   1. cross-engine trajectory agreement at every size (not just 216)
#   2. the Verlet-skin sweep that explains why `cached` is slower than `naive`
#   3. formal model-identity check: re-export the npz and diff it
set -u
W=$HOME/si-ace/molly/work
R=$W/results
JL="julia +1.11 --project=."
PY=$HOME/si-ace/spikeenv/bin/python
mkdir -p "$R"

echo "########## VERIFY (trajectory agreement) $(date -Is)"
cd "$HOME/si-ace/molly"
for rep in 3 4 5 6; do
  taskset -c 0-15 $JL -t 16 $W/md_molly.jl --model $W/si_model.json \
      --state $W/state_rep$rep.npz --mode naive --inner 10 --dt 0.25 \
      --verify --verify-legs 6 --tag verify_rep$rep > "$R/verify_jl_rep$rep.log" 2>&1
  grep -E "^  leg|^VERIFY" "$R/verify_jl_rep$rep.log"
done
cd "$HOME/si-ace/ACEpotentials/acejax/spike_distmd"
for rep in 3 4 5 6; do
  XLA_FLAGS="--xla_force_host_platform_device_count=1" taskset -c 0-15 \
    env PYTHONPATH=. "$PY" ace_md.py --model ../fixtures/si_fitted.npz \
      --rep $rep --ranks 1 --dt 0.25 --outer 6 --inner 10 \
      > "$R/verify_jax_rep$rep.log" 2>&1
  grep -E "^  leg|final PE|final \|dF\|" "$R/verify_jax_rep$rep.log"
done

echo "########## SKIN SWEEP (216 atoms, 1 thread) $(date -Is)"
cd "$HOME/si-ace/molly"
for s in 0.25 0.5 1.0 1.5; do
  for inner in 5 10; do
    T="skin_s${s}_i${inner}"
    taskset -c 0 $JL -t 1 $W/md_molly.jl --model $W/si_model.json \
        --state $W/state_rep3.npz --mode cached --inner $inner --dt 0.25 \
        --skin $s --steps 60 --reps 2 --tag "$T" > "$R/$T.log" 2>&1
    grep -hE "^RESULT|nlist builds" "$R/$T.log"
  done
done

echo "########## ALTERNATIVE JULIA BACKENDS $(date -Is)"
cd "$HOME/si-ace/molly"
$JL $W/try_backends.jl $W/si_model.json $W/state_rep3.npz 2>&1 | tee "$R/backends.log"

echo "########## MODEL IDENTITY $(date -Is)"
cd "$HOME/si-ace/ACEpotentials/acejax/julia"
$JL --project="$HOME/si-ace/molly" export_model.jl "$W/si_fresh.npz" ace1 \
    > "$R/export_fresh.log" 2>&1 && echo "export ok" || echo "EXPORT FAILED"
"$PY" "$W/check_model_match.py" "$W/si_fresh.npz" \
    "$HOME/si-ace/ACEpotentials/acejax/fixtures/si_fitted.npz" \
    | tee "$R/model_match.log"
echo "########## EXTRAS DONE $(date -Is)"
