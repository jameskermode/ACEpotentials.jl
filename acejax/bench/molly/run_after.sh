#!/usr/bin/env bash
# tuned series (two passes), then the model-identity export that failed for want
# of Lux in the benchmark project
set -u
W=$HOME/si-ace/molly/work
R=$W/results
bash $W/run_tuned.sh t1
bash $W/run_tuned.sh t2
echo "########## MODEL IDENTITY $(date -Is)"
cd "$HOME/si-ace/molly"
julia +1.11 --project=. -e 'using Pkg; Pkg.add("Lux")' > "$R/add_lux.log" 2>&1
cd "$HOME/si-ace/ACEpotentials/acejax/julia"
julia +1.11 --project="$HOME/si-ace/molly" export_model.jl "$W/si_fresh.npz" ace1 \
    > "$R/export_fresh.log" 2>&1 && echo "export ok" || echo "EXPORT FAILED"
"$HOME/si-ace/spikeenv/bin/python" "$W/check_model_match.py" "$W/si_fresh.npz" \
    "$HOME/si-ace/ACEpotentials/acejax/fixtures/si_fitted.npz" \
    > "$R/model_match.log" 2>&1
tail -3 "$R/model_match.log"
echo "########## AFTER DONE $(date -Is)"
