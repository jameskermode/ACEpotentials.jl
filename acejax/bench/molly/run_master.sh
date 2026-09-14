#!/usr/bin/env bash
# Full experiment: two passes over the size series, Julia then JAX each time,
# never concurrently (both are CPU-bound on the same 16 cores).
set -u
R=$HOME/si-ace/molly/work/results
mkdir -p "$R"
for pass in p1 p2; do
  echo "########## JULIA $pass  $(date -Is)"
  (cd "$HOME/si-ace/molly" && bash work/run_julia.sh "$pass" "$R")
  echo "########## JAX $pass  $(date -Is)"
  (cd "$HOME/si-ace/ACEpotentials/acejax/spike_distmd" && bash "$HOME/si-ace/molly/work/run_jax.sh" "$pass" "$R")
done
echo "########## DONE $(date -Is)"
