#!/usr/bin/env bash
# Validate the Si ACE bundle against our Python calculator on 1 and 2 ranks.
# Modelled on lammps-jax-build/run/test_eam_bundle.sh.
#
# NOT YET RUN TO COMPLETION: lestrade's LAMMPS build cannot start on that host
# (AVX-512 binary, non-AVX-512 CPU -- see FINDINGS_lammps.md).  Run this on a
# host whose CPU matches the build, or after rebuilding for the run host.
#
# Usage: test_si_bundle.sh <bundle.json> [tag]
set -euo pipefail
BUNDLE=${1:-si_ace.lammps-jax.json}; TAG=${2:-si}
V=/storage/eng/essswb/venvs/lammps-jax
export PJRT=$V/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export LAMMPS_PLUGIN_PATH=/storage/eng/essswb/lammps-jax-build/build-plugin-shared-cudart
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
KK="-k on g 1 -sf kk -pk kokkos newton on neigh half"

echo "### $TAG : 1 rank, run 0 ###"
$V/bin/lmp $KK -var pjrt $PJRT -var bundle "$BUNDLE" -var dump_path $TAG.np1.dump \
    -in in.mlip_si 2>&1 | grep -vE "^I[0-9]|StreamExecutor" \
  | grep -E "^ *0 +216|Step|ERROR|LAMMPS-JAX|Pair *\|"

echo "### $TAG : 2 ranks, run 0 ###"
$V/bin/lmp-mpirun -np 2 $V/lib/lmp.real $KK -var pjrt $PJRT -var bundle "$BUNDLE" \
    -var dump_path $TAG.np2.dump -in in.mlip_si 2>&1 \
  | grep -vE "^I[0-9]|StreamExecutor" | grep -E "^ *0 +216|Step|ERROR" | head -4

echo "### $TAG np1 vs np2 ###"
python3 cmpdump.py $TAG.np1.dump $TAG.np2.dump 1e-8

echo "### $TAG vs the acejax Python calculator (1 rank) ###"
PE=$(grep -A2 "^ *Step" log.lammps | awk 'NR==2{print $3}')
python3 check_vs_python.py $TAG.np1.dump ../si_fitted.npz ${PE:-}
