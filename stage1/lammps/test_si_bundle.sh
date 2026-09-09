#!/usr/bin/env bash
# Validate the Si ACE bundle against the acejax Python calculator, 1 and 2 ranks.
# Shape follows lammps-jax-build/run/test_eam_bundle.sh.
#
# HOST: must be moriarty, not lestrade.  The LAMMPS build is
# `build-SKX-AMPERE86` (Skylake-X + Ampere 8.6); moriarty is a Xeon Silver 4216
# with AVX-512 and an RTX A4500.  lestrade is an i9-14900K with no AVX-512 at
# all, where liblammps.so dies with SIGILL in static init before reaching main.
#
# `gpu/aware off` is REQUIRED: with GPU-aware MPI the 2-rank run aborts inside
# CommKokkos::borders_device with a PSM3 fabric assertion.  That happens with
# the stock lj bundle too, so it is the host's MPI/CUDA interaction, not ours.
#
# PYTHON must have numpy + jax + equinox + acejax's deps; the system python3
# does not.  Override with PYTHON=..., default is the sibling export venv.
#
# Usage: [PYTHON=/path/to/python] test_si_bundle.sh [bundle.json] [tag]
set -euo pipefail
BUNDLE=${1:-si_ace.lammps-jax.json}; TAG=${2:-si}
V=/storage/eng/essswb/venvs/lammps-jax
PYTHON=${PYTHON:-$HOME/si-ace/.venv/bin/python}
export PJRT=$V/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export LAMMPS_PLUGIN_PATH=/storage/eng/essswb/lammps-jax-build/build-plugin-shared-cudart
cd "$(dirname "$0")"
KK="-k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware off"

# Build the geometry ONCE.  `displace_atoms random` is decomposition dependent,
# so generating it per-run would give the two rank counts different systems.
[ -f si.data ] || $V/bin/lmp -in in.si_setup > setup.log 2>&1

echo "### $TAG : 1 rank ###"
$V/bin/lmp $KK -var pjrt $PJRT -var bundle "$BUNDLE" -var dump_path $TAG.np1.dump \
    -in in.mlip_si > $TAG.np1.log 2>&1
grep -E "^ *0 +216|Nghost:" $TAG.np1.log | head -2

echo "### $TAG : 2 ranks ###"
$V/bin/lmp-mpirun -np 2 $V/lib/lmp.real $KK -var pjrt $PJRT -var bundle "$BUNDLE" \
    -var dump_path $TAG.np2.dump -in in.mlip_si > $TAG.np2.log 2>&1
grep -E "^ *0 +216|Nghost:" $TAG.np2.log | head -2

echo "### $TAG np1 vs np2 ###"
$PYTHON cmpdump.py $TAG.np1.dump $TAG.np2.dump 1e-9

for R in np1 np2; do
  echo "### $TAG $R vs the acejax Python calculator ###"
  PE=$(grep -A1 "^ *Step" $TAG.$R.log | awk 'NR==2{print $3}')
  $PYTHON check_vs_python.py $TAG.$R.dump ../si_fitted.npz "$PE"
done
