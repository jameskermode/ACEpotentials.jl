#!/bin/bash
set -e
cd /storage/eng/essswb/phase13
export PYTHONPATH=$HOME/si-ace/lammps-jax/python JAX_PLATFORMS=cpu
PY=/storage/eng/essswb/phase13/venv-p13/bin/python
$PY scripts/export_all.py m0b3med mace-mp-0b3-medium-jax comm 2 3 4 5 6
$PY scripts/export_all.py m0b2sml mace-0b2-small-jax      comm 2 3 4 5 6
$PY scripts/export_all.py mp0sml  mace-mp-0-small-jax     comm 2 3 4 5 6
echo EXPORTDONE
