#!/bin/bash
# Build the MACE-JAX GPU environment on moriarty, exactly.
#
#   ./make_env.sh [VENV_PATH]
#
# Default VENV_PATH is /storage/eng/essswb/macejax-gpu/venv.
#
# Everything is pinned by requirements.lock.txt, which was frozen from the
# Phase-13 environment (/storage/eng/essswb/phase13/venv-p13) and extended with
# the jax CUDA plugin pins taken from the working GPU venv (~/si-ace/.venv).
# The jax version (0.11.1) is deliberately the SAME as acejax uses, so that a
# MACE-vs-ACE timing comparison is like-for-like.
set -euo pipefail

VENV=${1:-/storage/eng/essswb/macejax-gpu/venv}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PATH=$HOME/.local/bin:$PATH          # uv

command -v uv >/dev/null || { echo "uv not on PATH"; exit 1; }

uv venv --python 3.12.8 "$VENV"

# torch is the +cpu build: MACE-JAX only needs torch to read the torch
# checkpoint, and a CUDA torch would fight jax over the card.
VIRTUAL_ENV="$VENV" uv pip install \
    -r "$HERE/requirements.lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match

echo
echo "Built $VENV"
VIRTUAL_ENV="$VENV" uv pip freeze | grep -E "^(jax|jaxlib|jax-cuda12-plugin|mace-jax|mace-torch|torch|e3nn-jax)[ =@]" || true

# Optional acejax extras: MACEJAX_WITH_ACEJAX=1 ./make_env.sh
if [ "${MACEJAX_WITH_ACEJAX:-0}" = "1" ]; then
    VIRTUAL_ENV="$VENV" uv pip install -r "$HERE/requirements-acejax-extra.txt"
    echo "acejax extras installed (use PYTHONPATH=/path/to/acejax)"
fi
