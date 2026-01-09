# ACE Export Tools

This directory provides the IREE compiler tools needed by `create_package.jl`.

It is **not** a Python package - just a uv environment with IREE installed.

## Setup

```bash
cd export/tools
uv sync --extra iree
```

## Usage

The tools are used by `create_package.jl` to compile StableHLO MLIR to IREE VMFB:

```bash
# From export/ directory:
source tools/.venv/bin/activate
IREE_COMPILE=$(which iree-compile) julia +1.11 --project=. scripts/create_package.jl --test-model --name mypotential --output /tmp/pkg
```

## Contents

- `pyproject.toml` - Defines IREE dependencies
- `.venv/` - Virtual environment with iree-base-compiler and iree-base-runtime
- `uv.lock` - Locked dependency versions
