# Package Template

This directory contains **template Python files** used by `create_package.jl` to generate redistributable ACE potential packages.

## Contents

```
package/
└── src/mypotential/
    ├── calculator.py      # ASE Calculator implementation
    ├── _device.py         # Device selection logic
    └── _iree_wrapper.py   # IREE runtime wrapper
```

## How It Works

When you run `create_package.jl`, it:

1. Copies these template files to the output package
2. Replaces `mypotential` with the actual package name
3. Generates `__init__.py`, `pyproject.toml`, and test files from templates in the script
4. Adds compiled VMFB files and model parameters

## Usage

Do not use this directory directly. Instead, use `create_package.jl`:

```bash
cd export
source tools/.venv/bin/activate
IREE_COMPILE=$(which iree-compile) julia +1.11 --project=. scripts/create_package.jl \
    --test-model --name silicon_ace --output /tmp/pkg
```

This generates a complete, pip-installable package at `/tmp/pkg/silicon_ace/`.
