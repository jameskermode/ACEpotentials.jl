# ACE-IREE: ASE Calculator for IREE-compiled ACE Potentials

This Python package provides an ASE-compatible Calculator that uses ACE models
compiled via Reactant.jl to IREE VMFB format. The same compiled model can be
used from both Python/ASE and LAMMPS, ensuring consistent results.

## Installation

```bash
# Basic installation (uses subprocess for IREE)
pip install .

# With IREE runtime (faster, recommended)
pip install .[iree]

# Development installation
pip install -e .[dev,iree]
```

## Quick Start

```python
from ase.build import bulk
from ace_iree import ACECalculator

# Load the calculator
calc = ACECalculator(
    vmfb_path="ace_model_cpu.vmfb",
    constants_path="ace_constants.npz",
    metadata_path="ace_metadata.json",  # optional
    device="local-task",  # CPU with threading
)

# Create a structure
atoms = bulk("Si", "diamond", a=5.43) * (2, 2, 2)
atoms.calc = calc

# Calculate properties
energy = atoms.get_potential_energy()  # eV
forces = atoms.get_forces()            # eV/Angstrom
stress = atoms.get_stress()            # eV/Angstrom^3 (Voigt)
```

## Exporting Models from Julia

First, export your ACE model from Julia:

```julia
using ACEExport

# Load your fitted model
calc = load_ace_potential("silicon.json")

# Compile for deployment
compiled = compile_model(calc)

# Export to IREE format
export_to_iree(compiled, "export_dir/")
```

This creates:
- `ace_model.mlir` - StableHLO IR
- `ace_model_cpu.vmfb` - CPU compiled module
- `ace_model_cuda.vmfb` - GPU compiled module (optional)
- `ace_constants.npz` - Model parameters
- `ace_metadata.json` - Model configuration

## Device Selection

The calculator supports different compute backends:

```python
# CPU with multi-threading (default, recommended for most cases)
calc = ACECalculator(vmfb_path, constants_path, device="local-task")

# CPU single-threaded (for debugging)
calc = ACECalculator(vmfb_path, constants_path, device="local-sync")

# NVIDIA GPU (requires CUDA VMFB)
calc = ACECalculator("ace_model_cuda.vmfb", constants_path, device="cuda")

# Apple GPU (requires Metal VMFB)
calc = ACECalculator("ace_model_metal.vmfb", constants_path, device="metal")
```

## Runtime Backends

The calculator can use two backends for IREE execution:

1. **iree-runtime** (preferred): Direct Python bindings, faster
2. **subprocess**: Falls back to `iree-run-module` CLI

By default, it tries iree-runtime first:

```python
# Prefer iree-runtime (default)
calc = ACECalculator(vmfb_path, constants_path, prefer_runtime=True)

# Force subprocess (useful if version mismatch)
calc = ACECalculator(vmfb_path, constants_path, prefer_runtime=False)
```

## Supported Properties

- `energy` - Total potential energy (eV)
- `forces` - Forces on atoms (eV/Angstrom)
- `stress` - Stress tensor in Voigt notation (eV/Angstrom^3)

## Limitations

- Maximum atoms and neighbors are fixed at compilation time (default: 4096 atoms, 50 neighbors)
- All atomic species must be declared in the model
- Periodic boundary conditions require a cell with non-zero volume

## Requirements

- Python >= 3.9
- numpy >= 1.21
- ase >= 3.22
- matscipy >= 0.8
- iree-runtime >= 2.0 (optional but recommended)
