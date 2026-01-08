# MyPotential - Portable ACE Interatomic Potential

A pre-compiled ACE (Atomic Cluster Expansion) interatomic potential packaged for
portable deployment. Works across CPU and GPU without requiring Julia, PyTorch,
or JAX at runtime.

## Quick Start

```python
from mypotential import Calculator
from ase.build import bulk

# Create silicon diamond structure
atoms = bulk('Si', 'diamond', a=5.43)

# Create calculator (auto-selects best device: CUDA > Vulkan > CPU)
atoms.calc = Calculator()

# Calculate properties
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()

print(f"Energy: {energy:.4f} eV")
print(f"Forces shape: {forces.shape}")
```

## Installation

```bash
pip install mypotential

# For CUDA GPU support
pip install mypotential[cuda]

# For Vulkan GPU support
pip install mypotential[vulkan]
```

## Device Selection

The calculator automatically selects the best available device:

```python
from mypotential import Calculator, print_device_info

# See available devices
print_device_info()

# Auto-select (default)
calc = Calculator()  # CUDA > Vulkan > CPU

# Force specific device
calc = Calculator(device='cpu')
calc = Calculator(device='cuda')
calc = Calculator(device='vulkan')
```

## Architecture

This package uses a split computation architecture:

1. **IREE Model** (GPU-accelerated): Computes per-pair force contributions
2. **Host Accumulation** (CPU): Scatters pair forces to atomic forces

This design avoids IREE scatter operation compilation issues while maintaining
GPU acceleration for the expensive ACE model evaluation.

```
┌─────────────────────────────────────────────────────────────┐
│                   Input: ASE Atoms                          │
│  positions, atomic_numbers, cell, pbc                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Neighbor List (matscipy)                       │
│  pair_i, pair_j, rij                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              IREE Model (GPU-accelerated)                   │
│  compute_pair_forces(rij, pool_matrix, params)              │
│  → pair_forces [n_pairs, 3]                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Force Accumulation (CPU)                       │
│  np.add.at(forces, pair_i, -pair_forces)                    │
│  np.add.at(forces, pair_j, +pair_forces)                    │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Calculator

```python
class Calculator(ASECalculator):
    """ASE Calculator for IREE-compiled ACE potentials."""

    def __init__(
        self,
        device: str = 'auto',
        models_dir: Optional[Path] = None,
        params_path: Optional[Path] = None,
    ):
        """
        Args:
            device: 'auto', 'cuda', 'vulkan', or 'cpu'
            models_dir: Custom directory with model files
            params_path: Path to parameters NPZ file
        """
```

### Utilities

```python
from mypotential import (
    get_device_info,     # Get device availability dict
    print_device_info,   # Print device status
    select_device,       # Select device and get model path
    list_available_models,  # List available model files
)
```

## Creating a Custom Package

To create a package for your own ACE model:

1. **Export from Julia**:
   ```bash
   julia scripts/export_from_julia.jl your_model.json -o ./models
   ```

2. **Compile to VMFB**:
   ```bash
   python scripts/compile_model.py models/model.mlir -o models/ --backends all
   ```

3. **Copy to package**:
   ```bash
   cp models/*.vmfb src/mypotential/models/
   cp models/params.npz src/mypotential/models/
   cp models/metadata.json src/mypotential/models/
   ```

4. **Build and install**:
   ```bash
   pip install .
   ```

## Model Files

The package contains:

- `models/model_cpu.vmfb` - CPU backend (llvm-cpu)
- `models/model_cuda.vmfb` - NVIDIA GPU (CUDA)
- `models/model_vulkan.vmfb` - Portable GPU (Vulkan/SPIR-V)
- `models/params.npz` - Model parameters
- `models/metadata.json` - Model metadata

## Requirements

- Python >= 3.9
- numpy >= 1.20
- ase >= 3.22
- matscipy >= 0.8
- iree-runtime >= 2.5

## License

MIT License - See LICENSE file for details.
