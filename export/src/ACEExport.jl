"""
    ACEExport

Reactant-based MLIR export for ACEpotentials.jl models.

Compiles ETACE models (including StackedCalculator for 1+2+many body)
to portable IREE VMFB files usable from Python/ASE and LAMMPS.

# Requirements
- Julia 1.11+ (required for Reactant.jl)
- IREE compiler for VMFB generation

# Quick Start
```julia
using ACEpotentials
using ACEExport

# Load or create an ETACE model
calc = load_potential("model.json")

# Compile for export
compiled = compile_model(calc)

# Export to IREE format
export_to_iree(compiled, "output_dir/"; backends=[:cpu, :cuda])
```

# Exported Files
- `ace_model_cpu.vmfb` / `ace_model_cuda.vmfb` - Compiled IREE modules
- `ace_constants.npz` - Model parameters in NumPy format
- `ace_metadata.json` - Model configuration and shapes
"""
module ACEExport

using LinearAlgebra
using StaticArrays

# Reactant and Enzyme for compilation
using Reactant
using Enzyme

# For model parameter serialization
using NPZ
using JSON3

# ACEpotentials imports
using ACEpotentials
import ACEpotentials: ETModels
import EquivariantTensors as ET

# Include submodules
include("reactant_state.jl")
include("reactant_embeddings.jl")
include("reactant_ace_kernel.jl")
include("reactant_stacked.jl")
include("compile_model.jl")
include("export_stablehlo.jl")
include("constants_serialization.jl")

# Public API
export compile_model
export export_to_iree
export CompiledACEModel
export ReactantETACEState
export ReactantStackedModel
export DEFAULT_SHAPES

end # module
