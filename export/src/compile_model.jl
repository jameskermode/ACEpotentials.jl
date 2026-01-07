#=
Model Compilation with Reactant
===============================

Orchestrates Reactant compilation of ACE models.
Produces compiled functions ready for IREE export.
=#

## ============================================================================
## Compilation Configuration
## ============================================================================

"""
    CompiledShapes

Fixed tensor shapes for compiled model.
Inputs must be padded to these shapes.
"""
struct CompiledShapes
    max_atoms::Int      # Maximum number of atoms
    max_neigs::Int      # Maximum neighbors per atom
    max_edges::Int      # Maximum total edges
end

# Keyword argument constructor
CompiledShapes(; max_atoms::Int, max_neigs::Int, max_edges::Int) =
    CompiledShapes(max_atoms, max_neigs, max_edges)

"""
Default shapes for compilation.
Suitable for systems up to ~4000 atoms with ~50 neighbors each.
"""
const DEFAULT_SHAPES = CompiledShapes(4096, 50, 200_000)

"""
    CompiledACEModel{T}

Container for compiled ACE model.

Stores:
- Compiled functions for energy and EFV computation
- Fixed shapes used during compilation
- Original model state (for serialization)
- Input RArrays (for function signatures)
"""
struct CompiledACEModel{T}
    # Compiled functions (keyed by backend)
    energy_fns::Dict{Symbol, Any}
    efv_fns::Dict{Symbol, Any}

    # Compilation configuration
    shapes::CompiledShapes
    backends::Vector{Symbol}

    # Model state
    model::ReactantStackedModel{T}

    # Input arrays (for reference)
    input_shapes::NamedTuple
end

## ============================================================================
## Model Compilation
## ============================================================================

"""
    compile_model(calc; shapes=DEFAULT_SHAPES, backends=[:cpu], include_forces=true)

Compile ACE calculator for Reactant export.

# Arguments
- `calc`: StackedCalculator or ETACEPotential
- `shapes`: CompiledShapes specifying fixed tensor sizes
- `backends`: Vector of backends to compile for (:cpu, :cuda)
- `include_forces`: Whether to compile EFV function (slower compilation)

# Returns
- `CompiledACEModel` containing compiled functions

# Example
```julia
calc = load_potential("model.json")
compiled = compile_model(calc; backends=[:cpu, :cuda])
```
"""
function compile_model(calc;
                       shapes::CompiledShapes=DEFAULT_SHAPES,
                       backends::Vector{Symbol}=[:cpu],
                       include_forces::Bool=true)

    @info "Converting model to Reactant format..."

    # Convert to ReactantStackedModel
    model = ReactantStackedModel(calc)
    T = Float32  # Use Float32 for better performance

    @info "Preparing input arrays..." shapes

    # Create input arrays with fixed shapes
    positions = zeros(T, shapes.max_atoms, 3)
    atomic_numbers = zeros(Int64, shapes.max_atoms)
    edge_i = zeros(Int64, shapes.max_edges)
    edge_j = zeros(Int64, shapes.max_edges)
    edge_rij = zeros(T, shapes.max_edges, 3)

    # Store input shapes
    input_shapes = (
        positions = (shapes.max_atoms, 3),
        atomic_numbers = (shapes.max_atoms,),
        edge_i = (shapes.max_edges,),
        edge_j = (shapes.max_edges,),
        edge_rij = (shapes.max_edges, 3),
    )

    # Compile for each backend
    energy_fns = Dict{Symbol, Any}()
    efv_fns = Dict{Symbol, Any}()

    for backend in backends
        @info "Compiling for backend: $backend"

        # Set Reactant backend
        backend_str = backend == :cuda ? "gpu" : "cpu"
        Reactant.set_default_backend(backend_str)

        # Convert inputs to RArrays
        positions_ra = Reactant.to_rarray(positions)
        atomic_numbers_ra = Reactant.to_rarray(atomic_numbers)
        edge_i_ra = Reactant.to_rarray(edge_i)
        edge_j_ra = Reactant.to_rarray(edge_j)
        edge_rij_ra = Reactant.to_rarray(edge_rij)

        # Compile energy function
        @info "  Compiling energy function..."
        try
            energy_fn = Reactant.@compile stacked_energy_from_edges(
                edge_rij_ra, atomic_numbers_ra, edge_i_ra, edge_j_ra,
                Int32(0), Int32(0),  # n_atoms, n_edges (runtime values)
                model
            )
            energy_fns[backend] = energy_fn
        catch e
            @warn "Failed to compile energy function for $backend" exception=e
        end

        # Compile EFV function (if requested)
        if include_forces
            @info "  Compiling EFV function..."
            try
                efv_fn = Reactant.@compile stacked_efv_from_edges(
                    edge_rij_ra, atomic_numbers_ra, edge_i_ra, edge_j_ra,
                    Int32(0), Int32(0),
                    model
                )
                efv_fns[backend] = efv_fn
            catch e
                @warn "Failed to compile EFV function for $backend" exception=e
            end
        end
    end

    @info "Compilation complete" backends=collect(keys(energy_fns))

    return CompiledACEModel{T}(
        energy_fns,
        efv_fns,
        shapes,
        backends,
        model,
        input_shapes
    )
end

## ============================================================================
## Evaluation Helpers
## ============================================================================

"""
    evaluate_energy(compiled::CompiledACEModel, positions, atomic_numbers,
                    edge_i, edge_j, edge_rij, n_atoms, n_edges;
                    backend=:cpu)

Evaluate energy using compiled model.
"""
function evaluate_energy(compiled::CompiledACEModel{T},
                         positions::AbstractMatrix,
                         atomic_numbers::AbstractVector{Int},
                         edge_i::AbstractVector{Int},
                         edge_j::AbstractVector{Int},
                         edge_rij::AbstractMatrix,
                         n_atoms::Int,
                         n_edges::Int;
                         backend::Symbol=:cpu) where T

    haskey(compiled.energy_fns, backend) ||
        error("Model not compiled for backend: $backend")

    # Pad inputs to compiled shapes
    positions_pad = pad_array(positions, compiled.input_shapes.positions, T)
    atomic_numbers_pad = pad_array(atomic_numbers, compiled.input_shapes.atomic_numbers, Int64)
    edge_i_pad = pad_array(edge_i, compiled.input_shapes.edge_i, Int64)
    edge_j_pad = pad_array(edge_j, compiled.input_shapes.edge_j, Int64)
    edge_rij_pad = pad_array(edge_rij, compiled.input_shapes.edge_rij, T)

    # Call compiled function
    fn = compiled.energy_fns[backend]
    energy = fn(edge_rij_pad, atomic_numbers_pad, edge_i_pad, edge_j_pad,
                Int32(n_atoms), Int32(n_edges))

    return energy
end

"""
    evaluate_efv(compiled::CompiledACEModel, positions, atomic_numbers,
                 edge_i, edge_j, edge_rij, n_atoms, n_edges;
                 backend=:cpu)

Evaluate energy, forces, and virial using compiled model.
"""
function evaluate_efv(compiled::CompiledACEModel{T},
                      positions::AbstractMatrix,
                      atomic_numbers::AbstractVector{Int},
                      edge_i::AbstractVector{Int},
                      edge_j::AbstractVector{Int},
                      edge_rij::AbstractMatrix,
                      n_atoms::Int,
                      n_edges::Int;
                      backend::Symbol=:cpu) where T

    haskey(compiled.efv_fns, backend) ||
        error("EFV not compiled for backend: $backend")

    # Pad inputs
    positions_pad = pad_array(positions, compiled.input_shapes.positions, T)
    atomic_numbers_pad = pad_array(atomic_numbers, compiled.input_shapes.atomic_numbers, Int64)
    edge_i_pad = pad_array(edge_i, compiled.input_shapes.edge_i, Int64)
    edge_j_pad = pad_array(edge_j, compiled.input_shapes.edge_j, Int64)
    edge_rij_pad = pad_array(edge_rij, compiled.input_shapes.edge_rij, T)

    # Call compiled function
    fn = compiled.efv_fns[backend]
    energy, forces_pad, virial = fn(edge_rij_pad, atomic_numbers_pad,
                                     edge_i_pad, edge_j_pad,
                                     Int32(n_atoms), Int32(n_edges))

    # Unpad forces
    forces = forces_pad[1:n_atoms, :]

    return energy, forces, virial
end

## ============================================================================
## Utility Functions
## ============================================================================

"""
    pad_array(arr, target_shape, T)

Pad array to target shape with zeros.
"""
function pad_array(arr::AbstractArray, target_shape::Tuple, T::Type)
    result = zeros(T, target_shape...)
    src_shape = size(arr)

    # Copy source data
    indices = ntuple(i -> 1:min(src_shape[i], target_shape[i]), length(target_shape))
    result[indices...] = T.(arr[indices...])

    return result
end
