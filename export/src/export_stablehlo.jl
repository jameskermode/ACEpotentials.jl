#=
StableHLO / IREE Export
=======================

Export compiled models to StableHLO format and compile to IREE VMFB.
=#

using NPZ
using JSON3

## ============================================================================
## IREE Export
## ============================================================================

"""
    export_to_iree(compiled::CompiledACEModel, output_dir::String;
                   backends=[:cpu], compile_vmfb=true)

Export compiled model to IREE format.

Creates:
- `ace_model.mlir` - StableHLO IR
- `ace_model_cpu.vmfb` / `ace_model_cuda.vmfb` - Compiled IREE modules
- `ace_constants.npz` - Model parameters
- `ace_metadata.json` - Model configuration

# Arguments
- `compiled`: CompiledACEModel from compile_model()
- `output_dir`: Directory to write output files
- `backends`: Which backends to compile VMFB for
- `compile_vmfb`: Whether to run iree-compile (requires IREE installed)

# Example
```julia
compiled = compile_model(calc)
export_to_iree(compiled, "export_output/"; backends=[:cpu, :cuda])
```
"""
function export_to_iree(compiled::CompiledACEModel{T}, output_dir::String;
                        backends::Vector{Symbol}=[:cpu],
                        compile_vmfb::Bool=true) where T

    mkpath(output_dir)

    @info "Exporting to IREE format..." output_dir

    # Export StableHLO via Reactant
    mlir_path = joinpath(output_dir, "ace_model.mlir")

    @info "  Exporting StableHLO IR..."
    try
        # Get the first available compiled function for export
        backend = first(keys(compiled.efv_fns))
        fn = compiled.efv_fns[backend]

        # Create dummy inputs for export
        shapes = compiled.shapes
        edge_rij = zeros(T, shapes.max_edges, 3)
        atomic_numbers = zeros(Int64, shapes.max_atoms)
        edge_i = zeros(Int64, shapes.max_edges)
        edge_j = zeros(Int64, shapes.max_edges)

        # Convert to RArrays
        edge_rij_ra = Reactant.to_rarray(edge_rij)
        atomic_numbers_ra = Reactant.to_rarray(atomic_numbers)
        edge_i_ra = Reactant.to_rarray(edge_i)
        edge_j_ra = Reactant.to_rarray(edge_j)

        # Export using Reactant's serialization
        Reactant.Serialization.export_to_enzymejax(
            stacked_efv_from_edges,
            edge_rij_ra, atomic_numbers_ra, edge_i_ra, edge_j_ra,
            Int32(0), Int32(0), compiled.model;
            output_dir=output_dir,
            function_name="ace_efv"
        )

        @info "  StableHLO exported to $mlir_path"
    catch e
        @warn "StableHLO export failed" exception=e
        @info "  Creating placeholder MLIR file..."
        # Create a placeholder
        write(mlir_path, "// Placeholder - StableHLO export failed\n")
    end

    # Export model constants
    constants_path = joinpath(output_dir, "ace_constants.npz")
    @info "  Exporting model constants..."
    export_constants(compiled.model, constants_path)

    # Export metadata
    metadata_path = joinpath(output_dir, "ace_metadata.json")
    @info "  Exporting metadata..."
    export_metadata(compiled, metadata_path)

    # Compile to VMFB if requested
    if compile_vmfb
        @info "  Compiling to VMFB..."
        for backend in backends
            vmfb_path = joinpath(output_dir, "ace_model_$(backend).vmfb")
            compile_to_vmfb(mlir_path, vmfb_path, backend)
        end
    end

    @info "Export complete" output_dir

    return output_dir
end

"""
    compile_to_vmfb(mlir_path::String, vmfb_path::String, backend::Symbol)

Compile StableHLO to IREE VMFB using iree-compile.
"""
function compile_to_vmfb(mlir_path::String, vmfb_path::String, backend::Symbol)
    iree_backend = backend == :cuda ? "cuda" : "llvm-cpu"

    # Check if iree-compile is available
    iree_compile = Sys.which("iree-compile")
    if isnothing(iree_compile)
        @warn "iree-compile not found in PATH, skipping VMFB compilation"
        return nothing
    end

    try
        cmd = `$iree_compile
            --iree-input-type=stablehlo
            --iree-hal-target-backends=$iree_backend
            $mlir_path
            -o $vmfb_path`

        run(cmd)
        @info "  Compiled VMFB: $vmfb_path"
        return vmfb_path
    catch e
        @warn "VMFB compilation failed" backend exception=e
        return nothing
    end
end

## ============================================================================
## Test Input Export
## ============================================================================

"""
    export_test_inputs(compiled::CompiledACEModel, output_dir::String;
                       n_atoms=64, n_edges=1000)

Export test inputs for IREE verification.
"""
function export_test_inputs(compiled::CompiledACEModel{T}, output_dir::String;
                            n_atoms::Int=64, n_edges::Int=1000) where T

    test_dir = joinpath(output_dir, "test_inputs")
    mkpath(test_dir)

    shapes = compiled.shapes

    # Create random test inputs
    positions = rand(T, shapes.max_atoms, 3)
    atomic_numbers = zeros(Int64, shapes.max_atoms)
    atomic_numbers[1:n_atoms] .= 14  # Silicon

    edge_i = zeros(Int64, shapes.max_edges)
    edge_j = zeros(Int64, shapes.max_edges)
    edge_rij = zeros(T, shapes.max_edges, 3)

    # Create simple neighbor list
    for e in 1:n_edges
        i = rand(1:n_atoms)
        j = rand(1:n_atoms)
        while j == i
            j = rand(1:n_atoms)
        end
        edge_i[e] = i
        edge_j[e] = j
        edge_rij[e, :] = positions[j, :] - positions[i, :]
    end

    # Save as raw binary files (for iree-run-module)
    write(joinpath(test_dir, "positions.bin"), positions)
    write(joinpath(test_dir, "atomic_numbers.bin"), atomic_numbers)
    write(joinpath(test_dir, "edge_i.bin"), edge_i)
    write(joinpath(test_dir, "edge_j.bin"), edge_j)
    write(joinpath(test_dir, "edge_rij.bin"), edge_rij)

    # Save shapes info
    shapes_info = Dict(
        "n_atoms" => n_atoms,
        "n_edges" => n_edges,
        "max_atoms" => shapes.max_atoms,
        "max_edges" => shapes.max_edges,
        "positions_shape" => [shapes.max_atoms, 3],
        "edge_rij_shape" => [shapes.max_edges, 3],
    )
    open(joinpath(test_dir, "shapes.json"), "w") do f
        JSON3.pretty(f, shapes_info)
    end

    @info "Test inputs exported to $test_dir"
    return test_dir
end
