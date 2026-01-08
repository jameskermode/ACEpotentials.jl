#=
Constants Serialization
=======================

Export model constants to NPZ format for Python/C++ consumption.
=#

using NPZ
using JSON3

## ============================================================================
## NPZ Export
## ============================================================================

"""
    export_constants(model::ReactantStackedModel, path::String)

Export model constants to NPZ format.

The NPZ file contains all parameters needed to reconstruct the model:
- Species configuration
- Radial basis parameters
- ACE tensor specifications
- Readout weights
"""
function export_constants(model::ReactantStackedModel{T}, path::String) where T

    # Collect all arrays
    arrays = Dict{String, Any}()

    # Species info
    arrays["n_species"] = Int32[model.n_species]
    arrays["species_Z"] = Int32.(model.species_Z)
    arrays["rcut"] = Float32[model.rcut]

    # One-body
    arrays["E0"] = Float32.(model.E0)

    # ACE state
    ace = model.ace_state

    # ACE tensor specs
    arrays["spec_R"] = Int64.(ace.spec_R)
    arrays["spec_Y"] = Int64.(ace.spec_Y)
    arrays["A2Bmap"] = Float32.(ace.A2Bmap)

    # Specs matrices (per-order)
    for (i, spec_mat) in enumerate(ace.specs_mats)
        arrays["specs_mat_$i"] = Int64.(spec_mat)
    end
    arrays["n_orders"] = Int32[length(ace.specs_mats)]

    # Radial basis
    arrays["n_polys"] = Int32[ace.n_polys]
    arrays["n_rnl"] = Int32[ace.n_rnl]
    arrays["agnesi_params"] = Float32.(ace.agnesi_params)
    arrays["poly_A"] = Float32.(ace.poly_A)
    arrays["poly_B"] = Float32.(ace.poly_B)
    arrays["poly_C"] = Float32.(ace.poly_C)
    arrays["W_radial"] = Float32.(ace.W_radial)

    # Angular basis
    arrays["maxl"] = Int32[ace.maxl]
    arrays["nYlm"] = Int32[ace.nYlm]

    # Readout
    arrays["n_basis"] = Int32[ace.n_basis]
    arrays["W_readout"] = Float32.(ace.W_readout)

    # Pair potential (if present)
    arrays["has_pair"] = Int32[model.has_pair ? 1 : 0]
    if model.has_pair && !isnothing(model.pair_state)
        pair = model.pair_state
        arrays["pair_n_basis"] = Int32[pair.n_basis]
        arrays["pair_n_species"] = Int32[pair.n_species]
        arrays["pair_n_pairs"] = Int32[pair.n_pairs]
        arrays["pair_n_polys"] = Int32[pair.n_polys]
        arrays["pair_rcut"] = Float32[pair.rcut]
        arrays["pair_agnesi_params"] = Float32.(pair.agnesi_params)
        arrays["pair_poly_A"] = Float32.(pair.poly_A)
        arrays["pair_poly_B"] = Float32.(pair.poly_B)
        arrays["pair_poly_C"] = Float32.(pair.poly_C)
        arrays["pair_W_radial"] = Float32.(pair.W_radial)
        arrays["pair_rcut_outer"] = Float32[pair.rcut_outer]
        arrays["pair_p_outer"] = Int32[pair.p_outer]
        arrays["pair_W_readout"] = Float32.(pair.W_readout)
    end

    # Write NPZ file
    NPZ.npzwrite(path, arrays)

    @info "Constants exported to $path"
    return path
end

## ============================================================================
## Metadata Export
## ============================================================================

"""
    export_metadata(compiled::CompiledACEModel, path::String)

Export model metadata to JSON format.

Contains:
- Compiled shapes
- Species information
- Model architecture summary
- Compilation info
"""
function export_metadata(compiled::CompiledACEModel{T}, path::String) where T

    model = compiled.model
    shapes = compiled.shapes

    metadata = Dict{String, Any}(
        # Version info
        "format_version" => "1.0",
        "exporter" => "ACEExport.jl",

        # Compilation shapes
        "shapes" => Dict(
            "max_atoms" => shapes.max_atoms,
            "max_neigs" => shapes.max_neigs,
            "max_edges" => shapes.max_edges,
        ),

        # Species
        "n_species" => model.n_species,
        "species_Z" => model.species_Z,
        "rcut" => model.rcut,

        # Model architecture
        "has_onebody" => true,
        "has_pair" => model.has_pair,
        "has_ace" => true,

        # ACE config
        "ace" => Dict(
            "n_basis" => model.ace_state.n_basis,
            "n_rnl" => model.ace_state.n_rnl,
            "n_polys" => model.ace_state.n_polys,
            "maxl" => model.ace_state.maxl,
            "nYlm" => model.ace_state.nYlm,
            "n_A" => length(model.ace_state.spec_R),
            "n_AA" => sum(size(m, 1) for m in model.ace_state.specs_mats),
        ),

        # Compiled backends
        "compiled_backends" => String.(compiled.backends),

        # Input shapes for runtime
        "input_shapes" => Dict(
            "positions" => [shapes.max_atoms, 3],
            "atomic_numbers" => [shapes.max_atoms],
            "edge_i" => [shapes.max_edges],
            "edge_j" => [shapes.max_edges],
            "edge_rij" => [shapes.max_edges, 3],
        ),

        # Data types
        "dtype" => string(T),
    )

    # Write JSON
    open(path, "w") do f
        JSON3.pretty(f, metadata)
    end

    @info "Metadata exported to $path"
    return path
end

## ============================================================================
## Loading Functions (for validation)
## ============================================================================

"""
    load_constants(path::String)

Load model constants from NPZ file.
Returns a Dict with all arrays.
"""
function load_constants(path::String)
    return NPZ.npzread(path)
end

"""
    load_metadata(path::String)

Load model metadata from JSON file.
"""
function load_metadata(path::String)
    return JSON3.read(read(path, String))
end
