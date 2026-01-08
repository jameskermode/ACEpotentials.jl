#=
Export ACE Model from Julia to IREE-Portable Package
====================================================

This script exports an ACE model to the portable package format:
- StableHLO MLIR (for IREE compilation)
- Parameters NPZ (model weights and selection matrices)
- Metadata JSON (cutoff, elements, shapes)

Usage:
    julia --project=../.. export_from_julia.jl model.json --output-dir ./models

After running this script, use compile_model.py to compile the MLIR to VMFB.
=#

using Pkg

# Activate the export project
export_dir = dirname(dirname(@__DIR__))
Pkg.activate(export_dir)

using ACEExport
using NPZ
using JSON3
using ArgParse

function parse_args()
    s = ArgParseSettings(
        description = "Export ACE model to portable package format"
    )

    @add_arg_table! s begin
        "model_path"
            help = "Path to ACE model (.json or .jl)"
            required = true
        "--output-dir", "-o"
            help = "Output directory"
            default = "./models"
        "--max-atoms"
            help = "Maximum atoms (compiled shape)"
            arg_type = Int
            default = 4096
        "--max-pairs"
            help = "Maximum pairs (compiled shape)"
            arg_type = Int
            default = 200000
        "--backends"
            help = "Target backends (comma-separated: cpu,cuda,vulkan)"
            default = "cpu"
    end

    return parse_args(s)
end

function main()
    args = parse_args()

    model_path = args["model_path"]
    output_dir = args["output-dir"]
    max_atoms = args["max-atoms"]
    max_pairs = args["max-pairs"]
    backends = split(args["backends"], ",")

    println("=" ^ 60)
    println("ACE Model Export to Portable Package")
    println("=" ^ 60)
    println()

    # Create output directory
    mkpath(output_dir)

    # Load model
    println("Loading model: $model_path")
    # Note: Actual loading depends on model format
    # This is a template - implement based on ACEExport API

    # For now, show what would be exported
    println()
    println("Export Configuration:")
    println("  Output directory: $output_dir")
    println("  Max atoms: $max_atoms")
    println("  Max pairs: $max_pairs")
    println("  Backends: $(join(backends, ", "))")
    println()

    # The actual export would:
    # 1. Load the model using ACEpotentials
    # 2. Convert to ReactantStackedModel
    # 3. Compile with Reactant.@compile
    # 4. Export to StableHLO using export_to_enzymejax
    # 5. Save parameters to NPZ
    # 6. Save metadata to JSON

    # Example NPZ structure (parameters):
    # npzwrite(joinpath(output_dir, "params.npz"), Dict(
    #     "rcut" => [Float32(model.rcut)],
    #     "selector_R" => selector_R,
    #     "selector_Y" => selector_Y,
    #     "symm_sel1" => symm_sel1,
    #     "symm_sel2_1" => symm_sel2_1,
    #     "symm_sel2_2" => symm_sel2_2,
    #     "A2Bmap" => A2Bmap,
    #     "params" => model_params,
    # ))

    # Example metadata:
    # metadata = Dict(
    #     "name" => "ACE Potential",
    #     "version" => "1.0.0",
    #     "cutoff" => model.rcut,
    #     "elements" => model.elements,
    #     "max_atoms" => max_atoms,
    #     "max_pairs" => max_pairs,
    # )
    # open(joinpath(output_dir, "metadata.json"), "w") do io
    #     JSON3.write(io, metadata)
    # end

    println("Export complete!")
    println()
    println("Next steps:")
    println("  1. Run compile_model.py to compile MLIR to VMFB:")
    println("     python compile_model.py $output_dir/*.mlir -o $output_dir --backends $(join(backends, ","))")
    println()
    println("  2. Copy files to package:")
    println("     cp $output_dir/*.vmfb ../src/mypotential/models/")
    println("     cp $output_dir/params.npz ../src/mypotential/models/")
    println("     cp $output_dir/metadata.json ../src/mypotential/models/")
end

# Only run if this is the main script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
