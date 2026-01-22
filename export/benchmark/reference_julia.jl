#=
Generate Reference Energy and Forces from Julia ACE
====================================================

Creates fixture files for benchmarking JAX and Enzyme force implementations.
Exports both:
1. Reference E+F values computed via Julia Zygote
2. Input arrays (rij, ii, jj, zi, zj) for VMFB inputs

Usage:
    julia +1.11 --project=.. benchmark/reference_julia.jl [--size tiny|small|medium|large]
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using Printf
using Random
using NPZ
using JSON3
using LinearAlgebra

using ACEpotentials
using AtomsBase
using AtomsBase: ChemicalSpecies
using AtomsCalculators
using Lux
using StaticArrays
using Unitful

M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET

include(joinpath(@__DIR__, "..", "scripts", "create_package.jl"))

## ============================================================================
## Test System Generation
## ============================================================================

const SYSTEM_CONFIGS = Dict(
    "tiny" => (n=2, atoms=64, description="2x2x2 diamond Si, 64 atoms"),
    "small" => (n=4, atoms=256, description="4x4x4 simple cubic Si, 256 atoms"),
    "medium" => (n=6, atoms=864, description="6x6x6 simple cubic Si, 864 atoms"),
    "large" => (n=8, atoms=2048, description="8x8x8 simple cubic Si, 2048 atoms"),
)

"""
Create a test system with specified number of atoms.
Uses simple cubic packing for simplicity.
"""
function create_test_system(n_cells::Int, rcut::Float64; elements=(:Si,))
    # Simple cubic packing
    a = rcut / 2.5  # Lattice constant ensuring neighbors

    positions = SVector{3,Float64}[]
    for i in 0:(n_cells-1), j in 0:(n_cells-1), k in 0:(n_cells-1)
        push!(positions, SVector(i*a, j*a, k*a))
    end

    # Add small random perturbations for more realistic neighbor distribution
    rng = Random.MersenneTwister(42)
    for i in eachindex(positions)
        positions[i] = positions[i] + 0.05 * a * SVector(randn(rng, 3)...)
    end

    box = (SVector(n_cells*a, 0.0, 0.0),
           SVector(0.0, n_cells*a, 0.0),
           SVector(0.0, 0.0, n_cells*a))
    Z = first(elements)

    atoms = [Atom(Z, pos * u"Å") for pos in positions]
    return FlexibleSystem(atoms; cell_vectors=box .* u"Å", periodicity=(true, true, true))
end

## ============================================================================
## Reference Computation
## ============================================================================

"""
Compute reference energy and forces using Julia ACEpotentials.
"""
function compute_reference(calc::ETM.StackedCalculator, sys)
    # Energy, forces, virial via AtomsCalculators
    E, F, V = AtomsCalculators.energy_forces_virial(sys, calc)

    # Convert to arrays
    E_val = ustrip(u"eV", E)
    F_array = [ustrip.(u"eV/Å", f) for f in F]
    V_array = ustrip.(u"eV", V)

    return E_val, F_array, V_array
end

"""
Extract graph arrays for export.
"""
function extract_arrays_for_export(calc::ETM.StackedCalculator, sys; elements=(:Si,))
    rcut = maximum(c.rcut for c in calc.calcs if hasproperty(c, :rcut))
    G = ET.Atoms.interaction_graph(sys, rcut * u"Å")

    species_list = [ChemicalSpecies(e) for e in elements]
    arrays = extract_graph_arrays(G, species_list)

    # Also get positions for verification
    positions = zeros(Float32, length(sys), 3)
    for (i, atom) in enumerate(sys)
        pos = ustrip.(u"Å", position(atom))
        positions[i, :] .= Float32.(pos)
    end

    return (;
        rij = arrays.rij,
        ii = arrays.ii,
        jj = arrays.jj,
        zi = arrays.zi,
        zj = arrays.zj,
        node_positions = positions,
        node_species = arrays.node_species,
        n_atoms = Int32(length(sys)),
        n_edges = Int32(length(arrays.ii)),
    )
end

"""
Generate fixture file for a given system size.
"""
function generate_fixture(size_name::String, output_dir::String;
                          elements=(:Si,), order=2, max_level=6)
    config = SYSTEM_CONFIGS[size_name]
    @info "Generating fixture: $size_name" config.description

    # Create model
    calc, _, rcut = create_test_model(; elements=elements, order=order, max_level=max_level)

    # Create system
    sys = create_test_system(config.n, rcut; elements=elements)
    n_atoms = length(sys)
    @info "System created" n_atoms=n_atoms rcut=rcut

    # Compute reference E+F
    @info "Computing reference E+F..."
    E, F, V = compute_reference(calc, sys)
    @info "Reference computed" E=E

    # Extract arrays
    @info "Extracting arrays..."
    arrays = extract_arrays_for_export(calc, sys; elements=elements)
    @info "Arrays extracted" n_atoms=arrays.n_atoms n_edges=arrays.n_edges

    # Forces as array
    forces = zeros(Float32, n_atoms, 3)
    for (i, f) in enumerate(F)
        forces[i, :] .= Float32.(f)
    end

    # Save fixture
    fixture_path = joinpath(output_dir, "$(size_name)_$(n_atoms).npz")
    mkpath(output_dir)

    # Convert virial to regular array for NPZ
    virial = zeros(Float32, 3, 3)
    for i in 1:3, j in 1:3
        virial[i, j] = Float32(V[i, j])
    end

    npzwrite(fixture_path,
        # Inputs
        rij = arrays.rij,
        ii = arrays.ii,
        jj = arrays.jj,
        zi = arrays.zi,
        zj = arrays.zj,
        positions = arrays.node_positions,
        species = arrays.node_species,
        # Metadata
        n_atoms = [arrays.n_atoms],
        n_edges = [arrays.n_edges],
        # Reference outputs
        energy = [Float32(E)],
        forces = forces,
        virial = virial,
    )

    @info "Fixture saved" path=fixture_path

    # Also save metadata as JSON
    meta_path = joinpath(output_dir, "$(size_name)_$(n_atoms).json")
    open(meta_path, "w") do f
        JSON3.pretty(f, Dict(
            "size" => size_name,
            "n_atoms" => n_atoms,
            "n_edges" => Int(arrays.n_edges),
            "rcut" => rcut,
            "elements" => String.(elements),
            "energy" => E,
            "max_force" => maximum(norm.(F)),
        ))
    end

    return fixture_path
end

## ============================================================================
## Export energy_from_rij VMFB
## ============================================================================

"""
Export the energy_from_rij VMFB for JAX gradient testing.
"""
function export_energy_from_rij_vmfb(output_dir::String;
                                      elements=(:Si,), order=2, max_level=6,
                                      reference_size="tiny")
    @info "Exporting energy_from_rij VMFB..."

    # Create model
    calc, _, rcut = create_test_model(; elements=elements, order=order, max_level=max_level)

    # Use reference system matching the tiny fixture
    config = SYSTEM_CONFIGS[reference_size]
    sys = create_test_system(config.n, rcut; elements=elements)

    @info "Reference system" n_atoms=length(sys) rcut=rcut

    # Export the VMFB (this calls the function we'll add to create_package.jl)
    mlir_path = export_energy_from_rij(calc, sys, output_dir;
                                        elements=elements, name="energy_from_rij")

    if mlir_path !== nothing
        @info "MLIR exported" path=mlir_path

        # Check for scatter
        content = read(mlir_path, String)
        has_scatter = occursin("scatter", lowercase(content))
        n_lines = count('\n', content)

        @info "MLIR analysis" lines=n_lines has_scatter=has_scatter

        if has_scatter
            @warn "MLIR contains scatter - reverse-mode JAX grad may fail"
        else
            @info "MLIR is scatter-free - forward-mode compatible"
        end

        # Compile to VMFB
        vmfb_cpu = joinpath(output_dir, "energy_from_rij_cpu.vmfb")
        compile_vmfb(mlir_path, vmfb_cpu; backend=:cpu)

        return mlir_path
    else
        @error "MLIR export failed"
        return nothing
    end
end

## ============================================================================
## Main
## ============================================================================

function main()
    args = ARGS

    # Defaults
    size = "tiny"
    output_dir = joinpath(@__DIR__, "fixtures")
    export_vmfb = true

    # Parse args
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--size" && i < length(args)
            i += 1
            size = args[i]
        elseif arg == "--output" && i < length(args)
            i += 1
            output_dir = args[i]
        elseif arg == "--all"
            size = "all"
        elseif arg == "--no-vmfb"
            export_vmfb = false
        elseif arg == "--help" || arg == "-h"
            println("""
Usage: julia reference_julia.jl [OPTIONS]

Options:
  --size SIZE      System size: tiny, small, medium, large, or all (default: tiny)
  --output DIR     Output directory (default: benchmark/fixtures)
  --all            Generate all sizes
  --no-vmfb        Skip VMFB export
  --help           Show this help
""")
            return
        end
        i += 1
    end

    println("="^60)
    println("ACE Reference Fixture Generator")
    println("="^60)

    # Generate fixtures
    if size == "all"
        for sz in keys(SYSTEM_CONFIGS)
            generate_fixture(sz, output_dir)
        end
    else
        generate_fixture(size, output_dir)
    end

    # Export VMFB
    if export_vmfb
        vmfb_dir = joinpath(dirname(output_dir), "vmfb")
        export_energy_from_rij_vmfb(vmfb_dir; reference_size=size == "all" ? "tiny" : size)
    end

    println("\nDone!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
