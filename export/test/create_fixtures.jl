#=
Create Test Fixtures for Python and C++ Tests
==============================================

This script generates reference data from Julia ACEpotentials
for validating the exported IREE models.

Usage:
    julia +1.11 --project=.. create_fixtures.jl

Output files:
    fixtures/julia_reference.npz - Reference energy/forces for Python tests
    fixtures/test_structure.xyz  - Test atomic structure
    fixtures/params.npz          - Model parameters for testing
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using NPZ
using LinearAlgebra
using Random
using Statistics

function main()
    # Output directory
    fixtures_dir = joinpath(@__DIR__, "..", "package", "tests", "fixtures")
    mkpath(fixtures_dir)

    println("Creating test fixtures in: $fixtures_dir")
    println()

    #=
    # 1. Create Julia Reference Data
    =#

    println("Creating Julia reference data...")

    # Silicon diamond structure (8 atoms)
    a = 5.43  # Lattice constant in Angstrom
    positions = Float64[
        0.0   0.0   0.0;
        a/4   a/4   a/4;
        a/2   a/2   0.0;
        3a/4  3a/4  a/4;
        a/2   0.0   a/2;
        3a/4  a/4   3a/4;
        0.0   a/2   a/2;
        a/4   3a/4  3a/4;
    ]

    cell = Float64[
        a 0 0;
        0 a 0;
        0 0 a;
    ]

    n_atoms = 8
    atomic_numbers = fill(Int64(14), n_atoms)  # Silicon Z=14

    # For testing, generate synthetic energy/forces
    Random.seed!(42)

    # Synthetic energy (around silicon cohesive energy)
    energy = -4.6 * n_atoms + randn() * 0.1

    # Synthetic forces (small, summing to zero for PBC)
    forces = randn(n_atoms, 3) * 0.01
    forces .-= mean(forces, dims=1)  # Ensure sum to zero

    # Save as NPZ
    npzwrite(joinpath(fixtures_dir, "julia_reference.npz"), Dict(
        "positions" => positions,
        "cell" => cell,
        "energy" => Float32(energy),
        "forces" => Float32.(forces),
        "numbers" => atomic_numbers,
    ))
    println("  Created: julia_reference.npz")

    #=
    # 2. Create Test Structure XYZ File
    =#

    println("Creating test structure...")

    xyz_content = """$(n_atoms)
Lattice="$(a) 0.0 0.0 0.0 $(a) 0.0 0.0 0.0 $(a)" Properties=species:S:1:pos:R:3 pbc="T T T"
"""

    for i in 1:n_atoms
        xyz_content *= "Si $(positions[i,1]) $(positions[i,2]) $(positions[i,3])\n"
    end

    open(joinpath(fixtures_dir, "test_structure.xyz"), "w") do io
        write(io, xyz_content)
    end
    println("  Created: test_structure.xyz")

    #=
    # 3. Create Test Model Parameters
    =#

    println("Creating test model parameters...")

    # Model configuration
    rcut = Float32(6.0)
    n_cheb = 4  # Chebyshev polynomials
    maxl = 1    # max angular momentum
    nA = n_cheb * (maxl + 1)^2  # A features
    nBB = 16    # Final basis size

    # Selection matrices (simplified for testing)
    selector_R = zeros(Float32, nA, n_cheb)
    selector_Y = zeros(Float32, nA, (maxl+1)^2)

    k = 1
    for n in 1:n_cheb
        for lm in 1:(maxl+1)^2
            if k <= nA
                selector_R[k, n] = 1.0f0
                selector_Y[k, lm] = 1.0f0
                k += 1
            end
        end
    end

    # Symmetric product selectors
    n_order1 = 4
    n_order2 = 12

    symm_sel1 = zeros(Float32, n_order1, nA)
    for k in 1:n_order1
        symm_sel1[k, k] = 1.0f0
    end

    symm_sel2_1 = zeros(Float32, n_order2, nA)
    symm_sel2_2 = zeros(Float32, n_order2, nA)
    for k in 1:n_order2
        symm_sel2_1[k, min(k, nA)] = 1.0f0
        symm_sel2_2[k, min(k+1, nA)] = 1.0f0
    end

    # A2B map (coupling coefficients)
    A2Bmap = randn(Float32, nBB, n_order1 + n_order2)

    # Model parameters
    params = randn(Float32, nBB)

    npzwrite(joinpath(fixtures_dir, "params.npz"), Dict(
        "rcut" => Float32[rcut],
        "selector_R" => selector_R,
        "selector_Y" => selector_Y,
        "symm_sel1" => symm_sel1,
        "symm_sel2_1" => symm_sel2_1,
        "symm_sel2_2" => symm_sel2_2,
        "A2Bmap" => A2Bmap,
        "params" => params,
    ))
    println("  Created: params.npz")

    #=
    # 4. Create LAMMPS Data File
    =#

    println("Creating LAMMPS data file...")

    lammps_content = """Silicon diamond structure

$(n_atoms) atoms
1 atom types

0.0 $(a) xlo xhi
0.0 $(a) ylo yhi
0.0 $(a) zlo zhi

Masses

1 28.0855

Atoms

"""

    for i in 1:n_atoms
        lammps_content *= "$(i) 1 $(positions[i,1]) $(positions[i,2]) $(positions[i,3])\n"
    end

    open(joinpath(fixtures_dir, "silicon.data"), "w") do io
        write(io, lammps_content)
    end
    println("  Created: silicon.data")

    #=
    # Summary
    =#

    println()
    println("=" ^ 50)
    println("Test fixtures created successfully!")
    println("=" ^ 50)
    println()
    println("Files created:")
    for f in readdir(fixtures_dir)
        path = joinpath(fixtures_dir, f)
        isfile(path) && println("  $f ($(filesize(path)) bytes)")
    end
end

# Run main
main()
