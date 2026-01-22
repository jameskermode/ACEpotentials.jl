#=
Generate Julia Reference Data for LAMMPS Integration Test
==========================================================

Creates a Si diamond lattice, computes neighbor pairs, and calculates
energy and forces using the same model as the exported VMFB.
Saves reference data for comparison with LAMMPS results.
=#

using LinearAlgebra
using NPZ
using StaticArrays
using Random
using NeighbourLists

println("=" ^ 70)
println("Generate Julia Reference for LAMMPS Integration Test")
println("=" ^ 70)

## ============================================================================
## Energy model (same as export_energy_gradient.jl)
## ============================================================================

"""
Per-edge energy function - must match VMFB exactly.
"""
function per_edge_energy_scalar(rij::AbstractMatrix{T}) where T
    n_edges = size(rij, 1)

    r_sq = sum(rij .^ 2, dims=2)
    r = sqrt.(r_sq .+ T(1e-20))

    r0 = T(2.5)
    rcut = T(5.5)
    y = T(1) .- T(2) ./ (T(1) .+ r ./ r0)

    P0 = ones(T, n_edges, 1)
    P1 = y
    P2 = T(2) .* y .* P1 .- P0
    P3 = T(2) .* y .* P2 .- P1
    P4 = T(2) .* y .* P3 .- P2
    P5 = T(2) .* y .* P4 .- P3

    x = r ./ rcut
    # Mask for valid edges (r² > epsilon means non-zero, not padding)
    # This ensures zero-padded entries contribute 0 energy
    valid_mask = r_sq .> T(1e-10)
    envelope = (T(1) .- x) .^ 2 .* (x .< T(1)) .* valid_mask

    w = T.([0.1, 0.2, 0.15, 0.1, 0.05, 0.02])
    e_edge = (P0 .* w[1] .+ P1 .* w[2] .+ P2 .* w[3] .+
              P3 .* w[4] .+ P4 .* w[5] .+ P5 .* w[6]) .* envelope

    return sum(e_edge)
end

"""
Compute gradient of energy w.r.t. rij using finite differences.
"""
function compute_gradient_fd(rij::Matrix{Float64}; eps=1e-6)
    grad = zeros(size(rij))
    for i in axes(rij, 1)
        for j in axes(rij, 2)
            rij_plus = copy(rij)
            rij_minus = copy(rij)
            rij_plus[i, j] += eps
            rij_minus[i, j] -= eps
            grad[i, j] = (per_edge_energy_scalar(rij_plus) - per_edge_energy_scalar(rij_minus)) / (2 * eps)
        end
    end
    return grad
end

## ============================================================================
## Si Diamond Lattice Generation
## ============================================================================

"""
Generate Si diamond lattice positions.
"""
function generate_si_diamond(nx::Int, ny::Int, nz::Int, a::Float64=5.43)
    # Diamond basis (fractional coordinates)
    basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ]

    positions = Float64[]
    for ix in 0:nx-1
        for iy in 0:ny-1
            for iz in 0:nz-1
                for b in basis
                    x = (ix + b[1]) * a
                    y = (iy + b[2]) * a
                    z = (iz + b[3]) * a
                    push!(positions, x, y, z)
                end
            end
        end
    end

    n_atoms = nx * ny * nz * 8
    return reshape(positions, 3, n_atoms)', [nx * a, ny * a, nz * a]
end

"""
Scatter pair forces to atomic forces.
"""
function scatter_forces(pair_forces::Matrix{Float64}, pairs_i::Vector{Int},
                        pairs_j::Vector{Int}, n_atoms::Int)
    forces = zeros(n_atoms, 3)
    for e in 1:length(pairs_i)
        i = pairs_i[e]
        j = pairs_j[e]
        # Force on i from pair (i,j): -dE/drij
        # Force on j from pair (i,j): +dE/drij
        forces[i, :] .-= pair_forces[e, :]
        forces[j, :] .+= pair_forces[e, :]
    end
    return forces
end

## ============================================================================
## Generate Reference Data
## ============================================================================

println("\n--- Generating Si diamond lattice ---")
nx, ny, nz = 1, 1, 1  # 8 atoms
a = 5.43  # Angstrom

positions, box = generate_si_diamond(nx, ny, nz, a)
n_atoms = size(positions, 1)
println("  Number of atoms: $n_atoms")
println("  Box size: $box")

# Apply small random displacement for non-zero forces
println("\n--- Applying random displacement ---")
Random.seed!(12345)
displacement = 0.05 * randn(n_atoms, 3)
positions_displaced = positions + displacement
println("  Max displacement: $(maximum(abs.(displacement)))")

println("\n--- Building neighbor list using NeighbourLists.jl ---")
# Use cutoff 5.5 to match the VMFB model's rcut parameter
rcut = 5.5

# Wrap positions into primary cell [0, L) - required for NeighbourLists.jl
function wrap_position(p, box)
    wrapped = copy(p)
    for d in 1:3
        while wrapped[d] < 0
            wrapped[d] += box[d]
        end
        while wrapped[d] >= box[d]
            wrapped[d] -= box[d]
        end
    end
    return wrapped
end

positions_wrapped = [wrap_position(positions_displaced[i, :], box) for i in 1:n_atoms]

# Convert positions to SVector format for NeighbourLists.jl
X = [SVector{3,Float64}(positions_wrapped[i]) for i in 1:n_atoms]

# Create cell/box matrix (diagonal for orthorhombic)
cell = SMatrix{3,3,Float64}(
    box[1], 0.0, 0.0,
    0.0, box[2], 0.0,
    0.0, 0.0, box[3]
)

# Build neighbor list with PBC
# NeighbourLists.jl handles all periodic images correctly
nlist = PairList(X, rcut, cell, (true, true, true))  # PBC in all directions

# Extract pairs and rij vectors from PairList
# The PairList stores i, j indices and S shift vectors
# Rij = X[j] + cell * S - X[i]
n_pairs = length(nlist.i)
pairs_i = Int.(nlist.i)
pairs_j = Int.(nlist.j)

rij = zeros(n_pairs, 3)
for idx in 1:n_pairs
    i = nlist.i[idx]
    j = nlist.j[idx]
    s = nlist.S[idx]
    Rij = X[j] + cell * s - X[i]
    rij[idx, :] = Rij
end

println("  Number of pairs: $n_pairs")
println("  Pairs per atom: $(n_pairs / n_atoms)")

println("\n--- Padding to VMFB size (2000 edges) ---")
vmfb_size = 2000
n_actual_pairs = n_pairs

# Pad rij to VMFB size (LAMMPS plugin does the same)
rij_padded = zeros(vmfb_size, 3)
rij_padded[1:n_pairs, :] = rij
println("  Actual pairs: $n_pairs")
println("  Padded size: $vmfb_size")

println("\n--- Computing energy and gradient ---")
# Energy with padded input (matches LAMMPS)
energy = per_edge_energy_scalar(rij_padded)
println("  Total energy (padded): $energy eV")

# Also compute energy of just actual pairs for reference
energy_actual = per_edge_energy_scalar(rij)
println("  Energy (actual pairs only): $energy_actual eV")

# Compute gradient using finite differences on padded input
println("  Computing gradient (finite differences)...")
gradient = compute_gradient_fd(rij_padded)
println("  Gradient shape: $(size(gradient))")

# Extract gradient for actual pairs
gradient_actual = gradient[1:n_pairs, :]
println("  Max gradient (actual pairs): $(maximum(abs.(gradient_actual)))")

# Convert gradient to forces (only for actual pairs, not padding)
pair_forces = -gradient_actual  # Force = -dE/dr
atomic_forces = scatter_forces(pair_forces, pairs_i, pairs_j, n_atoms)
println("  Atomic forces shape: $(size(atomic_forces))")
println("  Max atomic force: $(maximum(abs.(atomic_forces)))")

# Verify Newton's 3rd law
force_sum = sum(atomic_forces, dims=1)
println("  Force sum (should be ~0): $force_sum")

## ============================================================================
## Save Reference Data
## ============================================================================

println("\n--- Saving reference data ---")
output_file = joinpath(@__DIR__, "julia_reference.npz")

npzwrite(output_file, Dict(
    "positions" => positions_displaced,
    "box" => box,
    "pairs_i" => pairs_i .- 1,  # Convert to 0-indexed for Python/LAMMPS
    "pairs_j" => pairs_j .- 1,
    "rij" => rij,
    "rij_padded" => rij_padded,
    "energy" => [energy],
    "energy_actual" => [energy_actual],
    "gradient" => gradient,
    "gradient_actual" => gradient_actual,
    "pair_forces" => pair_forces,
    "atomic_forces" => atomic_forces,
    "n_atoms" => [n_atoms],
    "n_pairs" => [n_pairs],
    "vmfb_size" => [vmfb_size],
    "rcut" => [rcut],
))

println("  Saved to: $output_file")

## ============================================================================
## Also save LAMMPS data file
## ============================================================================

println("\n--- Generating LAMMPS data file ---")
lammps_data = joinpath(@__DIR__, "si_test.data")

open(lammps_data, "w") do f
    println(f, "Si diamond test system for IREE integration test")
    println(f, "")
    println(f, "$n_atoms atoms")
    println(f, "1 atom types")
    println(f, "")
    println(f, "0.0 $(box[1]) xlo xhi")
    println(f, "0.0 $(box[2]) ylo yhi")
    println(f, "0.0 $(box[3]) zlo zhi")
    println(f, "")
    println(f, "Masses")
    println(f, "")
    println(f, "1 28.0855")
    println(f, "")
    println(f, "Atoms # atomic")
    println(f, "")
    for i in 1:n_atoms
        x, y, z = positions_wrapped[i]  # Use wrapped positions for LAMMPS
        println(f, "$i 1 $x $y $z")
    end
end

println("  Saved LAMMPS data file: $lammps_data")

## ============================================================================
## Summary
## ============================================================================

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println("""
Generated Julia reference data for LAMMPS integration test:

  System: Si diamond $(nx)x$(ny)x$(nz) = $n_atoms atoms
  Box: $(box[1]) x $(box[2]) x $(box[3]) Å
  Cutoff: $rcut Å
  Actual pairs: $n_pairs (using NeighbourLists.jl)
  VMFB padded size: $vmfb_size

  Energy (padded, matches LAMMPS): $energy eV
  Energy (actual pairs only): $energy_actual eV
  Max force: $(maximum(abs.(atomic_forces))) eV/Å
  Force sum: $(maximum(abs.(force_sum))) (Newton's 3rd law check)

Output files:
  - julia_reference.npz (reference data)
  - si_test.data (LAMMPS data file)
""")
println("=" ^ 70)
