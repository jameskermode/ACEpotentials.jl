#=
Export Bucket Energy+Gradient VMFBs for LAMMPS
==============================================

Creates multiple energy+gradient VMFBs at different "bucket" sizes.
At runtime, LAMMPS selects the smallest bucket that fits the system.

Usage:
    julia +1.11 --project=export export/benchmark/export_bucket_energy_gradient.jl
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using Reactant
using Reactant: @compile
using Enzyme
using LinearAlgebra
using NPZ

println("=" ^ 70)
println("Exporting Bucket Energy+Gradient VMFBs for LAMMPS")
println("=" ^ 70)

# Bucket configurations: (max_edges, approx_max_atoms)
# Based on ~26 edges/atom average for typical ACE cutoffs
buckets = [
    (2000, 77),      # Small: ~77 atoms
    (10000, 385),    # Medium: ~385 atoms
    (50000, 1923),   # Large: ~1923 atoms
    (100000, 3846),  # XLarge: ~3846 atoms
]

output_dir = joinpath(@__DIR__, "bucket_energy_gradient")
mkpath(output_dir)

## ============================================================================
## Energy function (same as export_energy_gradient.jl)
## ============================================================================

"""
Per-edge energy function using Float64 for type matching with LAMMPS F_FLOAT (double).
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
    valid_mask = r_sq .> T(1e-10)
    envelope = (T(1) .- x) .^ 2 .* (x .< T(1)) .* valid_mask

    w = T.([0.1, 0.2, 0.15, 0.1, 0.05, 0.02])
    e_edge = (P0 .* w[1] .+ P1 .* w[2] .+ P2 .* w[3] .+
              P3 .* w[4] .+ P4 .* w[5] .+ P5 .* w[6]) .* envelope

    return sum(e_edge)
end

## ============================================================================
## Combined energy + gradient function using Enzyme
## ============================================================================

function compute_energy_and_gradient(rij::AbstractMatrix{T}) where T
    # Compute energy
    energy = per_edge_energy_scalar(rij)

    # Compute gradient via Enzyme
    drij = zero(rij)
    Enzyme.autodiff(Reverse, Const(per_edge_energy_scalar), Active, Duplicated(copy(rij), drij))

    # Return as tuple (energy as 1-element array for IREE compatibility)
    return (reshape([energy], 1), drij)
end

## ============================================================================
## Export each bucket
## ============================================================================

iree_compile = expanduser("~/iree/bin/iree-compile")
iree_compile_cuda = expanduser("~/iree-cuda/bin/iree-compile")

if !isfile(iree_compile)
    iree_compile = Sys.which("iree-compile")
end

for (max_edges, max_atoms) in buckets
    println("\n" * "=" ^ 50)
    println("Bucket: $max_edges edges (≤$max_atoms atoms)")
    println("=" ^ 50)

    bucket_dir = joinpath(output_dir, "bucket_$(max_edges)")
    mkpath(bucket_dir)

    # Test native computation first
    println("\n  Testing native computation...")
    rij_test = randn(Float64, max_edges, 3) .* 2.0
    E_native = per_edge_energy_scalar(rij_test)
    energy_arr, grad_native = compute_energy_and_gradient(rij_test)
    println("    Energy: $(energy_arr[1])")
    println("    Gradient shape: $(size(grad_native))")
    println("    Max gradient: $(maximum(abs.(grad_native)))")

    # Export MLIR
    println("\n  Exporting MLIR via Reactant...")
    rij_ra = Reactant.to_rarray(rij_test)

    try
        t0 = time()
        Reactant.Serialization.export_to_enzymejax(
            compute_energy_and_gradient,
            rij_ra;
            output_dir=bucket_dir,
            function_name="energy_gradient"
        )
        t_export = time() - t0
        println("    Export time: $(round(t_export, digits=1))s")
    catch e
        println("    EXPORT FAILED: $e")
        continue
    end

    # Find the MLIR file
    mlir_files = filter(f -> endswith(f, ".mlir") && !contains(f, "inputs"), readdir(bucket_dir))
    if isempty(mlir_files)
        println("    No MLIR file found!")
        continue
    end

    # Sort by modification time and take latest
    mlir_files_with_time = [(f, mtime(joinpath(bucket_dir, f))) for f in mlir_files]
    sort!(mlir_files_with_time, by=x->x[2], rev=true)
    mlir_path = joinpath(bucket_dir, mlir_files_with_time[1][1])

    # Verify MLIR has correct size
    mlir_content = read(mlir_path, String)
    m = match(r"tensor<(\d+)x3xf64>", mlir_content)
    if !isnothing(m)
        found_size = parse(Int, m.captures[1])
        if found_size != max_edges
            println("    WARNING: MLIR has size $found_size, expected $max_edges")
        else
            println("    MLIR size verified: $max_edges edges")
        end
    end

    # Compile CPU VMFB
    vmfb_cpu = joinpath(bucket_dir, "energy_gradient_f64_cpu.vmfb")
    println("\n  Compiling CPU VMFB...")
    try
        run(pipeline(`$iree_compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu $mlir_path -o $vmfb_cpu`, stderr=devnull))
        println("    CPU VMFB: $(filesize(vmfb_cpu)) bytes")
    catch e
        println("    CPU compilation FAILED: $e")
    end

    # Compile CUDA VMFB
    if isfile(iree_compile_cuda)
        vmfb_cuda = joinpath(bucket_dir, "energy_gradient_f64_cuda.vmfb")
        println("  Compiling CUDA VMFB...")
        try
            run(pipeline(`$iree_compile_cuda --iree-input-type=stablehlo --iree-hal-target-backends=cuda $mlir_path -o $vmfb_cuda`, stderr=devnull))
            println("    CUDA VMFB: $(filesize(vmfb_cuda)) bytes")
        catch e
            println("    CUDA compilation FAILED: $e")
        end
    else
        println("  Skipping CUDA VMFB (no iree-cuda compiler)")
    end

    # Save metadata
    npzwrite(joinpath(bucket_dir, "metadata.npz"), Dict(
        "max_edges" => [max_edges],
        "max_atoms" => [max_atoms],
        "rcut" => [5.5],
    ))

    # Save test data for verification
    npzwrite(joinpath(bucket_dir, "test_data.npz"), Dict(
        "rij" => rij_test,
        "energy" => [E_native],
        "gradient" => grad_native,
    ))
end

## ============================================================================
## Summary
## ============================================================================

println("\n" * "=" ^ 70)
println("BUCKET ENERGY+GRADIENT VMFBs CREATED")
println("=" ^ 70)

println("\nBucket sizes and files:")
for (max_edges, max_atoms) in buckets
    bucket_dir = joinpath(output_dir, "bucket_$(max_edges)")
    vmfb_cpu = joinpath(bucket_dir, "energy_gradient_f64_cpu.vmfb")
    vmfb_cuda = joinpath(bucket_dir, "energy_gradient_f64_cuda.vmfb")

    cpu_size = isfile(vmfb_cpu) ? "$(filesize(vmfb_cpu)) bytes" : "NOT CREATED"
    cuda_size = isfile(vmfb_cuda) ? "$(filesize(vmfb_cuda)) bytes" : "NOT CREATED"

    println("  bucket_$(max_edges) (≤$max_atoms atoms):")
    println("    CPU:  $cpu_size")
    println("    CUDA: $cuda_size")
end

println("""

Output directory: $output_dir

VMFB Interface (Float64):
  Input:  rij (n_edges, 3) - edge displacement vectors (f64)
          Note: IREE expects (3, n_edges) due to column-major conversion
  Output: (energy (1,) f64, gradient (3, n_edges) f64)

Usage in LAMMPS:
  pair_style iree/kk <vmfb_path> cuda
  # Plugin should auto-select appropriate bucket based on neighbor count

Usage in Python:
  module = iree_rt.load_vm_flatbuffer_file("bucket_X/energy_gradient_f64_cpu.vmfb", "local-task")
  energy, grad = module.main(rij.T.astype(np.float64))  # Input: (3, n_edges) f64
  forces = -grad.T                                       # Output: (n_edges, 3) f64
""")
println("=" ^ 70)
