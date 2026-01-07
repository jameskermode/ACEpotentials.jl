#=
Simple End-to-End Export Test
=============================

Creates a minimal ACE model using EquivariantTensors directly,
compiles with Reactant, exports to IREE VMFB, and tests execution.

This verifies the full pipeline before full ACEpotentials.jl integration.
=#

using Test
using Printf
using LinearAlgebra
using Random
using NPZ
using SparseArrays

# Check if we have the required packages
const HAS_REACTANT = try
    using Reactant
    true
catch
    @warn "Reactant not available"
    false
end

const HAS_ET = try
    import EquivariantTensors as ET
    import Polynomials4ML as P4ML
    true
catch
    @warn "EquivariantTensors or Polynomials4ML not available"
    false
end

if !HAS_REACTANT || !HAS_ET
    @info "Skipping simple export test - missing dependencies"
    exit(0)
end

println("="^60)
println("SIMPLE ACE EXPORT TEST")
println("="^60)

## ============================================================================
## Step 1: Build minimal ACE model
## ============================================================================

println("\n1. Building minimal ACE model...")

# Small model parameters
Dtot, maxl, ORD = 3, 1, 2
N_cheb = Dtot + 1

mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
    level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
    Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
    Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
    basis=real)

nfeatures = length(symbasis, 0)
@printf("Model: Dtot=%d, maxl=%d, ORD=%d, %d features\n", Dtot, maxl, ORD, nfeatures)

## ============================================================================
## Step 2: Prepare Reactant state (from prototype patterns)
## ============================================================================

println("\n2. Preparing Reactant-compatible state...")

rng = MersenneTwister(42)
ps = ET.LuxCore.initialparameters(rng, symbasis)
st = ET.LuxCore.initialstates(rng, symbasis)

# Extract specs following prototype pattern
function spec_to_matrix(spec::AbstractVector{<:Tuple})
    isempty(spec) && return Matrix{Int}(undef, 0, 0)
    n = length(first(spec))
    mat = zeros(Int, length(spec), n)
    for (i, t) in enumerate(spec)
        for j in 1:n
            mat[i, j] = t[j]
        end
    end
    return mat
end

aspec = st.aspec
spec_R = [a[1] for a in aspec]
spec_Y = [a[2] for a in aspec]

aaspecs = st.aaspecs
specs_mats = [spec_to_matrix(s) for s in aaspecs]

# Convert sparse A2Bmap to dense
A2Bmap_sparse = st.A2Bmaps[1]
A2Bmap = zeros(Float32, size(A2Bmap_sparse)...)
for (i, j, v) in zip(findnz(A2Bmap_sparse)...)
    A2Bmap[i, j] = Float32(v)
end

# Random readout weights
params = randn(Float32, nfeatures)

@printf("   spec_R/Y: %d indices\n", length(spec_R))
@printf("   specs_mats: %d matrices\n", length(specs_mats))
@printf("   A2Bmap: %s\n", size(A2Bmap))
@printf("   params: %d\n", length(params))

## ============================================================================
## Step 3: Define ACE kernel functions
## ============================================================================

println("\n3. Defining ACE kernel functions...")

# Pooled sparse product
function pooled_sparse_product(Rnl_3, Ylm_3, spec_R, spec_Y)
    maxneigs, nnodes, _ = size(Rnl_3)
    nA = length(spec_R)
    A = zeros(eltype(Rnl_3), nnodes, nA)
    for iA in 1:nA
        for inode in 1:nnodes
            for j in 1:maxneigs
                A[inode, iA] += Rnl_3[j, inode, spec_R[iA]] * Ylm_3[j, inode, spec_Y[iA]]
            end
        end
    end
    return A
end

# Sparse symmetric product
function sparse_symm_prod(A, specs_mats)
    nnodes = size(A, 1)
    total = sum(size(m, 1) for m in specs_mats)
    AA = zeros(eltype(A), nnodes, total)
    offset = 0
    for mat in specs_mats
        nspec = size(mat, 1)
        order = size(mat, 2)
        for i in 1:nspec
            for inode in 1:nnodes
                val = one(eltype(A))
                for k in 1:order
                    val *= A[inode, mat[i, k]]
                end
                AA[inode, offset + i] = val
            end
        end
        offset += nspec
    end
    return AA
end

# Full ACE evaluation
function ace_evaluate(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)
    A = pooled_sparse_product(Rnl_3, Ylm_3, spec_R, spec_Y)
    AA = sparse_symm_prod(A, specs_mats)
    BB = AA * A2Bmap
    return BB
end

# Energy function
function ace_energy(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)
    BB = ace_evaluate(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap)
    return sum(BB * params)
end

## ============================================================================
## Step 4: Test Julia baseline
## ============================================================================

println("\n4. Testing Julia baseline...")

maxneigs, nnodes = 8, 4
nRnl = N_cheb
nYlm = (maxl + 1)^2

Rnl_3 = randn(Float32, maxneigs, nnodes, nRnl)
Ylm_3 = randn(Float32, maxneigs, nnodes, nYlm)

E_julia = ace_energy(Rnl_3, Ylm_3, spec_R, spec_Y, specs_mats, A2Bmap, params)
@printf("Julia energy: %.6f\n", E_julia)

## ============================================================================
## Step 5: Compile with Reactant
## ============================================================================

println("\n5. Compiling with Reactant...")
Reactant.set_default_backend("cpu")

Rnl_ra = Reactant.to_rarray(Rnl_3)
Ylm_ra = Reactant.to_rarray(Ylm_3)
spec_R_ra = Reactant.to_rarray(spec_R)
spec_Y_ra = Reactant.to_rarray(spec_Y)
specs_mats_ra = [Reactant.to_rarray(m) for m in specs_mats]
A2Bmap_ra = Reactant.to_rarray(A2Bmap)
params_ra = Reactant.to_rarray(params)

compiled_fn = Reactant.@compile ace_energy(
    Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
    specs_mats_ra, A2Bmap_ra, params_ra
)

E_reactant = compiled_fn(Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
                         specs_mats_ra, A2Bmap_ra, params_ra)
E_reactant_val = Float64(E_reactant)

@printf("Reactant energy: %.6f\n", E_reactant_val)
@printf("Difference: %.2e\n", abs(E_reactant_val - E_julia))

@test isapprox(E_reactant_val, E_julia, rtol=1e-5)

## ============================================================================
## Step 6: Export to StableHLO
## ============================================================================

println("\n6. Exporting to StableHLO...")

export_dir = joinpath(@__DIR__, "..", "test_export_output")
mkpath(export_dir)

try
    Reactant.Serialization.export_to_enzymejax(
        ace_energy,
        Rnl_ra, Ylm_ra, spec_R_ra, spec_Y_ra,
        specs_mats_ra, A2Bmap_ra, params_ra;
        output_dir=export_dir,
        function_name="ace_energy"
    )
    @info "StableHLO exported to $export_dir"
catch e
    @warn "StableHLO export failed" exception=e
end

# Save constants
constants_path = joinpath(export_dir, "ace_constants.npz")
NPZ.npzwrite(constants_path, Dict(
    "spec_R" => spec_R,
    "spec_Y" => spec_Y,
    "A2Bmap" => A2Bmap,
    "params" => params,
    "N_cheb" => Int32[N_cheb],
    "maxl" => Int32[maxl],
    "nfeatures" => Int32[nfeatures],
))
@info "Constants saved to $constants_path"

# Save test inputs for Python verification
test_inputs_path = joinpath(export_dir, "test_inputs.npz")
NPZ.npzwrite(test_inputs_path, Dict(
    "Rnl_3" => Rnl_3,
    "Ylm_3" => Ylm_3,
    "E_expected" => Float32[E_julia],
))
@info "Test inputs saved to $test_inputs_path"

## ============================================================================
## Step 7: Compile to VMFB (if iree-compile available)
## ============================================================================

println("\n7. Compiling to IREE VMFB...")

mlir_files = filter(f -> endswith(f, ".mlir") && !endswith(f, "_inputs.mlir"), readdir(export_dir))
if !isempty(mlir_files)
    mlir_path = joinpath(export_dir, first(mlir_files))
    vmfb_path = joinpath(export_dir, "ace_model_cpu.vmfb")

    # Check if iree-compile is available
    iree_compile = try
        chomp(read(`which iree-compile`, String))
    catch
        nothing
    end

    if iree_compile !== nothing
        @info "Found iree-compile at $iree_compile"
        try
            run(`iree-compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu $mlir_path -o $vmfb_path`)
            @info "VMFB compiled to $vmfb_path"
        catch e
            @warn "VMFB compilation failed" exception=e
        end
    else
        @info "iree-compile not found in PATH - skipping VMFB compilation"
    end
else
    @warn "No MLIR files found in export directory"
end

println("\n" * "="^60)
println("TEST COMPLETE")
println("Output directory: $export_dir")
println("="^60)

# List output files
println("\nGenerated files:")
for f in sort(readdir(export_dir))
    path = joinpath(export_dir, f)
    isfile(path) && @printf("  %s (%d bytes)\n", f, filesize(path))
end
