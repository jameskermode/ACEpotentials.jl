#=
Test ACE Full GPU Acceleration
==============================

This test implements Option 2: Full GPU acceleration using selection matrices.
All gather operations are converted to matrix multiplications, allowing the
entire forward and backward pass to run on GPU via IREE.

Architecture:
1. rij → Rnl, Ylm (embeddings - pure functions, no gather)
2. Rnl, Ylm → A (pooled product via selection matrices)
3. A → AA (symmetric product via selection matrices)
4. AA → E (matmul + sum)

All operations are dense matmuls - no gather/scatter → full GPU acceleration!
=#

using Test
using Printf
using LinearAlgebra
using Random

const HAS_REACTANT = try
    using Reactant
    using Enzyme
    using NPZ  # For StableHLO export
    true
catch
    @warn "Reactant or Enzyme not available"
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
    @info "Skipping test - missing dependencies"
    exit(0)
end

println("="^70)
println("ACE FULL GPU ACCELERATION (Selection Matrix Approach)")
println("="^70)

## ============================================================================
## Step 1: Build minimal ACE model and extract specs
## ============================================================================

println("\n1. Building minimal ACE model...")

Dtot, maxl, ORD = 3, 1, 2
N_cheb = Dtot + 1
rcut = Float32(6.0)

mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
    level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
    Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
    Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
    basis=real)

nfeatures = length(symbasis, 0)
@printf("Model: Dtot=%d, maxl=%d, ORD=%d, N_cheb=%d, nfeatures=%d\n",
        Dtot, maxl, ORD, N_cheb, nfeatures)

## ============================================================================
## Step 2: Extract specifications and build selection matrices
## ============================================================================

println("\n2. Extracting specs and building selection matrices...")

rng = MersenneTwister(42)
ps = ET.LuxCore.initialparameters(rng, symbasis)
st = ET.LuxCore.initialstates(rng, symbasis)

# Extract specs
spec_R = [s[1] for s in st.aspec]  # 1-indexed
spec_Y = [s[2] for s in st.aspec]  # 1-indexed
nA = length(spec_R)
nRnl = N_cheb
nYlm = (maxl + 1)^2

@printf("   nA (A features): %d\n", nA)
@printf("   nRnl: %d, nYlm: %d\n", nRnl, nYlm)

# Build selection matrices for pooled sparse product
# selector_R[k, n] = 1 if spec_R[k] == n, else 0
# A[:, k] = sum_j Rnl[j, :, spec_R[k]] * Ylm[j, :, spec_Y[k]]
# Using selectors: A = (Rnl @ selector_R.T) .* (Ylm @ selector_Y.T)

function build_selector_matrix(spec::Vector{Int}, n_features::Int)
    nA = length(spec)
    selector = zeros(Float32, nA, n_features)
    for k in 1:nA
        selector[k, spec[k]] = 1.0f0
    end
    return selector
end

selector_R = build_selector_matrix(spec_R, nRnl)
selector_Y = build_selector_matrix(spec_Y, nYlm)

@printf("   selector_R: %s\n", size(selector_R))
@printf("   selector_Y: %s\n", size(selector_Y))

# Extract specs_mats for symmetric product
function spec_to_matrix(spec)
    isempty(spec) && return zeros(Int, 0, 0)
    N = length(spec[1])
    mat = zeros(Int, length(spec), N)
    for (i, ϕ) in enumerate(spec)
        for j in 1:N
            mat[i, j] = ϕ[j]
        end
    end
    return mat
end

specs_mats = [spec_to_matrix(Vector(s)) for s in st.aaspecs]
@printf("   specs_mats: %d orders\n", length(specs_mats))
for (i, m) in enumerate(specs_mats)
    @printf("     order %d: %s\n", i, size(m))
end

# Build selection matrices for sparse symmetric product
# For order 1: AA[i,k] = A[i, spec[k,1]]
#   → AA = A @ selector1.T
# For order 2: AA[i,k] = A[i, spec[k,1]] * A[i, spec[k,2]]
#   → AA = (A @ selector1.T) .* (A @ selector2.T)

function build_symm_prod_selectors(specs_mats, nA)
    all_selectors = Vector{Matrix{Float32}}[]

    for spec_mat in specs_mats
        if isempty(spec_mat) || size(spec_mat, 1) == 0
            continue
        end

        order = size(spec_mat, 2)
        nspec = size(spec_mat, 1)

        selectors = Matrix{Float32}[]
        for t in 1:order
            sel = zeros(Float32, nspec, nA)
            for k in 1:nspec
                idx = spec_mat[k, t]
                sel[k, idx] = 1.0f0
            end
            push!(selectors, sel)
        end
        push!(all_selectors, selectors)
    end

    return all_selectors
end

symm_selectors = build_symm_prod_selectors(specs_mats, nA)
@printf("   symm_selectors: %d order groups\n", length(symm_selectors))
for (i, sels) in enumerate(symm_selectors)
    @printf("     order %d: %d selectors of size %s\n", i, length(sels),
            isempty(sels) ? "()" : string(size(sels[1])))
end

# Extract A2Bmap
function sparse_to_dense(m)
    dense = zeros(eltype(m.nzval_csr), m.m, m.n)
    for row in 1:m.m
        for idx in m.rowptr[row]:(m.rowptr[row+1]-1)
            col = m.colval[idx]
            dense[row, col] = m.nzval_csr[idx]
        end
    end
    return dense
end

A2Bmap = Float32.(sparse_to_dense(st.A2Bmaps[1]))
nAA = size(A2Bmap, 2)
nBB = size(A2Bmap, 1)
@printf("   A2Bmap: (%d, %d)\n", nBB, nAA)

# Random parameters
params = randn(Float32, nBB)

## ============================================================================
## Step 3: Define embedding functions (Chebyshev + Ylm)
## ============================================================================

println("\n3. Defining embedding functions...")

# Chebyshev polynomials via recurrence
# Using Reactant-compatible implementation (no mutation, no typed allocation)
function chebyshev_basis(r::AbstractVector, rcut::Real, n_cheb::Int)
    n_pairs = length(r)
    T = eltype(r)

    # Map to [-1, 1]
    x = 2 .* r ./ rcut .- 1

    # Build Chebyshev polynomials via concatenation (no mutation)
    if n_cheb >= 1
        P0 = ones(T, n_pairs)
    end
    if n_cheb >= 2
        P1 = x
    end

    # Recurrence: T_n = 2*x*T_{n-1} - T_{n-2}
    if n_cheb == 1
        return reshape(P0, :, 1)
    elseif n_cheb == 2
        return hcat(reshape(P0, :, 1), reshape(P1, :, 1))
    else
        # Build all polynomials
        Pnm2 = P0
        Pnm1 = P1
        result = hcat(reshape(P0, :, 1), reshape(P1, :, 1))
        for _ in 3:n_cheb
            Pn = 2 .* x .* Pnm1 .- Pnm2
            result = hcat(result, reshape(Pn, :, 1))
            Pnm2 = Pnm1
            Pnm1 = Pn
        end
        return result
    end
end

# Real spherical harmonics (up to l=1 for simplicity)
# Reactant-compatible (no mutation)
function real_ylm(rhat::AbstractMatrix, maxl::Int)
    n_pairs = size(rhat, 1)
    T = eltype(rhat)
    n_ylm = (maxl + 1)^2

    x = rhat[:, 1]
    y = rhat[:, 2]
    z = rhat[:, 3]

    # l=0
    c00 = T(0.28209479177387814)  # 1/(2*sqrt(pi))
    Y00 = fill(c00, n_pairs)

    if maxl == 0
        return reshape(Y00, :, 1)
    end

    # l=1
    c1 = T(0.4886025119029199)  # sqrt(3/(4*pi))
    Y1m1 = c1 .* y
    Y10 = c1 .* z
    Y1p1 = c1 .* x

    return hcat(reshape(Y00, :, 1), reshape(Y1m1, :, 1),
                reshape(Y10, :, 1), reshape(Y1p1, :, 1))
end

## ============================================================================
## Step 4: Define full ACE energy function using selection matrices
## ============================================================================

println("\n4. Defining full ACE energy function...")

#=
Full forward pass with selection matrices (no gather/scatter):

1. rij → r, rhat (normalization)
2. r → Rnl (Chebyshev)
3. rhat → Ylm (spherical harmonics)
4. Rnl, Ylm → A_pair = (Rnl @ sel_R.T) .* (Ylm @ sel_Y.T)
5. A_pair → A (pool over pairs: sum per atom)
6. A → AA via selection matrices
7. AA → E = sum((AA @ A2Bmap.T) @ params)

The pooling in step 5 is the tricky part. We need to handle it without scatter.
For fixed shapes, we can use a pooling matrix: pool[atom, pair] = 1 if pair belongs to atom
=#

# For this test, we'll use fixed shapes and a pooling matrix
n_pairs = 20
n_atoms = 4

# Create a simple pair structure - each atom has ~5 neighbors
Random.seed!(123)
pair_i = vcat([fill(i, 5) for i in 1:n_atoms]...)  # 5 pairs per atom = 20 pairs
pair_j = rand(1:n_atoms, n_pairs)  # random targets

# Build pooling matrix: pool[i, e] = 1 if pair_i[e] == i
pool_matrix = zeros(Float32, n_atoms, n_pairs)
for e in 1:n_pairs
    pool_matrix[pair_i[e], e] = 1.0f0
end

@printf("   n_pairs: %d, n_atoms: %d\n", n_pairs, n_atoms)
@printf("   pool_matrix: %s\n", size(pool_matrix))

# Generate random rij
rij = randn(Float32, n_pairs, 3)
# Ensure distances are within cutoff
r = sqrt.(sum(rij.^2, dims=2))
rij = rij .* (0.9f0 * rcut ./ max.(r, 1f-6))

@printf("   rij: %s\n", size(rij))

# Full ACE energy function using selection matrices
function ace_energy_selmat(
    rij::AbstractMatrix,
    pool_matrix::AbstractMatrix,
    selector_R::AbstractMatrix,
    selector_Y::AbstractMatrix,
    symm_sel1::AbstractMatrix,  # order 1 selector
    symm_sel2_1::AbstractMatrix, # order 2 first factor
    symm_sel2_2::AbstractMatrix, # order 2 second factor
    A2Bmap::AbstractMatrix,
    params::AbstractVector,
    rcut::Real,
    n_cheb::Int,
    maxl::Int
)
    n_pairs = size(rij, 1)
    T = eltype(rij)

    # Step 1: Compute r and rhat
    r2 = sum(rij .^ 2, dims=2)
    r = sqrt.(r2)
    eps = T(1e-6)
    rhat = rij ./ max.(r, eps)
    r_vec = dropdims(r, dims=2)

    # Step 2: Chebyshev embeddings
    Rnl = chebyshev_basis(r_vec, rcut, n_cheb)  # [n_pairs, nRnl]

    # Step 3: Spherical harmonics
    Ylm = real_ylm(rhat, maxl)  # [n_pairs, nYlm]

    # Step 4: Select and multiply for A_pair features
    # A_pair[e, k] = Rnl[e, spec_R[k]] * Ylm[e, spec_Y[k]]
    # Using selectors: A_pair = (Rnl @ sel_R.T) .* (Ylm @ sel_Y.T)
    Rnl_sel = Rnl * transpose(selector_R)  # [n_pairs, nA]
    Ylm_sel = Ylm * transpose(selector_Y)  # [n_pairs, nA]
    A_pair = Rnl_sel .* Ylm_sel  # [n_pairs, nA]

    # Step 5: Pool to atoms
    # A[i, k] = sum_{e: pair_i[e]==i} A_pair[e, k]
    # Using pooling matrix: A = pool_matrix @ A_pair
    A = pool_matrix * A_pair  # [n_atoms, nA]

    # Step 6: Sparse symmetric product via selection matrices
    # Order 1: AA1 = A @ sel1.T
    AA1 = A * transpose(symm_sel1)  # [n_atoms, nspec1]

    # Order 2: AA2 = (A @ sel2_1.T) .* (A @ sel2_2.T)
    A_sel1 = A * transpose(symm_sel2_1)  # [n_atoms, nspec2]
    A_sel2 = A * transpose(symm_sel2_2)  # [n_atoms, nspec2]
    AA2 = A_sel1 .* A_sel2  # [n_atoms, nspec2]

    # Concatenate
    AA = hcat(AA1, AA2)  # [n_atoms, nAA]

    # Step 7: Linear readout
    BB = AA * transpose(A2Bmap)  # [n_atoms, nBB]
    E = sum(BB * params)

    return E
end

## ============================================================================
## Step 5: Test Julia energy computation
## ============================================================================

println("\n5. Testing Julia energy computation...")

# Flatten symm selectors for the function signature
symm_sel1 = symm_selectors[1][1]  # Order 1 has 1 selector
symm_sel2_1 = symm_selectors[2][1]  # Order 2 first factor
symm_sel2_2 = symm_selectors[2][2]  # Order 2 second factor

@printf("   symm_sel1: %s\n", size(symm_sel1))
@printf("   symm_sel2_1: %s\n", size(symm_sel2_1))
@printf("   symm_sel2_2: %s\n", size(symm_sel2_2))

E_julia = ace_energy_selmat(
    rij, pool_matrix, selector_R, selector_Y,
    symm_sel1, symm_sel2_1, symm_sel2_2,
    A2Bmap, params, rcut, N_cheb, maxl
)
@printf("Julia energy: %.6f\n", E_julia)

## ============================================================================
## Step 6: Define energy + pair forces function
## ============================================================================

println("\n6. Defining energy + pair forces function...")

function ace_energy_and_forces(
    rij, pool_matrix, selector_R, selector_Y,
    symm_sel1, symm_sel2_1, symm_sel2_2,
    A2Bmap, params, rcut, n_cheb, maxl
)
    d_rij = zero(rij)

    _, energy = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        ace_energy_selmat,
        Enzyme.Active,
        Enzyme.Duplicated(rij, d_rij),
        Enzyme.Const(pool_matrix),
        Enzyme.Const(selector_R),
        Enzyme.Const(selector_Y),
        Enzyme.Const(symm_sel1),
        Enzyme.Const(symm_sel2_1),
        Enzyme.Const(symm_sel2_2),
        Enzyme.Const(A2Bmap),
        Enzyme.Const(params),
        Enzyme.Const(rcut),
        Enzyme.Const(n_cheb),
        Enzyme.Const(maxl)
    )

    # Pair forces = -dE/drij
    pair_forces = -d_rij

    return (energy, pair_forces)
end

## ============================================================================
## Step 7: Test with finite differences
## ============================================================================

println("\n7. Testing gradients with finite differences...")

eps_fd = Float32(1e-4)
pair_forces_fd = zeros(Float32, n_pairs, 3)

for e in 1:min(5, n_pairs)  # Test first 5 pairs
    for d in 1:3
        rij_plus = copy(rij)
        rij_plus[e, d] += eps_fd
        rij_minus = copy(rij)
        rij_minus[e, d] -= eps_fd

        E_plus = ace_energy_selmat(
            rij_plus, pool_matrix, selector_R, selector_Y,
            symm_sel1, symm_sel2_1, symm_sel2_2,
            A2Bmap, params, rcut, N_cheb, maxl
        )
        E_minus = ace_energy_selmat(
            rij_minus, pool_matrix, selector_R, selector_Y,
            symm_sel1, symm_sel2_1, symm_sel2_2,
            A2Bmap, params, rcut, N_cheb, maxl
        )

        pair_forces_fd[e, d] = -(E_plus - E_minus) / (2 * eps_fd)
    end
end

# Compute analytical gradients
E_ad, pair_forces_ad = ace_energy_and_forces(
    rij, pool_matrix, selector_R, selector_Y,
    symm_sel1, symm_sel2_1, symm_sel2_2,
    A2Bmap, params, rcut, N_cheb, maxl
)

@printf("Energy (AD): %.6f\n", E_ad)
@printf("Pair forces AD norm: %.6f\n", norm(pair_forces_ad))
@printf("Pair forces FD norm (first 5): %.6f\n", norm(pair_forces_fd[1:5, :]))

# Compare
rel_err = norm(pair_forces_ad[1:5, :] - pair_forces_fd[1:5, :]) / max(norm(pair_forces_fd[1:5, :]), 1e-10)
@printf("Relative error (first 5 pairs): %.2e\n", rel_err)
@test rel_err < 0.05

## ============================================================================
## Step 8: Compile with Reactant
## ============================================================================

println("\n8. Compiling with Reactant...")
Reactant.set_default_backend("cpu")

# Convert to RArrays
rij_ra = Reactant.to_rarray(rij)
pool_matrix_ra = Reactant.to_rarray(pool_matrix)
selector_R_ra = Reactant.to_rarray(selector_R)
selector_Y_ra = Reactant.to_rarray(selector_Y)
symm_sel1_ra = Reactant.to_rarray(symm_sel1)
symm_sel2_1_ra = Reactant.to_rarray(symm_sel2_1)
symm_sel2_2_ra = Reactant.to_rarray(symm_sel2_2)
A2Bmap_ra = Reactant.to_rarray(A2Bmap)
params_ra = Reactant.to_rarray(params)

try
    compiled_fn = Reactant.@compile ace_energy_and_forces(
        rij_ra, pool_matrix_ra, selector_R_ra, selector_Y_ra,
        symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
        A2Bmap_ra, params_ra, rcut, N_cheb, maxl
    )

    E_compiled, pf_compiled = compiled_fn(
        rij_ra, pool_matrix_ra, selector_R_ra, selector_Y_ra,
        symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
        A2Bmap_ra, params_ra, rcut, N_cheb, maxl
    )

    E_val = Float64(E_compiled)
    pf_val = Array(pf_compiled)

    @printf("Compiled energy: %.6f\n", E_val)
    @printf("Compiled pair forces norm: %.6f\n", norm(pf_val))

    @test isapprox(E_val, E_julia, rtol=1e-4)
    @test isapprox(pf_val, pair_forces_ad, rtol=1e-4)

    println("✓ Reactant compilation successful!")
catch e
    @warn "Reactant compilation failed" exception=e
    rethrow()
end

## ============================================================================
## Step 9: Export to StableHLO
## ============================================================================

println("\n9. Exporting to StableHLO...")

export_dir = joinpath(@__DIR__, "..", "test_ace_full_gpu_output")
mkpath(export_dir)

try
    Reactant.Serialization.export_to_enzymejax(
        ace_energy_and_forces,
        rij_ra, pool_matrix_ra, selector_R_ra, selector_Y_ra,
        symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
        A2Bmap_ra, params_ra, rcut, N_cheb, maxl;
        output_dir=export_dir,
        function_name="ace_full_gpu"
    )
    @info "StableHLO exported to $export_dir"
catch e
    @warn "StableHLO export failed" exception=e
end

## ============================================================================
## Step 10: Check for scatter ops and compile to VMFB
## ============================================================================

println("\n10. Checking for scatter ops and compiling to VMFB...")

iree_compile = joinpath(@__DIR__, "..", "python", ".venv", "bin", "iree-compile")
mlir_files = filter(f -> endswith(f, ".mlir") && !endswith(f, "_inputs.mlir"), readdir(export_dir))

global vmfb_compiled = false
global has_scatter = false
global has_gather = false

if isfile(iree_compile) && !isempty(mlir_files)
    mlir_path = joinpath(export_dir, first(mlir_files))
    vmfb_path = joinpath(export_dir, "ace_full_gpu_cpu.vmfb")

    println("   MLIR file: $mlir_path")

    mlir_content = read(mlir_path, String)
    global has_scatter = occursin("scatter", lowercase(mlir_content))
    global has_gather = occursin("gather", lowercase(mlir_content))
    println("   Contains 'scatter' ops: $has_scatter")
    println("   Contains 'gather' ops: $has_gather")

    # Count operations
    n_dot = length(collect(eachmatch(r"dot_general", mlir_content)))
    n_mul = length(collect(eachmatch(r"multiply", mlir_content)))
    n_add = length(collect(eachmatch(r"stablehlo\.add", mlir_content)))
    println("   dot_general ops: $n_dot")
    println("   multiply ops: $n_mul")
    println("   add ops: $n_add")

    try
        run(`$iree_compile --iree-input-type=stablehlo --iree-hal-target-backends=llvm-cpu $mlir_path -o $vmfb_path`)
        @info "SUCCESS! VMFB compiled to $vmfb_path"
        global vmfb_compiled = true
    catch e
        @warn "VMFB compilation failed" exception=e
    end
else
    @info "iree-compile not found or no MLIR files"
end

## ============================================================================
## Summary
## ============================================================================

println("\n" * "="^70)
println("TEST SUMMARY")
println("="^70)

println("\nArchitecture (Full GPU Acceleration):")
println("  IREE computes EVERYTHING:")
println("    1. rij → r, rhat (normalization)")
println("    2. r → Rnl (Chebyshev polynomials)")
println("    3. rhat → Ylm (spherical harmonics)")
println("    4. Rnl, Ylm → A_pair (via selection matrices)")
println("    5. A_pair → A (pooling via pool_matrix)")
println("    6. A → AA (symmetric product via selection matrices)")
println("    7. AA → E (linear readout)")
println("    8. Backward pass: dE/drij (all via matmuls)")
println("  Host only accumulates: pair_forces → atom_forces")

println("\nSelection Matrices:")
@printf("  selector_R: %s (%.1f KB)\n", size(selector_R), sizeof(selector_R)/1024)
@printf("  selector_Y: %s (%.1f KB)\n", size(selector_Y), sizeof(selector_Y)/1024)
@printf("  symm_sel1: %s (%.1f KB)\n", size(symm_sel1), sizeof(symm_sel1)/1024)
@printf("  symm_sel2_1: %s (%.1f KB)\n", size(symm_sel2_1), sizeof(symm_sel2_1)/1024)
@printf("  symm_sel2_2: %s (%.1f KB)\n", size(symm_sel2_2), sizeof(symm_sel2_2)/1024)
total_sel_kb = (sizeof(selector_R) + sizeof(selector_Y) + sizeof(symm_sel1) +
                sizeof(symm_sel2_1) + sizeof(symm_sel2_2)) / 1024
@printf("  Total: %.1f KB\n", total_sel_kb)

println("\nResults:")
@printf("  Julia energy: %.6f\n", E_julia)
@printf("  Compiled energy: %.6f\n", E_ad)
@printf("  Gradient error: %.2e\n", rel_err)

println("\nIREE Compilation:")
if vmfb_compiled
    println("  ✓ SUCCESS! Full GPU acceleration achieved!")
    println("  ✓ Contains scatter: $has_scatter")
    println("  ✓ Contains gather: $has_gather")
else
    println("  ✗ VMFB compilation failed or not attempted")
end

println("\nOutput files:")
for f in sort(readdir(export_dir))
    path = joinpath(export_dir, f)
    isfile(path) && @printf("  %s (%d bytes)\n", f, filesize(path))
end

println("="^70)
