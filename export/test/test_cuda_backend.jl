#=
Test ACE Full GPU - CUDA Backend
================================
Tests that the selection matrix approach compiles to CUDA VMFB.
=#

using Reactant
using Enzyme
using NPZ
using LinearAlgebra
using Random
import EquivariantTensors as ET
import Polynomials4ML as P4ML

println("="^70)
println("ACE FULL GPU - CUDA BACKEND TEST")
println("="^70)

# Build minimal ACE model
Dtot, maxl, ORD = 3, 1, 2
N_cheb = Dtot + 1
rcut = Float32(6.0)

mb_spec = ET.sparse_nnll_set(; L=0, ORD=ORD, minn=0, maxn=Dtot, maxl=maxl,
    level=bb->sum((b.n+b.l) for b in bb; init=0), maxlevel=Dtot)

symbasis = ET.sparse_equivariant_tensor(; L=0, mb_spec=mb_spec,
    Rnl_spec=P4ML.natural_indices(P4ML.ChebBasis(N_cheb)),
    Ylm_spec=P4ML.natural_indices(P4ML.real_solidharmonics(maxl)),
    basis=real)

rng = MersenneTwister(42)
ps = ET.LuxCore.initialparameters(rng, symbasis)
st = ET.LuxCore.initialstates(rng, symbasis)

# Extract specs
spec_R = [s[1] for s in st.aspec]
spec_Y = [s[2] for s in st.aspec]
nA = length(spec_R)
nRnl = N_cheb
nYlm = (maxl + 1)^2

# Build selection matrices
function build_selector_matrix(spec, n_features)
    nA = length(spec)
    selector = zeros(Float32, nA, n_features)
    for k in 1:nA
        selector[k, spec[k]] = 1.0f0
    end
    return selector
end

selector_R = build_selector_matrix(spec_R, nRnl)
selector_Y = build_selector_matrix(spec_Y, nYlm)

function spec_to_matrix(spec)
    isempty(spec) && return zeros(Int, 0, 0)
    N = length(spec[1])
    mat = zeros(Int, length(spec), N)
    for (i, p) in enumerate(spec)
        for j in 1:N
            mat[i, j] = p[j]
        end
    end
    return mat
end

specs_mats = [spec_to_matrix(Vector(s)) for s in st.aaspecs]

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
                sel[k, spec_mat[k, t]] = 1.0f0
            end
            push!(selectors, sel)
        end
        push!(all_selectors, selectors)
    end
    return all_selectors
end

symm_selectors = build_symm_prod_selectors(specs_mats, nA)
symm_sel1 = symm_selectors[1][1]
symm_sel2_1 = symm_selectors[2][1]
symm_sel2_2 = symm_selectors[2][2]

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
nBB = size(A2Bmap, 1)
params = randn(Float32, nBB)

# Test setup
n_pairs = 20
n_atoms = 4
Random.seed!(123)
pair_i = vcat([fill(i, 5) for i in 1:n_atoms]...)

pool_matrix = zeros(Float32, n_atoms, n_pairs)
for e in 1:n_pairs
    pool_matrix[pair_i[e], e] = 1.0f0
end

rij = randn(Float32, n_pairs, 3)
r = sqrt.(sum(rij.^2, dims=2))
rij = rij .* (0.9f0 * rcut ./ max.(r, 1f-6))

# Define functions (Reactant-compatible)
function chebyshev_basis(r, rcut, n_cheb)
    n_pairs = length(r)
    T = eltype(r)
    x = 2 .* r ./ rcut .- 1

    P0 = ones(T, n_pairs)
    if n_cheb == 1
        return reshape(P0, :, 1)
    end

    P1 = x
    if n_cheb == 2
        return hcat(reshape(P0, :, 1), reshape(P1, :, 1))
    end

    Pnm2, Pnm1 = P0, P1
    result = hcat(reshape(P0, :, 1), reshape(P1, :, 1))
    for _ in 3:n_cheb
        Pn = 2 .* x .* Pnm1 .- Pnm2
        result = hcat(result, reshape(Pn, :, 1))
        Pnm2, Pnm1 = Pnm1, Pn
    end
    return result
end

function real_ylm(rhat, maxl)
    n_pairs = size(rhat, 1)
    T = eltype(rhat)
    x, y, z = rhat[:, 1], rhat[:, 2], rhat[:, 3]

    c00 = T(0.28209479177387814)
    Y00 = fill(c00, n_pairs)

    if maxl == 0
        return reshape(Y00, :, 1)
    end

    c1 = T(0.4886025119029199)
    return hcat(reshape(Y00, :, 1), reshape(c1 .* y, :, 1),
                reshape(c1 .* z, :, 1), reshape(c1 .* x, :, 1))
end

function ace_energy_selmat(rij, pool_matrix, selector_R, selector_Y,
                           symm_sel1, symm_sel2_1, symm_sel2_2,
                           A2Bmap, params, rcut, n_cheb, maxl)
    T = eltype(rij)
    r2 = sum(rij .^ 2, dims=2)
    r = sqrt.(r2)
    rhat = rij ./ max.(r, T(1e-6))
    r_vec = dropdims(r, dims=2)

    Rnl = chebyshev_basis(r_vec, rcut, n_cheb)
    Ylm = real_ylm(rhat, maxl)

    Rnl_sel = Rnl * transpose(selector_R)
    Ylm_sel = Ylm * transpose(selector_Y)
    A_pair = Rnl_sel .* Ylm_sel

    A = pool_matrix * A_pair

    AA1 = A * transpose(symm_sel1)
    A_sel1 = A * transpose(symm_sel2_1)
    A_sel2 = A * transpose(symm_sel2_2)
    AA2 = A_sel1 .* A_sel2
    AA = hcat(AA1, AA2)

    BB = AA * transpose(A2Bmap)
    return sum(BB * params)
end

function ace_energy_and_forces(rij, pool_matrix, selector_R, selector_Y,
                               symm_sel1, symm_sel2_1, symm_sel2_2,
                               A2Bmap, params, rcut, n_cheb, maxl)
    d_rij = zero(rij)
    _, energy = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal, ace_energy_selmat, Enzyme.Active,
        Enzyme.Duplicated(rij, d_rij),
        Enzyme.Const(pool_matrix), Enzyme.Const(selector_R), Enzyme.Const(selector_Y),
        Enzyme.Const(symm_sel1), Enzyme.Const(symm_sel2_1), Enzyme.Const(symm_sel2_2),
        Enzyme.Const(A2Bmap), Enzyme.Const(params),
        Enzyme.Const(rcut), Enzyme.Const(n_cheb), Enzyme.Const(maxl)
    )
    return (energy, -d_rij)
end

# Julia reference
E_julia = ace_energy_selmat(rij, pool_matrix, selector_R, selector_Y,
                            symm_sel1, symm_sel2_1, symm_sel2_2,
                            A2Bmap, params, rcut, N_cheb, maxl)
println("\nJulia reference energy: ", E_julia)

# Compile with CUDA backend
println("\nCompiling with CUDA backend...")
Reactant.set_default_backend("gpu")

rij_ra = Reactant.to_rarray(rij)
pool_matrix_ra = Reactant.to_rarray(pool_matrix)
selector_R_ra = Reactant.to_rarray(selector_R)
selector_Y_ra = Reactant.to_rarray(selector_Y)
symm_sel1_ra = Reactant.to_rarray(symm_sel1)
symm_sel2_1_ra = Reactant.to_rarray(symm_sel2_1)
symm_sel2_2_ra = Reactant.to_rarray(symm_sel2_2)
A2Bmap_ra = Reactant.to_rarray(A2Bmap)
params_ra = Reactant.to_rarray(params)

compiled_cuda = Reactant.@compile ace_energy_and_forces(
    rij_ra, pool_matrix_ra, selector_R_ra, selector_Y_ra,
    symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
    A2Bmap_ra, params_ra, rcut, N_cheb, maxl
)

E_cuda, pf_cuda = compiled_cuda(
    rij_ra, pool_matrix_ra, selector_R_ra, selector_Y_ra,
    symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
    A2Bmap_ra, params_ra, rcut, N_cheb, maxl
)

println("CUDA compiled energy: ", Float64(E_cuda))
println("CUDA pair forces norm: ", norm(Array(pf_cuda)))
println("CUDA Reactant compilation successful!")

# Export StableHLO
export_dir = joinpath(@__DIR__, "..", "test_ace_full_gpu_cuda_output")
mkpath(export_dir)

Reactant.Serialization.export_to_enzymejax(
    ace_energy_and_forces,
    rij_ra, pool_matrix_ra, selector_R_ra, selector_Y_ra,
    symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
    A2Bmap_ra, params_ra, rcut, N_cheb, maxl;
    output_dir=export_dir, function_name="ace_cuda"
)

# Compile to CUDA VMFB
iree_compile = joinpath(@__DIR__, "..", "python", ".venv", "bin", "iree-compile")
mlir_files = filter(f -> endswith(f, ".mlir") && !endswith(f, "_inputs.mlir"), readdir(export_dir))

if isfile(iree_compile) && !isempty(mlir_files)
    mlir_path = joinpath(export_dir, first(mlir_files))
    vmfb_path = joinpath(export_dir, "ace_cuda.vmfb")

    mlir_content = read(mlir_path, String)
    println("\nMLIR analysis:")
    println("  Contains scatter: ", occursin("scatter", lowercase(mlir_content)))
    println("  Contains gather: ", occursin("gather", lowercase(mlir_content)))

    println("\nCompiling to CUDA VMFB...")
    run(`$iree_compile --iree-input-type=stablehlo --iree-hal-target-backends=cuda $mlir_path -o $vmfb_path`)
    println("SUCCESS: CUDA VMFB compiled: ", vmfb_path)
    println("  Size: ", filesize(vmfb_path), " bytes")
end

println("\n" * "="^70)
println("CUDA TEST COMPLETE - Full GPU acceleration verified!")
println("="^70)

# List output files
println("\nOutput files:")
for f in sort(readdir(export_dir))
    path = joinpath(export_dir, f)
    isfile(path) && println("  ", f, " (", filesize(path), " bytes)")
end
