#=
Numerical Equivalence Tests
============================

Verify that the Reactant-compiled ACE model produces the same results
as the native Julia implementation within acceptable tolerances.
=#

using Test
using ACEpotentials
using LinearAlgebra
using Random

# Conditionally load ACEExport
const HAS_EXPORT = try
    using ACEExport
    true
catch
    false
end

@testset "Numerical Equivalence" begin

    # Skip if ACEExport not available
    if !HAS_EXPORT
        @info "Skipping numerical equivalence tests - ACEExport not available"
        @test_skip "ACEExport required"
        return
    end

    @testset "ReactantETACEState extraction" begin
        # Create a simple model
        model = ace1_model(
            elements = [:Si],
            order = 2,
            totaldegree = 6
        )
        calc = ACEpotentials.ACEPotential(model)

        # Test state extraction (when implemented with real ETACE)
        @test_skip "Full ETACE extraction not yet implemented"
    end

    @testset "Embedding functions" begin
        # Test Agnesi transform
        r = 3.0
        pcut, pin, rin, req, rcut = 2.0, 2.0, 1.0, 2.5, 6.0
        y = ACEExport.compute_agnesi_transform(r, pcut, pin, rin, req, rcut)
        @test -1 <= y <= 1

        # Test at boundaries: Agnesi transform = (1-(r/rcut)^pcut) / (1+(r/req)^pin)
        # At r=rcut, cutoff factor = 0, so y = 0
        y_at_rcut = ACEExport.compute_agnesi_transform(rcut, pcut, pin, rin, req, rcut)
        @test y_at_rcut ≈ 0 atol=1e-6

        # Test envelope
        @test ACEExport.compute_envelope(0.0) ≈ 1.0 atol=1e-10
        @test ACEExport.compute_envelope(1.0) ≈ 0.0 atol=1e-10
        @test ACEExport.compute_envelope(-1.0) ≈ 0.0 atol=1e-10
        @test ACEExport.compute_envelope(0.5) > 0

        # Test Chebyshev basis
        n_polys = 5
        A = ones(n_polys)
        B = zeros(n_polys)
        C = zeros(n_polys)
        P = ACEExport.compute_chebyshev_basis(0.0, n_polys, A, B, C)
        @test length(P) == n_polys
        @test P[1] ≈ 1.0 atol=1e-10  # T_0(0) = 1
        @test P[2] ≈ 0.0 atol=1e-10  # T_1(0) = 0

        # Test spherical harmonics
        using StaticArrays
        rhat = SVector(1.0, 0.0, 0.0)
        Ylm = ACEExport.compute_ylm_reactant(rhat, 2)
        @test length(Ylm) == 9  # (2+1)^2

        # Y_00 should be 1/sqrt(4π) ≈ 0.28209
        @test Ylm[1] ≈ 0.28209479 atol=1e-5
    end

    @testset "ACE kernel components" begin
        # Test pooled sparse product
        maxneigs, nnodes, nRnl, nYlm = 5, 3, 4, 9

        Random.seed!(42)
        Rnl_3 = rand(Float32, maxneigs, nnodes, nRnl)
        Ylm_3 = rand(Float32, maxneigs, nnodes, nYlm)
        spec_R = [1, 2, 1, 3]
        spec_Y = [1, 1, 2, 3]

        A = ACEExport.pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)
        @test size(A) == (nnodes, length(spec_R))

        # Verify the pooling manually for first element
        expected_A11 = sum(Rnl_3[:, 1, spec_R[1]] .* Ylm_3[:, 1, spec_Y[1]])
        @test A[1, 1] ≈ expected_A11 atol=1e-5

        # Test sparse symmetric product
        specs_mats = [
            Int64[1 2; 2 3],
            Int64[1 2 3; 2 3 4],
        ]
        AA = ACEExport.sparse_symm_prod_reactant(A, specs_mats)
        @test size(AA, 1) == nnodes
        # Total AA columns = sum of nspec for each order
        @test size(AA, 2) == 2 + 2
    end

    @testset "Data structure conversion" begin
        # Test spec_to_matrix
        spec = [(1, 2), (3, 4), (5, 6)]
        mat = ACEExport.spec_to_matrix(spec)
        @test size(mat) == (3, 2)
        @test mat[1, :] == [1, 2]

        # Test with longer tuples
        spec3 = [(1, 2, 3), (4, 5, 6)]
        mat3 = ACEExport.spec_to_matrix(spec3)
        @test size(mat3) == (2, 3)

        # Empty spec
        empty_mat = ACEExport.spec_to_matrix(Tuple{Int,Int}[])
        @test size(empty_mat) == (0, 0)
    end

    @testset "CompiledShapes" begin
        shapes = ACEExport.DEFAULT_SHAPES
        @test shapes.max_atoms == 4096
        @test shapes.max_neigs == 50
        @test shapes.max_edges == 200_000

        # Custom shapes
        custom = ACEExport.CompiledShapes(
            max_atoms = 8192,
            max_neigs = 100,
            max_edges = 500_000
        )
        @test custom.max_atoms == 8192
    end

end  # @testset "Numerical Equivalence"
