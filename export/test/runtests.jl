using Test
using ACEExport

@testset "ACEExport.jl" begin

    @testset "Module loading" begin
        @test isdefined(ACEExport, :compile_model)
        @test isdefined(ACEExport, :export_to_iree)
        @test isdefined(ACEExport, :ReactantETACEState)
        @test isdefined(ACEExport, :ReactantStackedModel)
        @test isdefined(ACEExport, :DEFAULT_SHAPES)
    end

    @testset "Data structure conversion" begin
        # Test spec_to_matrix
        spec = [(1, 2), (3, 4), (5, 6)]
        mat = ACEExport.spec_to_matrix(spec)
        @test size(mat) == (3, 2)
        @test mat[1, :] == [1, 2]
        @test mat[2, :] == [3, 4]
        @test mat[3, :] == [5, 6]

        # Test empty spec
        empty_mat = ACEExport.spec_to_matrix(Tuple{Int,Int}[])
        @test size(empty_mat) == (0, 0)
    end

    @testset "Embeddings" begin
        using StaticArrays

        # Test Agnesi transform
        r = 3.0
        y = ACEExport.compute_agnesi_transform(r, 2.0, 2.0, 1.0, 2.5, 6.0)
        @test -1 <= y <= 1

        # Test envelope
        env = ACEExport.compute_envelope(0.5)
        @test env > 0
        @test ACEExport.compute_envelope(1.0) ≈ 0 atol=1e-10
        @test ACEExport.compute_envelope(-1.0) ≈ 0 atol=1e-10

        # Test Chebyshev basis
        n_polys = 5
        A = ones(n_polys)
        B = zeros(n_polys)
        C = zeros(n_polys)
        P = ACEExport.compute_chebyshev_basis(0.5, n_polys, A, B, C)
        @test length(P) == n_polys

        # Test spherical harmonics
        rhat = SVector(1.0, 0.0, 0.0) / 1.0
        Ylm = ACEExport.compute_ylm_reactant(rhat, 2)
        @test length(Ylm) == 9  # (2+1)^2
    end

    @testset "ACE kernel" begin
        # Test pooled sparse product
        maxneigs, nnodes, nRnl, nYlm = 5, 3, 4, 9
        Rnl_3 = rand(maxneigs, nnodes, nRnl)
        Ylm_3 = rand(maxneigs, nnodes, nYlm)
        spec_R = [1, 2, 1, 3]
        spec_Y = [1, 1, 2, 3]

        A = ACEExport.pooled_sparse_product_reactant(Rnl_3, Ylm_3, spec_R, spec_Y)
        @test size(A) == (nnodes, length(spec_R))

        # Test sparse symmetric product
        specs_mats = [
            [1 2; 2 3],  # Order 2
            [1 2 3; 2 3 4],  # Order 3
        ]
        AA = ACEExport.sparse_symm_prod_reactant(A, specs_mats)
        @test size(AA, 1) == nnodes
        @test size(AA, 2) == 2 + 2  # Sum of nspec for each order
    end

    @testset "CompiledShapes" begin
        shapes = ACEExport.DEFAULT_SHAPES
        @test shapes.max_atoms == 4096
        @test shapes.max_neigs == 50
        @test shapes.max_edges == 200_000
    end

end
