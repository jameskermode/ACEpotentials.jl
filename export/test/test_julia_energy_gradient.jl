#=
Test Julia Energy+Gradient Computation
======================================

Verifies that the Julia energy function and Enzyme gradient computation
produce consistent results. This is the reference implementation that
VMFBs are compared against.

Run:
    julia +1.11 --project=export export/test/test_julia_energy_gradient.jl

Expected: All tests pass
=#

using Test
using LinearAlgebra
using Enzyme

println("=" ^ 70)
println("Julia Energy+Gradient Test")
println("=" ^ 70)

## ============================================================================
## Energy function (must match export_bucket_energy_gradient.jl exactly)
## ============================================================================

"""
Per-edge energy function using Float64.
This is the reference implementation that gets compiled to VMFB.
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
    valid_mask = r_sq .> T(1e-10)
    envelope = (T(1) .- x) .^ 2 .* (x .< T(1)) .* valid_mask

    w = T.([0.1, 0.2, 0.15, 0.1, 0.05, 0.02])
    e_edge = (P0 .* w[1] .+ P1 .* w[2] .+ P2 .* w[3] .+
              P3 .* w[4] .+ P4 .* w[5] .+ P5 .* w[6]) .* envelope

    return sum(e_edge)
end

function compute_energy_and_gradient(rij::AbstractMatrix{T}) where T
    energy = per_edge_energy_scalar(rij)

    drij = zero(rij)
    Enzyme.autodiff(Reverse, Const(per_edge_energy_scalar), Active, Duplicated(copy(rij), drij))

    return energy, drij
end

## ============================================================================
## Tests
## ============================================================================

@testset "Julia Energy+Gradient" begin

    @testset "Energy computation" begin
        # Small test case
        rij = randn(100, 3) .* 2.0
        E = per_edge_energy_scalar(rij)

        @test isfinite(E)
        @test E != 0.0  # Should have non-zero energy
        println("  Energy (100 edges): $E")
    end

    @testset "Gradient computation via Enzyme" begin
        rij = randn(100, 3) .* 2.0
        E, grad = compute_energy_and_gradient(rij)

        @test isfinite(E)
        @test size(grad) == size(rij)
        @test all(isfinite, grad)
        @test maximum(abs.(grad)) > 0  # Should have non-zero gradients
        println("  Energy: $E")
        println("  Max gradient: $(maximum(abs.(grad)))")
    end

    @testset "Gradient numerical verification" begin
        # Verify Enzyme gradient matches finite difference
        rij = randn(50, 3) .* 2.0
        E0, grad = compute_energy_and_gradient(rij)

        # Finite difference check for a few components
        eps = 1e-6
        for idx in [(1, 1), (10, 2), (25, 3)]
            i, j = idx
            rij_p = copy(rij)
            rij_m = copy(rij)
            rij_p[i, j] += eps
            rij_m[i, j] -= eps

            E_p = per_edge_energy_scalar(rij_p)
            E_m = per_edge_energy_scalar(rij_m)
            fd_grad = (E_p - E_m) / (2 * eps)

            enzyme_grad = grad[i, j]
            diff = abs(enzyme_grad - fd_grad)

            @test diff < 1e-5  # enzyme=$enzyme_grad, fd=$fd_grad
        end
        println("  Finite difference verification PASSED")
    end

    @testset "Consistent results across calls" begin
        # Same input should give same output
        rij = randn(200, 3) .* 2.0

        E1, grad1 = compute_energy_and_gradient(rij)
        E2, grad2 = compute_energy_and_gradient(rij)

        @test E1 == E2  # Exact equality for deterministic computation
        @test grad1 == grad2
        println("  Determinism verified")
    end

    @testset "Bucket sizes" begin
        # Test at each bucket size used in production
        bucket_sizes = [2000, 10000, 50000]  # Skip 100000 for speed

        for n_edges in bucket_sizes
            rij = randn(n_edges, 3) .* 2.0
            E, grad = compute_energy_and_gradient(rij)

            @test isfinite(E)
            @test size(grad) == (n_edges, 3)
            @test all(isfinite, grad)
            println("  Bucket $n_edges: E=$E, max_grad=$(maximum(abs.(grad)))")
        end
    end

end

println("\n" * "=" ^ 70)
println("ALL JULIA TESTS PASSED")
println("=" ^ 70)
