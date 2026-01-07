#=
ACE Reactant Export Benchmark Suite
====================================

Compare performance of different ACE evaluation backends:
1. Julia native (ETACEPotential with Zygote)
2. Reactant CPU (compiled via Reactant.jl)
3. Reactant GPU (CUDA backend)
4. IREE CPU (deployed VMFB)
5. IREE GPU (CUDA VMFB)

Also validates numerical equivalence between backends.
=#

using ACEpotentials
using BenchmarkTools
using Statistics
using Printf
using Random
using LinearAlgebra

# Only load ACEExport if available
const HAS_EXPORT = try
    using ACEExport
    true
catch
    @warn "ACEExport not available. Install with: using Pkg; Pkg.develop(path=\"export/\")"
    false
end

## ============================================================================
## Test System Generation
## ============================================================================

"""
    generate_test_systems()

Generate a variety of test systems for benchmarking.
"""
function generate_test_systems()
    systems = Dict{String, Any}()

    # Silicon diamond structures of various sizes
    for n in [2, 4, 6, 8]
        atoms = n^3 * 8  # diamond structure has 8 atoms per cubic cell
        name = "Si_diamond_$(n)x$(n)x$(n)_$(atoms)atoms"
        systems[name] = (
            elements = [:Si],
            structure = :diamond,
            repeat = (n, n, n),
            a = 5.43,
            expected_atoms = atoms
        )
    end

    # FCC metals
    for (el, a) in [(:Al, 4.05), (:Cu, 3.615), (:Au, 4.078)]
        for n in [3, 5, 7]
            atoms = n^3 * 4  # FCC has 4 atoms per cubic cell
            name = "$(el)_fcc_$(n)x$(n)x$(n)_$(atoms)atoms"
            systems[name] = (
                elements = [el],
                structure = :fcc,
                repeat = (n, n, n),
                a = a,
                expected_atoms = atoms
            )
        end
    end

    return systems
end

"""
    build_system(spec)

Build an atomic system from specification.
"""
function build_system(spec)
    # Use AtomsBuilder to create the structure
    using AtomsBuilder

    if spec.structure == :diamond
        sys = bulk(spec.elements[1]; cubic=true, pbc=true) * spec.repeat
    elseif spec.structure == :fcc
        sys = bulk(spec.elements[1], :fcc; a=spec.a, cubic=true, pbc=true) * spec.repeat
    else
        error("Unknown structure: $(spec.structure)")
    end

    # Add small random perturbations
    rng = Random.MersenneTwister(42)
    positions = copy(position(sys))
    for i in 1:length(positions)
        positions[i] += 0.01 * randn(rng, 3)
    end

    return sys
end

## ============================================================================
## Backend Implementations
## ============================================================================

"""
    benchmark_julia_native(calc, system; n_runs=100)

Benchmark native Julia ACE evaluation.
"""
function benchmark_julia_native(calc, system; n_runs=100)
    # Warmup
    AtomsCalculators.potential_energy(system, calc)

    # Benchmark
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed begin
            AtomsCalculators.potential_energy(system, calc)
        end
        push!(times, t)
    end

    return (
        mean = mean(times) * 1000,  # ms
        std = std(times) * 1000,
        min = minimum(times) * 1000,
        median = median(times) * 1000,
        throughput = length(system) / mean(times)  # atoms/sec
    )
end

"""
    benchmark_julia_efv(calc, system; n_runs=100)

Benchmark native Julia energy+forces+virial.
"""
function benchmark_julia_efv(calc, system; n_runs=100)
    # Warmup
    AtomsCalculators.energy_forces_virial(system, calc)

    # Benchmark
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed begin
            AtomsCalculators.energy_forces_virial(system, calc)
        end
        push!(times, t)
    end

    return (
        mean = mean(times) * 1000,
        std = std(times) * 1000,
        min = minimum(times) * 1000,
        median = median(times) * 1000,
        throughput = length(system) / mean(times)
    )
end

## ============================================================================
## Numerical Validation
## ============================================================================

"""
    validate_efv(ref_E, ref_F, ref_V, test_E, test_F, test_V;
                 E_rtol=1e-5, F_rtol=1e-4, V_rtol=1e-4)

Validate energy, forces, virial against reference.
"""
function validate_efv(ref_E, ref_F, ref_V, test_E, test_F, test_V;
                      E_rtol=1e-5, F_rtol=1e-4, V_rtol=1e-4)
    # Energy
    E_err = abs(test_E - ref_E) / (abs(ref_E) + 1e-10)
    E_pass = E_err < E_rtol

    # Forces
    F_err = norm(test_F .- ref_F) / (norm(ref_F) + 1e-10)
    F_pass = F_err < F_rtol

    # Virial
    V_err = norm(test_V .- ref_V) / (norm(ref_V) + 1e-10)
    V_pass = V_err < V_rtol

    return (
        energy = (pass=E_pass, error=E_err),
        forces = (pass=F_pass, error=F_err),
        virial = (pass=V_pass, error=V_err),
        all_pass = E_pass && F_pass && V_pass
    )
end

## ============================================================================
## Main Benchmark Runner
## ============================================================================

"""
    run_benchmarks(calc; systems=nothing, n_runs=100)

Run full benchmark suite.
"""
function run_benchmarks(calc; systems=nothing, n_runs=100)
    if isnothing(systems)
        all_systems = generate_test_systems()
        # Filter to just a few representative systems
        system_names = [
            "Si_diamond_2x2x2_64atoms",
            "Si_diamond_4x4x4_512atoms",
            "Al_fcc_5x5x5_500atoms",
        ]
        systems = Dict(k => all_systems[k] for k in system_names if haskey(all_systems, k))
    end

    results = Dict{String, Any}()

    for (name, spec) in systems
        @info "Benchmarking $name..."

        try
            sys = build_system(spec)

            # Julia native energy only
            energy_stats = benchmark_julia_native(calc, sys; n_runs=n_runs)

            # Julia native EFV
            efv_stats = benchmark_julia_efv(calc, sys; n_runs=n_runs)

            results[name] = Dict(
                "n_atoms" => length(sys),
                "julia_energy" => energy_stats,
                "julia_efv" => efv_stats,
            )

            @info "  Energy: $(round(energy_stats.mean, digits=3)) ± $(round(energy_stats.std, digits=3)) ms"
            @info "  EFV:    $(round(efv_stats.mean, digits=3)) ± $(round(efv_stats.std, digits=3)) ms"
            @info "  Throughput: $(round(efv_stats.throughput/1000, digits=1))k atoms/sec"

        catch e
            @warn "Failed to benchmark $name: $e"
            results[name] = Dict("error" => string(e))
        end
    end

    return results
end

"""
    print_results_table(results)

Print results in a formatted table.
"""
function print_results_table(results)
    println("\n" * "="^80)
    println("BENCHMARK RESULTS")
    println("="^80)
    println()

    # Header
    @printf "%-30s %8s %10s %10s %12s\n" "System" "Atoms" "E (ms)" "EFV (ms)" "Throughput"
    println("-"^80)

    for (name, data) in sort(collect(results), by=x->get(x[2], "n_atoms", 0))
        if haskey(data, "error")
            @printf "%-30s %8s ERROR: %s\n" name "-" data["error"]
        else
            n_atoms = data["n_atoms"]
            e_time = data["julia_energy"].mean
            efv_time = data["julia_efv"].mean
            throughput = data["julia_efv"].throughput / 1000

            @printf "%-30s %8d %10.2f %10.2f %10.1fk/s\n" name n_atoms e_time efv_time throughput
        end
    end

    println("="^80)
end

## ============================================================================
## Quick Test
## ============================================================================

"""
    quick_test()

Run a quick test to verify the benchmark infrastructure.
"""
function quick_test()
    @info "Running quick benchmark test..."

    # Create a simple Si model
    model = ace1_model(
        elements = [:Si],
        order = 2,
        totaldegree = 6,
        pure = false,
        pure2b = false
    )

    # Create calculator
    calc = ACEpotentials.ACEPotential(model)

    # Run on small system
    results = run_benchmarks(calc;
        systems = Dict(
            "test_Si_64" => (
                elements = [:Si],
                structure = :diamond,
                repeat = (2, 2, 2),
                a = 5.43,
                expected_atoms = 64
            )
        ),
        n_runs = 10
    )

    print_results_table(results)

    return results
end

## ============================================================================
## Entry Point
## ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    quick_test()
end
