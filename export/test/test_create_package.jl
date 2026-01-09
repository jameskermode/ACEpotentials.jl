"""
Test create_package.jl pipeline automation.

This test verifies the full package creation pipeline:
1. Package creation completes successfully
2. MLIR has no scatter/gather ops (Reactant-compatible)
3. params.npz contains all required arrays
4. Package structure is correct
5. Python tests pass (if pytest available)

Run with:
    julia +1.11 --project=. test/test_create_package.jl
"""

using Test
using NPZ

# Path to create_package.jl
const CREATE_PACKAGE_SCRIPT = joinpath(@__DIR__, "..", "scripts", "create_package.jl")

# Required keys in params.npz
const REQUIRED_PARAM_KEYS = [
    "selector_R",
    "selector_Y",
    "symm_sel1",
    "symm_sel2_1",
    "symm_sel2_2",
    "A2Bmap",
    "params",
    "rcut",
    "n_polys",
    "maxl",
    "n_species",
    "species_Z",
]

# Required files in package structure
const REQUIRED_PACKAGE_FILES = [
    "pyproject.toml",
    "src/{name}/__init__.py",
    "src/{name}/calculator.py",
    "src/{name}/_iree_wrapper.py",
    "src/{name}/_device.py",
    "src/{name}/models/metadata.json",
    "src/{name}/models/params.npz",
    "tests/test_calculator.py",
    "tests/conftest.py",
]

"""
Run create_package.jl and return the output directory.
"""
function run_create_package(output_dir::String; name::String="testpkg")
    cmd = `julia +1.11 --project=$(dirname(@__DIR__)) $CREATE_PACKAGE_SCRIPT --test-model --name $name --output $output_dir`

    # Capture output
    output = IOBuffer()
    err = IOBuffer()

    try
        run(pipeline(cmd, stdout=output, stderr=err))
        return true, String(take!(output)), String(take!(err))
    catch e
        return false, String(take!(output)), String(take!(err))
    end
end

"""
Check MLIR file for scatter/gather operations.
"""
function check_mlir_no_scatter(mlir_path::String)
    if !isfile(mlir_path)
        return false, "MLIR file not found: $mlir_path"
    end

    content = read(mlir_path, String)

    # Check for scatter/gather ops that would fail in IREE
    forbidden_ops = ["mhlo.scatter", "stablehlo.scatter", "mhlo.gather", "stablehlo.gather"]

    for op in forbidden_ops
        if occursin(op, content)
            return false, "Found forbidden operation: $op"
        end
    end

    return true, "No scatter/gather ops found"
end

"""
Check params.npz contains all required keys.
"""
function check_params_npz(npz_path::String)
    if !isfile(npz_path)
        return false, String[], "params.npz not found: $npz_path"
    end

    params = NPZ.npzread(npz_path)
    keys_found = collect(keys(params))

    missing_keys = String[]
    for key in REQUIRED_PARAM_KEYS
        if !(key in keys_found)
            push!(missing_keys, key)
        end
    end

    if isempty(missing_keys)
        return true, keys_found, "All required keys present"
    else
        return false, keys_found, "Missing keys: $(join(missing_keys, ", "))"
    end
end

"""
Check package structure contains required files.
"""
function check_package_structure(pkg_dir::String, name::String)
    missing_files = String[]

    for file_template in REQUIRED_PACKAGE_FILES
        file = replace(file_template, "{name}" => name)
        full_path = joinpath(pkg_dir, file)
        if !isfile(full_path)
            push!(missing_files, file)
        end
    end

    if isempty(missing_files)
        return true, "All required files present"
    else
        return false, "Missing files: $(join(missing_files, ", "))"
    end
end

"""
Check if VMFB was compiled (optional - depends on iree-compile availability).
"""
function check_vmfb_compiled(models_dir::String)
    vmfb_cpu = joinpath(models_dir, "model_cpu.vmfb")

    if isfile(vmfb_cpu)
        size_kb = filesize(vmfb_cpu) / 1024
        return true, "model_cpu.vmfb present ($size_kb KB)"
    else
        return false, "model_cpu.vmfb not found (iree-compile may not be available)"
    end
end

"""
Run pytest on the generated package (optional - depends on Python/pytest availability).
"""
function run_pytest(pkg_dir::String)
    # Check if pytest is available
    pytest_available = try
        run(pipeline(`which pytest`, stdout=devnull, stderr=devnull))
        true
    catch
        false
    end

    if !pytest_available
        return nothing, "pytest not available"
    end

    # Check if VMFB exists (tests will skip without it)
    vmfb_path = joinpath(pkg_dir, "src", basename(pkg_dir), "models", "model_cpu.vmfb")
    if !isfile(vmfb_path)
        return nothing, "Skipping pytest - no VMFB compiled"
    end

    # Create venv and run tests
    try
        cd(pkg_dir) do
            # Install package
            run(`uv venv .venv`)
            run(`uv pip install -e . pytest -q`)

            # Run tests
            output = read(`uv run pytest tests/ -v`, String)

            # Check for failures
            if occursin("failed", lowercase(output)) && !occursin("0 failed", lowercase(output))
                return false, output
            else
                return true, output
            end
        end
    catch e
        return false, "pytest failed: $e"
    end
end


@testset "create_package.jl Pipeline" begin
    # Create temporary output directory
    output_dir = mktempdir()
    pkg_name = "testpkg"
    pkg_dir = joinpath(output_dir, pkg_name)

    @testset "Package Creation" begin
        @info "Running create_package.jl..." output_dir
        success, stdout_str, stderr_str = run_create_package(output_dir; name=pkg_name)

        if !success
            @error "Package creation failed" stdout=stdout_str stderr=stderr_str
        end

        @test success
        @test isdir(pkg_dir)
        # Check for package summary in stdout or success message in stderr
        @test occursin("Package Summary", stdout_str) || occursin("Package created successfully", stderr_str)
    end

    @testset "MLIR Generation" begin
        mlir_path = joinpath(output_dir, ".build", "main_0.mlir")

        @test isfile(mlir_path)

        if isfile(mlir_path)
            # Check file size is reasonable
            size_kb = filesize(mlir_path) / 1024
            @test size_kb > 1  # At least 1KB
            @info "MLIR file size: $size_kb KB"

            # Check no scatter/gather ops
            ok, msg = check_mlir_no_scatter(mlir_path)
            @test ok
            if !ok
                @error msg
            end
        end
    end

    @testset "Parameters Export" begin
        params_path = joinpath(pkg_dir, "src", pkg_name, "models", "params.npz")

        @test isfile(params_path)

        if isfile(params_path)
            ok, keys_found, msg = check_params_npz(params_path)
            @test ok
            if !ok
                @error msg keys_found
            else
                @info "params.npz keys: $(join(keys_found, ", "))"
            end
        end
    end

    @testset "Package Structure" begin
        ok, msg = check_package_structure(pkg_dir, pkg_name)
        @test ok
        if !ok
            @error msg
        end
    end

    @testset "Metadata" begin
        metadata_path = joinpath(pkg_dir, "src", pkg_name, "models", "metadata.json")

        @test isfile(metadata_path)

        if isfile(metadata_path)
            content = read(metadata_path, String)

            # Check required fields
            @test occursin("\"cutoff\"", content)
            @test occursin("\"elements\"", content)
            @test occursin("\"max_atoms\"", content)
            @test occursin("\"species_Z\"", content)
        end
    end

    @testset "VMFB Compilation" begin
        models_dir = joinpath(pkg_dir, "src", pkg_name, "models")
        ok, msg = check_vmfb_compiled(models_dir)

        # This is optional - mark as broken if not available
        if ok
            @test ok
            @info msg
        else
            @warn msg
            @test_skip "VMFB not compiled - iree-compile not available"
        end
    end

    @testset "Python Tests" begin
        result, msg = run_pytest(pkg_dir)

        if result === nothing
            @warn msg
            @test_skip msg
        elseif result
            @test true
            @info "Python tests passed"
        else
            @test false
            @error "Python tests failed" output=msg
        end
    end

    # Cleanup
    @info "Test complete. Output directory: $output_dir"
    # Note: Not removing output_dir to allow inspection on failure
end
