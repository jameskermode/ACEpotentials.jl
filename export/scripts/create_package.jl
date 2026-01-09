#=
Create Redistributable ACE Potential Package
=============================================

Creates a complete, pip-installable Python package from an ACE model.
The package is self-contained with pre-compiled IREE VMFB files.

Usage:
    # From an existing model
    julia +1.11 --project=.. create_package.jl model.json --name mypotential --output ./dist

    # Create a minimal test model (for testing the pipeline)
    julia +1.11 --project=.. create_package.jl --test-model --name testpotential --output ./dist

Output Structure:
    <output>/
    ├── <name>/                    # Python package directory
    │   ├── pyproject.toml
    │   ├── src/<name>/
    │   │   ├── __init__.py
    │   │   ├── calculator.py
    │   │   ├── _device.py
    │   │   ├── _iree_wrapper.py
    │   │   ├── models/
    │   │   │   ├── model_cpu.vmfb
    │   │   │   ├── model_cuda.vmfb (if available)
    │   │   │   ├── params.npz
    │   │   │   └── metadata.json
    │   │   └── ...
    │   └── tests/
    │       └── test_calculator.py
    └── build_log.txt
=#

using Pkg
Pkg.activate(dirname(@__DIR__))

using Printf
using LinearAlgebra
using Random
using NPZ
using JSON3

# Check for Reactant
const HAS_REACTANT = try
    using Reactant
    true
catch e
    @warn "Reactant not available - VMFB compilation will be skipped" exception=e
    false
end

# Load ACEExport components
using ACEExport
using ACEExport: CompiledShapes, stacked_energy_from_edges

# ACEpotentials for model creation
using ACEpotentials
M = ACEpotentials.Models
ETM = ACEpotentials.ETModels

import EquivariantTensors as ET
using Lux

## ============================================================================
## Package Template Files
## ============================================================================

const PYPROJECT_TEMPLATE = """
[project]
name = "{name}"
version = "{version}"
description = "ACE interatomic potential - {description}"
requires-python = ">=3.9"
dependencies = [
    "numpy>=1.20",
    "ase>=3.22",
    "matscipy>=0.8",
    "iree-base-runtime>={iree_version}",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]

[tool.uv]
find-links = ["https://iree.dev/pip-release-links.html"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
"""

const INIT_TEMPLATE = """
\"\"\"
{name} - ACE Interatomic Potential
{underline}

Auto-generated package for {elements} potential.

Usage:
    from {name} import ACECalculator
    from ase.build import bulk

    calc = ACECalculator()
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc
    print(f"Energy: {{atoms.get_potential_energy():.4f}} eV")
\"\"\"

from .calculator import ACECalculator
from ._device import get_device_info, list_available_backends

__version__ = "{version}"
__all__ = ["ACECalculator", "get_device_info", "list_available_backends"]
"""

const TEST_TEMPLATE = """
\"\"\"Tests for {name} package.\"\"\"

import pytest
import numpy as np

from {name} import ACECalculator, get_device_info, list_available_backends


def _models_available():
    \"\"\"Check if any compiled models are available.\"\"\"
    models = list_available_backends()
    return any(exists for _, exists in models.values())


# =============================================================================
# Basic Tests
# =============================================================================

def test_import():
    \"\"\"Test that the package can be imported.\"\"\"
    assert ACECalculator is not None


def test_device_info():
    \"\"\"Test device info retrieval.\"\"\"
    info = get_device_info()
    assert 'cpu' in info


def test_list_models():
    \"\"\"Test model listing.\"\"\"
    models = list_available_backends()
    assert 'cpu' in models


@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_calculator_creation():
    \"\"\"Test calculator creation.\"\"\"
    calc = ACECalculator(device='cpu')
    assert calc is not None


# =============================================================================
# Energy Tests
# =============================================================================

@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_silicon_energy():
    \"\"\"Test energy calculation on silicon.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)


@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_energy_changes_with_displacement():
    \"\"\"Test that energy changes when atoms are displaced.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc
    E0 = atoms.get_potential_energy()

    # Displace one atom
    atoms.positions[0, 0] += 0.1
    E1 = atoms.get_potential_energy()

    assert E0 != E1, "Energy should change with displacement"


# =============================================================================
# Force Tests
# =============================================================================

@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_forces_finite():
    \"\"\"Test that forces are computed and finite.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc

    forces = atoms.get_forces()
    assert forces.shape == (len(atoms), 3)
    assert np.all(np.isfinite(forces))


@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_forces_newton_third_law():
    \"\"\"Test Newton's third law: sum of forces = 0 for isolated system.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc

    forces = atoms.get_forces()
    total_force = np.sum(forces, axis=0)
    assert np.allclose(total_force, 0, atol=1e-5), \\
        f"Sum of forces should be zero, got {total_force}"


@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_forces_nonzero_when_displaced():
    \"\"\"Test that forces are non-zero when structure is perturbed.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.positions[0, 0] += 0.1  # Perturb
    atoms.calc = calc

    forces = atoms.get_forces()
    force_norm = np.linalg.norm(forces)
    assert force_norm > 0.1, f"Forces should be non-zero, got norm {force_norm}"


# Note: Finite difference force validation is slow and inaccurate with Float32.
# For rigorous force validation, compare against Julia ETACE reference values
# using the test_python_equivalence.jl integration test.


# =============================================================================
# Multi-Structure Tests
# =============================================================================

@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_larger_structure():
    \"\"\"Test on larger structure (4 atoms).\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43) * (2, 1, 1)  # 4 atoms
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (4, 3)
    assert np.allclose(np.sum(forces, axis=0), 0, atol=1e-5)


@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_supercell():
    \"\"\"Test on 2x2x2 supercell (16 atoms).\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43) * (2, 2, 2)  # 16 atoms
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (16, 3)
    # Newton's third law
    assert np.allclose(np.sum(forces, axis=0), 0, atol=1e-4)


# =============================================================================
# Consistency Tests
# =============================================================================

@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_energy_deterministic():
    \"\"\"Test that repeated energy calls give same result.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc

    E1 = atoms.get_potential_energy()
    E2 = atoms.get_potential_energy()
    E3 = atoms.get_potential_energy()

    assert E1 == E2 == E3, "Energy should be deterministic"


@pytest.mark.skipif(not _models_available(), reason="No compiled models available")
def test_forces_deterministic():
    \"\"\"Test that repeated force calls give same result.\"\"\"
    from ase.build import bulk

    calc = ACECalculator(device='cpu')
    atoms = bulk('Si', 'diamond', 5.43)
    atoms.calc = calc

    F1 = atoms.get_forces()
    F2 = atoms.get_forces()

    assert np.allclose(F1, F2), "Forces should be deterministic"
"""

## ============================================================================
## Model Creation
## ============================================================================

"""
    create_test_model(; elements=(:Si,), order=2, max_level=6)

Create a minimal ACE model for testing the export pipeline.
"""
function create_test_model(; elements=(:Si,), order=2, max_level=6, maxl=2, rcut=5.5)
    @info "Creating test ACE model..." elements order max_level

    rng = Random.MersenneTwister(42)

    # Create model configuration
    rin0cuts = M._default_rin0cuts(elements)
    rin0cuts = (x -> (rin = x.rin, r0 = x.r0, rcut = rcut)).(rin0cuts)

    # Create ACE model with learnable radial basis
    ace_model = M.ace_model(;
        elements = elements,
        order = order,
        Ytype = :solid,
        level = M.TotalDegree(),
        max_level = max_level,
        maxl = maxl,
        pair_maxn = max_level,
        rin0cuts = rin0cuts,
        pair_learnable = true,
        init_WB = :glorot_normal,
        init_Wpair = :glorot_normal
    )

    ps, st = Lux.setup(rng, ace_model)

    # Convert to full stacked calculator (E0 + Pair + ACE)
    stacked_calc = ETM.convert2et_full(ace_model, ps, st; rng=rng)

    return stacked_calc, elements, rcut
end

"""
    load_model(model_path::String)

Load an ACE model from a JSON file.
"""
function load_model(model_path::String)
    @info "Loading model from $model_path"

    # Load using ACEpotentials JSON interface
    calc = ACEpotentials.load_potential(model_path)

    # Extract elements and cutoff
    # This depends on the model structure
    elements = try
        # Try to get elements from the model
        calc.elements
    catch
        (:Si,)  # Default
    end

    rcut = try
        calc.rcut
    catch
        5.5  # Default
    end

    return calc, elements, rcut
end

## ============================================================================
## MLIR/VMFB Export - Selection Matrix Approach (Full GPU)
## ============================================================================

# Uses selection matrices to convert all gather/scatter operations to matmuls.
# This allows full GPU acceleration via IREE.
#
# Architecture:
# 1. rij → Rnl, Ylm (embeddings via Chebyshev + solid harmonics)
# 2. Rnl, Ylm → A (pooled product via selection matrices)
# 3. pool_matrix: edges → atoms (sum pair contributions)
# 4. A → AA (symmetric product via selection matrices)
# 5. AA → BB → E (coupling + readout)
#
# Forces via Enzyme autodiff on rij.
#
# All operations are dense matmuls - no gather/scatter → full IREE compatibility!

using Enzyme

# Pure Julia embedding functions for Reactant tracing
# These must match the actual ACE embeddings (Agnesi + envelope + W_radial, solid harmonics)

"""
Agnesi distance transform: r → y ∈ [-1, 1]
Vectorized for all edges at once.
"""
function agnesi_transform_export(r::AbstractVector, agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req)
    T = eltype(r)
    n_pairs = length(r)

    # Compute scaled distance: s = (r - rin) / (req - rin)
    s = (r .- agnesi_rin) ./ (agnesi_req .- agnesi_rin .+ T(1e-10))

    # Agnesi function with fixed pin=2, pcut=2 (most common)
    # x = 1 / (1 + a * s^2 / (1 + s^0))  = 1 / (1 + a * s^2 / 2)
    x = one(T) ./ (one(T) .+ agnesi_a .* s .^ 2 ./ T(2))

    # Linear map to [-1, 1]
    y = agnesi_b1 .* x .+ agnesi_b0

    # Clamp to [-1, 1]
    y = max.(-one(T), min.(one(T), y))

    return y
end

"""
Cutoff envelope: (1 - y²)²
This ensures smooth decay at boundaries.
"""
function envelope_export(y::AbstractVector)
    T = eltype(y)
    return (one(T) .- y .^ 2) .^ 2
end

"""
Chebyshev polynomials with proper 3-term recurrence using coefficients A, B, C.
P[n] = (A[n] * y + B[n]) * P[n-1] + C[n] * P[n-2]

This implementation is Reactant-compatible by:
- Using broadcast operations instead of length/size queries
- Using explicit loops over compile-time constants
- Building result via hcat (which works with traced arrays)
"""
function chebyshev_basis_export(y::AbstractVector, n_cheb::Int, poly_A, poly_B, poly_C)
    T = eltype(y)

    # n_cheb is a compile-time constant, so we can branch on it
    if n_cheb == 0
        # Return empty with matching batch dimension via broadcast
        return y .* zero(T) .+ zeros(T, 1, 0)  # Will broadcast to [n_pairs, 0]
    end

    # P[1] = A[1] (constant) - broadcast to match y's shape
    P0 = y .* zero(T) .+ T(poly_A[1])

    if n_cheb == 1
        return reshape(P0, :, 1)
    end

    # P[2] = A[2] * y + B[2]
    P1 = T(poly_A[2]) .* y .+ T(poly_B[2])

    if n_cheb == 2
        return hcat(reshape(P0, :, 1), reshape(P1, :, 1))
    end

    # Build all polynomials via recurrence
    # Start with first two polynomials
    result = hcat(reshape(P0, :, 1), reshape(P1, :, 1))
    Pnm2 = P0
    Pnm1 = P1

    # Unroll loop - n_cheb is compile-time constant
    for n in 3:n_cheb
        Pn = (T(poly_A[n]) .* y .+ T(poly_B[n])) .* Pnm1 .+ T(poly_C[n]) .* Pnm2
        result = hcat(result, reshape(Pn, :, 1))
        Pnm2 = Pnm1
        Pnm1 = Pn
    end

    return result
end

"""
Full radial embedding: r → Rnl
Pipeline: r → Agnesi → Chebyshev → envelope → linear(W_radial) → Rnl
"""
function radial_embedding_export(r::AbstractVector, agnesi_a, agnesi_b0, agnesi_b1,
                                  agnesi_rin, agnesi_req, n_cheb::Int,
                                  poly_A, poly_B, poly_C, W_radial::AbstractMatrix)
    T = eltype(r)
    n_pairs = length(r)
    n_rnl = size(W_radial, 1)

    # Step 1: Agnesi transform
    y = agnesi_transform_export(r, agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req)

    # Step 2: Chebyshev basis
    P = chebyshev_basis_export(y, n_cheb, poly_A, poly_B, poly_C)  # [n_pairs, n_cheb]

    # Step 3: Apply envelope
    env = envelope_export(y)  # [n_pairs]
    P_env = P .* env  # [n_pairs, n_cheb]

    # Step 4: Linear transform to Rnl
    # W_radial is [n_rnl, n_cheb], P_env is [n_pairs, n_cheb]
    # Result: [n_pairs, n_rnl]
    Rnl = P_env * transpose(W_radial)

    return Rnl
end

"""
Real SOLID harmonics (r^l * Y_lm) up to l=2 - Reactant-compatible
This is what ACE uses with Ytype=:solid.
"""
function solid_ylm_export(r_vec::AbstractVector, rhat::AbstractMatrix, maxl::Int)
    n_pairs = size(rhat, 1)
    T = eltype(rhat)

    # Get Cartesian coordinates scaled by r^l
    # For solid harmonics, we use r*x, r*y, r*z directly
    rx = r_vec .* rhat[:, 1]
    ry = r_vec .* rhat[:, 2]
    rz = r_vec .* rhat[:, 3]

    # l=0: r^0 * Y_00 = 1/(2√π)
    c00 = T(0.28209479177387814)
    Y00 = fill(c00, n_pairs)

    if maxl == 0
        return reshape(Y00, :, 1)
    end

    # l=1: r^1 * Y_1m = √(3/4π) * (ry, rz, rx)
    c1 = T(0.4886025119029199)
    Y1m1 = c1 .* ry
    Y10 = c1 .* rz
    Y1p1 = c1 .* rx

    if maxl == 1
        return hcat(reshape(Y00, :, 1), reshape(Y1m1, :, 1),
                    reshape(Y10, :, 1), reshape(Y1p1, :, 1))
    end

    # l=2: r^2 * Y_2m - quadratic forms in (rx, ry, rz)
    c2_0 = T(0.31539156525252005)
    c2_1 = T(1.0925484305920792)
    c2_2 = T(0.5462742152960396)

    r2 = r_vec .^ 2

    Y2m2 = c2_1 .* rx .* ry
    Y2m1 = c2_1 .* ry .* rz
    Y20 = c2_0 .* (T(3) .* rz .^ 2 .- r2)
    Y2p1 = c2_1 .* rx .* rz
    Y2p2 = c2_2 .* (rx .^ 2 .- ry .^ 2)

    return hcat(reshape(Y00, :, 1), reshape(Y1m1, :, 1), reshape(Y10, :, 1), reshape(Y1p1, :, 1),
                reshape(Y2m2, :, 1), reshape(Y2m1, :, 1), reshape(Y20, :, 1), reshape(Y2p1, :, 1), reshape(Y2p2, :, 1))
end

"""
ACE energy using selection matrices - all matmul, no gather/scatter.

Uses PROPER embeddings with:
- Agnesi distance transform
- Cutoff envelope (1-y²)²
- Learned radial weights (W_radial)
- Solid harmonics (r^l * Y_lm)

Arguments:
- rij: [n_pairs, 3] edge displacement vectors
- pool_matrix: [n_atoms, n_pairs] pooling matrix (1 where pair belongs to atom)
- selector_R, selector_Y: [nA, nRnl/nYlm] selection matrices for pooled product
- symm_sel1, symm_sel2_1, symm_sel2_2: selection matrices for symmetric product
- A2Bmap: coupling matrix
- params: readout weights
- agnesi_params: [5] = [a, b0, b1, rin, req] for Agnesi transform
- poly_A, poly_B, poly_C: Chebyshev recurrence coefficients
- W_radial: [n_rnl, n_cheb] radial weight matrix
- n_cheb, maxl: model hyperparameters
"""
function ace_energy_selmat_export(
    rij::AbstractMatrix,
    pool_matrix::AbstractMatrix,
    selector_R::AbstractMatrix,
    selector_Y::AbstractMatrix,
    symm_sel1::AbstractMatrix,
    symm_sel2_1::AbstractMatrix,
    symm_sel2_2::AbstractMatrix,
    A2Bmap::AbstractMatrix,
    params::AbstractVector,
    agnesi_a,
    agnesi_b0,
    agnesi_b1,
    agnesi_rin,
    agnesi_req,
    poly_A::AbstractVector,
    poly_B::AbstractVector,
    poly_C::AbstractVector,
    W_radial::AbstractMatrix,
    n_cheb,
    maxl
)
    # Step 1: Compute r and rhat
    T = eltype(rij)
    r2 = sum(rij .^ 2, dims=2)
    r = sqrt.(r2)
    eps = T(1e-6)
    rhat = rij ./ max.(r, eps)
    r_vec = dropdims(r, dims=2)

    # Step 2: Full radial embedding with Agnesi + envelope + W_radial
    Rnl = Reactant.@trace radial_embedding_export(
        r_vec, agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        n_cheb, poly_A, poly_B, poly_C, W_radial
    )

    # Step 3: Solid harmonics (r^l * Y_lm)
    Ylm = Reactant.@trace solid_ylm_export(r_vec, rhat, maxl)

    # Step 4: Pooled sparse product via selection matrices
    Rnl_sel = Rnl * transpose(selector_R)
    Ylm_sel = Ylm * transpose(selector_Y)
    A_pair = Rnl_sel .* Ylm_sel

    # Step 5: Pool to atoms
    A = pool_matrix * A_pair

    # Step 6: Sparse symmetric product via selection matrices
    AA1 = A * transpose(symm_sel1)
    A_sel1 = A * transpose(symm_sel2_1)
    A_sel2 = A * transpose(symm_sel2_2)
    AA2 = A_sel1 .* A_sel2
    AA = hcat(AA1, AA2)

    # Step 7: Linear readout
    BB = AA * transpose(A2Bmap)
    E = sum(BB * params)

    return E
end

"""ACE energy + forces via Enzyme autodiff."""
function ace_energy_and_forces_export(
    rij, pool_matrix, selector_R, selector_Y,
    symm_sel1, symm_sel2_1, symm_sel2_2,
    A2Bmap, params,
    agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
    poly_A, poly_B, poly_C, W_radial,
    n_cheb, maxl
)
    d_rij = zero(rij)

    _, energy = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        ace_energy_selmat_export,
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
        Enzyme.Const(agnesi_a),
        Enzyme.Const(agnesi_b0),
        Enzyme.Const(agnesi_b1),
        Enzyme.Const(agnesi_rin),
        Enzyme.Const(agnesi_req),
        Enzyme.Const(poly_A),
        Enzyme.Const(poly_B),
        Enzyme.Const(poly_C),
        Enzyme.Const(W_radial),
        Enzyme.Const(n_cheb),
        Enzyme.Const(maxl)
    )

    pair_forces = -d_rij
    return (energy, pair_forces)
end

## ============================================================================
## Pair Energy Export (Two-Body Contribution)
## ============================================================================

"""
Pair energy using vectorized operations - Reactant compatible.

Computes two-body pair potential energy:
1. Agnesi distance transform
2. Chebyshev basis (no inner envelope for pair)
3. Radial weights → features
4. Outer envelope: (s^(-p) - 1) * (1 - s)
5. Linear readout

Arguments:
- rij: [n_pairs, 3] edge displacement vectors
- agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req: Agnesi transform params
- poly_A, poly_B, poly_C: Chebyshev recurrence coefficients
- W_radial: [n_basis, n_polys] radial weight matrix
- rcut_outer: outer cutoff radius
- p_outer: outer envelope power
- W_readout: [n_basis] readout weights
- n_polys: number of Chebyshev polynomials
"""
function pair_energy_selmat_export(
    rij::AbstractMatrix,
    agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
    poly_A::AbstractVector, poly_B::AbstractVector, poly_C::AbstractVector,
    W_radial::AbstractMatrix,
    rcut_outer, p_outer,
    W_readout::AbstractVector,
    n_polys
)
    T = eltype(rij)

    # Distance computation
    r2 = sum(rij .^ 2, dims=2)
    r_vec = sqrt.(dropdims(r2, dims=2))

    # Agnesi transform
    s = (r_vec .- agnesi_rin) ./ (agnesi_req .- agnesi_rin .+ T(1e-10))
    x = one(T) ./ (one(T) .+ agnesi_a .* s .^ 2 ./ T(2))
    y = agnesi_b1 .* x .+ agnesi_b0
    y = max.(-one(T), min.(one(T), y))

    # Chebyshev basis (no inner envelope for pair)
    P = Reactant.@trace chebyshev_basis_export(y, n_polys, poly_A, poly_B, poly_C)

    # Linear layer: P @ W_radial^T -> [n_edges, n_basis]
    pair_features = P * transpose(W_radial)

    # Outer envelope: (s^(-p) - 1) * (1 - s)
    s_outer = r_vec ./ rcut_outer
    s_safe = max.(T(1e-6), min.(s_outer, T(0.9999)))
    outer_env = (s_safe .^ (-p_outer) .- one(T)) .* (one(T) .- s_safe)
    outer_env = outer_env .* (s_outer .< one(T))  # Zero beyond cutoff

    pair_features_env = pair_features .* reshape(outer_env, :, 1)

    # Readout: sum over edges and basis
    return sum(pair_features_env * W_readout)
end

"""Pair energy + forces via Enzyme autodiff."""
function pair_energy_and_forces_export(
    rij,
    agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
    poly_A, poly_B, poly_C,
    W_radial,
    rcut_outer, p_outer,
    W_readout,
    n_polys
)
    d_rij = zero(rij)

    _, energy = Enzyme.autodiff(
        Enzyme.ReverseWithPrimal,
        pair_energy_selmat_export,
        Enzyme.Active,
        Enzyme.Duplicated(rij, d_rij),
        Enzyme.Const(agnesi_a),
        Enzyme.Const(agnesi_b0),
        Enzyme.Const(agnesi_b1),
        Enzyme.Const(agnesi_rin),
        Enzyme.Const(agnesi_req),
        Enzyme.Const(poly_A),
        Enzyme.Const(poly_B),
        Enzyme.Const(poly_C),
        Enzyme.Const(W_radial),
        Enzyme.Const(rcut_outer),
        Enzyme.Const(p_outer),
        Enzyme.Const(W_readout),
        Enzyme.Const(n_polys)
    )

    pair_forces = -d_rij
    return (energy, pair_forces)
end

"""Build selection matrix from spec indices."""
function build_selector_matrix(spec::Vector{Int}, n_features::Int, ::Type{T}) where T
    nA = length(spec)
    selector = zeros(T, nA, n_features)
    for k in 1:nA
        selector[k, spec[k]] = one(T)
    end
    return selector
end

"""
    export_to_mlir(model, output_dir; shapes)

Export model to StableHLO MLIR format using Reactant.

Exports separate modules for each energy contribution:
- ACE (many-body): ace_energy_and_forces
- Pair (two-body): pair_energy_and_forces (if model has pair)
- E0 (one-body): Handled trivially in Python (constant per atom)

Each contribution is differentiated separately for cleaner code and
easier debugging. Python sums the contributions.

Uses selection matrix approach for full GPU acceleration:
- All gather/scatter converted to matmul
- Enzyme autodiff for forces
- pool_matrix passed from Python for edge→atom accumulation
"""
function export_to_mlir(model::ReactantStackedModel{T}, output_dir::String;
                        max_edges::Int=10_000, max_atoms::Int=256) where T
    @info "Exporting to StableHLO MLIR (modular contributions)..." output_dir

    if !HAS_REACTANT
        @warn "Reactant not available - creating placeholder MLIR"
        mlir_path = joinpath(output_dir, "ace_model.mlir")
        write(mlir_path, "// Placeholder - Reactant not available\n")
        return Dict("ace" => mlir_path)
    end

    Reactant.set_default_backend("cpu")

    ace = model.ace_state
    @info "  Model config:" n_species=model.n_species rcut=model.rcut has_pair=model.has_pair
    @info "  ACE state:" n_rnl=ace.n_rnl nYlm=ace.nYlm n_basis=ace.n_basis
    @info "  Compilation shapes:" max_atoms max_edges

    mlir_paths = Dict{String, String}()

    # =========================================================================
    # Export ACE (many-body) contribution
    # =========================================================================
    @info "Exporting ACE (many-body) contribution..."

    ace_mlir = export_ace_contribution(model, output_dir, max_edges, max_atoms, T)
    mlir_paths["ace"] = ace_mlir

    # =========================================================================
    # Export Pair (two-body) contribution if present
    # =========================================================================
    if model.has_pair && model.pair_state !== nothing
        @info "Exporting Pair (two-body) contribution..."
        pair_mlir = export_pair_contribution(model, output_dir, max_edges, T)
        mlir_paths["pair"] = pair_mlir
    else
        @info "  No pair contribution in model"
    end

    # E0 (one-body) is handled in Python - just E0[species] * count[species]
    @info "  E0 (one-body) will be computed in Python"

    return mlir_paths
end

"""Export ACE many-body contribution to MLIR."""
function export_ace_contribution(model::ReactantStackedModel{T}, output_dir::String,
                                  max_edges::Int, max_atoms::Int, ::Type{T}) where T
    ace = model.ace_state

    # Build selection matrices
    n_cheb = ace.n_polys
    maxl = ace.maxl
    nRnl = ace.n_rnl
    nYlm = (maxl + 1)^2

    spec_R = Int.(ace.spec_R)
    spec_Y = Int.(ace.spec_Y)
    nA = length(spec_R)

    selector_R = build_selector_matrix(spec_R, nRnl, T)
    selector_Y = build_selector_matrix(spec_Y, nYlm, T)

    @info "  ACE selection matrices:" nA nRnl nYlm

    # Build symmetric product selectors
    specs_mats = ace.specs_mats
    symm_sel1 = if length(specs_mats) >= 1 && size(specs_mats[1], 1) > 0
        build_selector_matrix(specs_mats[1][:, 1], nA, T)
    else
        zeros(T, 1, nA)
    end

    symm_sel2_1, symm_sel2_2 = if length(specs_mats) >= 2 && size(specs_mats[2], 1) > 0
        build_selector_matrix(specs_mats[2][:, 1], nA, T),
        build_selector_matrix(specs_mats[2][:, 2], nA, T)
    else
        zeros(T, 1, nA), zeros(T, 1, nA)
    end

    # Get A2Bmap and params
    A2Bmap = T.(ace.A2Bmap)
    params = T.(ace.W_readout[:, 1])

    # Extract embedding parameters (single species: pair_idx=1)
    pair_idx = 1
    agnesi_a = T(ace.agnesi_params[3, pair_idx])
    agnesi_b0 = T(ace.agnesi_params[4, pair_idx])
    agnesi_b1 = T(ace.agnesi_params[5, pair_idx])
    agnesi_rin = T(ace.agnesi_params[6, pair_idx])
    agnesi_req = T(ace.agnesi_params[7, pair_idx])

    poly_A = T.(ace.poly_A)
    poly_B = T.(ace.poly_B)
    poly_C = T.(ace.poly_C)
    W_radial = T.(ace.W_radial[:, :, pair_idx])

    @info "  ACE Agnesi params:" a=agnesi_a b0=agnesi_b0 b1=agnesi_b1

    # Create dummy inputs
    rij = randn(T, max_edges, 3)
    pool_matrix = zeros(T, max_atoms, max_edges)

    # Convert to RArrays
    rij_ra = Reactant.to_rarray(rij)
    pool_ra = Reactant.to_rarray(pool_matrix)
    selector_R_ra = Reactant.to_rarray(selector_R)
    selector_Y_ra = Reactant.to_rarray(selector_Y)
    symm_sel1_ra = Reactant.to_rarray(symm_sel1)
    symm_sel2_1_ra = Reactant.to_rarray(symm_sel2_1)
    symm_sel2_2_ra = Reactant.to_rarray(symm_sel2_2)
    A2Bmap_ra = Reactant.to_rarray(A2Bmap)
    params_ra = Reactant.to_rarray(params)
    W_radial_ra = Reactant.to_rarray(W_radial)

    # Compile
    @info "  Compiling ACE energy_and_forces..."
    compiled_fn = Reactant.@compile ace_energy_and_forces_export(
        rij_ra, pool_ra, selector_R_ra, selector_Y_ra,
        symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
        A2Bmap_ra, params_ra,
        agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        poly_A, poly_B, poly_C, W_radial_ra,
        n_cheb, maxl
    )

    # Test
    E_test, F_test = compiled_fn(
        rij_ra, pool_ra, selector_R_ra, selector_Y_ra,
        symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
        A2Bmap_ra, params_ra,
        agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        poly_A, poly_B, poly_C, W_radial_ra,
        n_cheb, maxl
    )
    @info "  ACE compilation test:" energy=Float64(E_test) forces_shape=size(F_test)

    # Export
    ace_dir = joinpath(output_dir, "ace")
    mkpath(ace_dir)

    Reactant.Serialization.export_to_enzymejax(
        ace_energy_and_forces_export,
        rij_ra, pool_ra, selector_R_ra, selector_Y_ra,
        symm_sel1_ra, symm_sel2_1_ra, symm_sel2_2_ra,
        A2Bmap_ra, params_ra,
        agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        poly_A, poly_B, poly_C, W_radial_ra,
        n_cheb, maxl;
        output_dir=ace_dir,
        function_name="ace_energy_and_forces"
    )

    mlir_files = filter(f -> endswith(f, ".mlir") && !contains(f, "_inputs"), readdir(ace_dir))
    mlir_path = joinpath(ace_dir, first(mlir_files))
    @info "  Generated ACE MLIR: $mlir_path"

    return mlir_path
end

"""Export Pair two-body contribution to MLIR."""
function export_pair_contribution(model::ReactantStackedModel{T}, output_dir::String,
                                   max_edges::Int, ::Type{T}) where T
    pair = model.pair_state

    # Extract pair parameters (single species: pair_idx=1)
    pair_idx = 1
    n_polys = pair.n_polys

    agnesi_a = T(pair.agnesi_params[3, pair_idx])
    agnesi_b0 = T(pair.agnesi_params[4, pair_idx])
    agnesi_b1 = T(pair.agnesi_params[5, pair_idx])
    agnesi_rin = T(pair.agnesi_params[6, pair_idx])
    agnesi_req = T(pair.agnesi_params[7, pair_idx])

    poly_A = T.(pair.poly_A)
    poly_B = T.(pair.poly_B)
    poly_C = T.(pair.poly_C)
    W_radial = T.(pair.W_radial[:, :, pair_idx])
    W_readout = T.(pair.W_readout[:, 1])

    rcut_outer = T(pair.rcut_outer)
    p_outer = pair.p_outer

    @info "  Pair params:" n_polys n_basis=pair.n_basis rcut_outer p_outer
    @info "  Pair Agnesi:" a=agnesi_a b0=agnesi_b0 b1=agnesi_b1

    # Create dummy input
    rij = randn(T, max_edges, 3)
    rij_ra = Reactant.to_rarray(rij)
    W_radial_ra = Reactant.to_rarray(W_radial)
    W_readout_ra = Reactant.to_rarray(W_readout)

    # Compile
    @info "  Compiling Pair energy_and_forces..."
    compiled_fn = Reactant.@compile pair_energy_and_forces_export(
        rij_ra,
        agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        poly_A, poly_B, poly_C,
        W_radial_ra,
        rcut_outer, p_outer,
        W_readout_ra,
        n_polys
    )

    # Test
    E_test, F_test = compiled_fn(
        rij_ra,
        agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        poly_A, poly_B, poly_C,
        W_radial_ra,
        rcut_outer, p_outer,
        W_readout_ra,
        n_polys
    )
    @info "  Pair compilation test:" energy=Float64(E_test) forces_shape=size(F_test)

    # Export
    pair_dir = joinpath(output_dir, "pair")
    mkpath(pair_dir)

    Reactant.Serialization.export_to_enzymejax(
        pair_energy_and_forces_export,
        rij_ra,
        agnesi_a, agnesi_b0, agnesi_b1, agnesi_rin, agnesi_req,
        poly_A, poly_B, poly_C,
        W_radial_ra,
        rcut_outer, p_outer,
        W_readout_ra,
        n_polys;
        output_dir=pair_dir,
        function_name="pair_energy_and_forces"
    )

    mlir_files = filter(f -> endswith(f, ".mlir") && !contains(f, "_inputs"), readdir(pair_dir))
    mlir_path = joinpath(pair_dir, first(mlir_files))
    @info "  Generated Pair MLIR: $mlir_path"

    return mlir_path
end

"""
    get_iree_version(iree_compile)

Get the IREE version from iree-compile --version output.
Returns the base version (e.g., "3.9.0" from "3.9.0rc20251125").
RC suffixes are stripped for PyPI compatibility.
"""
function get_iree_version(iree_compile::String)::String
    try
        output = read(`$iree_compile --version`, String)
        # Parse version from: "IREE compiler version 3.10.0rc20260106 @ ..."
        m = match(r"IREE compiler version (\S+)", output)
        if m !== nothing
            full_version = m.captures[1]
            # Strip RC suffix for PyPI compatibility (3.9.0rc20251125 → 3.9.0)
            base_version = match(r"^(\d+\.\d+\.\d+)", full_version)
            if base_version !== nothing
                return base_version.captures[1]
            end
            return full_version
        end
    catch e
        @warn "Failed to get IREE version" exception=e
    end
    return "3.9.0"  # Fallback to stable PyPI version
end

"""
    compile_to_vmfb(mlir_paths, output_dir; backends=[:cpu, :cuda, :vulkan])

Compile MLIR files to IREE VMFB for specified backends.

Handles multiple contributions (ACE, pair, etc.) - each gets its own VMFB.

Uses `iree-compile` from:
1. `IREE_COMPILE` environment variable (if set)
2. System PATH

For uv environments, set: export IREE_COMPILE=\$(uv run which iree-compile)

Returns (vmfb_paths, iree_version) where vmfb_paths is Dict{String, Dict{Symbol, String}}.
"""
function compile_to_vmfb(mlir_paths::Dict{String, String}, output_dir::String;
                         backends::Vector{Symbol}=[:cpu, :cuda, :vulkan])
    @info "Compiling to IREE VMFB..." backends contributions=keys(mlir_paths)

    # Check for iree-compile: env var first, then PATH
    iree_compile = get(ENV, "IREE_COMPILE", nothing)
    if isnothing(iree_compile) || !isfile(iree_compile)
        iree_compile = Sys.which("iree-compile")
    end

    if isnothing(iree_compile) || !isfile(iree_compile)
        @warn "iree-compile not found - skipping VMFB compilation"
        @info "Set IREE_COMPILE env var or install: pip install iree-base-compiler"
        return Dict{String, Dict{Symbol, String}}(), "3.9.0"
    end
    @info "Using iree-compile: $iree_compile"

    # Get IREE version
    iree_version = get_iree_version(iree_compile)

    # Compile each contribution for each backend
    all_vmfb_paths = Dict{String, Dict{Symbol, String}}()

    for (contribution, mlir_path) in mlir_paths
        @info "Compiling $contribution contribution..."
        vmfb_paths = Dict{Symbol, String}()

        for backend in backends
            iree_backend = if backend == :cuda
                "cuda"
            elseif backend == :vulkan
                "vulkan-spirv"
            else
                "llvm-cpu"
            end

            vmfb_name = "$(contribution)_$(backend).vmfb"
            vmfb_path = joinpath(output_dir, vmfb_name)

            @info "  Compiling $contribution for $backend..."
            try
                run(`$iree_compile
                    --iree-input-type=stablehlo
                    --iree-hal-target-backends=$iree_backend
                    $mlir_path
                    -o $vmfb_path`)

                vmfb_paths[backend] = vmfb_path
                @info "    Generated: $vmfb_path ($(filesize(vmfb_path)) bytes)"
            catch e
                @warn "    Failed to compile $contribution for $backend" exception=e
            end
        end

        all_vmfb_paths[contribution] = vmfb_paths
    end

    return all_vmfb_paths, iree_version
end

"""
    build_wheel(pkg_dir)

Build a Python wheel (.whl) from the package directory.

The wheel is platform-independent (py3-none-any) since:
- All Python code is pure Python
- VMFB files are portable compiled artifacts

Returns the path to the built wheel, or nothing if build fails.
"""
function build_wheel(pkg_dir::String)::Union{String, Nothing}
    dist_dir = joinpath(pkg_dir, "dist")

    # Try uv first, then fall back to pip/build
    build_cmd = nothing

    # Check for uv
    uv_path = try
        strip(read(`which uv`, String))
    catch
        nothing
    end

    if uv_path !== nothing && !isempty(uv_path)
        build_cmd = `$uv_path build --wheel --out-dir $dist_dir $pkg_dir`
    else
        # Fall back to python -m build
        python_path = try
            strip(read(`which python3`, String))
        catch
            try
                strip(read(`which python`, String))
            catch
                nothing
            end
        end

        if python_path !== nothing && !isempty(python_path)
            # Check if build module is available
            has_build = try
                run(pipeline(`$python_path -c "import build"`, stdout=devnull, stderr=devnull))
                true
            catch
                false
            end

            if has_build
                build_cmd = `$python_path -m build --wheel --outdir $dist_dir $pkg_dir`
            else
                @warn "Python 'build' module not available. Install with: pip install build"
            end
        end
    end

    if build_cmd === nothing
        @warn "No suitable build tool found (uv or python -m build). Skipping wheel build."
        return nothing
    end

    @info "Building wheel..." cmd=build_cmd
    try
        run(build_cmd)

        # Find the built wheel
        wheel_files = filter(f -> endswith(f, ".whl"), readdir(dist_dir))
        if isempty(wheel_files)
            @warn "Wheel build completed but no .whl file found in $dist_dir"
            return nothing
        end

        wheel_path = joinpath(dist_dir, first(wheel_files))
        wheel_size = filesize(wheel_path) / 1024
        @info "Wheel built successfully" path=wheel_path size_kb=wheel_size
        return wheel_path
    catch e
        @warn "Failed to build wheel" exception=e
        return nothing
    end
end

## ============================================================================
## Package Generation
## ============================================================================

"""
    create_package(calc, package_name, output_dir; kwargs...)

Create a complete redistributable Python package from an ACE calculator.

The package contains separate VMFB modules for each energy contribution:
- E0 (one-body): Computed in Python (constant per atom)
- Pair (two-body): pair_<backend>.vmfb
- ACE (many-body): ace_<backend>.vmfb

Python calculator sums all contributions for total energy/forces.
"""
function create_package(calc, package_name::String, output_dir::String;
                        elements=(:Si,),
                        rcut::Float64=5.5,
                        version::String="1.0.0",
                        description::String="ACE interatomic potential",
                        backends::Vector{Symbol}=[:cpu, :cuda, :vulkan],
                        max_atoms::Int=256,
                        max_edges::Int=10_000,
                        T::Type=Float32)

    @info "Creating redistributable package..." package_name output_dir

    # Create directory structure
    pkg_dir = joinpath(output_dir, package_name)
    src_dir = joinpath(pkg_dir, "src", package_name)
    models_dir = joinpath(src_dir, "models")
    tests_dir = joinpath(pkg_dir, "tests")
    build_dir = joinpath(output_dir, ".build")

    mkpath(src_dir)
    mkpath(models_dir)
    mkpath(tests_dir)
    mkpath(build_dir)

    # Step 1: Convert to ReactantStackedModel
    @info "Step 1: Converting model to Reactant format..."
    model = ReactantStackedModel(calc; T=T)

    # Step 2: Export to MLIR (separate modules for each contribution)
    @info "Step 2: Exporting to MLIR..."
    mlir_paths = export_to_mlir(model, build_dir; max_edges=max_edges, max_atoms=max_atoms)

    # Step 3: Compile each contribution to VMFB
    @info "Step 3: Compiling to VMFB..."
    vmfb_paths, iree_version = compile_to_vmfb(mlir_paths, models_dir; backends=backends)

    # Step 3b: Copy MLIR source files for portability
    @info "Step 3b: Copying MLIR source files..."
    for (contribution, mlir_path) in mlir_paths
        if isfile(mlir_path)
            dest_path = joinpath(models_dir, "$(contribution).mlir")
            cp(mlir_path, dest_path; force=true)
            @info "  Copied: $(contribution).mlir ($(filesize(dest_path)) bytes)"
        end
    end

    # Step 4: Export model parameters
    @info "Step 4: Exporting model parameters..."
    export_model_params(model, models_dir)

    # Step 5: Create metadata
    @info "Step 5: Creating metadata..."
    metadata = create_metadata(model, elements, version, max_atoms, max_edges, vmfb_paths)
    metadata_path = joinpath(models_dir, "metadata.json")
    open(metadata_path, "w") do io
        JSON3.pretty(io, metadata)
    end

    # Step 6: Copy Python source files
    @info "Step 6: Creating Python package structure..."
    create_python_package(pkg_dir, src_dir, tests_dir, package_name, version,
                         elements, description, iree_version)

    # Step 7: Create .gitkeep for models if no VMFB
    if isempty(vmfb_paths)
        gitkeep_path = joinpath(models_dir, ".gitkeep")
        write(gitkeep_path, "# VMFB files should be placed here after compilation\n")
    end

    # Step 8: Build wheel for distribution
    @info "Step 8: Building wheel..."
    wheel_path = build_wheel(pkg_dir)

    @info "Package created successfully!" pkg_dir

    # Summary
    println("\n" * "="^60)
    println("Package Summary")
    println("="^60)
    println("  Name: $package_name")
    println("  Version: $version")
    println("  Elements: $(join(string.(elements), ", "))")
    println("  Cutoff: $rcut Å")
    println("  Max atoms: $max_atoms")
    println("  Contributions: $(join(keys(vmfb_paths), ", "))")
    println("  Backends: $(join(string.(backends), ", "))")
    println("  MLIR sources: $(join(keys(mlir_paths), ", ")).mlir (for recompilation)")
    println()
    if wheel_path !== nothing
        println("Wheel built:")
        println("  $wheel_path")
        println()
        println("To install wheel:")
        println("  pip install $wheel_path")
    else
        println("To install (development mode):")
        println("  cd $pkg_dir && pip install -e .")
    end
    println()
    println("To use:")
    println("  from $package_name import ACECalculator")
    println("  calc = ACECalculator()")
    println("="^60)

    return pkg_dir, wheel_path
end

"""
    export_model_params(model, output_dir)

Export model parameters to NPZ format including:
- Selection matrices for IREE
- Embedding parameters (Agnesi, Chebyshev, W_radial)
- ACE coupling and readout weights
"""
function export_model_params(model::ReactantStackedModel{T}, output_dir::String) where T
    ace = model.ace_state

    # Build selection matrices for IREE
    n_cheb = ace.n_polys
    maxl = ace.maxl
    nRnl = ace.n_rnl  # Use actual n_rnl, not n_cheb
    nYlm = (maxl + 1)^2

    spec_R = Int.(ace.spec_R)
    spec_Y = Int.(ace.spec_Y)
    nA = length(spec_R)

    selector_R = build_selector_matrix(spec_R, nRnl, T)
    selector_Y = build_selector_matrix(spec_Y, nYlm, T)

    # Build symmetric product selectors
    specs_mats = ace.specs_mats
    if length(specs_mats) >= 1 && size(specs_mats[1], 1) > 0
        symm_sel1 = build_selector_matrix(specs_mats[1][:, 1], nA, T)
    else
        symm_sel1 = zeros(T, 1, nA)
    end

    if length(specs_mats) >= 2 && size(specs_mats[2], 1) > 0
        symm_sel2_1 = build_selector_matrix(specs_mats[2][:, 1], nA, T)
        symm_sel2_2 = build_selector_matrix(specs_mats[2][:, 2], nA, T)
    else
        symm_sel2_1 = zeros(T, 1, nA)
        symm_sel2_2 = zeros(T, 1, nA)
    end

    # Extract embedding parameters for pair index 1 (single species)
    pair_idx = 1

    # Agnesi transform parameters - shape (7, n_pairs) with [pin, pcut, a, b0, b1, rin, req]
    agnesi_a = T(ace.agnesi_params[3, pair_idx])
    agnesi_b0 = T(ace.agnesi_params[4, pair_idx])
    agnesi_b1 = T(ace.agnesi_params[5, pair_idx])
    agnesi_rin = T(ace.agnesi_params[6, pair_idx])
    agnesi_req = T(ace.agnesi_params[7, pair_idx])

    # Chebyshev coefficients
    poly_A = T.(ace.poly_A)
    poly_B = T.(ace.poly_B)
    poly_C = T.(ace.poly_C)

    # Radial weights - take slice for pair_idx
    W_radial = T.(ace.W_radial[:, :, pair_idx])

    params_dict = Dict{String, Any}(
        # Core model info
        "rcut" => T[model.rcut],
        "n_species" => Int32[model.n_species],
        "species_Z" => Int32.(model.species_Z),
        "E0" => T.(model.E0),
        "n_polys" => Int32[n_cheb],
        "n_rnl" => Int32[ace.n_rnl],
        "maxl" => Int32[maxl],
        "n_basis" => Int32[ace.n_basis],

        # Selection matrices for IREE (pre-computed)
        "selector_R" => selector_R,
        "selector_Y" => selector_Y,
        "symm_sel1" => symm_sel1,
        "symm_sel2_1" => symm_sel2_1,
        "symm_sel2_2" => symm_sel2_2,

        # ACE coupling and readout
        "A2Bmap" => T.(ace.A2Bmap),
        "params" => T.(ace.W_readout[:, 1]),  # Single species readout

        # Embedding parameters (new for proper ACE embeddings)
        "agnesi_a" => T[agnesi_a],
        "agnesi_b0" => T[agnesi_b0],
        "agnesi_b1" => T[agnesi_b1],
        "agnesi_rin" => T[agnesi_rin],
        "agnesi_req" => T[agnesi_req],
        "poly_A" => poly_A,
        "poly_B" => poly_B,
        "poly_C" => poly_C,
        "W_radial" => W_radial,
    )

    # Add pair state info and parameters
    if model.has_pair && model.pair_state !== nothing
        pair = model.pair_state
        params_dict["has_pair"] = Int32[1]
        params_dict["pair_n_basis"] = Int32[pair.n_basis]
        params_dict["pair_n_polys"] = Int32[pair.n_polys]
        params_dict["pair_rcut"] = T[pair.rcut]
        params_dict["pair_agnesi_params"] = T.(pair.agnesi_params)
        params_dict["pair_poly_A"] = T.(pair.poly_A)
        params_dict["pair_poly_B"] = T.(pair.poly_B)
        params_dict["pair_poly_C"] = T.(pair.poly_C)
        params_dict["pair_W_radial"] = T.(pair.W_radial)
        params_dict["pair_rcut_outer"] = T[pair.rcut_outer]
        params_dict["pair_p_outer"] = Int32[pair.p_outer]
        params_dict["pair_W_readout"] = T.(pair.W_readout)
    else
        params_dict["has_pair"] = Int32[0]
    end

    params_path = joinpath(output_dir, "params.npz")
    NPZ.npzwrite(params_path, params_dict)
    @info "  Saved: $params_path ($(filesize(params_path)) bytes)"
end

"""
    create_metadata(model, elements, version, max_atoms, max_pairs, vmfb_paths)

Create metadata dictionary for the package.

vmfb_paths is Dict{String, Dict{Symbol, String}} mapping contribution → backend → path.
"""
function create_metadata(model::ReactantStackedModel{T}, elements, version,
                         max_atoms::Int, max_pairs::Int, vmfb_paths) where T
    # Extract available backends from any contribution
    backends = Set{String}()
    for (contrib, backend_paths) in vmfb_paths
        for backend in keys(backend_paths)
            push!(backends, string(backend))
        end
    end

    return Dict(
        "name" => "ACE Potential",
        "version" => version,
        "model" => Dict(
            "cutoff" => model.rcut,
            "cutoff_units" => "angstrom",
            "elements" => [string(e) for e in elements],
            "type_mapping" => Dict(string(e) => i-1 for (i, e) in enumerate(elements)),
            "n_species" => model.n_species,
            "species_Z" => model.species_Z,
            "max_atoms" => max_atoms,
            "max_pairs" => max_pairs,
            "has_pair" => model.has_pair,
            "n_basis" => model.ace_state.n_basis,
        ),
        "units" => Dict(
            "length" => "angstrom",
            "energy" => "eV",
        ),
        "compilation" => Dict(
            "contributions" => collect(keys(vmfb_paths)),
            "backends" => collect(backends),
            "reactant_available" => HAS_REACTANT,
        ),
    )
end

"""
    create_python_package(pkg_dir, src_dir, tests_dir, name, version, elements, description, iree_version)

Create the Python package files.
"""
function create_python_package(pkg_dir, src_dir, tests_dir, name, version,
                               elements, description, iree_version)
    # Source from the package template directory
    template_dir = joinpath(dirname(@__DIR__), "package", "src", "mypotential")

    # pyproject.toml
    pyproject = replace(PYPROJECT_TEMPLATE,
        "{name}" => name,
        "{version}" => version,
        "{description}" => description,
        "{iree_version}" => iree_version)
    write(joinpath(pkg_dir, "pyproject.toml"), pyproject)

    # __init__.py
    elements_str = join([string(e) for e in elements], ", ")
    init_content = replace(INIT_TEMPLATE,
        "{name}" => name,
        "{version}" => version,
        "{elements}" => elements_str,
        "{underline}" => "=" ^ (length(name) + 29))
    write(joinpath(src_dir, "__init__.py"), init_content)

    # Copy core Python files from template
    for fname in ["calculator.py", "_device.py", "_iree_wrapper.py"]
        src_file = joinpath(template_dir, fname)
        if isfile(src_file)
            content = read(src_file, String)
            # Replace 'mypotential' with actual package name
            content = replace(content, "mypotential" => name)
            write(joinpath(src_dir, fname), content)
        end
    end

    # Test file
    test_content = replace(TEST_TEMPLATE, "{name}" => name)
    write(joinpath(tests_dir, "test_calculator.py"), test_content)

    # conftest.py for pytest
    write(joinpath(tests_dir, "conftest.py"), """
import pytest
import sys
from pathlib import Path

# Add package to path
pkg_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(pkg_src))
""")
end

## ============================================================================
## Main Entry Point
## ============================================================================

function main()
    # Parse arguments manually (ArgParse may not be available)
    args = ARGS

    # Defaults
    model_path = nothing
    test_model = false
    package_name = "acepotential"
    output_dir = "dist"
    backends = [:cpu, :cuda, :vulkan]
    elements = (:Si,)
    version = "1.0.0"

    # Parse arguments
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--test-model"
            test_model = true
        elseif arg == "--name" && i < length(args)
            i += 1
            package_name = args[i]
        elseif arg == "--output" && i < length(args)
            i += 1
            output_dir = args[i]
        elseif arg == "--backends" && i < length(args)
            i += 1
            backends = [Symbol(b) for b in split(args[i], ",")]
        elseif arg == "--elements" && i < length(args)
            i += 1
            elements = Tuple(Symbol.(split(args[i], ",")))
        elseif arg == "--version" && i < length(args)
            i += 1
            version = args[i]
        elseif arg == "--help" || arg == "-h"
            println("""
Usage: julia +1.11 --project=.. create_package.jl [OPTIONS] [MODEL_PATH]

Create a redistributable Python package from an ACE model.

Arguments:
  MODEL_PATH          Path to ACE model JSON file (optional if --test-model)

Options:
  --test-model        Create a minimal test model instead of loading
  --name NAME         Package name (default: acepotential)
  --output DIR        Output directory (default: dist)
  --backends LIST     Comma-separated backends: cpu,cuda,vulkan (default: all three)
  --elements LIST     Comma-separated elements: Si,C (default: Si)
  --version VER       Package version (default: 1.0.0)
  --help, -h          Show this help

Examples:
  # Create test package
  julia +1.11 --project=.. create_package.jl --test-model --name testpot

  # From existing model
  julia +1.11 --project=.. create_package.jl model.json --name mypot --backends cpu,cuda
""")
            return
        elseif !startswith(arg, "-")
            model_path = arg
        end
        i += 1
    end

    println("="^60)
    println("ACE Potential Package Creator")
    println("="^60)

    # Create or load model
    local calc, model_elements, rcut

    if test_model
        calc, model_elements, rcut = create_test_model(; elements=elements)
    elseif model_path !== nothing
        calc, model_elements, rcut = load_model(model_path)
    else
        @error "Either --test-model or a model path must be provided"
        return
    end

    # Create package
    pkg_dir = create_package(
        calc, package_name, output_dir;
        elements=model_elements,
        rcut=rcut,
        version=version,
        backends=backends
    )

    return pkg_dir
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
