# ACE Export Tests

This directory contains tests validating numerical equivalence between Julia reference values and exported VMFBs (for Python/ASE and LAMMPS).

## Important Note: Simplified Test Model

The bucket VMFBs use a **simplified polynomial energy function** for testing, not a real ACE model. This is intentional - it allows testing the export pipeline without requiring a full ACE model.

For production use with a real ACE model:
1. Export your model using `create_package.jl`
2. Generate matching Julia reference data
3. The same tests will validate numerical equivalence

## Test Suite Overview

| Test | Location | Purpose |
|------|----------|---------|
| Julia Energy+Gradient | `test/test_julia_energy_gradient.jl` | Validate Julia energy/gradient computation |
| VMFB Equivalence | `test/test_vmfb_equivalence.py` | VMFB matches Julia reference |
| Numerical Validation | `test/test_numerical_validation.py` | Comprehensive force/energy validation |
| ASE Calculator | `test/test_calculator.py` | Calculator integration test |
| Package Tests | `package/tests/` | pytest suite for package |
| LAMMPS Equivalence | `lammps/build_test/` | LAMMPS matches Julia |

## Test Details

### 1. Julia Energy+Gradient Test

Verifies the Julia energy function and Enzyme gradient computation are correct.

```bash
julia +1.11 --project=export export/test/test_julia_energy_gradient.jl
```

**What it tests:**
- Energy computation produces finite values
- Enzyme gradient computation works
- Gradients match finite difference approximation
- Results are deterministic
- Works at all bucket sizes (2000, 10000, 50000 edges)

### 2. VMFB Equivalence Test

Verifies exported VMFBs match Julia reference values stored in `test_data.npz`.

```bash
cd export/tools && uv run python ../test/test_vmfb_equivalence.py
```

**What it tests:**
- VMFB energy matches Julia reference energy
- VMFB gradients match Julia reference gradients
- Tests all bucket sizes in `benchmark/bucket_energy_gradient/`

**Expected precision:** < 1e-10 (float64)

### 3. Numerical Validation Test

Comprehensive validation of forces and conservation laws.

```bash
cd export/tools && uv run python ../test/test_numerical_validation.py
```

**What it tests:**
- **VMFB vs Julia**: Direct comparison of VMFB output to reference data
- **Calculator vs VMFB**: ASE Calculator correctly uses VMFB
- **Force Finite Difference**: Forces = -dE/dx (validates gradient correctness)
- **Newton's 3rd Law**: Total force on periodic system is zero
- **Force Sign Convention**: Displaced atoms have restoring forces

### 4. ASE Calculator Test

Quick integration test of the calculator with bucket VMFBs.

```bash
cd export/tools && uv run python ../test/test_calculator.py
```

**What it tests:**
- Calculator loads bucket VMFBs correctly
- Energy computation works
- Force computation works
- Newton's 3rd law conservation

### 5. Package Tests (pytest)

Comprehensive pytest suite for the Python package.

```bash
cd export/tools && uv run python -m pytest ../package/tests/ -v
```

**test_calculator.py:**
- Import and basic module structure
- Device info functions
- Calculator creation and configuration
- Energy/force computation on various systems
- Stress calculation
- Bucket selection at runtime

**test_numerical.py:**
- Finite difference force validation (single and multiple configs)
- Newton's 3rd law (total force zero)
- Stress tensor properties
- Result consistency across calls
- Julia reference comparison (skipped unless fixtures match)

### 6. LAMMPS Equivalence Test

Verifies LAMMPS pair_iree plugin matches Julia reference forces.

```bash
cd export/lammps/build_test && python compare_forces.py
```

**Prerequisites:**
- LAMMPS built with pair_iree_kokkos plugin
- Run LAMMPS first to generate `lammps_forces.dump`
- Julia reference generated via `generate_julia_reference.jl`

**What it tests:**
- Energy matches (accounting for full vs half neighbor list)
- Per-atom forces match
- Newton's 3rd law conservation

See `lammps/build_test/BENCHMARK_RESULTS.md` for detailed benchmark procedures.

## Running All Tests

```bash
# From ACEpotentials.jl-main directory

# 1. Julia test (validates Julia energy/gradient)
julia +1.11 --project=export export/test/test_julia_energy_gradient.jl

# 2. Python VMFB test (validates export pipeline)
cd export/tools && uv run python ../test/test_vmfb_equivalence.py

# 3. Numerical validation (comprehensive)
cd export/tools && uv run python ../test/test_numerical_validation.py

# 4. ASE Calculator test
cd export/tools && uv run python ../test/test_calculator.py

# 5. Package pytest suite
cd export/tools && uv run python -m pytest ../package/tests/ -v

# 6. LAMMPS test (if LAMMPS is built)
cd export/lammps/build_test && python compare_forces.py
```

## Test Data

Test data is stored in `benchmark/bucket_energy_gradient/bucket_*/test_data.npz`:
- `rij`: Input displacement vectors (n_edges, 3)
- `energy`: Julia reference energy (1,)
- `gradient`: Julia reference dE/drij (n_edges, 3)

## Regenerating Test Data

If you modify the energy function, regenerate the bucket VMFBs:

```bash
julia +1.11 --project=export export/benchmark/export_bucket_energy_gradient.jl
```

This creates new VMFBs and test_data.npz files in each bucket directory.

## Numerical Precision

| Comparison | Tolerance | Notes |
|------------|-----------|-------|
| VMFB vs Julia energy | < 1e-10 relative | Float64 precision |
| VMFB vs Julia gradient | < 1e-10 absolute | Float64 precision |
| Force finite difference | < 1e-4 relative | Limited by FD step size |
| Newton's 3rd law | < 1e-10 absolute | Machine precision |

## Limitations

1. **Simplified Model**: Tests use a polynomial energy function, not a real ACE model
2. **Julia Reference Required**: VMFB validation requires matching Julia reference data
3. **LAMMPS Optional**: LAMMPS tests require the built plugin
4. **CPU Only**: Current tests only validate CPU backend

## Test Coverage Summary

| Layer | What's Tested | Coverage |
|-------|---------------|----------|
| Julia | Energy+gradient computation | Full |
| VMFB | Export numerical accuracy | Full |
| Python | Calculator interface | Full |
| Numerical | Forces via autodiff | Full |
| Conservation | Newton's 3rd law | Full |
| LAMMPS | Plugin integration | Full (when built) |
