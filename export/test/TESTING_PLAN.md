# Testing Plans for Phase 8 and Phase 9

## Overview

This document outlines comprehensive testing strategies for:
- **Phase 8**: Portable Python Package (`export/package/`)
- **Phase 9**: LAMMPS + Kokkos Zero-Copy Integration (`export/lammps/`)

---

## Phase 8: Portable Python Package Testing

### 8.1 Unit Tests

#### 8.1.1 Device Selection (`_device.py`)

**Test file**: `package/tests/test_device.py`

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_cuda_detection` | Check CUDA driver detection | Returns True if libcuda.so.1 loadable |
| `test_vulkan_detection` | Check Vulkan driver detection | Returns True if libvulkan.so.1 loadable |
| `test_models_dir_location` | Verify models directory path | Returns path inside package |
| `test_select_device_auto` | Auto-selection logic | Returns best available device |
| `test_select_device_explicit` | Explicit device selection | Returns requested device if available |
| `test_select_device_missing_model` | Request device without model file | Raises FileNotFoundError |
| `test_get_device_info` | Device info dictionary | Contains cpu/cuda/vulkan keys |
| `test_list_available_models` | List model files | Returns dict with paths and existence |

```python
# Example test structure
def test_select_device_auto_fallback():
    """Test that auto falls back to CPU when GPU unavailable."""
    # Mock CUDA/Vulkan as unavailable
    with patch('mypotential._device._cuda_available', return_value=False):
        with patch('mypotential._device._vulkan_available', return_value=False):
            device, path = select_device('auto', models_dir=test_models_dir)
            assert device == 'local-task'
            assert 'cpu' in str(path)
```

#### 8.1.2 IREE Wrapper (`_iree_wrapper.py`)

**Test file**: `package/tests/test_iree_wrapper.py`

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_pad_array_2d` | Pad 2D array to target shape | Correct padding with fill value |
| `test_pad_array_1d` | Pad 1D array | Correct padding |
| `test_model_params_load` | Load NPZ parameters | All keys accessible |
| `test_iree_model_load` | Load VMFB module | No exceptions, functions found |
| `test_iree_model_missing_file` | Load nonexistent VMFB | Raises FileNotFoundError |
| `test_compute_energy_shape` | Energy output shape | Returns scalar float |
| `test_compute_pair_forces_shape` | Pair forces output shape | Returns [n_pairs, 3] array |

#### 8.1.3 Calculator (`calculator.py`)

**Test file**: `package/tests/test_calculator.py`

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_calculator_init` | Create calculator | No exceptions |
| `test_calculator_cutoff` | Cutoff property | Returns positive float |
| `test_implemented_properties` | Check properties list | Contains energy, forces, stress |
| `test_neighbor_list_empty` | System with no neighbors | Returns zero energy/forces |
| `test_pool_matrix_construction` | Pool matrix shape | [n_atoms, n_pairs] |
| `test_force_accumulation` | Pair to atom force scatter | Newton's 3rd law satisfied |
| `test_virial_calculation` | Virial tensor | 3x3 symmetric matrix |
| `test_stress_voigt` | Stress Voigt notation | 6-element array |

### 8.2 Integration Tests

**Test file**: `package/tests/test_integration.py`

#### 8.2.1 End-to-End Workflow

```python
@pytest.mark.integration
def test_silicon_bulk():
    """Complete workflow with silicon bulk."""
    from mypotential import Calculator
    from ase.build import bulk

    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.calc = Calculator(device='cpu')

    # Energy
    E = atoms.get_potential_energy()
    assert np.isfinite(E)

    # Forces
    F = atoms.get_forces()
    assert F.shape == (len(atoms), 3)
    assert np.allclose(F, 0, atol=1e-4)  # Perfect crystal

    # Stress
    S = atoms.get_stress()
    assert S.shape == (6,)
    assert np.all(np.isfinite(S))
```

#### 8.2.2 Multi-Backend Consistency

```python
@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", "cuda", "vulkan"])
def test_backend_consistency(device, reference_atoms):
    """All backends should give same results."""
    try:
        calc = Calculator(device=device)
    except (RuntimeError, FileNotFoundError):
        pytest.skip(f"{device} not available")

    reference_atoms.calc = calc
    E = reference_atoms.get_potential_energy()
    F = reference_atoms.get_forces()

    # Compare against CPU reference
    assert np.isclose(E, REFERENCE_ENERGY, rtol=1e-5)
    assert np.allclose(F, REFERENCE_FORCES, rtol=1e-4)
```

#### 8.2.3 MD Stability Test

```python
@pytest.mark.integration
@pytest.mark.slow
def test_md_stability():
    """Run short MD and check energy conservation."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase import units

    atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
    atoms.calc = Calculator(device='cpu')

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = VelocityVerlet(atoms, timestep=1*units.fs)

    energies = []
    for _ in range(100):
        dyn.run(1)
        energies.append(atoms.get_total_energy())

    # Energy should be conserved within tolerance
    E_drift = abs(energies[-1] - energies[0]) / len(atoms)
    assert E_drift < 0.01  # meV/atom drift over 100 steps
```

### 8.3 Numerical Validation

**Test file**: `package/tests/test_numerical.py`

#### 8.3.1 Against Julia Reference

```python
@pytest.mark.validation
def test_against_julia_reference():
    """Compare against Julia ACEpotentials output."""
    # Load pre-computed Julia reference
    ref = np.load('tests/fixtures/julia_reference.npz')

    atoms = read('tests/fixtures/test_structure.xyz')
    atoms.calc = Calculator(device='cpu')

    E_python = atoms.get_potential_energy()
    F_python = atoms.get_forces()

    assert np.isclose(E_python, ref['energy'], rtol=1e-5)
    assert np.allclose(F_python, ref['forces'], rtol=1e-4)
```

#### 8.3.2 Finite Difference Force Check

```python
@pytest.mark.validation
def test_forces_finite_difference():
    """Verify forces via finite difference."""
    atoms = bulk('Si', 'diamond', a=5.43)
    atoms.calc = Calculator(device='cpu')

    F_analytical = atoms.get_forces()

    # Numerical gradient
    delta = 1e-5
    F_numerical = np.zeros_like(F_analytical)

    for i in range(len(atoms)):
        for d in range(3):
            atoms_plus = atoms.copy()
            atoms_plus.positions[i, d] += delta
            atoms_plus.calc = Calculator(device='cpu')
            E_plus = atoms_plus.get_potential_energy()

            atoms_minus = atoms.copy()
            atoms_minus.positions[i, d] -= delta
            atoms_minus.calc = Calculator(device='cpu')
            E_minus = atoms_minus.get_potential_energy()

            F_numerical[i, d] = -(E_plus - E_minus) / (2 * delta)

    assert np.allclose(F_analytical, F_numerical, rtol=1e-3, atol=1e-6)
```

### 8.4 Performance Tests

**Test file**: `package/tests/test_performance.py`

```python
@pytest.mark.benchmark
def test_throughput_scaling():
    """Measure throughput for different system sizes."""
    import time

    sizes = [64, 216, 512, 1000, 2000, 4096]
    results = []

    for n_atoms in sizes:
        # Create system with approximately n_atoms
        rep = int(np.ceil((n_atoms / 8) ** (1/3)))
        atoms = bulk('Si', 'diamond', a=5.43) * (rep, rep, rep)
        atoms.calc = Calculator(device='cpu')

        # Warmup
        atoms.get_potential_energy()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(10):
            atoms.get_forces()
        t1 = time.perf_counter()

        throughput = 10 / (t1 - t0)
        results.append((len(atoms), throughput))
        print(f"  {len(atoms):5d} atoms: {throughput:.1f} evals/sec")

    # Verify reasonable performance
    assert results[-1][1] > 1.0  # At least 1 eval/sec for 4096 atoms
```

---

## Phase 9: LAMMPS + Kokkos Zero-Copy Testing

### 9.1 Build Tests

**Test script**: `lammps/test/test_build.sh`

```bash
#!/bin/bash
# Test CMake configuration and build

set -e

BUILD_DIR="build_test"
rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR

echo "=== Test 1: CPU-only build (no Kokkos) ==="
cmake .. -DBUILD_KOKKOS_PAIR=OFF
make pair_iree -j4
test -f libpair_iree.so && echo "PASS: pair_iree built" || echo "FAIL"

echo "=== Test 2: ace_forces library ==="
make ace_forces -j4
test -f libace_forces.so && echo "PASS: ace_forces built" || echo "FAIL"

echo "=== Test 3: Kokkos build (if available) ==="
if [ -n "$KOKKOS_ROOT" ]; then
    rm -rf *
    cmake .. -DBUILD_KOKKOS_PAIR=ON -DKokkos_ROOT=$KOKKOS_ROOT
    make pair_iree_kokkos -j4 || echo "Kokkos build skipped"
fi

echo "Build tests complete"
```

### 9.2 Unit Tests (C++)

**Test file**: `lammps/test/test_iree_kokkos_handle.cpp`

```cpp
#include <gtest/gtest.h>
#include "pair_iree_kokkos.h"

using namespace LAMMPS_NS;

class IREEKokkosHandleTest : public ::testing::Test {
protected:
    IREEKokkosHandle handle;
};

TEST_F(IREEKokkosHandleTest, InitialState) {
    EXPECT_FALSE(handle.is_valid());
    EXPECT_EQ(handle.device(), nullptr);
}

TEST_F(IREEKokkosHandleTest, LoadModule) {
    // Requires test VMFB file
    bool loaded = handle.load_module("test_model.vmfb", "local-task");
    if (loaded) {
        EXPECT_TRUE(handle.is_valid());
        EXPECT_NE(handle.device(), nullptr);
    }
}

TEST_F(IREEKokkosHandleTest, ImportBuffer) {
    handle.load_module("test_model.vmfb", "local-task");

    std::vector<float> data(100 * 3, 1.0f);
    std::vector<int64_t> shape = {100, 3};

    auto view = handle.import_gpu_buffer(
        data.data(),
        data.size() * sizeof(float),
        shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32
    );

    EXPECT_NE(view, nullptr);
    if (view) {
        iree_hal_buffer_view_release(view);
    }
}
```

### 9.3 Kokkos Integration Tests

**Test file**: `lammps/test/test_kokkos_views.cpp`

```cpp
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "pair_iree_kokkos.h"

#ifdef KOKKOS_ENABLE_CUDA
using DeviceType = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
#else
using DeviceType = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
#endif

class KokkosViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        Kokkos::initialize();
    }
    void TearDown() override {
        Kokkos::finalize();
    }
};

TEST_F(KokkosViewTest, ViewAllocation) {
    using View = Kokkos::View<float*[3], Kokkos::LayoutRight, DeviceType>;

    View rij("rij", 1000);
    EXPECT_EQ(rij.extent(0), 1000);
    EXPECT_EQ(rij.extent(1), 3);
}

TEST_F(KokkosViewTest, AtomicScatter) {
    using View1D = Kokkos::View<int*, DeviceType>;
    using View2D = Kokkos::View<double*[3], DeviceType>;

    int n_atoms = 100;
    int n_pairs = 500;

    View1D pair_i("pair_i", n_pairs);
    View2D pair_forces("pair_forces", n_pairs);
    View2D forces("forces", n_atoms);

    // Initialize pair data on host, copy to device
    auto h_pair_i = Kokkos::create_mirror_view(pair_i);
    auto h_pair_forces = Kokkos::create_mirror_view(pair_forces);

    for (int e = 0; e < n_pairs; e++) {
        h_pair_i(e) = e % n_atoms;
        h_pair_forces(e, 0) = 1.0;
        h_pair_forces(e, 1) = 0.0;
        h_pair_forces(e, 2) = 0.0;
    }

    Kokkos::deep_copy(pair_i, h_pair_i);
    Kokkos::deep_copy(pair_forces, h_pair_forces);
    Kokkos::deep_copy(forces, 0.0);

    // Scatter
    Kokkos::parallel_for("scatter", n_pairs,
        KOKKOS_LAMBDA(int e) {
            int i = pair_i(e);
            Kokkos::atomic_add(&forces(i, 0), pair_forces(e, 0));
            Kokkos::atomic_add(&forces(i, 1), pair_forces(e, 1));
            Kokkos::atomic_add(&forces(i, 2), pair_forces(e, 2));
        }
    );
    Kokkos::fence();

    // Verify
    auto h_forces = Kokkos::create_mirror_view(forces);
    Kokkos::deep_copy(h_forces, forces);

    double sum = 0.0;
    for (int i = 0; i < n_atoms; i++) {
        sum += h_forces(i, 0);
    }
    EXPECT_DOUBLE_EQ(sum, n_pairs);  // Each pair contributes 1.0
}
```

### 9.4 LAMMPS Integration Tests

**Test file**: `lammps/test/test_lammps_integration.py`

```python
"""LAMMPS integration tests using Python interface."""

import pytest
import numpy as np
import os

# Skip if LAMMPS not available
lammps = pytest.importorskip('lammps')


@pytest.fixture
def lmp():
    """Create LAMMPS instance with pair_iree loaded."""
    lmp = lammps.lammps(cmdargs=['-log', 'none', '-screen', 'none'])
    return lmp


@pytest.mark.lammps
def test_pair_style_load(lmp):
    """Test that pair_style iree can be loaded."""
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")

    # This will fail if plugin not loaded, but shouldn't crash
    try:
        lmp.command("pair_style iree test_model.vmfb")
    except Exception as e:
        if "Unknown pair style" in str(e):
            pytest.skip("pair_iree not loaded as plugin")
        raise


@pytest.mark.lammps
def test_silicon_energy(lmp, silicon_data_file):
    """Test energy calculation for silicon."""
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("read_data " + silicon_data_file)
    lmp.command("pair_style iree model_cpu.vmfb")
    lmp.command("pair_coeff * * Si")

    lmp.command("run 0")

    pe = lmp.get_thermo("pe")
    assert np.isfinite(pe)


@pytest.mark.lammps
@pytest.mark.slow
def test_nve_stability(lmp, silicon_data_file):
    """Test NVE MD stability."""
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("read_data " + silicon_data_file)
    lmp.command("pair_style iree model_cpu.vmfb")
    lmp.command("pair_coeff * * Si")

    lmp.command("velocity all create 300 12345")
    lmp.command("fix 1 all nve")
    lmp.command("thermo 10")

    # Run 100 steps
    lmp.command("run 100")

    # Check final energy is reasonable
    pe = lmp.get_thermo("pe")
    ke = lmp.get_thermo("ke")
    assert np.isfinite(pe + ke)
```

### 9.5 Zero-Copy Validation Tests

**Test file**: `lammps/test/test_zero_copy.cpp`

```cpp
#ifdef KOKKOS_ENABLE_CUDA
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "pair_iree_kokkos.h"

class ZeroCopyTest : public ::testing::Test {
protected:
    IREEKokkosHandle handle;

    void SetUp() override {
        handle.load_module("test_model.vmfb", "cuda");
    }
};

TEST_F(ZeroCopyTest, PointerPreservation) {
    // Allocate CUDA memory
    float* d_data;
    cudaMalloc(&d_data, 1000 * 3 * sizeof(float));

    // Fill with test pattern
    std::vector<float> h_data(1000 * 3);
    for (int i = 0; i < 1000 * 3; i++) {
        h_data[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    // Import into IREE
    std::vector<int64_t> shape = {1000, 3};
    auto view = handle.import_gpu_buffer(
        d_data, 1000 * 3 * sizeof(float), shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32
    );

    ASSERT_NE(view, nullptr);

    // Verify IREE is using the same pointer (zero-copy)
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(view);

    // Get device pointer from IREE buffer
    iree_hal_buffer_mapping_t mapping;
    iree_status_t status = iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ, 0,
        1000 * 3 * sizeof(float), &mapping
    );

    if (iree_status_is_ok(status)) {
        // For true zero-copy, the mapped pointer should be our original
        // Note: This depends on IREE implementation details
        EXPECT_EQ(mapping.contents.data, d_data);
        iree_hal_buffer_unmap_range(&mapping);
    }

    iree_hal_buffer_view_release(view);
    cudaFree(d_data);
}

TEST_F(ZeroCopyTest, NoHostTransfer) {
    // This test verifies no GPU->CPU->GPU transfer occurs
    // by checking CUDA profiler events or timing

    float* d_data;
    cudaMalloc(&d_data, 10000 * 3 * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Import should be instant (just pointer wrapping)
    for (int i = 0; i < 100; i++) {
        auto view = handle.import_gpu_buffer(
            d_data, 10000 * 3 * sizeof(float),
            {10000, 3}, IREE_HAL_ELEMENT_TYPE_FLOAT_32
        );
        iree_hal_buffer_view_release(view);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // 100 imports should take < 10ms (no data transfer)
    EXPECT_LT(ms, 10.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
}
#endif  // KOKKOS_ENABLE_CUDA
```

### 9.6 Performance Benchmarks

**Test file**: `lammps/test/benchmark_kokkos.cpp`

```cpp
#include <benchmark/benchmark.h>
#include <Kokkos_Core.hpp>
#include "pair_iree_kokkos.h"

static void BM_PairListBuild(benchmark::State& state) {
    int n_atoms = state.range(0);
    // Setup views and neighbor list
    // ...

    for (auto _ : state) {
        build_pair_lists();
        Kokkos::fence();
    }

    state.SetItemsProcessed(state.iterations() * n_atoms);
}
BENCHMARK(BM_PairListBuild)->Range(64, 4096);

static void BM_ForceScatter(benchmark::State& state) {
    int n_pairs = state.range(0);
    int n_atoms = n_pairs / 10;

    // Setup pair_forces, pair_i, forces views
    // ...

    for (auto _ : state) {
        Kokkos::parallel_for("scatter", n_pairs,
            KOKKOS_LAMBDA(int e) {
                // Scatter logic
            }
        );
        Kokkos::fence();
    }

    state.SetItemsProcessed(state.iterations() * n_pairs);
}
BENCHMARK(BM_ForceScatter)->Range(1000, 200000);

static void BM_FullCompute(benchmark::State& state) {
    // Full compute() call benchmark
    int n_atoms = state.range(0);
    // Setup LAMMPS-like environment
    // ...

    for (auto _ : state) {
        compute(1, 1);  // eflag=1, vflag=1
    }

    state.SetItemsProcessed(state.iterations());
    state.SetLabel(std::to_string(n_atoms) + " atoms");
}
BENCHMARK(BM_FullCompute)->Range(64, 4096);

BENCHMARK_MAIN();
```

---

## Test Fixtures and Data

### Creating Test Fixtures

**Script**: `test/create_fixtures.jl`

```julia
#=
Create test fixtures for Python and C++ tests
=#

using ACEExport
using NPZ
using JSON3

# Create reference silicon structure
function create_silicon_fixture()
    # ... create atoms, calculate E/F with Julia

    npzwrite("fixtures/julia_reference.npz", Dict(
        "positions" => positions,
        "energy" => energy,
        "forces" => forces,
        "cell" => cell,
    ))
end

# Create test VMFB for unit tests
function create_test_vmfb()
    # Export minimal model for testing
    # ...
end
```

### Test Data Files

```
test/
├── fixtures/
│   ├── julia_reference.npz      # Reference E/F from Julia
│   ├── test_structure.xyz       # Test atomic structure
│   ├── silicon_2x2x2.data       # LAMMPS data file
│   └── test_model.vmfb          # Minimal compiled model
├── test_device.py
├── test_iree_wrapper.py
├── test_calculator.py
├── test_integration.py
├── test_numerical.py
└── test_performance.py
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/test-export.yml
name: Export Package Tests

on:
  push:
    paths:
      - 'export/**'
  pull_request:
    paths:
      - 'export/**'

jobs:
  test-python-package:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest numpy ase matscipy iree-runtime
          pip install -e export/package

      - name: Run unit tests
        run: |
          cd export/package
          pytest tests/ -v --ignore=tests/test_integration.py

      - name: Run integration tests
        if: ${{ github.event_name == 'push' }}
        run: |
          cd export/package
          pytest tests/test_integration.py -v -m "not slow"

  test-lammps-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install IREE
        run: |
          pip install iree-compiler iree-runtime

      - name: Build LAMMPS plugins
        run: |
          cd export/lammps
          mkdir build && cd build
          cmake .. -DBUILD_KOKKOS_PAIR=OFF
          make -j4

      - name: Run C++ tests
        run: |
          cd export/lammps/build
          ctest --output-on-failure
```

---

## Test Execution Commands

### Phase 8 (Python Package)

```bash
# Run all unit tests
cd export/package
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/mypotential --cov-report=html

# Run specific test categories
pytest tests/ -v -m "not slow"           # Skip slow tests
pytest tests/ -v -m "integration"         # Only integration
pytest tests/ -v -m "validation"          # Only validation
pytest tests/ -v -m "benchmark"           # Only benchmarks

# Run against specific device
MYPOTENTIAL_DEVICE=cuda pytest tests/ -v
```

### Phase 9 (LAMMPS + Kokkos)

```bash
# Build tests
cd export/lammps
mkdir build && cd build
cmake .. -DBUILD_TESTING=ON -DBUILD_KOKKOS_PAIR=ON
make -j4

# Run C++ unit tests
ctest --output-on-failure

# Run benchmarks
./benchmark_kokkos --benchmark_format=json > benchmark_results.json

# LAMMPS integration (requires LAMMPS with plugin)
cd ../test
python test_lammps_integration.py
```

---

## Success Criteria

### Phase 8
- [ ] All unit tests pass on CPU
- [ ] Integration tests pass with pre-compiled model
- [ ] Numerical agreement with Julia reference < 1e-5 (energy), 1e-4 (forces)
- [ ] Finite difference forces match analytical < 1e-3
- [ ] MD energy drift < 0.01 meV/atom over 100 steps
- [ ] Performance: > 1 eval/sec for 4096 atoms on CPU

### Phase 9
- [ ] CMake configuration succeeds for all build variants
- [ ] pair_iree (CPU) builds and links
- [ ] pair_iree_kokkos builds with Kokkos (if available)
- [ ] Zero-copy buffer import verified (no GPU->CPU transfer)
- [ ] Atomic scatter produces correct forces
- [ ] LAMMPS NVE runs stably for 1000 steps
- [ ] Performance: GPU faster than CPU for > 1000 atoms
