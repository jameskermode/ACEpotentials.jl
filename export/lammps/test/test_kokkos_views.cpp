/**
 * Kokkos View Tests for pair_iree_kokkos
 * =======================================
 *
 * Unit tests for Kokkos view operations used in pair_iree_kokkos.
 * Tests atomic scatter, view allocation, and data transfer.
 *
 * Compile with:
 *   g++ -std=c++17 -I$KOKKOS_ROOT/include test_kokkos_views.cpp \
 *       -L$KOKKOS_ROOT/lib -lkokkos -lpthread -ldl -o test_kokkos_views
 *
 * Or use CMake with BUILD_TESTING=ON
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#ifdef USE_GTEST
#include <gtest/gtest.h>
#endif

#include <Kokkos_Core.hpp>

// Test configuration
#ifdef KOKKOS_ENABLE_CUDA
using DeviceType = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>;
#else
using DeviceType = Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>;
#endif

using execution_space = DeviceType::execution_space;
using memory_space = DeviceType::memory_space;

template<typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, DeviceType>;

template<typename T>
using View2D = Kokkos::View<T*[3], Kokkos::LayoutRight, DeviceType>;

// ============================================================================
// Test: View Allocation
// ============================================================================
bool test_view_allocation() {
    std::cout << "  Testing view allocation..." << std::endl;

    int n_pairs = 1000;

    View1D<int> pair_i("pair_i", n_pairs);
    View1D<int> pair_j("pair_j", n_pairs);
    View2D<float> rij("rij", n_pairs);
    View2D<float> pair_forces("pair_forces", n_pairs);

    // Check dimensions
    assert(pair_i.extent(0) == n_pairs);
    assert(rij.extent(0) == n_pairs);
    assert(rij.extent(1) == 3);

    std::cout << "    PASS: Views allocated correctly" << std::endl;
    return true;
}

// ============================================================================
// Test: Host-Device Data Transfer
// ============================================================================
bool test_data_transfer() {
    std::cout << "  Testing host-device transfer..." << std::endl;

    int n = 100;
    View1D<float> d_data("d_data", n);

    // Create and fill host mirror
    auto h_data = Kokkos::create_mirror_view(d_data);
    for (int i = 0; i < n; i++) {
        h_data(i) = static_cast<float>(i);
    }

    // Copy to device
    Kokkos::deep_copy(d_data, h_data);

    // Copy back
    auto h_result = Kokkos::create_mirror_view(d_data);
    Kokkos::deep_copy(h_result, d_data);

    // Verify
    for (int i = 0; i < n; i++) {
        assert(std::abs(h_result(i) - static_cast<float>(i)) < 1e-6f);
    }

    std::cout << "    PASS: Data transfer works" << std::endl;
    return true;
}

// ============================================================================
// Test: Atomic Scatter (Core operation for force accumulation)
// ============================================================================
bool test_atomic_scatter() {
    std::cout << "  Testing atomic scatter..." << std::endl;

    int n_atoms = 100;
    int n_pairs = 500;

    View1D<int> pair_i("pair_i", n_pairs);
    View2D<double> pair_forces("pair_forces", n_pairs);
    View2D<double> forces("forces", n_atoms);

    // Initialize on host
    auto h_pair_i = Kokkos::create_mirror_view(pair_i);
    auto h_pair_forces = Kokkos::create_mirror_view(pair_forces);

    for (int e = 0; e < n_pairs; e++) {
        h_pair_i(e) = e % n_atoms;  // Distribute pairs among atoms
        h_pair_forces(e, 0) = 1.0;
        h_pair_forces(e, 1) = 0.0;
        h_pair_forces(e, 2) = 0.0;
    }

    Kokkos::deep_copy(pair_i, h_pair_i);
    Kokkos::deep_copy(pair_forces, h_pair_forces);
    Kokkos::deep_copy(forces, 0.0);

    // Scatter using atomic operations
    Kokkos::parallel_for("scatter", n_pairs,
        KOKKOS_LAMBDA(int e) {
            int i = pair_i(e);
            Kokkos::atomic_add(&forces(i, 0), pair_forces(e, 0));
            Kokkos::atomic_add(&forces(i, 1), pair_forces(e, 1));
            Kokkos::atomic_add(&forces(i, 2), pair_forces(e, 2));
        }
    );
    Kokkos::fence();

    // Verify on host
    auto h_forces = Kokkos::create_mirror_view(forces);
    Kokkos::deep_copy(h_forces, forces);

    double sum = 0.0;
    for (int i = 0; i < n_atoms; i++) {
        sum += h_forces(i, 0);
    }

    // Each pair contributes 1.0 to x-force, total should be n_pairs
    assert(std::abs(sum - n_pairs) < 1e-10);

    std::cout << "    PASS: Atomic scatter sum = " << sum << " (expected " << n_pairs << ")" << std::endl;
    return true;
}

// ============================================================================
// Test: Newton's Third Law Scatter (f[i] -= pf, f[j] += pf)
// ============================================================================
bool test_newton_third_law_scatter() {
    std::cout << "  Testing Newton's 3rd law scatter..." << std::endl;

    int n_atoms = 10;
    int n_pairs = 20;

    View1D<int> pair_i("pair_i", n_pairs);
    View1D<int> pair_j("pair_j", n_pairs);
    View2D<double> pair_forces("pair_forces", n_pairs);
    View2D<double> forces("forces", n_atoms);

    // Initialize: pairs (0,1), (1,2), (2,3), ...
    auto h_pair_i = Kokkos::create_mirror_view(pair_i);
    auto h_pair_j = Kokkos::create_mirror_view(pair_j);
    auto h_pair_forces = Kokkos::create_mirror_view(pair_forces);

    for (int e = 0; e < n_pairs; e++) {
        h_pair_i(e) = e % n_atoms;
        h_pair_j(e) = (e + 1) % n_atoms;
        h_pair_forces(e, 0) = 1.0;
        h_pair_forces(e, 1) = 0.5;
        h_pair_forces(e, 2) = 0.0;
    }

    Kokkos::deep_copy(pair_i, h_pair_i);
    Kokkos::deep_copy(pair_j, h_pair_j);
    Kokkos::deep_copy(pair_forces, h_pair_forces);
    Kokkos::deep_copy(forces, 0.0);

    // Newton's 3rd law scatter: f[i] -= pf, f[j] += pf
    Kokkos::parallel_for("newton_scatter", n_pairs,
        KOKKOS_LAMBDA(int e) {
            int i = pair_i(e);
            int j = pair_j(e);

            Kokkos::atomic_add(&forces(i, 0), -pair_forces(e, 0));
            Kokkos::atomic_add(&forces(i, 1), -pair_forces(e, 1));
            Kokkos::atomic_add(&forces(i, 2), -pair_forces(e, 2));

            Kokkos::atomic_add(&forces(j, 0), +pair_forces(e, 0));
            Kokkos::atomic_add(&forces(j, 1), +pair_forces(e, 1));
            Kokkos::atomic_add(&forces(j, 2), +pair_forces(e, 2));
        }
    );
    Kokkos::fence();

    // Verify: total force should be zero (Newton's 3rd law)
    auto h_forces = Kokkos::create_mirror_view(forces);
    Kokkos::deep_copy(h_forces, forces);

    double total_fx = 0, total_fy = 0, total_fz = 0;
    for (int i = 0; i < n_atoms; i++) {
        total_fx += h_forces(i, 0);
        total_fy += h_forces(i, 1);
        total_fz += h_forces(i, 2);
    }

    assert(std::abs(total_fx) < 1e-10);
    assert(std::abs(total_fy) < 1e-10);
    assert(std::abs(total_fz) < 1e-10);

    std::cout << "    PASS: Total force = (" << total_fx << ", " << total_fy << ", " << total_fz << ")" << std::endl;
    return true;
}

// ============================================================================
// Test: Parallel Pair Building
// ============================================================================
bool test_parallel_pair_build() {
    std::cout << "  Testing parallel pair building..." << std::endl;

    int n_atoms = 50;
    int max_pairs = 1000;
    double cutoff_sq = 4.0;

    // Create random positions
    View2D<double> x("x", n_atoms);
    auto h_x = Kokkos::create_mirror_view(x);
    for (int i = 0; i < n_atoms; i++) {
        h_x(i, 0) = static_cast<double>(rand()) / RAND_MAX * 10.0;
        h_x(i, 1) = static_cast<double>(rand()) / RAND_MAX * 10.0;
        h_x(i, 2) = static_cast<double>(rand()) / RAND_MAX * 10.0;
    }
    Kokkos::deep_copy(x, h_x);

    // Output arrays
    View1D<int> pair_i("pair_i", max_pairs);
    View1D<int> pair_j("pair_j", max_pairs);
    View2D<double> rij("rij", max_pairs);
    Kokkos::View<int, DeviceType> d_npairs("npairs");

    // Build pairs in parallel
    Kokkos::parallel_for("build_pairs", n_atoms,
        KOKKOS_LAMBDA(int i) {
            for (int j = i + 1; j < n_atoms; j++) {
                double dx = x(j, 0) - x(i, 0);
                double dy = x(j, 1) - x(i, 1);
                double dz = x(j, 2) - x(i, 2);
                double rsq = dx*dx + dy*dy + dz*dz;

                if (rsq < cutoff_sq) {
                    int idx = Kokkos::atomic_fetch_add(&d_npairs(), 1);
                    if (idx < max_pairs) {
                        pair_i(idx) = i;
                        pair_j(idx) = j;
                        rij(idx, 0) = dx;
                        rij(idx, 1) = dy;
                        rij(idx, 2) = dz;
                    }
                }
            }
        }
    );
    Kokkos::fence();

    // Get pair count
    auto h_npairs = Kokkos::create_mirror_view(d_npairs);
    Kokkos::deep_copy(h_npairs, d_npairs);
    int npairs = h_npairs();

    std::cout << "    Found " << npairs << " pairs" << std::endl;
    assert(npairs >= 0 && npairs < max_pairs);

    // Verify distances
    auto h_rij = Kokkos::create_mirror_view(rij);
    Kokkos::deep_copy(h_rij, rij);

    for (int e = 0; e < std::min(npairs, 10); e++) {
        double rsq = h_rij(e, 0)*h_rij(e, 0) +
                     h_rij(e, 1)*h_rij(e, 1) +
                     h_rij(e, 2)*h_rij(e, 2);
        assert(rsq < cutoff_sq);
    }

    std::cout << "    PASS: Parallel pair building works" << std::endl;
    return true;
}

// ============================================================================
// Test: Performance Scaling
// ============================================================================
bool test_scatter_performance() {
    std::cout << "  Testing scatter performance..." << std::endl;

    std::vector<int> sizes = {1000, 10000, 100000};

    for (int n_pairs : sizes) {
        int n_atoms = n_pairs / 10;

        View1D<int> pair_i("pair_i", n_pairs);
        View2D<double> pair_forces("pair_forces", n_pairs);
        View2D<double> forces("forces", n_atoms);

        // Initialize
        auto h_pair_i = Kokkos::create_mirror_view(pair_i);
        auto h_pair_forces = Kokkos::create_mirror_view(pair_forces);

        for (int e = 0; e < n_pairs; e++) {
            h_pair_i(e) = e % n_atoms;
            h_pair_forces(e, 0) = 1.0;
            h_pair_forces(e, 1) = 0.5;
            h_pair_forces(e, 2) = 0.25;
        }

        Kokkos::deep_copy(pair_i, h_pair_i);
        Kokkos::deep_copy(pair_forces, h_pair_forces);
        Kokkos::deep_copy(forces, 0.0);

        // Warmup
        Kokkos::parallel_for("scatter", n_pairs,
            KOKKOS_LAMBDA(int e) {
                int i = pair_i(e);
                Kokkos::atomic_add(&forces(i, 0), pair_forces(e, 0));
            }
        );
        Kokkos::fence();

        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 100; rep++) {
            Kokkos::parallel_for("scatter", n_pairs,
                KOKKOS_LAMBDA(int e) {
                    int i = pair_i(e);
                    Kokkos::atomic_add(&forces(i, 0), pair_forces(e, 0));
                    Kokkos::atomic_add(&forces(i, 1), pair_forces(e, 1));
                    Kokkos::atomic_add(&forces(i, 2), pair_forces(e, 2));
                }
            );
        }
        Kokkos::fence();
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double time_per_call = elapsed_ms / 100.0;

        std::cout << "    " << n_pairs << " pairs: " << time_per_call << " ms/call" << std::endl;
    }

    std::cout << "    PASS: Performance test complete" << std::endl;
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);

    std::cout << "======================================" << std::endl;
    std::cout << "Kokkos View Tests for pair_iree_kokkos" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Execution space: " << execution_space::name() << std::endl;
    std::cout << "Memory space: " << memory_space::name() << std::endl;
    std::cout << std::endl;

    int failures = 0;

    try {
        if (!test_view_allocation()) failures++;
        if (!test_data_transfer()) failures++;
        if (!test_atomic_scatter()) failures++;
        if (!test_newton_third_law_scatter()) failures++;
        if (!test_parallel_pair_build()) failures++;
        if (!test_scatter_performance()) failures++;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        failures++;
    }

    Kokkos::finalize();

    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    if (failures == 0) {
        std::cout << "All tests PASSED" << std::endl;
        return 0;
    } else {
        std::cout << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
}
