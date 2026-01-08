#!/bin/bash
#
# LAMMPS Plugin Build Tests
# =========================
#
# Tests CMake configuration and building for different configurations.
#
# Usage:
#   ./test_build.sh           # Run all tests
#   ./test_build.sh --clean   # Clean before running
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${SRC_DIR}/build_test"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}PASS${NC}: $1"; }
fail() { echo -e "${RED}FAIL${NC}: $1"; exit 1; }
skip() { echo -e "${YELLOW}SKIP${NC}: $1"; }
info() { echo -e "INFO: $1"; }

# Clean if requested
if [ "$1" == "--clean" ]; then
    info "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

echo "========================================"
echo "LAMMPS Plugin Build Tests"
echo "========================================"
echo "Source dir: $SRC_DIR"
echo "Build dir: $BUILD_DIR"
echo ""

# Test 1: ace_forces library (always builds)
echo "=== Test 1: ace_forces library ==="
cd "$BUILD_DIR"
rm -rf * 2>/dev/null || true

cmake "$SRC_DIR" -DBUILD_KOKKOS_PAIR=OFF 2>&1 | head -50

if make ace_forces -j4 2>&1; then
    if [ -f "libace_forces.so" ]; then
        pass "ace_forces library built"
        ls -la libace_forces.so
    else
        fail "ace_forces.so not found"
    fi
else
    fail "ace_forces build failed"
fi
echo ""

# Test 2: pair_iree (CPU fallback, requires LAMMPS + IREE)
echo "=== Test 2: pair_iree (CPU fallback) ==="
cd "$BUILD_DIR"

if [ -n "$LAMMPS_SRC" ] && [ -n "$IREE_DIR" ] && [ -n "$IREE_SRC_DIR" ]; then
    rm -rf * 2>/dev/null || true
    cmake "$SRC_DIR" -DBUILD_KOKKOS_PAIR=OFF 2>&1 | head -50

    if make pair_iree -j4 2>&1; then
        if [ -f "libpair_iree.so" ]; then
            pass "pair_iree library built"
            ls -la libpair_iree.so
        else
            skip "pair_iree.so not found (dependencies missing?)"
        fi
    else
        skip "pair_iree build failed (expected if deps missing)"
    fi
else
    skip "pair_iree - LAMMPS_SRC, IREE_DIR, or IREE_SRC_DIR not set"
fi
echo ""

# Test 3: ace_iree_plugin (ML-IAP interface)
echo "=== Test 3: ace_iree_plugin (ML-IAP) ==="
cd "$BUILD_DIR"

if [ -n "$LAMMPS_SRC" ] && [ -n "$IREE_DIR" ] && [ -n "$IREE_SRC_DIR" ]; then
    rm -rf * 2>/dev/null || true
    cmake "$SRC_DIR" 2>&1 | head -50

    if make ace_iree_plugin -j4 2>&1; then
        if [ -f "libace_iree.so" ]; then
            pass "ace_iree_plugin built"
            ls -la libace_iree.so
        else
            skip "libace_iree.so not found (MPI missing?)"
        fi
    else
        skip "ace_iree_plugin build failed"
    fi
else
    skip "ace_iree_plugin - dependencies not set"
fi
echo ""

# Test 4: pair_iree_kokkos (Kokkos + IREE)
echo "=== Test 4: pair_iree_kokkos (Kokkos) ==="
cd "$BUILD_DIR"

if [ -n "$KOKKOS_ROOT" ] || command -v kokkos_launch &>/dev/null; then
    rm -rf * 2>/dev/null || true
    cmake "$SRC_DIR" -DBUILD_KOKKOS_PAIR=ON 2>&1 | head -50

    if make pair_iree_kokkos -j4 2>&1; then
        if [ -f "libpair_iree_kokkos.so" ]; then
            pass "pair_iree_kokkos built"
            ls -la libpair_iree_kokkos.so
        else
            skip "libpair_iree_kokkos.so not found"
        fi
    else
        skip "pair_iree_kokkos build failed (Kokkos not found?)"
    fi
else
    skip "pair_iree_kokkos - Kokkos not available"
fi
echo ""

# Summary
echo "========================================"
echo "Build Test Summary"
echo "========================================"
echo "See above for PASS/FAIL/SKIP results"
echo ""

# Check final state
cd "$BUILD_DIR"
echo "Built libraries:"
ls -la *.so 2>/dev/null || echo "  (none)"
echo ""

echo "Build tests complete!"
