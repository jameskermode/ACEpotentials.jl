cmake_minimum_required(VERSION 3.16)
project(ace_iree_lammps CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# IREE directories - from environment or command line
# IREE_DIR: Binary distribution with static libraries
# IREE_SRC_DIR: Source checkout for headers (binary dist has incomplete headers)
if(DEFINED ENV{IREE_DIR})
    set(IREE_DIR $ENV{IREE_DIR} CACHE PATH "IREE installation directory")
elseif(NOT DEFINED IREE_DIR)
    set(IREE_DIR "$ENV{HOME}/iree" CACHE PATH "IREE installation directory")
endif()

if(DEFINED ENV{IREE_SRC_DIR})
    set(IREE_SRC_DIR $ENV{IREE_SRC_DIR} CACHE PATH "IREE source directory for headers")
elseif(NOT DEFINED IREE_SRC_DIR)
    set(IREE_SRC_DIR "$ENV{HOME}/iree-src" CACHE PATH "IREE source directory for headers")
endif()

# LAMMPS source directory
if(DEFINED ENV{LAMMPS_SRC})
    set(LAMMPS_SRC_DIR $ENV{LAMMPS_SRC} CACHE PATH "LAMMPS source directory")
else()
    set(LAMMPS_SRC_DIR "$ENV{HOME}/lammps/lammps-22Jul2025/src" CACHE PATH "LAMMPS source directory")
endif()

# Check IREE libraries and source headers
if(EXISTS "${IREE_DIR}/lib/libiree_runtime_unified.a")
    message(STATUS "Found IREE libraries at: ${IREE_DIR}")
    set(IREE_LIBRARY_DIRS ${IREE_DIR}/lib)
    set(IREE_LIBS_FOUND TRUE)
else()
    message(STATUS "IREE libraries not found at ${IREE_DIR}/lib. Set IREE_DIR.")
    set(IREE_LIBS_FOUND FALSE)
endif()

# Headers come from source checkout (binary dist is incomplete)
if(EXISTS "${IREE_SRC_DIR}/runtime/src/iree/base/allocator.h")
    message(STATUS "Found IREE headers at: ${IREE_SRC_DIR}/runtime/src")
    set(IREE_INCLUDE_DIRS ${IREE_SRC_DIR}/runtime/src)
    set(IREE_HEADERS_FOUND TRUE)
else()
    message(STATUS "IREE headers not found at ${IREE_SRC_DIR}. Set IREE_SRC_DIR.")
    set(IREE_HEADERS_FOUND FALSE)
endif()

# Check LAMMPS
if(EXISTS "${LAMMPS_SRC_DIR}/lammps.h")
    message(STATUS "Found LAMMPS source at: ${LAMMPS_SRC_DIR}")
    set(LAMMPS_FOUND TRUE)
else()
    message(STATUS "LAMMPS source not found at ${LAMMPS_SRC_DIR}. Set LAMMPS_SRC.")
    set(LAMMPS_FOUND FALSE)
endif()

# Find MPI (optional, required only for LAMMPS ML-IAP plugin)
find_package(MPI QUIET)
if(MPI_FOUND)
    message(STATUS "MPI found")
else()
    message(STATUS "MPI not found")
endif()

# ==============================================================================
# ACE Forces Library (standalone, no IREE dependency for chain rule functions)
# ==============================================================================
# This library provides:
# - Chebyshev polynomial embeddings and derivatives
# - Spherical harmonics embeddings and derivatives
# - Chain rule to convert embedding gradients to atomic forces
#
# These are standard mathematical functions, NOT the ACE kernel.
# The ACE kernel (energy + gradients) comes from compiled VMFB.

add_library(ace_forces SHARED
    src/ace_forces.c
)

target_include_directories(ace_forces PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(ace_forces PRIVATE
    m  # math library
)

set_target_properties(ace_forces PROPERTIES
    OUTPUT_NAME "ace_forces"
    PREFIX "lib"
    C_STANDARD 99
)

# ==============================================================================
# ACE IREE Plugin (LAMMPS ML-IAP with IREE integration)
# ==============================================================================
# Collect IREE static libraries (only the essential ones for runtime)
set(IREE_STATIC_LIBS
    ${IREE_LIBRARY_DIRS}/libiree_runtime_unified.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_hal.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_drivers.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_local_task_task_driver.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_local_task_registration_registration.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_local_sync_sync_driver.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_local_sync_registration_registration.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_local.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_executable_loader.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_loaders_registration_registration.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_loaders_embedded_elf_loader.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_loaders_vmvx_module_loader.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_elf_elf_module.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_elf_arch.a
    ${IREE_LIBRARY_DIRS}/libiree_hal_local_elf_platform.a
    ${IREE_LIBRARY_DIRS}/libiree_vm_impl.a
    ${IREE_LIBRARY_DIRS}/libiree_vm_bytecode_module.a
    ${IREE_LIBRARY_DIRS}/libiree_modules_hal_hal.a
    ${IREE_LIBRARY_DIRS}/libiree_modules_hal_types.a
    ${IREE_LIBRARY_DIRS}/libiree_modules_vmvx_vmvx.a
    ${IREE_LIBRARY_DIRS}/libiree_task_task.a
    ${IREE_LIBRARY_DIRS}/libiree_task_api.a
    ${IREE_LIBRARY_DIRS}/libiree_base_base.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_arena.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_atomic_slist.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_cpu.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_dynamic_library.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_event_pool.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_flags.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_fpu_state.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_memory.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_path.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_synchronization.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_threading.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_time.a
    ${IREE_LIBRARY_DIRS}/libiree_base_internal_wait_handle.a
    ${IREE_LIBRARY_DIRS}/libflatcc_parsing.a
    ${IREE_LIBRARY_DIRS}/libflatcc_runtime.a
)

# Only build LAMMPS plugin if all dependencies are available
if(MPI_FOUND AND LAMMPS_FOUND AND IREE_LIBS_FOUND AND IREE_HEADERS_FOUND)
    message(STATUS "All LAMMPS plugin dependencies found - building ace_iree_plugin")

    add_library(ace_iree_plugin SHARED
        src/iree_wrapper.cpp
        src/mliap_ace_iree.cpp
    )

    target_compile_definitions(ace_iree_plugin PRIVATE
        IREE_AVAILABLE
        IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
    )

    # IMPORTANT: IREE includes must come BEFORE LAMMPS to avoid header conflicts
    # (LAMMPS has memory.h which conflicts with system <memory.h> used by IREE)
    target_include_directories(ace_iree_plugin PRIVATE
        ${IREE_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${LAMMPS_SRC_DIR}
        ${LAMMPS_SRC_DIR}/ML-IAP
        ${MPI_CXX_INCLUDE_DIRS}
    )

    target_link_libraries(ace_iree_plugin PRIVATE
        ${IREE_STATIC_LIBS}
        pthread
        dl
        m
    )

    set_target_properties(ace_iree_plugin PROPERTIES
        OUTPUT_NAME "ace_iree"
        PREFIX "lib"
    )

    set(BUILD_LAMMPS_PLUGIN TRUE)
else()
    message(STATUS "Skipping LAMMPS plugin (missing: MPI=${MPI_FOUND} LAMMPS=${LAMMPS_FOUND} IREE_LIBS=${IREE_LIBS_FOUND} IREE_HEADERS=${IREE_HEADERS_FOUND})")
    set(BUILD_LAMMPS_PLUGIN FALSE)
endif()

# ==============================================================================
# Kokkos-Enabled IREE Pair Style (optional - requires Kokkos + LAMMPS)
# ==============================================================================
# Provides zero-copy GPU integration between Kokkos and IREE
option(BUILD_KOKKOS_PAIR "Build Kokkos-enabled pair_iree_kokkos" OFF)

# For LAMMPS plugins, we don't link against Kokkos - we use LAMMPS's Kokkos at runtime
# We just need the Kokkos headers from the LAMMPS source tree
if(EXISTS "${LAMMPS_SRC_DIR}/KOKKOS/kokkos_type.h")
    message(STATUS "Found Kokkos headers in LAMMPS source")
    set(Kokkos_FOUND TRUE)
    # Check if LAMMPS was built with CUDA by looking for CUDA types
    if(EXISTS "${LAMMPS_SRC_DIR}/KOKKOS/atom_kokkos.h")
        set(HAS_KOKKOS_CUDA TRUE)
        message(STATUS "  Assuming Kokkos CUDA backend (plugin will use LAMMPS's Kokkos)")
    endif()
else()
    message(STATUS "Kokkos headers not found in LAMMPS - pair_iree_kokkos will not be built")
    set(Kokkos_FOUND FALSE)
    set(BUILD_KOKKOS_PAIR OFF)
endif()

# Additional IREE CUDA libraries (for Kokkos+GPU builds)
if(HAS_KOKKOS_CUDA AND EXISTS "${IREE_LIBRARY_DIRS}/libiree_hal_drivers_cuda_cuda.a")
    set(IREE_CUDA_LIBS
        ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_cuda_cuda.a
        ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_cuda_dynamic_symbols.a
        ${IREE_LIBRARY_DIRS}/libiree_hal_drivers_cuda_registration_registration.a
    )
    # Additional utility libraries that CUDA driver depends on
    file(GLOB IREE_CUDA_UTILS_LIBS "${IREE_LIBRARY_DIRS}/libiree_hal_utils_*.a")
    set(HAS_IREE_CUDA TRUE)
    message(STATUS "Found IREE CUDA libraries")
else()
    set(HAS_IREE_CUDA FALSE)
endif()

# Build Kokkos pair_style if requested and dependencies available
if(BUILD_KOKKOS_PAIR AND Kokkos_FOUND AND LAMMPS_FOUND AND IREE_LIBS_FOUND AND IREE_HEADERS_FOUND)
    message(STATUS "Building Kokkos-enabled pair_iree_kokkos")

    add_library(pair_iree_kokkos SHARED
        src/pair_iree_kokkos.cpp
    )

    target_compile_definitions(pair_iree_kokkos PRIVATE
        IREE_AVAILABLE
        IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
    )

    if(HAS_IREE_CUDA)
        target_compile_definitions(pair_iree_kokkos PRIVATE IREE_HAL_HAVE_CUDA_DRIVER)
    endif()

    # NOTE: We DON'T use find_package(Kokkos) or link Kokkos::kokkos
    # Plugins use LAMMPS's Kokkos at runtime - linking a separate Kokkos causes
    # finalization conflicts ("Execution space instance couldn't be found")
    # We just need headers from LAMMPS source tree and lib/kokkos
    set(LAMMPS_KOKKOS_DIR "${LAMMPS_SRC_DIR}/../lib/kokkos")
    set(LAMMPS_BUILD_DIR "${LAMMPS_SRC_DIR}/../build")
    target_include_directories(pair_iree_kokkos PRIVATE
        ${IREE_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${LAMMPS_SRC_DIR}
        ${LAMMPS_SRC_DIR}/KOKKOS
        ${LAMMPS_SRC_DIR}/PLUGIN
        ${LAMMPS_KOKKOS_DIR}/core/src
        ${LAMMPS_KOKKOS_DIR}/containers/src
        ${LAMMPS_KOKKOS_DIR}/algorithms/src
        ${LAMMPS_KOKKOS_DIR}/simd/src
        ${LAMMPS_KOKKOS_DIR}/tpls/desul/include  # desul atomics headers
        ${LAMMPS_KOKKOS_DIR}/tpls/mdspan/include  # mdspan headers
        ${LAMMPS_BUILD_DIR}/lib/kokkos  # Generated KokkosCore_config.h
        ${LAMMPS_BUILD_DIR}/lib/kokkos/core/src  # Generated desul Config.hpp
        ${MPI_CXX_INCLUDE_DIRS}
    )

    # Kokkos CUDA requires extended-lambda for device lambdas
    target_compile_options(pair_iree_kokkos PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-gpu-targets>
    )

    # Link IREE libraries
    # Note: CUDA libraries need --whole-archive to ensure static initializers are included
    # Order matters: cuda.a must come before registration.a to resolve dependencies
    if(HAS_IREE_CUDA)
        target_link_libraries(pair_iree_kokkos PRIVATE
            ${IREE_STATIC_LIBS}
            ${IREE_CUDA_UTILS_LIBS}
            -Wl,--whole-archive
            ${IREE_CUDA_LIBS}
            -Wl,--no-whole-archive
            cuda
            cudart
            pthread
            dl
            m
        )
    else()
        target_link_libraries(pair_iree_kokkos PRIVATE
            ${IREE_STATIC_LIBS}
            pthread
            dl
            m
        )
    endif()

    set_target_properties(pair_iree_kokkos PROPERTIES
        OUTPUT_NAME "pair_iree_kokkos"
        PREFIX "lib"
    )

    set(BUILD_KOKKOS_PAIR_STYLE TRUE)
else()
    set(BUILD_KOKKOS_PAIR_STYLE FALSE)
endif()

# ==============================================================================
# Non-Kokkos IREE Pair Style (CPU fallback)
# ==============================================================================
# Simpler pair_style for CPU-only deployments
if(LAMMPS_FOUND AND IREE_LIBS_FOUND AND IREE_HEADERS_FOUND)
    message(STATUS "Building non-Kokkos pair_iree (CPU fallback)")

    add_library(pair_iree SHARED
        src/pair_iree.cpp
        src/iree_wrapper.cpp
    )

    target_compile_definitions(pair_iree PRIVATE
        IREE_AVAILABLE
        IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
    )

    target_include_directories(pair_iree PRIVATE
        ${IREE_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${LAMMPS_SRC_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
    )

    target_link_libraries(pair_iree PRIVATE
        ${IREE_STATIC_LIBS}
        pthread
        dl
        m
    )

    set_target_properties(pair_iree PROPERTIES
        OUTPUT_NAME "pair_iree"
        PREFIX "lib"
    )

    set(BUILD_PAIR_IREE TRUE)

    # Build simpler pair_iree_gradient for Reactant-exported VMFBs
    message(STATUS "Building pair_iree_gradient (Reactant VMFB support)")

    add_library(pair_iree_gradient SHARED
        src/pair_iree_gradient.cpp
    )

    target_compile_definitions(pair_iree_gradient PRIVATE
        IREE_AVAILABLE
        IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
    )

    target_include_directories(pair_iree_gradient PRIVATE
        ${IREE_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/src
        ${LAMMPS_SRC_DIR}
        ${MPI_CXX_INCLUDE_DIRS}
    )

    target_link_libraries(pair_iree_gradient PRIVATE
        ${IREE_STATIC_LIBS}
        pthread
        dl
        m
    )

    set_target_properties(pair_iree_gradient PROPERTIES
        OUTPUT_NAME "pair_iree_gradient"
        PREFIX "lib"
    )

    set(BUILD_PAIR_IREE_GRADIENT TRUE)
else()
    set(BUILD_PAIR_IREE FALSE)
    set(BUILD_PAIR_IREE_GRADIENT FALSE)
endif()

# ==============================================================================
# Installation
# ==============================================================================
# Always install ace_forces
install(TARGETS ace_forces
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    src/ace_forces.h
    DESTINATION include/ace_iree
)

# Install LAMMPS plugin if built
if(BUILD_LAMMPS_PLUGIN)
    install(TARGETS ace_iree_plugin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    install(FILES
        src/mliap_ace_iree.h
        src/iree_wrapper.h
        DESTINATION include/ace_iree
    )
endif()

# Install pair_iree if built
if(BUILD_PAIR_IREE)
    install(TARGETS pair_iree
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    install(FILES
        src/pair_iree.h
        DESTINATION include/ace_iree
    )
endif()

# Install pair_iree_kokkos if built
if(BUILD_KOKKOS_PAIR_STYLE)
    install(TARGETS pair_iree_kokkos
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

    install(FILES
        src/pair_iree_kokkos.h
        DESTINATION include/ace_iree
    )
endif()

# ==============================================================================
# Print configuration summary
# ==============================================================================
message(STATUS "")
message(STATUS "=== ACE IREE Configuration ===")
message(STATUS "  ace_forces library: ENABLED (always built)")
message(STATUS "  LAMMPS ML-IAP plugin: ${BUILD_LAMMPS_PLUGIN}")
message(STATUS "  pair_iree (CPU): ${BUILD_PAIR_IREE}")
message(STATUS "  pair_iree_gradient (Reactant VMFBs): ${BUILD_PAIR_IREE_GRADIENT}")
message(STATUS "  pair_iree_kokkos (GPU): ${BUILD_KOKKOS_PAIR_STYLE}")
if(BUILD_LAMMPS_PLUGIN OR BUILD_PAIR_IREE OR BUILD_KOKKOS_PAIR_STYLE)
    message(STATUS "")
    message(STATUS "  IREE libraries: ${IREE_DIR}/lib")
    message(STATUS "  IREE headers: ${IREE_INCLUDE_DIRS}")
    message(STATUS "  LAMMPS source: ${LAMMPS_SRC_DIR}")
    if(Kokkos_FOUND)
        message(STATUS "  Kokkos: ${Kokkos_DIR}")
        message(STATUS "    CUDA: ${HAS_KOKKOS_CUDA}")
    endif()
endif()
message(STATUS "==============================")
message(STATUS "")
