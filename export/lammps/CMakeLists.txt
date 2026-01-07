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
else()
    message(WARNING "IREE libraries not found at ${IREE_DIR}/lib. Set IREE_DIR.")
endif()

# Headers come from source checkout (binary dist is incomplete)
if(EXISTS "${IREE_SRC_DIR}/runtime/src/iree/base/allocator.h")
    message(STATUS "Found IREE headers at: ${IREE_SRC_DIR}/runtime/src")
    set(IREE_INCLUDE_DIRS ${IREE_SRC_DIR}/runtime/src)
else()
    message(WARNING "IREE headers not found at ${IREE_SRC_DIR}. Set IREE_SRC_DIR.")
endif()

# Check LAMMPS
if(EXISTS "${LAMMPS_SRC_DIR}/lammps.h")
    message(STATUS "Found LAMMPS source at: ${LAMMPS_SRC_DIR}")
else()
    message(WARNING "LAMMPS source not found at ${LAMMPS_SRC_DIR}. Set LAMMPS_SRC.")
endif()

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

# Build shared library for ML-IAP plugin
add_library(ace_iree_plugin SHARED
    src/iree_wrapper.cpp
    src/mliap_ace_iree.cpp
)

# Enable native IREE if headers and libraries are found
if(EXISTS "${IREE_SRC_DIR}/runtime/src/iree/base/allocator.h" AND
   EXISTS "${IREE_DIR}/lib/libiree_runtime_unified.a")
    message(STATUS "Native IREE embedding enabled")
    target_compile_definitions(ace_iree_plugin PRIVATE
        IREE_AVAILABLE
        IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl
    )
    set(IREE_ENABLED TRUE)
else()
    message(STATUS "Native IREE disabled - missing headers or libraries")
    set(IREE_ENABLED FALSE)
endif()

# Find MPI (required for LAMMPS headers)
find_package(MPI REQUIRED)

# IMPORTANT: IREE includes must come BEFORE LAMMPS to avoid header conflicts
# (LAMMPS has memory.h which conflicts with system <memory.h> used by IREE)
target_include_directories(ace_iree_plugin PRIVATE
    ${IREE_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${LAMMPS_SRC_DIR}
    ${LAMMPS_SRC_DIR}/ML-IAP
    ${MPI_CXX_INCLUDE_DIRS}
)

# Link IREE static libs and system libs
target_link_libraries(ace_iree_plugin PRIVATE
    ${IREE_STATIC_LIBS}
    pthread
    dl
    m
)

# Set output name
set_target_properties(ace_iree_plugin PROPERTIES
    OUTPUT_NAME "ace_iree"
    PREFIX "lib"
)

# Installation
install(TARGETS ace_iree_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES
    src/mliap_ace_iree.h
    src/iree_wrapper.h
    DESTINATION include/ace_iree
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== ACE IREE LAMMPS Plugin Configuration ===")
message(STATUS "  IREE libraries: ${IREE_DIR}/lib")
message(STATUS "  IREE headers: ${IREE_INCLUDE_DIRS}")
message(STATUS "  IREE source: ${IREE_SRC_DIR}")
message(STATUS "  LAMMPS source: ${LAMMPS_SRC_DIR}")
message(STATUS "  Native IREE: ${IREE_ENABLED}")
message(STATUS "============================================")
message(STATUS "")
