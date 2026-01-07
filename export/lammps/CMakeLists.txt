cmake_minimum_required(VERSION 3.16)
project(ace_iree_lammps CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find IREE runtime
find_package(IREECompiler QUIET)
if(NOT IREECompiler_FOUND)
    # Try pkg-config
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(IREE iree_runtime)
    endif()
endif()

# If IREE not found via package, try environment variable
if(NOT IREE_FOUND AND NOT IREECompiler_FOUND)
    if(DEFINED ENV{IREE_DIR})
        set(IREE_ROOT $ENV{IREE_DIR})
        set(IREE_INCLUDE_DIRS ${IREE_ROOT}/include)
        set(IREE_LIBRARY_DIRS ${IREE_ROOT}/lib)
        find_library(IREE_HAL_LIBRARY
            NAMES iree_hal iree_hal_d
            PATHS ${IREE_LIBRARY_DIRS}
        )
        find_library(IREE_RUNTIME_LIBRARY
            NAMES iree_runtime iree_runtime_d
            PATHS ${IREE_LIBRARY_DIRS}
        )
        set(IREE_LIBRARIES ${IREE_HAL_LIBRARY} ${IREE_RUNTIME_LIBRARY})
    else()
        message(WARNING "IREE not found. Set IREE_DIR environment variable.")
        set(IREE_INCLUDE_DIRS "")
        set(IREE_LIBRARIES "")
    endif()
endif()

# Find LAMMPS
if(DEFINED ENV{LAMMPS_SRC})
    set(LAMMPS_SRC_DIR $ENV{LAMMPS_SRC})
else()
    set(LAMMPS_SRC_DIR "${CMAKE_SOURCE_DIR}/../../../lammps/src" CACHE PATH "LAMMPS source directory")
endif()

# Check LAMMPS headers exist
if(EXISTS "${LAMMPS_SRC_DIR}/lammps.h")
    message(STATUS "Found LAMMPS source at: ${LAMMPS_SRC_DIR}")
else()
    message(WARNING "LAMMPS source not found at ${LAMMPS_SRC_DIR}. Set LAMMPS_SRC environment variable.")
endif()

# Build shared library for ML-IAP plugin
add_library(ace_iree_plugin SHARED
    src/iree_wrapper.cpp
    src/mliap_ace_iree.cpp
)

target_include_directories(ace_iree_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${LAMMPS_SRC_DIR}
    ${LAMMPS_SRC_DIR}/ML-IAP
    ${IREE_INCLUDE_DIRS}
)

target_link_libraries(ace_iree_plugin PRIVATE
    ${IREE_LIBRARIES}
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
message(STATUS "=== ACE IREE LAMMPS Plugin Configuration ===")
message(STATUS "  IREE include dirs: ${IREE_INCLUDE_DIRS}")
message(STATUS "  IREE libraries: ${IREE_LIBRARIES}")
message(STATUS "  LAMMPS source: ${LAMMPS_SRC_DIR}")
message(STATUS "============================================")
