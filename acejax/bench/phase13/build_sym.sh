#!/bin/bash
set -ex
source /etc/profile.d/modules.sh 2>/dev/null || true
for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do [ -f $f ] && . $f && break; done
module purge || true
module load foss/2023b CUDA/12.9.1 || true
cd /storage/eng/essswb/phase13/lammps
# Flags copied EXACTLY from build-SKX-AMPERE86-mlpace (the Phase 8 binary), so
# routes 2 and 3 differ only in the pair style, not in how LAMMPS was compiled.
# PLUGIN is kept so the jax/kk plugin loads into this same binary.
cmake -B build-SKX-AMPERE86-symmetrix \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D CMAKE_CXX_COMPILER=/storage/eng/essswb/phase13/lammps/lib/kokkos/bin/nvcc_wrapper \
  -D BUILD_SHARED_LIBS=ON -D BUILD_MPI=ON -D BUILD_OMP=ON \
  -D PKG_KOKKOS=ON -D PKG_MANYBODY=ON -D PKG_PLUGIN=ON \
  -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_SKX=ON -D Kokkos_ARCH_AMPERE86=ON \
  -D SYMMETRIX_KOKKOS=ON -D SYMMETRIX_SPHERICART_CUDA=ON \
  cmake
cmake --build build-SKX-AMPERE86-symmetrix -j 24
echo BUILDDONE
