# Common environment for every Phase 13 LAMMPS invocation.
for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh; do [ -f $f ] && . $f && break; done
module purge >/dev/null 2>&1 || true
module load foss/2023b CUDA/12.9.1 >/dev/null 2>&1 || true
V=/storage/eng/essswb/venvs/lammps-jax
P13=/storage/eng/essswb/phase13
export PJRT=$V/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so
export LAMMPS_PLUGIN_PATH=/storage/eng/essswb/lammps-jax-build/build-plugin-shared-cudart
# Do not preallocate the card: a co-tenant has been OOM d by that before.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
LMP=${LMP:-$P13/lammps/build-SKX-AMPERE86-symmetrix/lmp}
# The build directory MUST precede $V/lib.  BUILD_SHARED_LIBS=ON means every
# style lives in liblammps.so, so with $V/lib first a new lmp silently loads the
# OLD library and pair styles vanish.  This has bitten three times.
export LD_LIBRARY_PATH=$(dirname $LMP):$V/lib:/software/easybuild/software/CUDA/12.9.1/lib64:/software/easybuild/software/OpenMPI/4.1.6-GCC-13.2.0/lib:${LD_LIBRARY_PATH:-}
KK="-k on g 1 -sf kk -pk kokkos newton on neigh half gpu/aware off"
