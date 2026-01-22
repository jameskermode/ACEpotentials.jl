/* -*- c++ -*- ----------------------------------------------------------
 * pair_iree_kokkos.h - Kokkos-Enabled IREE Pair Style with Zero-Copy
 * ====================================================================
 *
 * LAMMPS pair_style using IREE-compiled ACE models with Kokkos for
 * GPU portability and zero-copy data transfer.
 *
 * Architecture:
 * - Kokkos views are imported directly into IREE via HAL external buffers
 * - No GPU→CPU→GPU data transfers during force computation
 * - IREE computes pair_forces[npairs, 3], Kokkos scatters to atom forces
 * - Split computation avoids IREE scatter op compilation issues
 *
 * Memory Flow (all on GPU):
 * ┌─────────────────────────────────────────────────────────────────┐
 * │ LAMMPS/Kokkos: x[], f[], type[], neighbor lists                │
 * │        │                                                        │
 * │        ▼ (zero-copy import via IREE HAL)                       │
 * │ IREE: compute_pair_forces(x, rij, types) → pair_forces[]       │
 * │        │                                                        │
 * │        ▼ (Kokkos atomic_add)                                   │
 * │ LAMMPS/Kokkos: f[] += accumulated pair_forces                  │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Usage:
 *   package kokkos cuda/aware on neigh half
 *   pair_style iree/kk model_cuda.vmfb
 *   pair_coeff * * Si
 *   fix 1 all nve/kk
 *
 ------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(iree/kk,PairIREEKokkos<LMPDeviceType>);
PairStyle(iree/kk/device,PairIREEKokkos<LMPDeviceType>);
PairStyle(iree/kk/host,PairIREEKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_IREE_KOKKOS_H
#define LMP_PAIR_IREE_KOKKOS_H

#include "pair.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"

// IREE runtime headers
#include "iree/runtime/api.h"
#include "iree/hal/api.h"

// CUDA headers for context sharing
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#include <string>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

/**
 * IREE Model Handle for Kokkos integration.
 *
 * Manages IREE runtime, session, and compiled module.
 * Provides zero-copy buffer import and invocation.
 */
class IREEKokkosHandle {
public:
    IREEKokkosHandle();
    ~IREEKokkosHandle();

    // Non-copyable
    IREEKokkosHandle(const IREEKokkosHandle&) = delete;
    IREEKokkosHandle& operator=(const IREEKokkosHandle&) = delete;

    /**
     * Load compiled VMFB module.
     *
     * @param vmfb_path Path to compiled .vmfb file
     * @param device_type "cuda" or "local-task" (CPU)
     * @return true on success
     */
    bool load_module(const char* vmfb_path, const char* device_type);

    /**
     * Import external GPU buffer into IREE (zero-copy).
     *
     * @param ptr GPU pointer to data
     * @param size_bytes Size in bytes
     * @param shape Tensor shape
     * @param element_type IREE element type
     * @return Buffer view handle (caller must release)
     */
    iree_hal_buffer_view_t* import_gpu_buffer(
        void* ptr,
        size_t size_bytes,
        const std::vector<iree_hal_dim_t>& shape,
        iree_hal_element_type_t element_type
    );

    /**
     * Create device-local buffer (for outputs).
     */
    iree_hal_buffer_view_t* create_device_buffer(
        const std::vector<iree_hal_dim_t>& shape,
        iree_hal_element_type_t element_type
    );

    /**
     * Invoke compute_pair_forces function.
     *
     * @param inputs Input buffer views
     * @param num_inputs Number of inputs
     * @param output Output buffer view (pre-allocated)
     * @return true on success
     */
    bool invoke_pair_forces(
        iree_hal_buffer_view_t** inputs,
        int num_inputs,
        iree_hal_buffer_view_t* output
    );

    /**
     * Invoke compute_energy function.
     */
    bool invoke_energy(
        iree_hal_buffer_view_t** inputs,
        int num_inputs,
        float* energy_out
    );

    /**
     * Invoke energy+gradient function (combined output).
     * The VMFB returns (energy_scalar, gradient_tensor, input_passthrough).
     * Returns the device pointer to IREE's gradient buffer for zero-copy access.
     *
     * @param input Input buffer view (rij tensor)
     * @param energy_out Output for energy scalar
     * @param gradient_ptr_out Output for device pointer to gradient buffer
     * @param gradient_size_out Output for size in bytes of gradient buffer
     * @return true on success
     */
    bool invoke_energy_gradient_zerocopy(
        iree_hal_buffer_view_t* input,
        float* energy_out,
        void** gradient_ptr_out,
        size_t* gradient_size_out
    );

    /**
     * Invoke energy+gradient function (combined output).
     * The VMFB returns (energy_scalar, gradient_tensor, input_passthrough).
     * Copies gradient to raw device pointer via GPU->CPU->GPU transfer.
     *
     * @param input Input buffer view (rij tensor)
     * @param energy_out Output for energy scalar
     * @param gradient_dest_ptr Raw device pointer to copy gradient to
     * @param gradient_size Size in bytes of gradient buffer
     * @return true on success
     */
    bool invoke_energy_gradient(
        iree_hal_buffer_view_t* input,
        float* energy_out,
        void* gradient_dest_ptr,
        size_t gradient_size
    );

    /**
     * Invoke energy+gradient with IREE-owned input buffer.
     * Use this when input is created via create_device_buffer() and
     * data is transferred using iree_hal_device_transfer_h2d().
     */
    bool invoke_energy_gradient_direct(
        iree_hal_buffer_view_t* input_iree,
        float* energy_out,
        void* gradient_dest_ptr,
        size_t gradient_size
    );

    /**
     * Invoke energy+gradient with Float64 support and zero-copy.
     * For use with Float64 VMFBs that match Kokkos F_FLOAT (double).
     *
     * @param input_iree IREE-owned input buffer (Float64)
     * @param energy_out Output for energy scalar (double)
     * @param gradient_ptr_out Output for device pointer to gradient (zero-copy)
     * @param gradient_size_out Size of gradient in bytes
     * @return true on success
     */
    bool invoke_energy_gradient_f64_zerocopy(
        iree_hal_buffer_view_t* input_iree,
        double* energy_out,
        void** gradient_ptr_out,
        size_t* gradient_size_out
    );

    /**
     * Perform D2D copy of gradient from IREE buffer to Kokkos buffer.
     * Used when shared context allows direct GPU-to-GPU copy.
     *
     * @param iree_gradient_ptr Device pointer to IREE gradient buffer
     * @param kokkos_gradient_ptr Device pointer to Kokkos gradient buffer
     * @param size_bytes Number of bytes to copy
     * @return true on success
     */
    bool copy_gradient_d2d(
        void* iree_gradient_ptr,
        void* kokkos_gradient_ptr,
        size_t size_bytes
    );

    /**
     * Synchronize (wait for all GPU operations to complete).
     */
    void sync();

    bool is_valid() const { return session_ != nullptr; }
    iree_hal_device_t* device() const { return hal_device_; }
    bool is_cuda() const { return is_cuda_device_; }
    bool uses_shared_context() const { return shared_cuda_context_; }

private:
    iree_runtime_instance_t* instance_ = nullptr;
    iree_runtime_session_t* session_ = nullptr;
    iree_hal_device_t* hal_device_ = nullptr;

    iree_vm_function_t compute_pair_forces_fn_;
    iree_vm_function_t compute_energy_fn_;

    bool functions_loaded_ = false;
    bool is_cuda_device_ = false;
    bool shared_cuda_context_ = false;

#ifdef KOKKOS_ENABLE_CUDA
    CUcontext kokkos_cuda_context_ = nullptr;
    CUstream kokkos_cuda_stream_ = nullptr;
#endif

    // Cached gradient buffer view from last invocation (for zero-copy)
    iree_hal_buffer_view_t* cached_gradient_view_ = nullptr;
};

/**
 * PairIREEKokkos - Kokkos-enabled IREE pair_style.
 *
 * Template parameter DeviceType is LMPDeviceType or LMPHostType.
 *
 * Inherits from Pair directly (like other LAMMPS Kokkos pair styles).
 */
template<class DeviceType>
class PairIREEKokkos : public Pair {
public:
    // Kokkos type aliases
    typedef DeviceType device_type;
    typedef ArrayTypes<DeviceType> AT;

    enum {EnabledNeighFlags = FULL|HALFTHREAD|HALF};

    PairIREEKokkos(class LAMMPS*);
    ~PairIREEKokkos() override;

    void compute(int, int) override;
    void settings(int, char**) override;
    void coeff(int, char**) override;
    void init_style() override;
    void init_list(int, class NeighList*) override;  // needed for ptr to neighbor list
    double init_one(int, int) override;

protected:
    // Model configuration
    std::string vmfb_path_;
    double cutoff_;
    int max_atoms_;
    int max_pairs_;
    int vmfb_size_;  // Max edges the loaded VMFB can handle (from bucket size)

    // Element mapping
    std::vector<std::string> elements_;
    std::vector<int> type_map_;  // LAMMPS type -> model species index

    // IREE handle
    std::unique_ptr<IREEKokkosHandle> iree_;

    // Kokkos atom interface
    class AtomKokkos* atomKK;
    ExecutionSpace execution_space;
    int neighflag;
    int eflag, vflag;
    int newton_pair;

    // Kokkos views for positions/forces (mirrors of LAMMPS data)
    typename AT::t_x_array_randomread x;
    typename AT::t_f_array f;
    typename AT::t_int_1d_randomread type;

    // Kokkos views for pair data (persistent, resized as needed)
    typename AT::t_int_1d d_pair_i_;
    typename AT::t_int_1d d_pair_j_;
    typename AT::t_x_array d_rij_;
    typename AT::t_f_array d_pair_forces_;

    // Per-atom energy/virial for Kokkos
    DAT::tdual_efloat_1d k_eatom;
    DAT::tdual_virial_array k_vatom;
    typename AT::t_efloat_1d d_eatom;
    typename AT::t_virial_array d_vatom;

    // Pair count
    int npairs_;
    int nlocal, nall;

    // Methods
    void allocate();

public:
    // These methods must be public for CUDA extended lambda support
    void build_pair_lists();
    void scatter_forces();

protected:

    /**
     * Import Kokkos view into IREE (templated helper).
     */
    template<typename ViewType>
    iree_hal_buffer_view_t* import_view(const ViewType& view);
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_IREE_KOKKOS_H
#endif
