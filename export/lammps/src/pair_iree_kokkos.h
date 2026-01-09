/**
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
 *   pair_style iree model_cuda.vmfb
 *   pair_coeff * * Si
 *   fix 1 all nve/kk
 */

#ifndef LMP_PAIR_IREE_KOKKOS_H
#define LMP_PAIR_IREE_KOKKOS_H

#include "pair_kokkos.h"
#include "kokkos_type.h"
#include "neigh_list_kokkos.h"

// IREE runtime headers
#include "iree/runtime/api.h"
#include "iree/hal/api.h"

#include <string>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

// Forward declarations
template<class DeviceType> class PairIREEKokkos;

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
        const std::vector<int64_t>& shape,
        iree_hal_element_type_t element_type
    );

    /**
     * Create device-local buffer (for outputs).
     */
    iree_hal_buffer_view_t* create_device_buffer(
        const std::vector<int64_t>& shape,
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
     * Synchronize (wait for all GPU operations to complete).
     */
    void sync();

    bool is_valid() const { return session_ != nullptr; }
    iree_hal_device_t* device() const { return hal_device_; }

private:
    iree_runtime_instance_t* instance_ = nullptr;
    iree_runtime_session_t* session_ = nullptr;
    iree_hal_device_t* hal_device_ = nullptr;

    iree_vm_function_t compute_pair_forces_fn_;
    iree_vm_function_t compute_energy_fn_;

    bool functions_loaded_ = false;
};

/**
 * PairIREEKokkos - Kokkos-enabled IREE pair_style.
 *
 * Template parameter DeviceType determines execution space:
 * - Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace> for GPU
 * - Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace> for CPU
 */
template<class DeviceType>
class PairIREEKokkos : public PairKokkos<DeviceType> {
public:
    // Kokkos type aliases
    using device_type = DeviceType;
    using execution_space = typename DeviceType::execution_space;
    using memory_space = typename DeviceType::memory_space;

    template<typename T>
    using View = Kokkos::View<T, Kokkos::LayoutRight, DeviceType>;

    template<typename T>
    using ViewHost = Kokkos::View<T, Kokkos::LayoutRight, Kokkos::HostSpace>;

    // Inherited from Pair
    using Pair::lmp;
    using Pair::atom;
    using Pair::force;
    using Pair::neighbor;
    using Pair::comm;
    using Pair::list;
    using Pair::cutsq;
    using Pair::cutforce;
    using Pair::eflag;
    using Pair::vflag;
    using Pair::evflag;
    using Pair::eflag_global;
    using Pair::vflag_global;
    using Pair::eng_vdwl;

    PairIREEKokkos(class LAMMPS*);
    ~PairIREEKokkos() override;

    void compute(int, int) override;
    void settings(int, char**) override;
    void coeff(int, char**) override;
    void init_style() override;
    double init_one(int, int) override;

    // Kokkos-specific
    void cleanup_copy();

protected:
    // Model configuration
    std::string vmfb_path_;
    double cutoff_;
    int max_atoms_;
    int max_pairs_;

    // Element mapping
    std::vector<std::string> elements_;
    std::vector<int> type_map_;  // LAMMPS type -> model species index

    // IREE handle
    std::unique_ptr<IREEKokkosHandle> iree_;

    // Kokkos views for pair data (persistent, resized as needed)
    View<int*> d_pair_i_;
    View<int*> d_pair_j_;
    View<F_FLOAT*[3]> d_rij_;
    View<F_FLOAT*[3]> d_pair_forces_;

    // Pair count
    int npairs_;

    // Methods
    void allocate();
    void build_pair_lists();
    void scatter_forces();

    /**
     * Import Kokkos view into IREE (templated helper).
     */
    template<typename ViewType>
    iree_hal_buffer_view_t* import_view(const ViewType& view);
};

// Explicit instantiation declarations
#ifdef KOKKOS_ENABLE_CUDA
extern template class PairIREEKokkos<Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>;
#endif

#ifdef KOKKOS_ENABLE_OPENMP
extern template class PairIREEKokkos<Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>;
#endif

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_IREE_KOKKOS_H
