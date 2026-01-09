/**
 * pair_iree_kokkos.cpp - Implementation of Kokkos-Enabled IREE Pair Style
 * ======================================================================
 *
 * Zero-copy GPU integration between LAMMPS/Kokkos and IREE-compiled ACE models.
 */

#include "pair_iree_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor_kokkos.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "comm.h"

#include <cstring>
#include <cmath>

namespace LAMMPS_NS {

// ===========================================================================
// IREEKokkosHandle Implementation
// ===========================================================================

IREEKokkosHandle::IREEKokkosHandle() {
    memset(&compute_pair_forces_fn_, 0, sizeof(compute_pair_forces_fn_));
    memset(&compute_energy_fn_, 0, sizeof(compute_energy_fn_));
}

IREEKokkosHandle::~IREEKokkosHandle() {
    if (session_) {
        iree_runtime_session_release(session_);
    }
    if (instance_) {
        iree_runtime_instance_release(instance_);
    }
}

bool IREEKokkosHandle::load_module(const char* vmfb_path, const char* device_type) {
    iree_status_t status;

    // Create runtime instance
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);

    status = iree_runtime_instance_create(
        &instance_options,
        iree_allocator_system(),
        &instance_
    );
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Create session options
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    // Create HAL device based on requested type
    iree_hal_device_t* device = nullptr;
    iree_string_view_t device_uri = iree_make_cstring_view(device_type);

    status = iree_runtime_instance_try_create_default_device(
        instance_, device_uri, &device
    );
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }
    hal_device_ = device;

    // Create session with device
    status = iree_runtime_session_create_with_device(
        instance_,
        &session_options,
        device,
        iree_runtime_instance_host_allocator(instance_),
        &session_
    );
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Load VMFB module
    status = iree_runtime_session_append_bytecode_module_from_file(
        session_, vmfb_path
    );
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Look up exported functions
    status = iree_runtime_session_lookup_function(
        session_,
        iree_make_cstring_view("module.compute_pair_forces"),
        &compute_pair_forces_fn_
    );
    if (iree_status_is_ok(status)) {
        functions_loaded_ = true;
    } else {
        // Try alternative name
        iree_status_free(status);
        status = iree_runtime_session_lookup_function(
            session_,
            iree_make_cstring_view("module.main"),
            &compute_pair_forces_fn_
        );
        if (iree_status_is_ok(status)) {
            functions_loaded_ = true;
        }
    }

    // Optional: look up energy function
    iree_runtime_session_lookup_function(
        session_,
        iree_make_cstring_view("module.compute_energy"),
        &compute_energy_fn_
    );

    return functions_loaded_;
}

iree_hal_buffer_view_t* IREEKokkosHandle::import_gpu_buffer(
    void* ptr,
    size_t size_bytes,
    const std::vector<int64_t>& shape,
    iree_hal_element_type_t element_type
) {
    if (!hal_device_ || !ptr) return nullptr;

    iree_hal_buffer_t* buffer = nullptr;
    iree_hal_buffer_view_t* buffer_view = nullptr;
    iree_status_t status;

    // Set up external buffer descriptor pointing to GPU memory
    iree_hal_external_buffer_t external_buffer;
    memset(&external_buffer, 0, sizeof(external_buffer));
    external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION;
    external_buffer.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
    external_buffer.size = size_bytes;
    external_buffer.handle.device_allocation.ptr = (uint64_t)ptr;

    // Buffer parameters for device-local memory
    iree_hal_buffer_params_t buffer_params;
    memset(&buffer_params, 0, sizeof(buffer_params));
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;

    // Import the external buffer (zero-copy)
    status = iree_hal_allocator_import_buffer(
        iree_hal_device_allocator(hal_device_),
        buffer_params,
        &external_buffer,
        /*release_callback=*/nullptr,
        &buffer
    );

    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return nullptr;
    }

    // Create buffer view with shape/type metadata
    status = iree_hal_buffer_view_create(
        buffer,
        shape.size(),
        shape.data(),
        element_type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        iree_hal_device_host_allocator(hal_device_),
        &buffer_view
    );

    iree_hal_buffer_release(buffer);  // View holds reference

    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return nullptr;
    }

    return buffer_view;
}

iree_hal_buffer_view_t* IREEKokkosHandle::create_device_buffer(
    const std::vector<int64_t>& shape,
    iree_hal_element_type_t element_type
) {
    if (!hal_device_) return nullptr;

    // Calculate size
    size_t elem_size = iree_hal_element_dense_byte_count(element_type);
    size_t total_elems = 1;
    for (auto dim : shape) total_elems *= dim;
    size_t size_bytes = total_elems * elem_size;

    iree_hal_buffer_t* buffer = nullptr;
    iree_hal_buffer_view_t* buffer_view = nullptr;
    iree_status_t status;

    // Allocate device-local buffer
    iree_hal_buffer_params_t buffer_params;
    memset(&buffer_params, 0, sizeof(buffer_params));
    buffer_params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    buffer_params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT |
                          IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;
    buffer_params.access = IREE_HAL_MEMORY_ACCESS_ALL;

    status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(hal_device_),
        buffer_params,
        size_bytes,
        &buffer
    );

    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return nullptr;
    }

    // Create buffer view
    status = iree_hal_buffer_view_create(
        buffer,
        shape.size(),
        shape.data(),
        element_type,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        iree_hal_device_host_allocator(hal_device_),
        &buffer_view
    );

    iree_hal_buffer_release(buffer);

    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return nullptr;
    }

    return buffer_view;
}

bool IREEKokkosHandle::invoke_pair_forces(
    iree_hal_buffer_view_t** inputs,
    int num_inputs,
    iree_hal_buffer_view_t* output
) {
    if (!session_ || !functions_loaded_) return false;

    iree_status_t status;

    // Create call with inputs and output
    iree_runtime_call_t call;
    status = iree_runtime_call_initialize(session_, compute_pair_forces_fn_, &call);
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Push inputs
    for (int i = 0; i < num_inputs; i++) {
        status = iree_runtime_call_inputs_push_back_buffer_view(&call, inputs[i]);
        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }
    }

    // Invoke
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Pop output (the output buffer is filled in-place or we get result)
    iree_hal_buffer_view_t* result_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Copy result to output if needed (for now, assume output is the result)
    if (result_view != output) {
        // TODO: Handle buffer copy if result != preallocated output
        iree_hal_buffer_view_release(result_view);
    }

    iree_runtime_call_deinitialize(&call);
    return true;
}

bool IREEKokkosHandle::invoke_energy(
    iree_hal_buffer_view_t** inputs,
    int num_inputs,
    float* energy_out
) {
    if (!session_) return false;

    // Check if energy function exists
    if (compute_energy_fn_.module == nullptr) {
        // Use pair_forces function and sum would need to be done externally
        return false;
    }

    iree_status_t status;

    iree_runtime_call_t call;
    status = iree_runtime_call_initialize(session_, compute_energy_fn_, &call);
    if (!iree_status_is_ok(status)) {
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    for (int i = 0; i < num_inputs; i++) {
        status = iree_runtime_call_inputs_push_back_buffer_view(&call, inputs[i]);
        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }
    }

    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get scalar output
    iree_hal_buffer_view_t* result_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Read scalar value
    iree_hal_buffer_t* result_buffer = iree_hal_buffer_view_buffer(result_view);
    iree_hal_buffer_map_range(
        result_buffer,
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ,
        0, sizeof(float),
        (iree_hal_buffer_mapping_t*)energy_out
    );

    iree_hal_buffer_view_release(result_view);
    iree_runtime_call_deinitialize(&call);
    return true;
}

void IREEKokkosHandle::sync() {
    if (hal_device_) {
        iree_hal_device_wait_semaphores(
            hal_device_,
            IREE_HAL_WAIT_MODE_ALL,
            iree_hal_semaphore_list_empty(),
            iree_infinite_timeout()
        );
    }
}

// ===========================================================================
// PairIREEKokkos Implementation
// ===========================================================================

template<class DeviceType>
PairIREEKokkos<DeviceType>::PairIREEKokkos(LAMMPS* lmp) :
    PairKokkos<DeviceType>(lmp),
    cutoff_(5.0),
    max_atoms_(4096),
    max_pairs_(200000),
    npairs_(0)
{
    // Kokkos pair style flags
    this->kokkosable = 1;
    this->atomKK = (AtomKokkos*)this->atom;
    this->execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

    // Allocate IREE handle
    iree_ = std::make_unique<IREEKokkosHandle>();
}

template<class DeviceType>
PairIREEKokkos<DeviceType>::~PairIREEKokkos() {
    // Views cleaned up automatically by Kokkos
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::settings(int narg, char** arg) {
    if (narg < 1) {
        this->error->all(FLERR, "Illegal pair_style iree command");
    }

    vmfb_path_ = arg[0];

    // Optional: device type (default to cuda for GPU, local-task for CPU)
    const char* device_type = "cuda";
    if (narg > 1) {
        device_type = arg[1];
    }

    // Load IREE module
    if (!iree_->load_module(vmfb_path_.c_str(), device_type)) {
        this->error->all(FLERR, "Failed to load IREE module");
    }
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::coeff(int narg, char** arg) {
    if (narg < 3) {
        this->error->all(FLERR, "Incorrect args for pair coefficients");
    }

    // Parse element names (pair_coeff * * Si O)
    int ntypes = this->atom->ntypes;
    elements_.clear();
    type_map_.resize(ntypes + 1, -1);

    for (int i = 2; i < narg; i++) {
        elements_.push_back(arg[i]);
        // Map LAMMPS type i-1 to model element index i-2
        if (i - 2 < ntypes) {
            type_map_[i - 1] = i - 2;
        }
    }

    // Set cutoffs
    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            this->cutsq[i][j] = cutoff_ * cutoff_;
        }
    }
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::init_style() {
    // Request half neighbor list with Kokkos
    this->neighbor->add_request(this, NeighConst::REQ_DEFAULT);

    // Allocate pair data views
    allocate();
}

template<class DeviceType>
double PairIREEKokkos<DeviceType>::init_one(int i, int j) {
    return cutoff_;
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::allocate() {
    // Resize views for maximum expected pairs
    Kokkos::resize(d_pair_i_, max_pairs_);
    Kokkos::resize(d_pair_j_, max_pairs_);
    Kokkos::resize(d_rij_, max_pairs_);
    Kokkos::resize(d_pair_forces_, max_pairs_);
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
    this->eflag = eflag_in;
    this->vflag = vflag_in;

    if (this->eflag || this->vflag) {
        this->ev_setup(this->eflag, this->vflag);
    }

    // Sync atom data to device
    this->atomKK->sync(this->execution_space, X_MASK | TYPE_MASK);
    auto x = this->atomKK->k_x.view<DeviceType>();
    auto type = this->atomKK->k_type.view<DeviceType>();
    auto f = this->atomKK->k_f.view<DeviceType>();

    // Get neighbor list
    NeighListKokkos<DeviceType>* k_list =
        static_cast<NeighListKokkos<DeviceType>*>(this->list);
    auto d_numneigh = k_list->d_numneigh;
    auto d_neighbors = k_list->d_neighbors;
    auto d_ilist = k_list->d_ilist;
    int inum = k_list->inum;

    // Build pair arrays on GPU
    build_pair_lists();

    if (npairs_ == 0) {
        this->atomKK->modified(this->execution_space, F_MASK);
        return;
    }

    // Import Kokkos views into IREE (zero-copy)
    std::vector<int64_t> rij_shape = {npairs_, 3};
    auto rij_iree = iree_->import_gpu_buffer(
        d_rij_.data(),
        npairs_ * 3 * sizeof(F_FLOAT),
        rij_shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32
    );

    // Create output buffer for pair forces
    auto pair_forces_iree = iree_->create_device_buffer(
        rij_shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32
    );

    // TODO: Import additional inputs (types, pool_matrix, model params)
    // For now, simplified interface with just rij

    iree_hal_buffer_view_t* inputs[] = {rij_iree};
    int num_inputs = 1;

    // Invoke IREE compute_pair_forces
    bool success = iree_->invoke_pair_forces(inputs, num_inputs, pair_forces_iree);

    if (!success) {
        this->error->all(FLERR, "IREE pair forces computation failed");
    }

    // TODO: Copy IREE output to d_pair_forces_ view
    // For zero-copy, we'd import d_pair_forces_ as output directly

    // Scatter pair forces to atom forces using Kokkos
    scatter_forces();

    // Cleanup IREE buffer views
    if (rij_iree) iree_hal_buffer_view_release(rij_iree);
    if (pair_forces_iree) iree_hal_buffer_view_release(pair_forces_iree);

    // Mark forces as modified on device
    this->atomKK->modified(this->execution_space, F_MASK);
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::build_pair_lists() {
    // Get neighbor list views
    NeighListKokkos<DeviceType>* k_list =
        static_cast<NeighListKokkos<DeviceType>*>(this->list);

    auto d_numneigh = k_list->d_numneigh;
    auto d_neighbors = k_list->d_neighbors;
    auto d_ilist = k_list->d_ilist;
    int inum = k_list->inum;

    auto x = this->atomKK->k_x.view<DeviceType>();
    double cutsq = cutoff_ * cutoff_;

    // Count pairs first
    auto d_pair_i = d_pair_i_;
    auto d_pair_j = d_pair_j_;
    auto d_rij = d_rij_;

    // Use atomic counter for pair count
    Kokkos::View<int, DeviceType> d_npairs("npairs");
    Kokkos::deep_copy(d_npairs, 0);

    // Build pair arrays
    Kokkos::parallel_for("build_pairs", inum,
        KOKKOS_LAMBDA(int ii) {
            int i = d_ilist(ii);
            double xi = x(i, 0);
            double yi = x(i, 1);
            double zi = x(i, 2);

            int jnum = d_numneigh(i);
            for (int jj = 0; jj < jnum; jj++) {
                int j = d_neighbors(i, jj);
                j &= NEIGHMASK;

                double dx = x(j, 0) - xi;
                double dy = x(j, 1) - yi;
                double dz = x(j, 2) - zi;
                double rsq = dx*dx + dy*dy + dz*dz;

                if (rsq < cutsq) {
                    int idx = Kokkos::atomic_fetch_add(&d_npairs(), 1);
                    if (idx < d_pair_i.extent(0)) {
                        d_pair_i(idx) = i;
                        d_pair_j(idx) = j;
                        d_rij(idx, 0) = dx;
                        d_rij(idx, 1) = dy;
                        d_rij(idx, 2) = dz;
                    }
                }
            }
        }
    );

    Kokkos::fence();

    // Copy pair count back to host
    auto h_npairs = Kokkos::create_mirror_view(d_npairs);
    Kokkos::deep_copy(h_npairs, d_npairs);
    npairs_ = h_npairs();

    if (npairs_ > max_pairs_) {
        this->error->all(FLERR, "Too many pairs, increase max_pairs");
    }
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::scatter_forces() {
    auto f = this->atomKK->k_f.view<DeviceType>();
    auto d_pair_i = d_pair_i_;
    auto d_pair_j = d_pair_j_;
    auto d_pair_forces = d_pair_forces_;
    int npairs = npairs_;

    // Scatter pair forces to atom forces using atomic operations
    Kokkos::parallel_for("scatter_forces", npairs,
        KOKKOS_LAMBDA(int e) {
            int i = d_pair_i(e);
            int j = d_pair_j(e);

            F_FLOAT fx = d_pair_forces(e, 0);
            F_FLOAT fy = d_pair_forces(e, 1);
            F_FLOAT fz = d_pair_forces(e, 2);

            // f_i -= pair_force (action on i from j)
            Kokkos::atomic_add(&f(i, 0), -fx);
            Kokkos::atomic_add(&f(i, 1), -fy);
            Kokkos::atomic_add(&f(i, 2), -fz);

            // f_j += pair_force (Newton's 3rd law)
            Kokkos::atomic_add(&f(j, 0), fx);
            Kokkos::atomic_add(&f(j, 1), fy);
            Kokkos::atomic_add(&f(j, 2), fz);
        }
    );

    Kokkos::fence();
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::cleanup_copy() {
    // Called during neighbor list rebuild
}

template<class DeviceType>
template<typename ViewType>
iree_hal_buffer_view_t* PairIREEKokkos<DeviceType>::import_view(const ViewType& view) {
    // Helper to import arbitrary Kokkos view into IREE
    std::vector<int64_t> shape;
    for (unsigned i = 0; i < ViewType::rank; i++) {
        shape.push_back(view.extent(i));
    }

    size_t size_bytes = view.size() * sizeof(typename ViewType::value_type);

    iree_hal_element_type_t elem_type;
    if constexpr (std::is_same_v<typename ViewType::value_type, float>) {
        elem_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    } else if constexpr (std::is_same_v<typename ViewType::value_type, double>) {
        elem_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    } else if constexpr (std::is_same_v<typename ViewType::value_type, int>) {
        elem_type = IREE_HAL_ELEMENT_TYPE_INT_32;
    } else {
        elem_type = IREE_HAL_ELEMENT_TYPE_OPAQUE_32;
    }

    return iree_->import_gpu_buffer(
        (void*)view.data(),
        size_bytes,
        shape,
        elem_type
    );
}

// ===========================================================================
// Explicit Instantiations
// ===========================================================================

#ifdef KOKKOS_ENABLE_CUDA
template class PairIREEKokkos<Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>>;
#endif

#ifdef KOKKOS_ENABLE_OPENMP
template class PairIREEKokkos<Kokkos::Device<Kokkos::OpenMP, Kokkos::HostSpace>>;
#endif

}  // namespace LAMMPS_NS
