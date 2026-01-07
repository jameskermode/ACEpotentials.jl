/**
 * IREE Runtime Wrapper Implementation
 * ====================================
 *
 * C++ wrapper around IREE runtime C API for ACE model inference.
 * Based on IREE samples/dynamic_shapes/main.c pattern.
 */

#include "iree_wrapper.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cstdio>

#ifdef IREE_AVAILABLE
#include "iree/runtime/api.h"
#endif

namespace ace_iree {

// Default shapes if metadata not available
static const ModelShapes DEFAULT_SHAPES = {
    4096,    // max_atoms
    50,      // max_neigs
    200000   // max_edges
};

/**
 * Implementation class (PIMPL pattern).
 */
class IREEModel::Impl {
public:
#ifdef IREE_AVAILABLE
    iree_runtime_instance_t* instance = nullptr;
    iree_hal_device_t* device = nullptr;
    iree_runtime_session_t* session = nullptr;
#endif

    std::string vmfb_path;
    std::string device_str;
    bool initialized = false;

    // Padded input buffers (reused across calls)
    std::vector<float> edge_rij_padded;
    std::vector<int64_t> atomic_numbers_padded;
    std::vector<int64_t> edge_i_padded;
    std::vector<int64_t> edge_j_padded;

    // Output buffers
    std::vector<float> forces_out;

    Impl() = default;
    ~Impl() { cleanup(); }

    void cleanup() {
#ifdef IREE_AVAILABLE
        if (session) {
            iree_runtime_session_release(session);
            session = nullptr;
        }
        if (device) {
            iree_hal_device_release(device);
            device = nullptr;
        }
        if (instance) {
            iree_runtime_instance_release(instance);
            instance = nullptr;
        }
#endif
        initialized = false;
    }
};

#ifdef IREE_AVAILABLE
// Helper to format IREE status as string
static std::string format_status(iree_status_t status) {
    if (iree_status_is_ok(status)) return "OK";

    iree_allocator_t alloc = iree_allocator_system();
    char* msg = nullptr;
    iree_host_size_t msg_len = 0;

    if (iree_status_to_string(status, &alloc, &msg, &msg_len)) {
        std::string result(msg, msg_len);
        iree_allocator_free(alloc, msg);
        return result;
    }
    return "Unknown error";
}
#endif

// Constructor
IREEModel::IREEModel(const std::string& vmfb_path,
                     const std::string& metadata_path,
                     const std::string& device_str)
    : impl_(std::make_unique<Impl>()) {

    impl_->vmfb_path = vmfb_path;
    impl_->device_str = device_str;

    // Load metadata if provided
    if (!metadata_path.empty()) {
        load_metadata(metadata_path);
    } else {
        shapes_ = DEFAULT_SHAPES;
        metadata_.shapes = shapes_;
        metadata_.rcut = 6.0;  // Default cutoff
        metadata_.n_species = 0;
    }

    // Pre-allocate padded buffers
    impl_->edge_rij_padded.resize(shapes_.max_edges * 3);
    impl_->atomic_numbers_padded.resize(shapes_.max_atoms);
    impl_->edge_i_padded.resize(shapes_.max_edges);
    impl_->edge_j_padded.resize(shapes_.max_edges);
    impl_->forces_out.resize(shapes_.max_atoms * 3);

#ifdef IREE_AVAILABLE
    // Initialize IREE runtime
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_options_use_all_available_drivers(&instance_options);

    iree_status_t status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &impl_->instance);

    if (!iree_status_is_ok(status)) {
        std::string msg = format_status(status);
        iree_status_ignore(status);
        throw std::runtime_error("Failed to create IREE instance: " + msg);
    }

    // Create device
    iree_string_view_t driver_name = iree_make_cstring_view(device_str.c_str());
    status = iree_runtime_instance_try_create_default_device(
        impl_->instance, driver_name, &impl_->device);

    if (!iree_status_is_ok(status)) {
        std::string msg = format_status(status);
        iree_status_ignore(status);
        throw std::runtime_error("Failed to create device '" + device_str + "': " + msg);
    }

    // Create session
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    status = iree_runtime_session_create_with_device(
        impl_->instance, &session_options, impl_->device,
        iree_runtime_instance_host_allocator(impl_->instance), &impl_->session);

    if (!iree_status_is_ok(status)) {
        std::string msg = format_status(status);
        iree_status_ignore(status);
        throw std::runtime_error("Failed to create session: " + msg);
    }

    // Load module
    status = iree_runtime_session_append_bytecode_module_from_file(
        impl_->session, vmfb_path.c_str());

    if (!iree_status_is_ok(status)) {
        std::string msg = format_status(status);
        iree_status_ignore(status);
        throw std::runtime_error("Failed to load VMFB '" + vmfb_path + "': " + msg);
    }

    impl_->initialized = true;
#else
    throw std::runtime_error("IREE not available - rebuild with IREE support");
#endif
}

// Destructor
IREEModel::~IREEModel() = default;

// Move operations
IREEModel::IREEModel(IREEModel&&) noexcept = default;
IREEModel& IREEModel::operator=(IREEModel&&) noexcept = default;

void IREEModel::load_metadata(const std::string& path) {
    // Simple JSON parsing for metadata
    std::ifstream f(path);
    if (!f.is_open()) {
        shapes_ = DEFAULT_SHAPES;
        metadata_.shapes = shapes_;
        return;
    }

    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

    // Initialize with defaults
    shapes_ = DEFAULT_SHAPES;

    // Parse max_atoms
    auto pos = content.find("\"max_atoms\"");
    if (pos != std::string::npos) {
        pos = content.find(":", pos);
        if (pos != std::string::npos) {
            shapes_.max_atoms = std::stoi(content.substr(pos + 1));
        }
    }

    // Parse max_neigs
    pos = content.find("\"max_neigs\"");
    if (pos != std::string::npos) {
        pos = content.find(":", pos);
        if (pos != std::string::npos) {
            shapes_.max_neigs = std::stoi(content.substr(pos + 1));
        }
    }

    // Parse max_edges
    pos = content.find("\"max_edges\"");
    if (pos != std::string::npos) {
        pos = content.find(":", pos);
        if (pos != std::string::npos) {
            shapes_.max_edges = std::stoi(content.substr(pos + 1));
        }
    }

    // Parse rcut
    pos = content.find("\"rcut\"");
    if (pos != std::string::npos) {
        pos = content.find(":", pos);
        if (pos != std::string::npos) {
            metadata_.rcut = std::stod(content.substr(pos + 1));
        }
    }

    metadata_.shapes = shapes_;
}

EFVOutput IREEModel::compute_efv(
    const float* edge_rij,
    const int64_t* atomic_numbers,
    const int64_t* edge_i,
    const int64_t* edge_j,
    int32_t n_atoms,
    int32_t n_edges) {

#ifdef IREE_AVAILABLE
    if (!impl_->initialized) {
        throw std::runtime_error("IREE model not initialized");
    }

    EFVOutput result;
    result.forces.resize(shapes_.max_atoms * 3, 0.0f);
    std::memset(result.virial, 0, sizeof(result.virial));

    // Initialize call
    iree_runtime_call_t call;
    iree_status_t status = iree_runtime_call_initialize_by_name(
        impl_->session, iree_make_cstring_view("module.ace_efv"), &call);

    if (!iree_status_is_ok(status)) {
        std::string msg = format_status(status);
        iree_status_ignore(status);
        throw std::runtime_error("Failed to initialize call: " + msg);
    }

    // Pad inputs to compiled shapes
    std::fill(impl_->edge_rij_padded.begin(), impl_->edge_rij_padded.end(), 0.0f);
    std::fill(impl_->atomic_numbers_padded.begin(), impl_->atomic_numbers_padded.end(), 0);
    std::fill(impl_->edge_i_padded.begin(), impl_->edge_i_padded.end(), 0);
    std::fill(impl_->edge_j_padded.begin(), impl_->edge_j_padded.end(), 0);

    size_t copy_edges = std::min((size_t)n_edges, (size_t)shapes_.max_edges);
    size_t copy_atoms = std::min((size_t)n_atoms, (size_t)shapes_.max_atoms);

    std::memcpy(impl_->edge_rij_padded.data(), edge_rij, copy_edges * 3 * sizeof(float));
    std::memcpy(impl_->atomic_numbers_padded.data(), atomic_numbers, copy_atoms * sizeof(int64_t));
    std::memcpy(impl_->edge_i_padded.data(), edge_i, copy_edges * sizeof(int64_t));
    std::memcpy(impl_->edge_j_padded.data(), edge_j, copy_edges * sizeof(int64_t));

    // Create buffer views for inputs
    iree_hal_buffer_view_t* edge_rij_view = nullptr;
    iree_hal_buffer_view_t* atomic_numbers_view = nullptr;
    iree_hal_buffer_view_t* edge_i_view = nullptr;
    iree_hal_buffer_view_t* edge_j_view = nullptr;

    iree_hal_buffer_params_t buffer_params = {
        .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    };

    // edge_rij: [max_edges, 3]
    const iree_hal_dim_t edge_rij_shape[2] = {(iree_hal_dim_t)shapes_.max_edges, 3};
    status = iree_hal_buffer_view_allocate_buffer_copy(
        iree_runtime_session_device(impl_->session),
        iree_runtime_session_device_allocator(impl_->session),
        2, edge_rij_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
        iree_make_const_byte_span(impl_->edge_rij_padded.data(),
                                  shapes_.max_edges * 3 * sizeof(float)),
        &edge_rij_view);

    if (!iree_status_is_ok(status)) goto cleanup;

    // atomic_numbers: [max_atoms]
    {
        const iree_hal_dim_t atomic_numbers_shape[1] = {(iree_hal_dim_t)shapes_.max_atoms};
        status = iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(impl_->session),
            iree_runtime_session_device_allocator(impl_->session),
            1, atomic_numbers_shape, IREE_HAL_ELEMENT_TYPE_SINT_64,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
            iree_make_const_byte_span(impl_->atomic_numbers_padded.data(),
                                      shapes_.max_atoms * sizeof(int64_t)),
            &atomic_numbers_view);
    }

    if (!iree_status_is_ok(status)) goto cleanup;

    // edge_i: [max_edges]
    {
        const iree_hal_dim_t edge_i_shape[1] = {(iree_hal_dim_t)shapes_.max_edges};
        status = iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(impl_->session),
            iree_runtime_session_device_allocator(impl_->session),
            1, edge_i_shape, IREE_HAL_ELEMENT_TYPE_SINT_64,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
            iree_make_const_byte_span(impl_->edge_i_padded.data(),
                                      shapes_.max_edges * sizeof(int64_t)),
            &edge_i_view);
    }

    if (!iree_status_is_ok(status)) goto cleanup;

    // edge_j: [max_edges]
    {
        const iree_hal_dim_t edge_j_shape[1] = {(iree_hal_dim_t)shapes_.max_edges};
        status = iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(impl_->session),
            iree_runtime_session_device_allocator(impl_->session),
            1, edge_j_shape, IREE_HAL_ELEMENT_TYPE_SINT_64,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
            iree_make_const_byte_span(impl_->edge_j_padded.data(),
                                      shapes_.max_edges * sizeof(int64_t)),
            &edge_j_view);
    }

    if (!iree_status_is_ok(status)) goto cleanup;

    // Push inputs
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, edge_rij_view);
    if (!iree_status_is_ok(status)) goto cleanup;

    status = iree_runtime_call_inputs_push_back_buffer_view(&call, atomic_numbers_view);
    if (!iree_status_is_ok(status)) goto cleanup;

    status = iree_runtime_call_inputs_push_back_buffer_view(&call, edge_i_view);
    if (!iree_status_is_ok(status)) goto cleanup;

    status = iree_runtime_call_inputs_push_back_buffer_view(&call, edge_j_view);
    if (!iree_status_is_ok(status)) goto cleanup;

    // n_atoms and n_edges as scalar buffer views
    {
        const iree_hal_dim_t scalar_shape[1] = {1};
        iree_hal_buffer_view_t* n_atoms_view = nullptr;
        iree_hal_buffer_view_t* n_edges_view = nullptr;

        status = iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(impl_->session),
            iree_runtime_session_device_allocator(impl_->session),
            1, scalar_shape, IREE_HAL_ELEMENT_TYPE_SINT_32,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
            iree_make_const_byte_span(&n_atoms, sizeof(int32_t)),
            &n_atoms_view);
        if (!iree_status_is_ok(status)) goto cleanup;

        status = iree_runtime_call_inputs_push_back_buffer_view(&call, n_atoms_view);
        iree_hal_buffer_view_release(n_atoms_view);
        if (!iree_status_is_ok(status)) goto cleanup;

        status = iree_hal_buffer_view_allocate_buffer_copy(
            iree_runtime_session_device(impl_->session),
            iree_runtime_session_device_allocator(impl_->session),
            1, scalar_shape, IREE_HAL_ELEMENT_TYPE_SINT_32,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
            iree_make_const_byte_span(&n_edges, sizeof(int32_t)),
            &n_edges_view);
        if (!iree_status_is_ok(status)) goto cleanup;

        status = iree_runtime_call_inputs_push_back_buffer_view(&call, n_edges_view);
        iree_hal_buffer_view_release(n_edges_view);
        if (!iree_status_is_ok(status)) goto cleanup;
    }

    // Invoke
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    if (!iree_status_is_ok(status)) goto cleanup;

    // Get outputs
    {
        iree_hal_buffer_view_t* energy_view = nullptr;
        iree_hal_buffer_view_t* forces_view = nullptr;
        iree_hal_buffer_view_t* virial_view = nullptr;

        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
        if (!iree_status_is_ok(status)) goto cleanup;

        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &forces_view);
        if (!iree_status_is_ok(status)) {
            iree_hal_buffer_view_release(energy_view);
            goto cleanup;
        }

        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &virial_view);
        if (!iree_status_is_ok(status)) {
            iree_hal_buffer_view_release(energy_view);
            iree_hal_buffer_view_release(forces_view);
            goto cleanup;
        }

        // Transfer energy
        float energy_val = 0.0f;
        status = iree_hal_device_transfer_d2h(
            iree_runtime_session_device(impl_->session),
            iree_hal_buffer_view_buffer(energy_view), 0,
            &energy_val, sizeof(float),
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
            iree_infinite_timeout());
        result.energy = energy_val;

        // Transfer forces
        if (iree_status_is_ok(status)) {
            status = iree_hal_device_transfer_d2h(
                iree_runtime_session_device(impl_->session),
                iree_hal_buffer_view_buffer(forces_view), 0,
                result.forces.data(), shapes_.max_atoms * 3 * sizeof(float),
                IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                iree_infinite_timeout());
        }

        // Transfer virial
        if (iree_status_is_ok(status)) {
            status = iree_hal_device_transfer_d2h(
                iree_runtime_session_device(impl_->session),
                iree_hal_buffer_view_buffer(virial_view), 0,
                result.virial, 9 * sizeof(float),
                IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
                iree_infinite_timeout());
        }

        iree_hal_buffer_view_release(energy_view);
        iree_hal_buffer_view_release(forces_view);
        iree_hal_buffer_view_release(virial_view);
    }

cleanup:
    iree_hal_buffer_view_release(edge_rij_view);
    iree_hal_buffer_view_release(atomic_numbers_view);
    iree_hal_buffer_view_release(edge_i_view);
    iree_hal_buffer_view_release(edge_j_view);
    iree_runtime_call_deinitialize(&call);

    if (!iree_status_is_ok(status)) {
        std::string msg = format_status(status);
        iree_status_ignore(status);
        throw std::runtime_error("IREE call failed: " + msg);
    }

    return result;
#else
    (void)edge_rij; (void)atomic_numbers; (void)edge_i; (void)edge_j;
    (void)n_atoms; (void)n_edges;
    throw std::runtime_error("IREE not available");
#endif
}

void IREEModel::compute_efv_inplace(
    const float* edge_rij,
    const int64_t* atomic_numbers,
    const int64_t* edge_i,
    const int64_t* edge_j,
    int32_t n_atoms,
    int32_t n_edges,
    float* energy_out,
    float* forces_out,
    float* virial_out) {

    EFVOutput result = compute_efv(edge_rij, atomic_numbers, edge_i, edge_j, n_atoms, n_edges);

    *energy_out = result.energy;
    std::memcpy(forces_out, result.forces.data(), shapes_.max_atoms * 3 * sizeof(float));
    std::memcpy(virial_out, result.virial, 9 * sizeof(float));
}

bool iree_available() {
#ifdef IREE_AVAILABLE
    return true;
#else
    return false;
#endif
}

std::string iree_version() {
#ifdef IREE_AVAILABLE
    return "IREE runtime available";
#else
    return "IREE not available";
#endif
}

} // namespace ace_iree
