/**
 * IREE Runtime Wrapper Implementation
 * ====================================
 *
 * Implementation of C++ wrapper around IREE runtime C API.
 */

#include "iree_wrapper.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <iostream>

// Try to include IREE headers
#ifdef IREE_AVAILABLE
#include "iree/runtime/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
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
    iree_runtime_call_t call;
    bool call_initialized = false;
#endif

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
        if (call_initialized) {
            iree_runtime_call_deinitialize(&call);
            call_initialized = false;
        }
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
    }
};

/**
 * Parse JSON metadata file.
 * Simple parser - assumes well-formed JSON from our exporter.
 */
void IREEModel::load_metadata(const std::string& path) {
    if (path.empty()) {
        // Use defaults
        shapes_ = DEFAULT_SHAPES;
        metadata_.shapes = shapes_;
        metadata_.rcut = 6.0;
        metadata_.n_species = 1;
        metadata_.species_Z = {14};  // Silicon default
        metadata_.dtype = "Float32";
        return;
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open metadata file: " << path << std::endl;
        shapes_ = DEFAULT_SHAPES;
        metadata_.shapes = shapes_;
        metadata_.rcut = 6.0;
        metadata_.n_species = 1;
        metadata_.species_Z = {14};
        metadata_.dtype = "Float32";
        return;
    }

    // Read entire file
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();

    // Simple JSON parsing (for our specific format)
    auto find_int = [&json](const std::string& key) -> int32_t {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0;
        pos = json.find_first_of("-0123456789", pos);
        if (pos == std::string::npos) return 0;
        return std::stoi(json.substr(pos));
    };

    auto find_double = [&json](const std::string& key) -> double {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return 0.0;
        pos = json.find(':', pos);
        if (pos == std::string::npos) return 0.0;
        pos = json.find_first_of("-0123456789.", pos);
        if (pos == std::string::npos) return 0.0;
        return std::stod(json.substr(pos));
    };

    // Parse shapes
    shapes_.max_atoms = find_int("max_atoms");
    if (shapes_.max_atoms == 0) shapes_.max_atoms = DEFAULT_SHAPES.max_atoms;

    shapes_.max_neigs = find_int("max_neigs");
    if (shapes_.max_neigs == 0) shapes_.max_neigs = DEFAULT_SHAPES.max_neigs;

    shapes_.max_edges = find_int("max_edges");
    if (shapes_.max_edges == 0) shapes_.max_edges = DEFAULT_SHAPES.max_edges;

    metadata_.shapes = shapes_;

    // Parse model config
    metadata_.rcut = find_double("rcut");
    if (metadata_.rcut <= 0) metadata_.rcut = 6.0;

    metadata_.n_species = find_int("n_species");
    if (metadata_.n_species <= 0) metadata_.n_species = 1;

    // Parse species_Z array (simplified - just find first few integers after "species_Z")
    size_t pos = json.find("\"species_Z\"");
    if (pos != std::string::npos) {
        pos = json.find('[', pos);
        if (pos != std::string::npos) {
            size_t end = json.find(']', pos);
            std::string arr = json.substr(pos + 1, end - pos - 1);

            metadata_.species_Z.clear();
            size_t i = 0;
            while (i < arr.size()) {
                size_t num_start = arr.find_first_of("0123456789", i);
                if (num_start == std::string::npos) break;
                size_t num_end = arr.find_first_not_of("0123456789", num_start);
                if (num_end == std::string::npos) num_end = arr.size();
                metadata_.species_Z.push_back(std::stoi(arr.substr(num_start, num_end - num_start)));
                i = num_end;
            }
        }
    }

    if (metadata_.species_Z.empty()) {
        metadata_.species_Z = {14};  // Default to Si
    }
}

IREEModel::IREEModel(const std::string& vmfb_path,
                     const std::string& metadata_path,
                     const std::string& device)
    : impl_(std::make_unique<Impl>())
{
    // Load metadata first
    load_metadata(metadata_path);

    // Allocate padded buffers
    impl_->edge_rij_padded.resize(shapes_.max_edges * 3, 0.0f);
    impl_->atomic_numbers_padded.resize(shapes_.max_atoms, 0);
    impl_->edge_i_padded.resize(shapes_.max_edges, 0);
    impl_->edge_j_padded.resize(shapes_.max_edges, 0);
    impl_->forces_out.resize(shapes_.max_atoms * 3, 0.0f);

#ifdef IREE_AVAILABLE
    // Create IREE instance
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);

    iree_status_t status = iree_runtime_instance_create(
        &instance_options,
        iree_allocator_system(),
        &impl_->instance
    );
    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        throw std::runtime_error("Failed to create IREE instance");
    }

    // Create device
    iree_hal_driver_t* driver = nullptr;
    iree_string_view_t driver_name = iree_make_cstring_view(
        device == "cuda" ? "cuda" : "local-task"
    );

    status = iree_hal_driver_registry_try_create(
        iree_runtime_instance_driver_registry(impl_->instance),
        driver_name,
        iree_allocator_system(),
        &driver
    );
    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        throw std::runtime_error("Failed to create IREE driver");
    }

    status = iree_hal_driver_create_default_device(
        driver, iree_allocator_system(), &impl_->device
    );
    iree_hal_driver_release(driver);
    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        throw std::runtime_error("Failed to create IREE device");
    }

    // Create session
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    status = iree_runtime_session_create_with_device(
        impl_->instance,
        &session_options,
        impl_->device,
        iree_runtime_instance_host_allocator(impl_->instance),
        &impl_->session
    );
    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        throw std::runtime_error("Failed to create IREE session");
    }

    // Load VMFB module
    std::ifstream file(vmfb_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open VMFB file: " + vmfb_path);
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> vmfb_data(file_size);
    file.read(vmfb_data.data(), file_size);

    status = iree_runtime_session_append_bytecode_module_from_memory(
        impl_->session,
        iree_make_const_byte_span(vmfb_data.data(), vmfb_data.size()),
        iree_allocator_system()
    );
    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        throw std::runtime_error("Failed to load VMFB module");
    }

    // Initialize call for ace_efv function
    status = iree_runtime_call_initialize_by_name(
        impl_->session,
        iree_make_cstring_view("module.ace_efv"),
        &impl_->call
    );
    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        throw std::runtime_error("Failed to find ace_efv function in VMFB");
    }
    impl_->call_initialized = true;

#else
    (void)vmfb_path;
    (void)device;
    std::cerr << "Warning: IREE not available, model will return zeros" << std::endl;
#endif
}

IREEModel::~IREEModel() = default;

IREEModel::IREEModel(IREEModel&&) noexcept = default;
IREEModel& IREEModel::operator=(IREEModel&&) noexcept = default;

EFVOutput IREEModel::compute_efv(
    const float* edge_rij,
    const int64_t* atomic_numbers,
    const int64_t* edge_i,
    const int64_t* edge_j,
    int32_t n_atoms,
    int32_t n_edges
) {
    EFVOutput output;
    output.forces.resize(shapes_.max_atoms * 3);

    compute_efv_inplace(
        edge_rij, atomic_numbers, edge_i, edge_j,
        n_atoms, n_edges,
        &output.energy, output.forces.data(), output.virial
    );

    return output;
}

void IREEModel::compute_efv_inplace(
    const float* edge_rij,
    const int64_t* atomic_numbers,
    const int64_t* edge_i,
    const int64_t* edge_j,
    int32_t n_atoms,
    int32_t n_edges,
    float* energy,
    float* forces,
    float* virial
) {
#ifdef IREE_AVAILABLE
    if (!impl_ || !impl_->call_initialized) {
        *energy = 0.0f;
        std::memset(forces, 0, shapes_.max_atoms * 3 * sizeof(float));
        std::memset(virial, 0, 9 * sizeof(float));
        return;
    }

    // Pad inputs
    std::memset(impl_->edge_rij_padded.data(), 0, impl_->edge_rij_padded.size() * sizeof(float));
    std::memset(impl_->atomic_numbers_padded.data(), 0, impl_->atomic_numbers_padded.size() * sizeof(int64_t));
    std::memset(impl_->edge_i_padded.data(), 0, impl_->edge_i_padded.size() * sizeof(int64_t));
    std::memset(impl_->edge_j_padded.data(), 0, impl_->edge_j_padded.size() * sizeof(int64_t));

    std::memcpy(impl_->edge_rij_padded.data(), edge_rij, n_edges * 3 * sizeof(float));
    std::memcpy(impl_->atomic_numbers_padded.data(), atomic_numbers, n_atoms * sizeof(int64_t));
    std::memcpy(impl_->edge_i_padded.data(), edge_i, n_edges * sizeof(int64_t));
    std::memcpy(impl_->edge_j_padded.data(), edge_j, n_edges * sizeof(int64_t));

    // Reset call inputs/outputs
    iree_runtime_call_reset(&impl_->call);

    // Create buffer views for inputs
    iree_hal_buffer_view_t* edge_rij_view = nullptr;
    iree_hal_buffer_view_t* atomic_numbers_view = nullptr;
    iree_hal_buffer_view_t* edge_i_view = nullptr;
    iree_hal_buffer_view_t* edge_j_view = nullptr;

    iree_hal_dim_t edge_rij_shape[2] = {(iree_hal_dim_t)shapes_.max_edges, 3};
    iree_hal_dim_t atomic_numbers_shape[1] = {(iree_hal_dim_t)shapes_.max_atoms};
    iree_hal_dim_t edge_shape[1] = {(iree_hal_dim_t)shapes_.max_edges};

    iree_hal_device_allocator_t* allocator = iree_hal_device_allocator(impl_->device);

    // Create edge_rij buffer view
    iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
        impl_->device,
        allocator,
        2, edge_rij_shape,
        IREE_HAL_ELEMENT_TYPE_FLOAT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(impl_->edge_rij_padded.data(),
                                   impl_->edge_rij_padded.size() * sizeof(float)),
        &edge_rij_view
    );

    // Similar for other inputs...
    status = iree_hal_buffer_view_allocate_buffer_copy(
        impl_->device, allocator, 1, atomic_numbers_shape,
        IREE_HAL_ELEMENT_TYPE_INT_64, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(impl_->atomic_numbers_padded.data(),
                                   impl_->atomic_numbers_padded.size() * sizeof(int64_t)),
        &atomic_numbers_view
    );

    status = iree_hal_buffer_view_allocate_buffer_copy(
        impl_->device, allocator, 1, edge_shape,
        IREE_HAL_ELEMENT_TYPE_INT_64, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(impl_->edge_i_padded.data(),
                                   impl_->edge_i_padded.size() * sizeof(int64_t)),
        &edge_i_view
    );

    status = iree_hal_buffer_view_allocate_buffer_copy(
        impl_->device, allocator, 1, edge_shape,
        IREE_HAL_ELEMENT_TYPE_INT_64, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span(impl_->edge_j_padded.data(),
                                   impl_->edge_j_padded.size() * sizeof(int64_t)),
        &edge_j_view
    );

    // Push inputs
    iree_runtime_call_inputs_push_back_buffer_view(&impl_->call, edge_rij_view);
    iree_runtime_call_inputs_push_back_buffer_view(&impl_->call, atomic_numbers_view);
    iree_runtime_call_inputs_push_back_buffer_view(&impl_->call, edge_i_view);
    iree_runtime_call_inputs_push_back_buffer_view(&impl_->call, edge_j_view);

    // Push scalar inputs (n_atoms, n_edges)
    iree_vm_value_t n_atoms_val = iree_vm_value_make_i32(n_atoms);
    iree_vm_value_t n_edges_val = iree_vm_value_make_i32(n_edges);
    iree_runtime_call_inputs_push_back_value(&impl_->call, &n_atoms_val);
    iree_runtime_call_inputs_push_back_value(&impl_->call, &n_edges_val);

    // Invoke
    status = iree_runtime_call_invoke(&impl_->call, /*flags=*/0);

    // Release input buffer views
    iree_hal_buffer_view_release(edge_rij_view);
    iree_hal_buffer_view_release(atomic_numbers_view);
    iree_hal_buffer_view_release(edge_i_view);
    iree_hal_buffer_view_release(edge_j_view);

    if (!iree_status_is_ok(status)) {
        iree_status_free(status);
        *energy = 0.0f;
        std::memset(forces, 0, shapes_.max_atoms * 3 * sizeof(float));
        std::memset(virial, 0, 9 * sizeof(float));
        return;
    }

    // Read outputs
    iree_hal_buffer_view_t* energy_view = nullptr;
    iree_hal_buffer_view_t* forces_view = nullptr;
    iree_hal_buffer_view_t* virial_view = nullptr;

    iree_runtime_call_outputs_pop_front_buffer_view(&impl_->call, &energy_view);
    iree_runtime_call_outputs_pop_front_buffer_view(&impl_->call, &forces_view);
    iree_runtime_call_outputs_pop_front_buffer_view(&impl_->call, &virial_view);

    // Map and copy energy
    iree_hal_buffer_mapping_t energy_mapping;
    status = iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(energy_view),
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ,
        0, sizeof(float),
        &energy_mapping
    );
    if (iree_status_is_ok(status)) {
        *energy = *reinterpret_cast<const float*>(energy_mapping.contents.data);
        iree_hal_buffer_unmap_range(&energy_mapping);
    }

    // Map and copy forces
    iree_hal_buffer_mapping_t forces_mapping;
    status = iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(forces_view),
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ,
        0, shapes_.max_atoms * 3 * sizeof(float),
        &forces_mapping
    );
    if (iree_status_is_ok(status)) {
        std::memcpy(forces, forces_mapping.contents.data, shapes_.max_atoms * 3 * sizeof(float));
        iree_hal_buffer_unmap_range(&forces_mapping);
    }

    // Map and copy virial
    iree_hal_buffer_mapping_t virial_mapping;
    status = iree_hal_buffer_map_range(
        iree_hal_buffer_view_buffer(virial_view),
        IREE_HAL_MAPPING_MODE_SCOPED,
        IREE_HAL_MEMORY_ACCESS_READ,
        0, 9 * sizeof(float),
        &virial_mapping
    );
    if (iree_status_is_ok(status)) {
        std::memcpy(virial, virial_mapping.contents.data, 9 * sizeof(float));
        iree_hal_buffer_unmap_range(&virial_mapping);
    }

    // Release output buffer views
    iree_hal_buffer_view_release(energy_view);
    iree_hal_buffer_view_release(forces_view);
    iree_hal_buffer_view_release(virial_view);

#else
    (void)edge_rij;
    (void)atomic_numbers;
    (void)edge_i;
    (void)edge_j;
    (void)n_atoms;
    (void)n_edges;

    // Return zeros when IREE not available
    *energy = 0.0f;
    std::memset(forces, 0, shapes_.max_atoms * 3 * sizeof(float));
    std::memset(virial, 0, 9 * sizeof(float));
#endif
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
    return "IREE runtime not available";
#endif
}

}  // namespace ace_iree
