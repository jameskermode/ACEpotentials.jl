/* ----------------------------------------------------------------------
 * pair_iree_kokkos.cpp - Implementation of Kokkos-Enabled IREE Pair Style
 * ======================================================================
 *
 * Zero-copy GPU integration between LAMMPS/Kokkos and IREE-compiled ACE models.
 * ------------------------------------------------------------------------- */

#include "pair_iree_kokkos.h"
#include "atom_kokkos.h"
#include "neighbor_kokkos.h"
#include "kokkos.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"

#include <cstring>
#include <cmath>

// CUDA runtime for direct GPU memcpy
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>  // For CUDA Driver API (cuMemcpy)
// IREE CUDA buffer API for getting device pointer (for D2D copy)
#include "iree/hal/drivers/cuda/cuda_buffer.h"
#endif

// IREE CUDA driver registration (explicit call needed since static init may not work)
#ifdef IREE_HAL_HAVE_CUDA_DRIVER
extern "C" iree_status_t iree_hal_cuda_driver_module_register(
    iree_hal_driver_registry_t* registry);
#endif

namespace LAMMPS_NS {

// ===========================================================================
// IREEKokkosHandle Implementation
// ===========================================================================

IREEKokkosHandle::IREEKokkosHandle() {
    memset(&compute_pair_forces_fn_, 0, sizeof(compute_pair_forces_fn_));
    memset(&compute_energy_fn_, 0, sizeof(compute_energy_fn_));
}

IREEKokkosHandle::~IREEKokkosHandle() {
    // Release cached gradient view first
    if (cached_gradient_view_) {
        iree_hal_buffer_view_release(cached_gradient_view_);
        cached_gradient_view_ = nullptr;
    }
    if (session_) {
        iree_runtime_session_release(session_);
    }
    if (instance_) {
        iree_runtime_instance_release(instance_);
    }
}

bool IREEKokkosHandle::load_module(const char* vmfb_path, const char* device_type) {
    iree_status_t status;

#ifdef KOKKOS_ENABLE_CUDA
    // Capture Kokkos CUDA context before IREE creates its own
    // This enables potential context sharing for zero-copy buffer access
    CUresult cu_result = cuCtxGetCurrent(&kokkos_cuda_context_);
    if (cu_result == CUDA_SUCCESS && kokkos_cuda_context_ != nullptr) {
        fprintf(stderr, "IREE: Captured Kokkos CUDA context: %p\n", (void*)kokkos_cuda_context_);
    } else {
        fprintf(stderr, "IREE: No active CUDA context from Kokkos\n");
        kokkos_cuda_context_ = nullptr;
    }
#endif

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

    // Explicitly register CUDA driver (static init may not work with nvcc_wrapper)
#ifdef IREE_HAL_HAVE_CUDA_DRIVER
    status = iree_hal_cuda_driver_module_register(
        iree_runtime_instance_driver_registry(instance_));
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "Warning: Failed to register CUDA driver\n");
        iree_status_free(status);
        // Continue anyway - might work with other drivers
    }
#endif

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
        fprintf(stderr, "IREE: Failed to create device '%s'\n", device_type);
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }
    hal_device_ = device;

    // Detect if this is a CUDA device
    iree_string_view_t device_id = iree_hal_device_id(device);
    is_cuda_device_ = (device_id.size >= 4 &&
                       strncmp(device_id.data, "cuda", 4) == 0);

    // Debug: print device info
    fprintf(stderr, "IREE: Created device: %.*s (CUDA: %s)\n",
            (int)device_id.size, device_id.data,
            is_cuda_device_ ? "yes" : "no");

#ifdef KOKKOS_ENABLE_CUDA
    // Check if IREE is using the same CUDA context as Kokkos
    // If so, we can do true zero-copy buffer sharing
    if (is_cuda_device_ && kokkos_cuda_context_ != nullptr) {
        CUcontext iree_context;
        cu_result = cuCtxGetCurrent(&iree_context);
        if (cu_result == CUDA_SUCCESS) {
            shared_cuda_context_ = (iree_context == kokkos_cuda_context_);
            fprintf(stderr, "IREE: CUDA context sharing: %s (Kokkos: %p, IREE: %p)\n",
                    shared_cuda_context_ ? "YES" : "no",
                    (void*)kokkos_cuda_context_, (void*)iree_context);
        }
    }
#endif

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
    // Try multiple naming conventions:
    // 1. Python IREE-compiled VMFBs use "main"
    // 2. Reactant-exported VMFBs use "reactant_compute___.main"
    // 3. Standard IREE VMFBs use "module.compute_pair_forces" or "module.main"
    const char* function_names[] = {
        "main",  // Python IREE compiler uses this
        "reactant_compute___.main",
        "module.compute_pair_forces",
        "module.main",
        nullptr
    };

    for (int i = 0; function_names[i] != nullptr; i++) {
        status = iree_runtime_session_lookup_function(
            session_,
            iree_make_cstring_view(function_names[i]),
            &compute_pair_forces_fn_
        );
        if (iree_status_is_ok(status)) {
            functions_loaded_ = true;
            break;
        }
        iree_status_free(status);
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
    const std::vector<iree_hal_dim_t>& shape,
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

    // Release callback - null means IREE won't free the memory
    iree_hal_buffer_release_callback_t release_callback;
    memset(&release_callback, 0, sizeof(release_callback));

    // Import the external buffer (zero-copy)
    status = iree_hal_allocator_import_buffer(
        iree_hal_device_allocator(hal_device_),
        buffer_params,
        &external_buffer,
        release_callback,
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
    const std::vector<iree_hal_dim_t>& shape,
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

bool IREEKokkosHandle::invoke_energy_gradient_zerocopy(
    iree_hal_buffer_view_t* input,
    float* energy_out,
    void** gradient_ptr_out,
    size_t* gradient_size_out
) {
    if (!session_ || !functions_loaded_) return false;

    iree_status_t status;

    // Initialize call
    iree_runtime_call_t call;
    status = iree_runtime_call_initialize(session_, compute_pair_forces_fn_, &call);
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "IREE: Failed to initialize call\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Push input
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, input);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to push input\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Invoke
    status = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Invoke failed\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get first output (energy - scalar f32)
    iree_hal_buffer_view_t* energy_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to get energy output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Read energy value using device-to-host transfer (small scalar is OK)
    float energy_value = 0.0f;
    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(energy_view),
        0,
        &energy_value,
        sizeof(float),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );
    if (iree_status_is_ok(status)) {
        *energy_out = energy_value;
    } else {
        fprintf(stderr, "IREE: Failed to transfer energy from device\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        *energy_out = 0.0f;
    }
    iree_hal_buffer_view_release(energy_view);

    // Get second output (gradient - same shape as input)
    iree_hal_buffer_view_t* gradient_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &gradient_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to get gradient output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get device pointer to IREE's gradient buffer (zero-copy)
    iree_hal_buffer_t* src_buffer = iree_hal_buffer_view_buffer(gradient_view);
    iree_device_size_t iree_size = iree_hal_buffer_view_byte_length(gradient_view);

#ifdef KOKKOS_ENABLE_CUDA
    // Get the underlying allocated buffer (in case it's a sub-buffer)
    iree_hal_buffer_t* allocated_buffer = iree_hal_buffer_allocated_buffer(src_buffer);
    iree_device_size_t buffer_offset = iree_hal_buffer_byte_offset(src_buffer);

    // Get CUDA device pointer from IREE's buffer
    CUdeviceptr src_ptr_base = iree_hal_cuda_buffer_device_pointer(allocated_buffer);
    CUdeviceptr src_ptr = src_ptr_base + buffer_offset;

    *gradient_ptr_out = (void*)src_ptr;
    *gradient_size_out = (size_t)iree_size;
#else
    *gradient_ptr_out = nullptr;
    *gradient_size_out = (size_t)iree_size;
#endif

    // NOTE: We do NOT release gradient_view here because the caller needs the pointer to remain valid
    // The caller must call sync() to ensure the buffer remains valid during use
    // This is a potential memory leak if not managed properly - TODO: add explicit cleanup

    // Skip third output (input passthrough)
    iree_hal_buffer_view_t* passthrough_view = nullptr;
    iree_runtime_call_outputs_pop_front_buffer_view(&call, &passthrough_view);
    if (passthrough_view) iree_hal_buffer_view_release(passthrough_view);

    iree_runtime_call_deinitialize(&call);
    return true;
}

bool IREEKokkosHandle::invoke_energy_gradient(
    iree_hal_buffer_view_t* input_external,
    float* energy_out,
    void* gradient_dest_ptr,
    size_t gradient_size
) {
    if (!session_ || !functions_loaded_) return false;

    iree_status_t status;

    // Get input size from external buffer
    size_t input_size = iree_hal_buffer_view_byte_length(input_external);
    std::vector<iree_hal_dim_t> input_shape;
    size_t rank = iree_hal_buffer_view_shape_rank(input_external);
    for (size_t i = 0; i < rank; i++) {
        input_shape.push_back(iree_hal_buffer_view_shape_dim(input_external, i));
    }

    // Create IREE-owned input buffer (to avoid cross-context issues)
    iree_hal_buffer_view_t* input_iree = create_device_buffer(
        input_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32
    );
    if (!input_iree) {
        fprintf(stderr, "IREE: Failed to create input buffer\n");
        return false;
    }

    // First, transfer external input to host using cudaMemcpy (not IREE HAL)
    // This avoids cross-context issues since cudaMemcpy works across contexts
    std::vector<uint8_t> input_staging(input_size);

#ifdef KOKKOS_ENABLE_CUDA
    // Get the raw pointer from the external buffer (this is Kokkos's device memory)
    // Since import_gpu_buffer just wraps the pointer, we can get it back
    iree_hal_buffer_t* ext_buffer = iree_hal_buffer_view_buffer(input_external);
    iree_hal_buffer_t* ext_alloc_buffer = iree_hal_buffer_allocated_buffer(ext_buffer);
    iree_device_size_t ext_offset = iree_hal_buffer_byte_offset(ext_buffer);

    // For an imported external buffer, we need to get the original pointer
    // The handle was stored as: external_buffer.handle.device_allocation.ptr
    // But IREE doesn't provide a way to get it back directly...
    // Instead, let's get the Kokkos pointer from the caller
    // For now, we'll have to modify the interface to pass host data directly

    // As a workaround, use IREE's HAL transfer which might use a different code path
    // Actually, let's try IREE HAL D2H - it might work for imported buffers
#endif

    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(input_external),
        0,
        input_staging.data(),
        input_size,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "IREE: Failed to D2H transfer input from external buffer\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        iree_hal_buffer_view_release(input_iree);
        return false;
    }

    // H2D transfer to IREE's internal buffer
    status = iree_hal_device_transfer_h2d(
        hal_device_,
        input_staging.data(),
        iree_hal_buffer_view_buffer(input_iree),
        0,
        input_size,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "IREE: Failed to H2D transfer input\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        iree_hal_buffer_view_release(input_iree);
        return false;
    }

    // Initialize call
    iree_runtime_call_t call;
    status = iree_runtime_call_initialize(session_, compute_pair_forces_fn_, &call);
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "IREE: Failed to initialize call\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        iree_hal_buffer_view_release(input_iree);
        return false;
    }

    // Push IREE-owned input
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, input_iree);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        iree_hal_buffer_view_release(input_iree);
        fprintf(stderr, "IREE: Failed to push input\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Invoke
    status = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        iree_hal_buffer_view_release(input_iree);
        fprintf(stderr, "IREE: Invoke failed\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Release input buffer view now (no longer needed)
    iree_hal_buffer_view_release(input_iree);

    // Get first output (energy - scalar f32)
    iree_hal_buffer_view_t* energy_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to get energy output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Read energy value using IREE HAL D2H transfer
    float energy_value = 0.0f;
    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(energy_view),
        0,
        &energy_value,
        sizeof(float),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );
    if (iree_status_is_ok(status)) {
        *energy_out = energy_value;
    } else {
        fprintf(stderr, "IREE: Failed to transfer energy from device\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        *energy_out = 0.0f;
    }
    iree_hal_buffer_view_release(energy_view);

    // Get second output (gradient - same shape as input)
    iree_hal_buffer_view_t* gradient_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &gradient_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to get gradient output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get gradient size
    iree_device_size_t iree_size = iree_hal_buffer_view_byte_length(gradient_view);
    size_t copy_size = (iree_size < gradient_size) ? (size_t)iree_size : gradient_size;

    // Allocate host staging buffer
    std::vector<uint8_t> host_staging(copy_size);

    // Use IREE HAL to transfer gradient from device to host
    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(gradient_view),
        0,
        host_staging.data(),
        copy_size,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );

    iree_hal_buffer_view_release(gradient_view);

    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to transfer gradient from device\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

#ifdef KOKKOS_ENABLE_CUDA
    // Copy from host to Kokkos's device buffer
    cudaError_t err = cudaMemcpy(gradient_dest_ptr, host_staging.data(), copy_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA H2D copy to Kokkos failed: %s\n", cudaGetErrorString(err));
        iree_runtime_call_deinitialize(&call);
        return false;
    }
    cudaDeviceSynchronize();
#else
    // CPU: direct copy
    memcpy(gradient_dest_ptr, host_staging.data(), copy_size);
#endif

    // Skip third output (input passthrough)
    iree_hal_buffer_view_t* passthrough_view = nullptr;
    iree_runtime_call_outputs_pop_front_buffer_view(&call, &passthrough_view);
    if (passthrough_view) iree_hal_buffer_view_release(passthrough_view);

    iree_runtime_call_deinitialize(&call);
    return true;
}

bool IREEKokkosHandle::invoke_energy_gradient_direct(
    iree_hal_buffer_view_t* input_iree,
    float* energy_out,
    void* gradient_dest_ptr,
    size_t gradient_size
) {
    // This function expects input_iree to be an IREE-owned buffer (created via create_device_buffer)
    // It directly invokes the VMFB without trying to transfer from external memory

    if (!session_ || !functions_loaded_) return false;

    iree_status_t status;

    // Initialize call
    iree_runtime_call_t call;
    status = iree_runtime_call_initialize(session_, compute_pair_forces_fn_, &call);
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "IREE: Failed to initialize call\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Push IREE-owned input
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, input_iree);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to push input\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Invoke
    status = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Invoke failed\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get first output (energy - scalar f32)
    iree_hal_buffer_view_t* energy_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to get energy output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Read energy value
    float energy_value = 0.0f;
    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(energy_view),
        0,
        &energy_value,
        sizeof(float),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );
    if (iree_status_is_ok(status)) {
        *energy_out = energy_value;
    } else {
        fprintf(stderr, "IREE: Failed to transfer energy from device\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        *energy_out = 0.0f;
    }
    iree_hal_buffer_view_release(energy_view);

    // Get second output (gradient)
    iree_hal_buffer_view_t* gradient_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &gradient_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to get gradient output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get gradient size
    iree_device_size_t iree_size = iree_hal_buffer_view_byte_length(gradient_view);
    size_t copy_size = (iree_size < gradient_size) ? (size_t)iree_size : gradient_size;

    // Transfer gradient to host
    std::vector<uint8_t> host_staging(copy_size);
    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(gradient_view),
        0,
        host_staging.data(),
        copy_size,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );

    iree_hal_buffer_view_release(gradient_view);

    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE: Failed to transfer gradient from device\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

#ifdef KOKKOS_ENABLE_CUDA
    // Copy from host to Kokkos's device buffer
    cudaError_t err = cudaMemcpy(gradient_dest_ptr, host_staging.data(), copy_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA H2D copy to Kokkos failed: %s\n", cudaGetErrorString(err));
        iree_runtime_call_deinitialize(&call);
        return false;
    }
    cudaDeviceSynchronize();
#else
    memcpy(gradient_dest_ptr, host_staging.data(), copy_size);
#endif

    // Skip third output (input passthrough)
    iree_hal_buffer_view_t* passthrough_view = nullptr;
    iree_runtime_call_outputs_pop_front_buffer_view(&call, &passthrough_view);
    if (passthrough_view) iree_hal_buffer_view_release(passthrough_view);

    iree_runtime_call_deinitialize(&call);
    return true;
}

bool IREEKokkosHandle::invoke_energy_gradient_f64_zerocopy(
    iree_hal_buffer_view_t* input_iree,
    double* energy_out,
    void** gradient_ptr_out,
    size_t* gradient_size_out
) {
    // Float64 zero-copy gradient extraction for Kokkos integration
    // This method returns a device pointer to IREE's gradient buffer
    // enabling D2D copy without going through host memory

    if (!session_ || !functions_loaded_) return false;

    iree_status_t status;

    // Release any cached gradient view from previous invocation
    if (cached_gradient_view_) {
        iree_hal_buffer_view_release(cached_gradient_view_);
        cached_gradient_view_ = nullptr;
    }

    // Initialize call
    iree_runtime_call_t call;
    status = iree_runtime_call_initialize(session_, compute_pair_forces_fn_, &call);
    if (!iree_status_is_ok(status)) {
        fprintf(stderr, "IREE F64: Failed to initialize call\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Push IREE-owned input
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, input_iree);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE F64: Failed to push input\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Invoke
    status = iree_runtime_call_invoke(&call, 0);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE F64: Invoke failed\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get first output (energy - scalar f64)
    iree_hal_buffer_view_t* energy_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE F64: Failed to get energy output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Read energy value (small D2H transfer - 8 bytes is negligible)
    double energy_value = 0.0;
    status = iree_hal_device_transfer_d2h(
        hal_device_,
        iree_hal_buffer_view_buffer(energy_view),
        0,
        &energy_value,
        sizeof(double),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()
    );
    if (iree_status_is_ok(status)) {
        *energy_out = energy_value;
    } else {
        fprintf(stderr, "IREE F64: Failed to transfer energy from device\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        *energy_out = 0.0;
    }
    iree_hal_buffer_view_release(energy_view);

    // Get second output (gradient - same shape as input, Float64)
    iree_hal_buffer_view_t* gradient_view = nullptr;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &gradient_view);
    if (!iree_status_is_ok(status)) {
        iree_runtime_call_deinitialize(&call);
        fprintf(stderr, "IREE F64: Failed to get gradient output\n");
        iree_status_fprint(stderr, status);
        iree_status_free(status);
        return false;
    }

    // Get gradient size
    iree_device_size_t iree_size = iree_hal_buffer_view_byte_length(gradient_view);
    *gradient_size_out = (size_t)iree_size;

#ifdef KOKKOS_ENABLE_CUDA
    if (is_cuda_device_) {
        // Get the underlying allocated buffer (in case it's a sub-buffer)
        iree_hal_buffer_t* src_buffer = iree_hal_buffer_view_buffer(gradient_view);
        iree_hal_buffer_t* allocated_buffer = iree_hal_buffer_allocated_buffer(src_buffer);
        iree_device_size_t buffer_offset = iree_hal_buffer_byte_offset(src_buffer);

        // Get CUDA device pointer from IREE's buffer
        CUdeviceptr src_ptr_base = iree_hal_cuda_buffer_device_pointer(allocated_buffer);
        CUdeviceptr src_ptr = src_ptr_base + buffer_offset;

        *gradient_ptr_out = (void*)src_ptr;

        // Cache the gradient view to keep the buffer valid
        cached_gradient_view_ = gradient_view;
    } else
#endif
    {
        *gradient_ptr_out = nullptr;
        iree_hal_buffer_view_release(gradient_view);
    }

    // Skip third output (input passthrough)
    iree_hal_buffer_view_t* passthrough_view = nullptr;
    iree_runtime_call_outputs_pop_front_buffer_view(&call, &passthrough_view);
    if (passthrough_view) iree_hal_buffer_view_release(passthrough_view);

    iree_runtime_call_deinitialize(&call);
    return true;
}

bool IREEKokkosHandle::copy_gradient_d2d(
    void* iree_gradient_ptr,
    void* kokkos_gradient_ptr,
    size_t size_bytes
) {
#ifdef KOKKOS_ENABLE_CUDA
    if (!iree_gradient_ptr || !kokkos_gradient_ptr) return false;

    // D2D copy - stays entirely on GPU
    cudaError_t err = cudaMemcpy(
        kokkos_gradient_ptr,
        iree_gradient_ptr,
        size_bytes,
        cudaMemcpyDeviceToDevice
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "IREE: D2D gradient copy failed: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
#else
    // CPU path - direct memcpy
    if (!iree_gradient_ptr || !kokkos_gradient_ptr) return false;
    memcpy(kokkos_gradient_ptr, iree_gradient_ptr, size_bytes);
    return true;
#endif
}

void IREEKokkosHandle::sync() {
    if (hal_device_) {
        iree_hal_device_wait_semaphores(
            hal_device_,
            IREE_HAL_WAIT_MODE_ALL,
            iree_hal_semaphore_list_empty(),
            iree_infinite_timeout(),
            IREE_HAL_WAIT_FLAG_DEFAULT
        );
    }
}

// ===========================================================================
// PairIREEKokkos Implementation
// ===========================================================================

template<class DeviceType>
PairIREEKokkos<DeviceType>::PairIREEKokkos(LAMMPS* lmp) :
    Pair(lmp),
    cutoff_(5.5),
    max_atoms_(4096),
    max_pairs_(200000),
    vmfb_size_(2000),  // Default, will be updated from path
    npairs_(0),
    nlocal(0),
    nall(0)
{
    // Kokkos pair style flags
    kokkosable = 1;
    atomKK = (AtomKokkos*)atom;
    execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

    // Pair flags
    single_enable = 0;
    restartinfo = 0;
    one_coeff = 1;
    manybody_flag = 1;

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
        error->all(FLERR, "Illegal pair_style iree command");
    }

    vmfb_path_ = arg[0];

    // Parse bucket size from path (e.g., "bucket_10000/energy..." -> 10000)
    // Look for "bucket_XXXXX/" pattern in path (with trailing slash to get folder name)
    vmfb_size_ = 2000;  // Default
    std::string path_str(vmfb_path_);

    // Find the last "bucket_" followed by digits and then a "/"
    size_t search_pos = 0;
    while (true) {
        size_t pos = path_str.find("bucket_", search_pos);
        if (pos == std::string::npos) break;

        size_t start = pos + 7;  // Length of "bucket_"
        size_t end = start;
        while (end < path_str.size() && isdigit(path_str[end])) {
            end++;
        }

        // Only accept if we found digits followed by "/" (folder name pattern)
        if (end > start && end < path_str.size() && path_str[end] == '/') {
            vmfb_size_ = std::stoi(path_str.substr(start, end - start));
        }
        search_pos = end;
    }

    if (comm->me == 0) {
        fprintf(screen, "IREE: VMFB path = %s\n", vmfb_path_.c_str());
        fprintf(screen, "IREE: Bucket size = %d edges\n", vmfb_size_);
    }

    // Optional: device type (default to cuda for GPU, can specify local-task for CPU)
    const char* device_type = "cuda";
    if (narg > 1) {
        device_type = arg[1];
    }

    // Load IREE module
    if (!iree_->load_module(vmfb_path_.c_str(), device_type)) {
        error->all(FLERR, "Failed to load IREE module");
    }
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::coeff(int narg, char** arg) {
    if (narg < 3) {
        error->all(FLERR, "Incorrect args for pair coefficients");
    }

    // Parse element names (pair_coeff * * Si O)
    int ntypes = atom->ntypes;
    elements_.clear();
    type_map_.resize(ntypes + 1, -1);

    for (int i = 2; i < narg; i++) {
        elements_.push_back(arg[i]);
        // Map LAMMPS type i-1 to model element index i-2
        if (i - 2 < ntypes) {
            type_map_[i - 1] = i - 2;
        }
    }

    // Allocate cutsq array if needed
    if (!allocated) allocate();

    // Set cutoffs
    for (int i = 1; i <= ntypes; i++) {
        for (int j = 1; j <= ntypes; j++) {
            cutsq[i][j] = cutoff_ * cutoff_;
            setflag[i][j] = 1;
        }
    }
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::init_style() {
    // Get Kokkos neighbor flag from LAMMPS
    neighflag = lmp->kokkos->neighflag;

    // Request neighbor list - use half list for efficiency
    // We always apply Newton's 3rd law (add forces to both atoms)
    neighbor->add_request(this, NeighConst::REQ_DEFAULT);

    // Configure neighbor list request for Kokkos
    auto request = neighbor->find_request(this);
    request->set_kokkos_host(std::is_same<DeviceType, LMPHostType>::value &&
                             !std::is_same<DeviceType, LMPDeviceType>::value);
    request->set_kokkos_device(std::is_same<DeviceType, LMPDeviceType>::value);

    // Half neighbor list is the default - we handle Newton's 3rd law ourselves
    // by adding forces to both atoms in the pair
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::init_list(int /*id*/, NeighList *ptr) {
    list = ptr;
}

template<class DeviceType>
double PairIREEKokkos<DeviceType>::init_one(int i, int j) {
    return cutoff_;
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::allocate() {
    allocated = 1;
    int n = atom->ntypes;

    // Standard Pair arrays
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++)
            setflag[i][j] = 0;

    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
    eflag = eflag_in;
    vflag = vflag_in;

    ev_init(eflag, vflag);

    // Sync atom data to device
    atomKK->sync(execution_space, X_MASK | TYPE_MASK);

    // Get views from atomKK
    x = atomKK->k_x.view<DeviceType>();
    f = atomKK->k_f.view<DeviceType>();
    type = atomKK->k_type.view<DeviceType>();
    nlocal = atom->nlocal;
    nall = atom->nlocal + atom->nghost;
    newton_pair = force->newton_pair;

    // Build pair arrays on GPU
    build_pair_lists();

    if (npairs_ == 0) {
        atomKK->modified(execution_space, F_MASK);
        return;
    }

    // Check against bucket VMFB size (parsed from path in settings())
    const int vmfb_size = vmfb_size_;
    if (npairs_ > vmfb_size) {
        char msg[256];
        snprintf(msg, sizeof(msg),
            "Too many pairs (%d) for current VMFB bucket (%d edges). "
            "Use a larger bucket VMFB.", npairs_, vmfb_size);
        error->all(FLERR, msg);
    }

    // Ensure rij buffer is sized for VMFB (padded with zeros)
    if (d_rij_.extent(0) < (size_t)vmfb_size) {
        Kokkos::resize(d_rij_, vmfb_size, 3);
        Kokkos::resize(d_pair_forces_, vmfb_size, 3);
    }

    // Zero-pad the unused entries (Kokkos fence ensures ordering)
    if (npairs_ < vmfb_size) {
        auto l_rij = d_rij_;
        int n_pairs = npairs_;
        int vmfb_sz = vmfb_size;
        Kokkos::parallel_for("zero_pad_rij",
            Kokkos::RangePolicy<typename DeviceType::execution_space>(n_pairs, vmfb_sz),
            KOKKOS_LAMBDA(int e) {
                l_rij(e, 0) = 0.0;
                l_rij(e, 1) = 0.0;
                l_rij(e, 2) = 0.0;
            }
        );
        Kokkos::fence();
    }

    // Transpose rij from (N, 3) row-major to (3, N) column-major for IREE VMFB
    // VMFB expects layout: [x0, x1, ..., xN, y0, y1, ..., yN, z0, z1, ..., zN]
    // Kokkos stores as: [x0, y0, z0, x1, y1, z1, ...]
    // Using F_FLOAT (double) for Float64 VMFB - enables zero-copy with Kokkos
    Kokkos::View<F_FLOAT*, typename DeviceType::memory_space> d_rij_transposed("rij_transposed", vmfb_size * 3);

    {
        auto l_rij = d_rij_;
        auto l_rij_t = d_rij_transposed;
        int vs = vmfb_size;
        int np = npairs_;

        Kokkos::parallel_for("transpose_rij",
            Kokkos::RangePolicy<typename DeviceType::execution_space>(0, np),
            KOKKOS_LAMBDA(int e) {
                // From row-major (e, dim) to column-major (dim * N + e)
                l_rij_t(0 * vs + e) = l_rij(e, 0);  // x
                l_rij_t(1 * vs + e) = l_rij(e, 1);  // y
                l_rij_t(2 * vs + e) = l_rij(e, 2);  // z
            }
        );

        // Zero-pad the rest (for unused edges)
        if (np < vs) {
            Kokkos::parallel_for("zero_pad_rij_transposed",
                Kokkos::RangePolicy<typename DeviceType::execution_space>(np, vs),
                KOKKOS_LAMBDA(int e) {
                    l_rij_t(0 * vs + e) = 0.0;
                    l_rij_t(1 * vs + e) = 0.0;
                    l_rij_t(2 * vs + e) = 0.0;
                }
            );
        }
        Kokkos::fence();
    }

    // =========================================================================
    // Float64 Zero-Copy Path
    // =========================================================================
    // With Float64 VMFB, types match between Kokkos (F_FLOAT=double) and IREE.
    // This eliminates all type conversions and enables true zero-copy.
    //
    // Data flow (all on GPU, no PCIe transfers except 8-byte energy scalar):
    //   1. Kokkos d_rij_transposed (GPU, double) -> IREE input (via D2D or import)
    //   2. IREE computes gradient (GPU, double)
    //   3. IREE gradient (GPU, double) -> Kokkos scatter (D2D copy or direct use)
    //   4. Kokkos scatter to f[] (GPU)
    // =========================================================================

#ifdef KOKKOS_ENABLE_CUDA
    // Sync Kokkos before IREE operations
    Kokkos::fence();
#endif

    // Create IREE-owned input buffer (Float64)
    std::vector<iree_hal_dim_t> rij_shape = {3, (iree_hal_dim_t)vmfb_size};
    auto rij_iree = iree_->create_device_buffer(rij_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_64);
    if (!rij_iree) {
        error->all(FLERR, "Failed to create IREE input buffer (Float64)");
    }

    // Transfer Kokkos GPU data to IREE buffer via D2D copy
    // TODO: If contexts are shared, use import_gpu_buffer for true zero-copy
    iree_status_t status;
#ifdef KOKKOS_ENABLE_CUDA
    if (iree_->is_cuda()) {
        // D2D copy: Kokkos GPU -> IREE GPU (stays on device)
        iree_hal_buffer_t* iree_buffer = iree_hal_buffer_view_buffer(rij_iree);
        iree_hal_buffer_t* allocated_buffer = iree_hal_buffer_allocated_buffer(iree_buffer);

        // Get IREE's CUDA device pointer
        CUdeviceptr iree_ptr = iree_hal_cuda_buffer_device_pointer(allocated_buffer);

        // D2D memcpy from Kokkos to IREE
        cudaError_t err = cudaMemcpy(
            (void*)iree_ptr,
            d_rij_transposed.data(),
            vmfb_size * 3 * sizeof(F_FLOAT),
            cudaMemcpyDeviceToDevice
        );
        if (err != cudaSuccess) {
            iree_hal_buffer_view_release(rij_iree);
            fprintf(stderr, "CUDA D2D copy to IREE failed: %s\n", cudaGetErrorString(err));
            error->all(FLERR, "CUDA D2D transfer failed");
        }
        cudaDeviceSynchronize();
    } else
#endif
    {
        // CPU fallback: H2D transfer
        auto h_rij_t = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_rij_transposed);
        status = iree_hal_device_transfer_h2d(
            iree_->device(),
            h_rij_t.data(),
            iree_hal_buffer_view_buffer(rij_iree),
            0,
            vmfb_size * 3 * sizeof(F_FLOAT),
            IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
            iree_infinite_timeout()
        );
        if (!iree_status_is_ok(status)) {
            iree_hal_buffer_view_release(rij_iree);
            fprintf(stderr, "IREE: Failed to transfer input to IREE buffer\n");
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            error->all(FLERR, "IREE H2D transfer failed");
        }
    }

    // Ensure d_pair_forces_ is sized correctly
    if ((int)d_pair_forces_.extent(0) < vmfb_size ||
        (int)d_pair_forces_.extent(1) < 3) {
        Kokkos::resize(d_pair_forces_, vmfb_size, 3);
    }

    // Allocate flat gradient buffer for IREE output (matches VMFB layout)
    Kokkos::View<F_FLOAT*, typename DeviceType::memory_space> d_gradient_flat("gradient_flat", vmfb_size * 3);

    // Call IREE to compute energy and gradient (Float64)
    double energy = 0.0;
    bool success = false;

#ifdef KOKKOS_ENABLE_CUDA
    if (iree_->is_cuda()) {
        // Zero-copy gradient path: get device pointer directly from IREE
        void* iree_grad_ptr = nullptr;
        size_t grad_size = 0;

        success = iree_->invoke_energy_gradient_f64_zerocopy(
            rij_iree, &energy,
            &iree_grad_ptr, &grad_size
        );

        if (success && iree_grad_ptr) {
            // D2D copy from IREE gradient buffer to Kokkos buffer
            success = iree_->copy_gradient_d2d(
                iree_grad_ptr,
                d_gradient_flat.data(),
                vmfb_size * 3 * sizeof(F_FLOAT)
            );
        }
        cudaDeviceSynchronize();
    } else
#endif
    {
        // CPU fallback: use standard invoke with D2H + H2D
        float energy_f32 = 0.0f;
        success = iree_->invoke_energy_gradient_direct(
            rij_iree, &energy_f32,
            d_gradient_flat.data(), vmfb_size * 3 * sizeof(F_FLOAT)
        );
        energy = static_cast<double>(energy_f32);
    }

#ifdef KOKKOS_ENABLE_CUDA
    cudaDeviceSynchronize();
    Kokkos::fence();
#endif

    if (!success) {
        if (rij_iree) iree_hal_buffer_view_release(rij_iree);
        error->all(FLERR, "IREE energy+gradient computation failed");
    }

    // Report energy
    if (eflag_global) {
        eng_vdwl += energy;
    }

    // =========================================================================
    // Scatter Forces - Simplified Float64 Version
    // =========================================================================
    // With Float64 VMFB, gradient is already in F_FLOAT (double) format.
    // No type conversions needed - direct access to gradient values.
    // =========================================================================

    {
        auto l_f = f;
        auto l_d_pair_i = d_pair_i_;
        auto l_d_pair_j = d_pair_j_;
        auto l_grad = d_gradient_flat;  // F_FLOAT (double) - types match!
        int np = npairs_;
        const int vs = vmfb_size;

        Kokkos::parallel_for("scatter_forces_f64",
            Kokkos::RangePolicy<typename DeviceType::execution_space>(0, np),
            KOKKOS_LAMBDA(int e) {
                int i = l_d_pair_i(e);
                int j = l_d_pair_j(e);

                // Direct access to Float64 gradient - no type conversion!
                // IREE layout is (3, N): [x0..xN, y0..yN, z0..zN]
                F_FLOAT gx = l_grad(0 * vs + e);
                F_FLOAT gy = l_grad(1 * vs + e);
                F_FLOAT gz = l_grad(2 * vs + e);

                // F_i = +gradient (force on i due to pair ij)
                Kokkos::atomic_add(&l_f(i, 0), gx);
                Kokkos::atomic_add(&l_f(i, 1), gy);
                Kokkos::atomic_add(&l_f(i, 2), gz);

                // F_j = -gradient (Newton's 3rd law)
                // With half neighbor list, we always add to both atoms
                // j can be a ghost atom - LAMMPS will handle communication
                Kokkos::atomic_add(&l_f(j, 0), -gx);
                Kokkos::atomic_add(&l_f(j, 1), -gy);
                Kokkos::atomic_add(&l_f(j, 2), -gz);
            }
        );
        Kokkos::fence();
    }

    // Cleanup IREE buffer views
    if (rij_iree) iree_hal_buffer_view_release(rij_iree);

    // Ensure all GPU work is complete
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::fence();
    cudaDeviceSynchronize();
#endif

    // Mark forces as modified on device
    atomKK->modified(execution_space, F_MASK);
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::build_pair_lists() {
    // Get neighbor list views
    NeighListKokkos<DeviceType>* k_list =
        static_cast<NeighListKokkos<DeviceType>*>(list);

    auto d_numneigh = k_list->d_numneigh;
    auto d_neighbors = k_list->d_neighbors;
    auto d_ilist = k_list->d_ilist;
    int inum = k_list->inum;

    auto l_x = x;  // local copy for lambda capture
    double cutsq_val = cutoff_ * cutoff_;

    // Resize pair arrays if needed
    if (d_pair_i_.extent(0) < (size_t)max_pairs_) {
        Kokkos::resize(d_pair_i_, max_pairs_);
        Kokkos::resize(d_pair_j_, max_pairs_);
        Kokkos::resize(d_rij_, max_pairs_, 3);
        Kokkos::resize(d_pair_forces_, max_pairs_, 3);
    }

    // Local copies for lambda
    auto l_d_pair_i = d_pair_i_;
    auto l_d_pair_j = d_pair_j_;
    auto l_d_rij = d_rij_;

    // Use atomic counter for pair count
    Kokkos::View<int, typename DeviceType::memory_space> d_npairs("npairs");
    Kokkos::deep_copy(d_npairs, 0);

    int max_p = max_pairs_;

    // Build pair arrays
    Kokkos::parallel_for("build_pairs",
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, inum),
        KOKKOS_LAMBDA(int ii) {
            int i = d_ilist(ii);
            double xi = l_x(i, 0);
            double yi = l_x(i, 1);
            double zi = l_x(i, 2);

            int jnum = d_numneigh(i);
            for (int jj = 0; jj < jnum; jj++) {
                int j = d_neighbors(i, jj);
                j &= NEIGHMASK;

                double dx = l_x(j, 0) - xi;
                double dy = l_x(j, 1) - yi;
                double dz = l_x(j, 2) - zi;
                double rsq = dx*dx + dy*dy + dz*dz;

                if (rsq < cutsq_val) {
                    int idx = Kokkos::atomic_fetch_add(&d_npairs(), 1);
                    if (idx < max_p) {
                        l_d_pair_i(idx) = i;
                        l_d_pair_j(idx) = j;
                        l_d_rij(idx, 0) = dx;
                        l_d_rij(idx, 1) = dy;
                        l_d_rij(idx, 2) = dz;
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
        error->all(FLERR, "Too many pairs, increase max_pairs");
    }
}

template<class DeviceType>
void PairIREEKokkos<DeviceType>::scatter_forces() {
    // =========================================================================
    // Float64 Scatter Forces - Simplified Version
    // =========================================================================
    // With Float64 VMFB, types match between IREE and Kokkos (both use double).
    // No type conversions (reinterpret_cast, static_cast) are needed.
    //
    // The VMFB returns dE/drij (gradient of energy w.r.t. rij)
    // where rij = xj - xi (displacement from i to j)
    //
    // Forces are F = -dE/dx:
    //   F_i = -dE/dxi = -dE/drij * drij/dxi = -dE/drij * (-1) = +dE/drij = +gradient
    //   F_j = -dE/dxj = -dE/drij * drij/dxj = -dE/drij * (+1) = -dE/drij = -gradient
    //
    // IREE layout is (3, vmfb_size): [x0..xN, y0..yN, z0..zN]
    // Flat index: component d, edge e -> d * vmfb_size + e
    // =========================================================================

    auto l_f = f;  // local copy for lambda capture
    auto l_d_pair_i = d_pair_i_;
    auto l_d_pair_j = d_pair_j_;
    auto l_d_pair_forces = d_pair_forces_;
    int npairs = npairs_;

    // Use the bucket VMFB size (parsed from path in settings())
    const int vmfb_size = vmfb_size_;

    // Ensure CUDA operations are complete before Kokkos kernel
#ifdef KOKKOS_ENABLE_CUDA
    cudaDeviceSynchronize();
#endif
    Kokkos::fence();

    // Direct pointer access - Float64 types match!
    F_FLOAT* grad_ptr = l_d_pair_forces.data();

    int vs = vmfb_size;  // capture for lambda
    Kokkos::parallel_for("scatter_forces_f64",
        Kokkos::RangePolicy<typename DeviceType::execution_space>(0, npairs),
        KOKKOS_LAMBDA(int e) {
            int i = l_d_pair_i(e);
            int j = l_d_pair_j(e);

            // Direct Float64 access - no type conversion!
            F_FLOAT gx = grad_ptr[0 * vs + e];
            F_FLOAT gy = grad_ptr[1 * vs + e];
            F_FLOAT gz = grad_ptr[2 * vs + e];

            // F_i = +gradient (force on i due to pair ij)
            Kokkos::atomic_add(&l_f(i, 0), gx);
            Kokkos::atomic_add(&l_f(i, 1), gy);
            Kokkos::atomic_add(&l_f(i, 2), gz);

            // F_j = -gradient (Newton's 3rd law)
            // With half neighbor list, we always add to both atoms
            // j can be a ghost atom - LAMMPS will handle communication
            Kokkos::atomic_add(&l_f(j, 0), -gx);
            Kokkos::atomic_add(&l_f(j, 1), -gy);
            Kokkos::atomic_add(&l_f(j, 2), -gz);
        }
    );

    Kokkos::fence();
}

template<class DeviceType>
template<typename ViewType>
iree_hal_buffer_view_t* PairIREEKokkos<DeviceType>::import_view(const ViewType& view) {
    // Helper to import arbitrary Kokkos view into IREE
    std::vector<iree_hal_dim_t> shape;
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
// Explicit Instantiations - Use LAMMPS types
// ===========================================================================

template class PairIREEKokkos<LMPDeviceType>;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
// Only instantiate host type if different from device type
#ifndef KOKKOS_ENABLE_SERIAL
template class PairIREEKokkos<LMPHostType>;
#endif
#endif

}  // namespace LAMMPS_NS

// ===========================================================================
// Plugin Registration
// ===========================================================================

#include "lammpsplugin.h"
#include "version.h"

using namespace LAMMPS_NS;

static Pair *iree_kk_creator(LAMMPS *lmp)
{
    return new PairIREEKokkos<LMPDeviceType>(lmp);
}

static Pair *iree_kk_host_creator(LAMMPS *lmp)
{
    return new PairIREEKokkos<LMPHostType>(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
    lammpsplugin_t plugin;
    lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

    // Register iree/kk pair style (device version)
    plugin.version = LAMMPS_VERSION;
    plugin.style = "pair";
    plugin.name = "iree/kk";
    plugin.info = "IREE-compiled ACE pair style with Kokkos GPU support v1.0";
    plugin.author = "ACEpotentials.jl";
    plugin.creator.v1 = (lammpsplugin_factory1 *) &iree_kk_creator;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);

    // Also register iree/kk/device variant
    plugin.name = "iree/kk/device";
    (*register_plugin)(&plugin, lmp);

    // Register host variant
    plugin.name = "iree/kk/host";
    plugin.info = "IREE-compiled ACE pair style with Kokkos CPU support v1.0";
    plugin.creator.v1 = (lammpsplugin_factory1 *) &iree_kk_host_creator;
    (*register_plugin)(&plugin, lmp);
}
