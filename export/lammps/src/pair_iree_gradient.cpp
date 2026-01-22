/**
 * pair_iree_gradient.cpp - Simple IREE Pair Style for Reactant-exported gradient VMFBs
 * =====================================================================================
 *
 * This pair style works with Reactant-exported VMFBs that compute gradients.
 * Input: rij (3, n_edges) - edge displacement vectors
 * Output: gradient (3, n_edges) - energy gradients for force computation
 *
 * Usage:
 *   pair_style iree/gradient gradient.vmfb [cutoff]
 *   pair_coeff * * Si
 */

#include "pair.h"
#include "atom.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "update.h"

#include <cstring>
#include <cmath>
#include <vector>
#include <string>

// IREE runtime headers
#include "iree/runtime/api.h"

namespace LAMMPS_NS {

class PairIREEGradient : public Pair {
public:
    PairIREEGradient(LAMMPS* lmp) :
        Pair(lmp),
        cutoff_(6.0),
        vmfb_size_(2000),  // Default VMFB expects 2000 edges
        session_(nullptr),
        instance_(nullptr),
        device_(nullptr)
    {
        single_enable = 0;
        restartinfo = 0;
        one_coeff = 1;
        manybody_flag = 1;
        no_virial_fdotr_compute = 1;
        centroidstressflag = CENTROID_NOTAVAIL;
    }

    ~PairIREEGradient() override {
        cleanup_iree();
        if (allocated) {
            memory->destroy(setflag);
            memory->destroy(cutsq);
        }
    }

    void compute(int eflag, int vflag) override {
        ev_init(eflag, vflag);

        double **x = atom->x;
        double **f = atom->f;
        int *type = atom->type;
        int nlocal = atom->nlocal;
        int newton_pair = force->newton_pair;

        double cutsq_val = cutoff_ * cutoff_;

        // Build rij, pair_i, pair_j arrays from neighbor list
        build_pair_data(x, cutsq_val);

        int n_pairs = (int)pair_i_.size();
        if (n_pairs == 0) return;

        if (n_pairs > vmfb_size_) {
            error->all(FLERR, "Too many pairs ({}) exceeds VMFB size ({})", n_pairs, vmfb_size_);
        }

        // Prepare padded input: (3, vmfb_size_)
        std::vector<float> rij_padded(3 * vmfb_size_, 0.0f);
        for (int e = 0; e < n_pairs; e++) {
            rij_padded[0 * vmfb_size_ + e] = rij_[e * 3 + 0];  // x
            rij_padded[1 * vmfb_size_ + e] = rij_[e * 3 + 1];  // y
            rij_padded[2 * vmfb_size_ + e] = rij_[e * 3 + 2];  // z
        }

        // Call IREE - returns both energy and gradient
        float energy = 0.0f;
        std::vector<float> gradient(3 * vmfb_size_, 0.0f);
        if (!call_iree(rij_padded.data(), &energy, gradient.data())) {
            error->all(FLERR, "IREE computation failed");
        }

        // Add energy contribution
        if (eflag_global) {
            eng_vdwl += (double)energy;
        }

        // Scatter gradients to forces
        // Forces = -gradient
        // f[i] -= gradient (force on center atom)
        // f[j] += gradient (Newton's 3rd law)
        for (int e = 0; e < n_pairs; e++) {
            int i = pair_i_[e];
            int j = pair_j_[e];

            double fx = gradient[0 * vmfb_size_ + e];
            double fy = gradient[1 * vmfb_size_ + e];
            double fz = gradient[2 * vmfb_size_ + e];

            f[i][0] -= fx;
            f[i][1] -= fy;
            f[i][2] -= fz;

            if (newton_pair || j < nlocal) {
                f[j][0] += fx;
                f[j][1] += fy;
                f[j][2] += fz;
            }
        }
    }

    void settings(int narg, char** arg) override {
        if (narg < 1) {
            error->all(FLERR, "Illegal pair_style iree/gradient command");
        }

        vmfb_path_ = arg[0];

        // Optional: cutoff
        if (narg > 1) {
            cutoff_ = utils::numeric(FLERR, arg[1], false, lmp);
        }

        // Optional: VMFB size
        if (narg > 2) {
            vmfb_size_ = utils::inumeric(FLERR, arg[2], false, lmp);
        }

        // Initialize IREE
        if (!init_iree()) {
            error->all(FLERR, "Failed to initialize IREE with VMFB: {}", vmfb_path_);
        }
    }

    void coeff(int narg, char** arg) override {
        if (narg < 3) {
            error->all(FLERR, "Incorrect args for pair coefficients");
        }

        if (!allocated) allocate();

        int ilo, ihi, jlo, jhi;
        utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
        utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

        int count = 0;
        for (int i = ilo; i <= ihi; i++) {
            for (int j = std::max(jlo, i); j <= jhi; j++) {
                setflag[i][j] = 1;
                count++;
            }
        }

        if (count == 0) {
            error->all(FLERR, "Incorrect args for pair coefficients");
        }
    }

    void init_style() override {
        neighbor->add_request(this, NeighConst::REQ_DEFAULT);
    }

    double init_one(int /*i*/, int /*j*/) override {
        return cutoff_;
    }

protected:
    std::string vmfb_path_;
    double cutoff_;
    int vmfb_size_;

    // IREE state
    iree_runtime_session_t* session_;
    iree_runtime_instance_t* instance_;
    iree_hal_device_t* device_;
    iree_vm_function_t function_;

    // Pair data buffers
    std::vector<int> pair_i_;
    std::vector<int> pair_j_;
    std::vector<float> rij_;

    void allocate() {
        allocated = 1;
        int n = atom->ntypes;
        memory->create(setflag, n + 1, n + 1, "pair:setflag");
        memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

        for (int i = 1; i <= n; i++) {
            for (int j = i; j <= n; j++) {
                setflag[i][j] = 0;
            }
        }
    }

    bool init_iree() {
        cleanup_iree();

        iree_status_t status;

        // Create instance with all available drivers
        iree_runtime_instance_options_t instance_options;
        iree_runtime_instance_options_initialize(&instance_options);
        iree_runtime_instance_options_use_all_available_drivers(&instance_options);

        status = iree_runtime_instance_create(&instance_options,
            iree_allocator_system(), &instance_);
        if (!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Create default device for "local-task" (CPU)
        status = iree_runtime_instance_try_create_default_device(
            instance_,
            iree_make_cstring_view("local-task"),
            &device_);
        if (!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Create session with device
        iree_runtime_session_options_t session_options;
        iree_runtime_session_options_initialize(&session_options);
        status = iree_runtime_session_create_with_device(
            instance_, &session_options, device_,
            iree_runtime_instance_host_allocator(instance_), &session_);
        if (!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Load VMFB
        status = iree_runtime_session_append_bytecode_module_from_file(
            session_, vmfb_path_.c_str());
        if (!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Look up function - try multiple naming conventions
        const char* function_names[] = {
            "reactant_compute___.main",  // Reactant-exported
            "main",                       // Simple name
            "module.main",               // Module-prefixed
            nullptr
        };

        bool found = false;
        for (int i = 0; function_names[i] != nullptr; i++) {
            status = iree_runtime_session_lookup_function(
                session_,
                iree_make_cstring_view(function_names[i]),
                &function_);
            if (iree_status_is_ok(status)) {
                found = true;
                if (comm->me == 0) {
                    utils::logmesg(lmp, "IREE: Using function '{}'\n", function_names[i]);
                }
                break;
            }
            iree_status_free(status);
        }

        if (!found) {
            error->warning(FLERR, "IREE: Could not find expected function in VMFB");
            return false;
        }

        return true;
    }

    void cleanup_iree() {
        if (session_) {
            iree_runtime_session_release(session_);
            session_ = nullptr;
        }
        if (device_) {
            iree_hal_device_release(device_);
            device_ = nullptr;
        }
        if (instance_) {
            iree_runtime_instance_release(instance_);
            instance_ = nullptr;
        }
    }

    bool call_iree(const float* rij_input, float* energy_output, float* gradient_output) {
        iree_status_t status;

        // Create input buffer view: (3, vmfb_size_)
        iree_hal_buffer_view_t* input_view = nullptr;
        const iree_hal_dim_t input_shape[2] = {3, (iree_hal_dim_t)vmfb_size_};

        iree_hal_buffer_params_t buffer_params = {
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        };

        status = iree_hal_buffer_view_allocate_buffer_copy(
            device_, iree_hal_device_allocator(device_),
            2, input_shape, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
            IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, buffer_params,
            iree_make_const_byte_span(rij_input, 3 * vmfb_size_ * sizeof(float)),
            &input_view);

        if (!iree_status_is_ok(status)) {
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Initialize call
        iree_runtime_call_t call;
        status = iree_runtime_call_initialize(session_, function_, &call);
        if (!iree_status_is_ok(status)) {
            iree_hal_buffer_view_release(input_view);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Push input
        status = iree_runtime_call_inputs_push_back_buffer_view(&call, input_view);
        iree_hal_buffer_view_release(input_view);  // call holds reference

        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Invoke
        status = iree_runtime_call_invoke(&call, 0);
        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Get first output (energy - shape (1,))
        iree_hal_buffer_view_t* energy_view = nullptr;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &energy_view);
        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Read energy
        iree_hal_buffer_mapping_t energy_mapping;
        status = iree_hal_buffer_map_range(
            iree_hal_buffer_view_buffer(energy_view),
            IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
            0, sizeof(float), &energy_mapping);

        if (iree_status_is_ok(status)) {
            std::memcpy(energy_output, energy_mapping.contents.data, sizeof(float));
            iree_hal_buffer_unmap_range(&energy_mapping);
        }
        iree_hal_buffer_view_release(energy_view);

        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            return false;
        }

        // Get second output (gradient - shape (3, vmfb_size_))
        iree_hal_buffer_view_t* gradient_view = nullptr;
        status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &gradient_view);
        if (!iree_status_is_ok(status)) {
            iree_runtime_call_deinitialize(&call);
            iree_status_fprint(stderr, status);
            iree_status_free(status);
            return false;
        }

        // Read gradient
        iree_hal_buffer_mapping_t gradient_mapping;
        status = iree_hal_buffer_map_range(
            iree_hal_buffer_view_buffer(gradient_view),
            IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
            0, 3 * vmfb_size_ * sizeof(float), &gradient_mapping);

        if (iree_status_is_ok(status)) {
            std::memcpy(gradient_output, gradient_mapping.contents.data, 3 * vmfb_size_ * sizeof(float));
            iree_hal_buffer_unmap_range(&gradient_mapping);
        }

        iree_hal_buffer_view_release(gradient_view);
        iree_runtime_call_deinitialize(&call);

        return iree_status_is_ok(status);
    }

    void build_pair_data(double **x, double cutsq_val) {
        int *ilist = list->ilist;
        int *numneigh = list->numneigh;
        int **firstneigh = list->firstneigh;
        int inum = list->inum;

        pair_i_.clear();
        pair_j_.clear();
        rij_.clear();

        for (int ii = 0; ii < inum; ii++) {
            int i = ilist[ii];
            double xi = x[i][0];
            double yi = x[i][1];
            double zi = x[i][2];

            int *jlist = firstneigh[i];
            int jnum = numneigh[i];

            for (int jj = 0; jj < jnum; jj++) {
                int j = jlist[jj];
                j &= NEIGHMASK;

                double dx = x[j][0] - xi;
                double dy = x[j][1] - yi;
                double dz = x[j][2] - zi;
                double rsq = dx*dx + dy*dy + dz*dz;

                if (rsq < cutsq_val) {
                    pair_i_.push_back(i);
                    pair_j_.push_back(j);
                    rij_.push_back(static_cast<float>(dx));
                    rij_.push_back(static_cast<float>(dy));
                    rij_.push_back(static_cast<float>(dz));
                }
            }
        }
    }
};

}  // namespace LAMMPS_NS

// =============================================================================
// Plugin Registration
// =============================================================================
#include "version.h"
#include "lammpsplugin.h"

using namespace LAMMPS_NS;

static Pair *iree_gradient_creator(LAMMPS *lmp)
{
    return new PairIREEGradient(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
    lammpsplugin_t plugin;
    lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

    plugin.version = LAMMPS_VERSION;
    plugin.style = "pair";
    plugin.name = "iree/gradient";
    plugin.info = "IREE Reactant gradient VMFB pair style v1.0";
    plugin.author = "ACEpotentials.jl";
    plugin.creator.v1 = (lammpsplugin_factory1 *) &iree_gradient_creator;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);
}
