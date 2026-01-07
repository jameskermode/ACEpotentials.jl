/**
 * ACE MLIAP IREE Plugin - Header
 * ==============================
 *
 * LAMMPS ML-IAP unified interface for ACE models compiled to IREE VMFB format.
 *
 * This plugin provides production-ready ACE evaluation with:
 * - No Julia/Python dependency at runtime
 * - Portable compiled model (.vmfb)
 * - CPU and GPU execution via IREE backends
 * - Complete force and virial computation
 *
 * The model is fully self-contained: all embeddings (radial basis, spherical
 * harmonics) are computed inside the compiled model, not in this plugin.
 *
 * Build Requirements:
 *   - LAMMPS with ML-IAP package
 *   - IREE runtime libraries
 *   - C++17 compiler
 *
 * Usage in LAMMPS:
 *   pair_style mliap unified ace_iree
 *   pair_coeff * * ace_model_cpu.vmfb ace_metadata.json Si O
 */

#ifndef MLIAP_ACE_IREE_H
#define MLIAP_ACE_IREE_H

#include "iree_wrapper.h"
#include <vector>
#include <memory>
#include <unordered_map>

// Forward declarations for LAMMPS types
namespace LAMMPS_NS {
    class LAMMPS;
    class MLIAPData;
}

namespace LAMMPS_NS {

/**
 * ACE model using IREE runtime for LAMMPS ML-IAP unified interface.
 *
 * This class wraps an ACE model compiled to IREE VMFB format and
 * implements the ML-IAP unified interface for energy and force computation.
 *
 * Key features:
 * - All embeddings are inside the compiled model (self-contained VMFB)
 * - Edge-based interface: pass edge vectors, get back energy/forces/virial
 * - Fixed-shape compilation with dynamic masking for varying atom counts
 */
class MLIAPModelACEIREE {
public:
    /**
     * Construct ACE IREE model.
     *
     * @param lmp         LAMMPS instance
     * @param vmfb_path   Path to compiled .vmfb model file
     * @param metadata_path Path to metadata .json file
     * @param device      IREE device ("local-task", "cuda", etc.)
     */
    MLIAPModelACEIREE(
        LAMMPS* lmp,
        const char* vmfb_path,
        const char* metadata_path = nullptr,
        const char* device = "local-task"
    );

    ~MLIAPModelACEIREE();

    // Prevent copying
    MLIAPModelACEIREE(const MLIAPModelACEIREE&) = delete;
    MLIAPModelACEIREE& operator=(const MLIAPModelACEIREE&) = delete;

    /**
     * Compute energy and forces.
     *
     * This is the main entry point called by LAMMPS. It:
     * 1. Builds edge list from LAMMPS neighbor lists
     * 2. Calls the IREE model (which computes embeddings internally)
     * 3. Accumulates forces from the model output
     *
     * @param data  ML-IAP data structure with neighbor lists and positions
     */
    void compute_forces(MLIAPData* data);

    /**
     * Compute descriptors (not used for unified interface).
     */
    void compute_descriptors(MLIAPData* data);

    /**
     * Get cutoff radius (from model metadata).
     */
    double get_rcutfac() const { return model_->rcut(); }

    /**
     * Get number of species.
     */
    int get_nspecies() const { return model_->n_species(); }

    /**
     * Get species atomic numbers.
     */
    const std::vector<int32_t>& get_species_Z() const { return model_->species_Z(); }

    /**
     * Check if model is ready.
     */
    bool is_valid() const { return model_ && model_->is_valid(); }

private:
    LAMMPS* lmp_;
    std::unique_ptr<ace_iree::IREEModel> model_;

    // Species mapping: LAMMPS type -> model species index
    std::vector<int32_t> type_to_species_;

    // Z -> species index mapping
    std::unordered_map<int32_t, int32_t> z_to_species_;

    // Working buffers (reused across calls)
    std::vector<float> edge_rij_;         // [max_edges, 3]
    std::vector<int64_t> atomic_numbers_; // [max_atoms]
    std::vector<int64_t> edge_i_;         // [max_edges]
    std::vector<int64_t> edge_j_;         // [max_edges]
    std::vector<float> forces_out_;       // [max_atoms, 3]
    float virial_out_[9];

    /**
     * Build edge list from LAMMPS neighbor lists.
     *
     * Converts LAMMPS per-atom neighbor lists to a flat edge list
     * suitable for the IREE model.
     *
     * @param data      ML-IAP data with neighbor lists
     * @param n_edges   Output: number of edges created
     */
    void build_edge_list(MLIAPData* data, int32_t& n_edges);

    /**
     * Accumulate forces from model output to LAMMPS atoms.
     *
     * @param data     ML-IAP data structure
     * @param n_atoms  Number of local atoms
     */
    void accumulate_forces(MLIAPData* data, int32_t n_atoms);

    /**
     * Accumulate virial contribution.
     *
     * @param data  ML-IAP data structure
     */
    void accumulate_virial(MLIAPData* data);

    /**
     * Map LAMMPS type to model species index.
     *
     * @param type  LAMMPS atom type (1-indexed)
     */
    int32_t type_to_species_idx(int type);
};

}  // namespace LAMMPS_NS

#endif  // MLIAP_ACE_IREE_H
