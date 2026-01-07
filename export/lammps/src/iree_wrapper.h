/**
 * IREE Runtime Wrapper for ACE Models
 * ====================================
 *
 * C++ wrapper around IREE runtime C API for loading and executing
 * ACE models compiled to VMFB format.
 *
 * The compiled model takes edge-based inputs:
 *   - edge_rij: [max_edges, 3] float32 - displacement vectors
 *   - atomic_numbers: [max_atoms] int64 - atomic numbers (0=padding)
 *   - edge_i: [max_edges] int64 - source atom indices
 *   - edge_j: [max_edges] int64 - target atom indices
 *   - n_atoms: int32 - actual number of atoms
 *   - n_edges: int32 - actual number of edges
 *
 * And returns:
 *   - energy: float32 scalar
 *   - forces: [max_atoms, 3] float32
 *   - virial: [3, 3] float32
 */

#ifndef ACE_IREE_WRAPPER_H
#define ACE_IREE_WRAPPER_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace ace_iree {

/**
 * Compiled model shapes (fixed at compilation time).
 */
struct ModelShapes {
    int32_t max_atoms;   // Maximum atoms (default: 4096)
    int32_t max_neigs;   // Maximum neighbors per atom (default: 50)
    int32_t max_edges;   // Maximum edges (default: 200000)
};

/**
 * Model metadata loaded from JSON.
 */
struct ModelMetadata {
    ModelShapes shapes;
    double rcut;
    int32_t n_species;
    std::vector<int32_t> species_Z;
    std::string dtype;
};

/**
 * Output structure for EFV computation.
 */
struct EFVOutput {
    float energy;
    std::vector<float> forces;    // [max_atoms * 3]
    float virial[9];              // [3, 3] row-major
};

/**
 * IREE Model wrapper class.
 *
 * Handles loading VMFB files and executing the compiled ACE model.
 */
class IREEModel {
public:
    /**
     * Create IREE model from VMFB file.
     *
     * @param vmfb_path Path to compiled .vmfb file
     * @param metadata_path Path to metadata .json file (optional)
     * @param device IREE device string ("local-task", "cuda", etc.)
     */
    IREEModel(const std::string& vmfb_path,
              const std::string& metadata_path = "",
              const std::string& device = "local-task");

    ~IREEModel();

    // Prevent copying
    IREEModel(const IREEModel&) = delete;
    IREEModel& operator=(const IREEModel&) = delete;

    // Allow moving
    IREEModel(IREEModel&&) noexcept;
    IREEModel& operator=(IREEModel&&) noexcept;

    /**
     * Check if model is loaded and ready.
     */
    bool is_valid() const { return impl_ != nullptr; }

    /**
     * Get model shapes.
     */
    const ModelShapes& shapes() const { return shapes_; }

    /**
     * Get model metadata.
     */
    const ModelMetadata& metadata() const { return metadata_; }

    /**
     * Get cutoff radius.
     */
    double rcut() const { return metadata_.rcut; }

    /**
     * Get number of species.
     */
    int32_t n_species() const { return metadata_.n_species; }

    /**
     * Get species atomic numbers.
     */
    const std::vector<int32_t>& species_Z() const { return metadata_.species_Z; }

    /**
     * Compute energy, forces, and virial.
     *
     * @param edge_rij Edge displacement vectors [n_edges, 3]
     * @param atomic_numbers Atomic numbers [n_atoms]
     * @param edge_i Source atom indices [n_edges]
     * @param edge_j Target atom indices [n_edges]
     * @param n_atoms Actual number of atoms
     * @param n_edges Actual number of edges
     *
     * @return EFVOutput with energy, forces, and virial
     */
    EFVOutput compute_efv(
        const float* edge_rij,
        const int64_t* atomic_numbers,
        const int64_t* edge_i,
        const int64_t* edge_j,
        int32_t n_atoms,
        int32_t n_edges
    );

    /**
     * Compute energy, forces, and virial with pre-allocated output buffers.
     *
     * @param edge_rij Edge displacement vectors [n_edges, 3]
     * @param atomic_numbers Atomic numbers [n_atoms]
     * @param edge_i Source atom indices [n_edges]
     * @param edge_j Target atom indices [n_edges]
     * @param n_atoms Actual number of atoms
     * @param n_edges Actual number of edges
     * @param energy Output energy (scalar)
     * @param forces Output forces [max_atoms * 3]
     * @param virial Output virial [9]
     */
    void compute_efv_inplace(
        const float* edge_rij,
        const int64_t* atomic_numbers,
        const int64_t* edge_i,
        const int64_t* edge_j,
        int32_t n_atoms,
        int32_t n_edges,
        float* energy,
        float* forces,
        float* virial
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;

    ModelShapes shapes_;
    ModelMetadata metadata_;

    void load_metadata(const std::string& path);
};

/**
 * Check if IREE runtime is available.
 */
bool iree_available();

/**
 * Get IREE runtime version string.
 */
std::string iree_version();

}  // namespace ace_iree

#endif  // ACE_IREE_WRAPPER_H
