/**
 * ACE MLIAP IREE Plugin - Implementation
 * =======================================
 *
 * Implementation of LAMMPS ML-IAP unified interface using IREE runtime.
 *
 * This implementation assumes:
 * - All embeddings are computed inside the IREE model
 * - The model takes edge_rij, atomic_numbers, edge_i, edge_j as inputs
 * - The model returns energy, forces, and virial
 */

#include "mliap_ace_iree.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// LAMMPS headers - adapt these paths as needed
// #include "lammps.h"
// #include "mliap_data.h"
// #include "atom.h"
// #include "neighbor.h"
// #include "error.h"

namespace LAMMPS_NS {

/**
 * Placeholder MLIAPData structure.
 * In real LAMMPS, this comes from the ML-IAP package.
 */
#ifndef MLIAP_DATA_DEFINED
struct MLIAPData {
    int nlistatoms;           // Number of local atoms
    int ntotal;               // Total atoms including ghosts
    double** x;               // Positions [ntotal][3]
    double** f;               // Forces [ntotal][3]
    int* type;                // Atom types [ntotal]
    int* numneighs;           // Neighbor counts [nlistatoms]
    int** firstneigh;         // Neighbor lists [nlistatoms][numneighs[i]]
    double energy;            // Total potential energy
    double virial[6];         // Virial tensor (Voigt: xx, yy, zz, yz, xz, xy)
};
#endif

MLIAPModelACEIREE::MLIAPModelACEIREE(
    LAMMPS* lmp,
    const char* vmfb_path,
    const char* metadata_path,
    const char* device
) : lmp_(lmp)
{
    // Load IREE model
    std::string meta_path = metadata_path ? metadata_path : "";
    model_ = std::make_unique<ace_iree::IREEModel>(vmfb_path, meta_path, device);

    if (!model_->is_valid()) {
        throw std::runtime_error("Failed to load ACE model from " + std::string(vmfb_path));
    }

    // Build Z -> species index mapping
    const auto& species_Z = model_->species_Z();
    for (size_t i = 0; i < species_Z.size(); ++i) {
        z_to_species_[species_Z[i]] = static_cast<int32_t>(i);
    }

    // Allocate working buffers
    const auto& shapes = model_->shapes();
    edge_rij_.resize(shapes.max_edges * 3);
    atomic_numbers_.resize(shapes.max_atoms);
    edge_i_.resize(shapes.max_edges);
    edge_j_.resize(shapes.max_edges);
    forces_out_.resize(shapes.max_atoms * 3);
}

MLIAPModelACEIREE::~MLIAPModelACEIREE() = default;

void MLIAPModelACEIREE::build_edge_list(MLIAPData* data, int32_t& n_edges) {
    const auto& shapes = model_->shapes();
    const double rcut = model_->rcut();
    const double rcut_sq = rcut * rcut;

    // Clear buffers
    std::memset(edge_rij_.data(), 0, edge_rij_.size() * sizeof(float));
    std::memset(atomic_numbers_.data(), 0, atomic_numbers_.size() * sizeof(int64_t));
    std::memset(edge_i_.data(), 0, edge_i_.size() * sizeof(int64_t));
    std::memset(edge_j_.data(), 0, edge_j_.size() * sizeof(int64_t));

    int32_t n_atoms = data->nlistatoms;
    n_edges = 0;

    // Check atom count
    if (n_atoms > shapes.max_atoms) {
        throw std::runtime_error(
            "Too many atoms: " + std::to_string(n_atoms) +
            " > max " + std::to_string(shapes.max_atoms)
        );
    }

    // Fill atomic numbers
    for (int i = 0; i < n_atoms; ++i) {
        // Note: LAMMPS type is 1-indexed, need to get actual Z
        // For now assume type == species index, actual Z stored in model
        int type = data->type[i];
        int32_t species_idx = type_to_species_idx(type);
        if (species_idx >= 0 && species_idx < static_cast<int32_t>(model_->species_Z().size())) {
            atomic_numbers_[i] = model_->species_Z()[species_idx];
        } else {
            atomic_numbers_[i] = 0;  // Unknown species
        }
    }

    // Build edge list from neighbor lists
    for (int i = 0; i < n_atoms; ++i) {
        double* xi = data->x[i];
        int numneigh = data->numneighs[i];
        int* neighs = data->firstneigh[i];

        for (int jj = 0; jj < numneigh; ++jj) {
            int j = neighs[jj];
            // LAMMPS uses high bits for special bonds
            j &= 0x3FFFFFFF;

            double* xj = data->x[j];

            // Compute displacement
            double dx = xj[0] - xi[0];
            double dy = xj[1] - xi[1];
            double dz = xj[2] - xi[2];
            double rsq = dx*dx + dy*dy + dz*dz;

            if (rsq < rcut_sq && rsq > 1e-10) {
                if (n_edges >= shapes.max_edges) {
                    throw std::runtime_error(
                        "Too many edges: " + std::to_string(n_edges) +
                        " >= max " + std::to_string(shapes.max_edges)
                    );
                }

                edge_i_[n_edges] = i;
                edge_j_[n_edges] = j < n_atoms ? j : j;  // Ghost atoms mapped directly
                edge_rij_[n_edges * 3 + 0] = static_cast<float>(dx);
                edge_rij_[n_edges * 3 + 1] = static_cast<float>(dy);
                edge_rij_[n_edges * 3 + 2] = static_cast<float>(dz);

                ++n_edges;
            }
        }
    }
}

int32_t MLIAPModelACEIREE::type_to_species_idx(int type) {
    // LAMMPS types are 1-indexed
    // For now, assume simple mapping: type 1 -> species 0, etc.
    // In a full implementation, this would be configured based on
    // the pair_coeff element list matching model's species_Z
    return type - 1;
}

void MLIAPModelACEIREE::accumulate_forces(MLIAPData* data, int32_t n_atoms) {
    // Copy forces from model output to LAMMPS force array
    for (int32_t i = 0; i < n_atoms; ++i) {
        data->f[i][0] += forces_out_[i * 3 + 0];
        data->f[i][1] += forces_out_[i * 3 + 1];
        data->f[i][2] += forces_out_[i * 3 + 2];
    }
}

void MLIAPModelACEIREE::accumulate_virial(MLIAPData* data) {
    // Model virial is [3, 3] row-major: [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]
    // LAMMPS virial is Voigt: [xx, yy, zz, yz, xz, xy]
    data->virial[0] += virial_out_[0];  // xx
    data->virial[1] += virial_out_[4];  // yy
    data->virial[2] += virial_out_[8];  // zz
    data->virial[3] += virial_out_[5];  // yz
    data->virial[4] += virial_out_[2];  // xz
    data->virial[5] += virial_out_[1];  // xy
}

void MLIAPModelACEIREE::compute_forces(MLIAPData* data) {
    if (!model_ || !model_->is_valid()) {
        std::cerr << "ACE IREE model not valid" << std::endl;
        data->energy = 0.0;
        return;
    }

    int32_t n_atoms = data->nlistatoms;
    if (n_atoms == 0) {
        data->energy = 0.0;
        return;
    }

    // Build edge list from neighbor lists
    int32_t n_edges = 0;
    build_edge_list(data, n_edges);

    if (n_edges == 0) {
        // No neighbors - isolated atoms, only one-body contribution
        // (handled inside the model)
        data->energy = 0.0;
        return;
    }

    // Call IREE model
    float energy = 0.0f;
    model_->compute_efv_inplace(
        edge_rij_.data(),
        atomic_numbers_.data(),
        edge_i_.data(),
        edge_j_.data(),
        n_atoms,
        n_edges,
        &energy,
        forces_out_.data(),
        virial_out_
    );

    // Store energy
    data->energy = static_cast<double>(energy);

    // Accumulate forces
    accumulate_forces(data, n_atoms);

    // Accumulate virial
    accumulate_virial(data);
}

void MLIAPModelACEIREE::compute_descriptors(MLIAPData* /* data */) {
    // Unified interface doesn't use descriptors
    // All computation is done in compute_forces
}

}  // namespace LAMMPS_NS

/**
 * Plugin factory function.
 * This is called by LAMMPS when loading the plugin.
 */
extern "C" {

// Plugin version
const char* mliap_plugin_version() {
    return "1.0.0";
}

// Plugin name
const char* mliap_plugin_name() {
    return "ace_iree";
}

// Create model instance
void* mliap_plugin_create(
    void* lmp,
    const char* vmfb_path,
    const char* metadata_path,
    const char* device
) {
    try {
        return new LAMMPS_NS::MLIAPModelACEIREE(
            static_cast<LAMMPS_NS::LAMMPS*>(lmp),
            vmfb_path,
            metadata_path,
            device ? device : "local-task"
        );
    } catch (const std::exception& e) {
        std::cerr << "Failed to create ACE IREE model: " << e.what() << std::endl;
        return nullptr;
    }
}

// Destroy model instance
void mliap_plugin_destroy(void* model) {
    delete static_cast<LAMMPS_NS::MLIAPModelACEIREE*>(model);
}

// Get cutoff
double mliap_plugin_get_cutoff(void* model) {
    auto* m = static_cast<LAMMPS_NS::MLIAPModelACEIREE*>(model);
    return m->get_rcutfac();
}

// Compute forces
void mliap_plugin_compute_forces(void* model, void* data) {
    auto* m = static_cast<LAMMPS_NS::MLIAPModelACEIREE*>(model);
    m->compute_forces(static_cast<LAMMPS_NS::MLIAPData*>(data));
}

}  // extern "C"
