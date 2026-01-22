/**
 * pair_iree.h - Non-Kokkos IREE Pair Style (CPU Fallback)
 * =======================================================
 *
 * LAMMPS pair_style using IREE-compiled ACE models for CPU execution.
 * This is the non-Kokkos fallback when GPU/Kokkos is not available.
 *
 * For GPU acceleration with zero-copy, use pair_iree_kokkos.h instead.
 *
 * Usage:
 *   pair_style iree model_cpu.vmfb
 *   pair_coeff * * Si
 */

#ifndef LMP_PAIR_IREE_H
#define LMP_PAIR_IREE_H

#include "pair.h"
#include "iree_wrapper.h"

#include <string>
#include <vector>
#include <memory>

namespace LAMMPS_NS {

/**
 * PairIREE - Non-Kokkos IREE pair_style for CPU.
 *
 * Uses iree_wrapper.h for IREE runtime management.
 * Implements split computation: IREE computes pair_forces,
 * host loop scatters to atom forces.
 */
class PairIREE : public Pair {
public:
    PairIREE(class LAMMPS*);
    ~PairIREE() override;

    void compute(int, int) override;
    void settings(int, char**) override;
    void coeff(int, char**) override;
    void init_style() override;
    double init_one(int, int) override;

protected:
    // Model configuration
    std::string vmfb_path_;
    double cutoff_;
    int max_atoms_;
    int max_pairs_;

    // Element mapping
    std::vector<std::string> elements_;
    std::vector<int> type_map_;  // LAMMPS type -> model species index

    // IREE model wrapper
    std::unique_ptr<ace_iree::IREEModel> iree_;

    // Working buffers (host memory)
    std::vector<int> pair_i_;
    std::vector<int> pair_j_;
    std::vector<float> rij_;          // [n_pairs * 3]
    std::vector<float> pair_forces_;  // [n_pairs * 3]

    // Methods
    void allocate();
    void build_pair_lists();
    void scatter_forces(int n_pairs);
};

}  // namespace LAMMPS_NS

#endif  // LMP_PAIR_IREE_H
