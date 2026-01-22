/**
 * pair_iree.cpp - Non-Kokkos IREE Pair Style (CPU Fallback)
 * =========================================================
 *
 * CPU implementation of IREE-compiled ACE models for LAMMPS.
 * For GPU with zero-copy, use pair_iree_kokkos.cpp instead.
 */

#include "pair_iree.h"
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

namespace LAMMPS_NS {

PairIREE::PairIREE(LAMMPS* lmp) :
    Pair(lmp),
    cutoff_(5.0),
    max_atoms_(4096),
    max_pairs_(200000)
{
    single_enable = 0;  // No single() function
    restartinfo = 0;    // No restart info
    one_coeff = 1;      // Only one pair_coeff * * line allowed
    manybody_flag = 1;  // Many-body potential

    // Pre-allocate working buffers
    pair_i_.reserve(max_pairs_);
    pair_j_.reserve(max_pairs_);
    rij_.reserve(max_pairs_ * 3);
    pair_forces_.reserve(max_pairs_ * 3);
}

PairIREE::~PairIREE() {
    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
}

void PairIREE::settings(int narg, char** arg) {
    if (narg < 1) {
        error->all(FLERR, "Illegal pair_style iree command");
    }

    vmfb_path_ = arg[0];

    // Optional: metadata path
    std::string metadata_path;
    if (narg > 1) {
        metadata_path = arg[1];
    }

    // Load IREE model with CPU device
    try {
        iree_ = std::make_unique<ace_iree::IREEModel>(
            vmfb_path_, metadata_path, "local-task"
        );
        cutoff_ = iree_->metadata().rcut;
    } catch (const std::exception& e) {
        error->all(FLERR, "Failed to load IREE model: {}", e.what());
    }
}

void PairIREE::coeff(int narg, char** arg) {
    if (narg < 3) {
        error->all(FLERR, "Incorrect args for pair coefficients");
    }

    // Parse pair_coeff * * element1 element2 ...
    if (!allocated) allocate();

    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

    // Parse element names
    int ntypes = atom->ntypes;
    elements_.clear();
    type_map_.resize(ntypes + 1, -1);

    for (int i = 2; i < narg; i++) {
        elements_.push_back(arg[i]);
    }

    // Map LAMMPS types to model element indices
    int nelements = elements_.size();
    for (int i = ilo; i <= ihi; i++) {
        if (i - 1 < nelements) {
            type_map_[i] = i - 1;
        }
    }

    // Set all pair coefficients
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

void PairIREE::init_style() {
    // Request half neighbor list
    neighbor->add_request(this, NeighConst::REQ_DEFAULT);

    // Ensure cutoffs are set
    for (int i = 1; i <= atom->ntypes; i++) {
        for (int j = i; j <= atom->ntypes; j++) {
            if (setflag[i][j]) {
                cutsq[i][j] = cutoff_ * cutoff_;
                cutsq[j][i] = cutsq[i][j];
            }
        }
    }
}

double PairIREE::init_one(int i, int j) {
    if (setflag[i][j] == 0) {
        error->all(FLERR, "All pair coeffs are not set");
    }
    return cutoff_;
}

void PairIREE::allocate() {
    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++) {
        for (int j = i; j <= n; j++) {
            setflag[i][j] = 0;
        }
    }

    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

void PairIREE::compute(int eflag, int vflag) {
    ev_init(eflag, vflag);

    double **x = atom->x;
    double **f = atom->f;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    // Get neighbor list
    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    // Build pair lists
    build_pair_lists();

    int n_pairs = pair_i_.size();
    if (n_pairs == 0) return;

    // Prepare inputs for IREE
    // Build pool_matrix and other required tensors for selection-matrix approach
    // For now, using the split computation: compute pair_forces, scatter to atoms

    // The compiled model expects:
    // - rij: [n_pairs, 3] displacement vectors
    // - pool_matrix: [n_atoms, n_pairs] pooling matrix
    // - selection matrices for R, Y, symm products
    // - model parameters

    // For this CPU fallback, we use the existing IREEModel::compute_efv interface
    // which handles the full EFV computation

    // Build edge data in the expected format
    std::vector<float> edge_rij(max_pairs_ * 3, 0.0f);
    std::vector<int64_t> atomic_numbers(max_atoms_, 0);
    std::vector<int64_t> edge_i(max_pairs_, 0);
    std::vector<int64_t> edge_j(max_pairs_, 0);

    // Copy pair data
    for (int e = 0; e < n_pairs; e++) {
        edge_rij[e * 3 + 0] = rij_[e * 3 + 0];
        edge_rij[e * 3 + 1] = rij_[e * 3 + 1];
        edge_rij[e * 3 + 2] = rij_[e * 3 + 2];
        edge_i[e] = pair_i_[e];
        edge_j[e] = pair_j_[e];
    }

    // Copy atomic numbers
    int n_atoms = nlocal;
    if (newton_pair) n_atoms = atom->nlocal + atom->nghost;
    for (int i = 0; i < std::min(n_atoms, max_atoms_); i++) {
        atomic_numbers[i] = type_map_[type[i]];
    }

    // Call IREE model
    try {
        auto result = iree_->compute_efv(
            edge_rij.data(),
            atomic_numbers.data(),
            edge_i.data(),
            edge_j.data(),
            n_atoms,
            n_pairs
        );

        // Add energy
        if (eflag_global) {
            eng_vdwl += result.energy;
        }

        // Add forces
        for (int i = 0; i < nlocal; i++) {
            f[i][0] += result.forces[i * 3 + 0];
            f[i][1] += result.forces[i * 3 + 1];
            f[i][2] += result.forces[i * 3 + 2];
        }

        // Add virial
        if (vflag_global) {
            virial[0] += result.virial[0];  // xx
            virial[1] += result.virial[4];  // yy
            virial[2] += result.virial[8];  // zz
            virial[3] += result.virial[1];  // xy
            virial[4] += result.virial[2];  // xz
            virial[5] += result.virial[5];  // yz
        }
    } catch (const std::exception& e) {
        error->all(FLERR, "IREE computation failed: {}", e.what());
    }
}

void PairIREE::build_pair_lists() {
    double **x = atom->x;
    int *type = atom->type;
    int nlocal = atom->nlocal;
    double cutsq_val = cutoff_ * cutoff_;

    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    // Clear existing pair data
    pair_i_.clear();
    pair_j_.clear();
    rij_.clear();

    // Build pair lists from neighbor list
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

    if (pair_i_.size() > (size_t)max_pairs_) {
        error->all(FLERR, "Too many pairs ({}) exceeds max_pairs ({})",
                   pair_i_.size(), max_pairs_);
    }
}

void PairIREE::scatter_forces(int n_pairs) {
    double **f = atom->f;

    // Scatter pair forces to atom forces
    // f_i -= pair_force (force on i from j)
    // f_j += pair_force (Newton's 3rd law)
    for (int e = 0; e < n_pairs; e++) {
        int i = pair_i_[e];
        int j = pair_j_[e];

        double fx = pair_forces_[e * 3 + 0];
        double fy = pair_forces_[e * 3 + 1];
        double fz = pair_forces_[e * 3 + 2];

        f[i][0] -= fx;
        f[i][1] -= fy;
        f[i][2] -= fz;

        f[j][0] += fx;
        f[j][1] += fy;
        f[j][2] += fz;
    }
}

}  // namespace LAMMPS_NS

// =============================================================================
// Plugin Registration
// =============================================================================
#include "version.h"
#include "lammpsplugin.h"

using namespace LAMMPS_NS;

static Pair *iree_creator(LAMMPS *lmp)
{
    return new PairIREE(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
    lammpsplugin_t plugin;
    lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

    // Register iree pair style (CPU non-Kokkos)
    plugin.version = LAMMPS_VERSION;
    plugin.style = "pair";
    plugin.name = "iree";
    plugin.info = "IREE-compiled ACE pair style (CPU, non-Kokkos) v1.0";
    plugin.author = "ACEpotentials.jl";
    plugin.creator.v1 = (lammpsplugin_factory1 *) &iree_creator;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);
}
