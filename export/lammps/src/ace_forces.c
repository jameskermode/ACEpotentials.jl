/**
 * ACE Forces Library Implementation
 * ==================================
 *
 * This file implements the embedding functions and chain rule for forces.
 * These are standard mathematical functions (Chebyshev, spherical harmonics),
 * NOT the ACE kernel itself.
 *
 * The ACE kernel (energy + gradients) comes from compiled code:
 * - Energy: IREE VMFB (works)
 * - Gradients: StableHLO via XLA (IREE can't compile due to scatter)
 */

#include "ace_forces.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Chebyshev Polynomials
 * ============================================================================
 *
 * Standard Chebyshev polynomials of the first kind via recurrence:
 *   T_0(x) = 1
 *   T_1(x) = x
 *   T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
 *
 * Derivatives:
 *   T'_0(x) = 0
 *   T'_1(x) = 1
 *   T'_n(x) = 2*T_{n-1}(x) + 2*x*T'_{n-1}(x) - T'_{n-2}(x)
 */

void ace_chebyshev(
    float r, float rcut, int N_cheb,
    float* Rnl, float* dRnl_dr
) {
    if (N_cheb < 1) return;

    /* Transform r to x in [-1, 1] for Chebyshev domain */
    /* Using x = 2*(r/rcut) - 1, so x=-1 at r=0, x=1 at r=rcut */
    float x = 2.0f * (r / rcut) - 1.0f;
    float dx_dr = 2.0f / rcut;

    /* T_0 */
    Rnl[0] = 1.0f;
    if (dRnl_dr) dRnl_dr[0] = 0.0f;

    if (N_cheb < 2) return;

    /* T_1 */
    Rnl[1] = x;
    if (dRnl_dr) dRnl_dr[1] = dx_dr;

    /* Recurrence for T_n, n >= 2 */
    for (int n = 2; n < N_cheb; n++) {
        Rnl[n] = 2.0f * x * Rnl[n-1] - Rnl[n-2];
        if (dRnl_dr) {
            /* d/dr T_n = (d/dx T_n) * dx/dr */
            /* d/dx T_n = n * U_{n-1}(x) where U is Chebyshev 2nd kind */
            /* Using: T'_n = 2*T_{n-1} + 2*x*T'_{n-1} - T'_{n-2} (chain rule) */
            dRnl_dr[n] = 2.0f * Rnl[n-1] * dx_dr
                       + 2.0f * x * dRnl_dr[n-1]
                       - dRnl_dr[n-2];
        }
    }

    /* Apply smooth cutoff envelope: f(r) = (1 - (r/rcut)^2)^2 for r < rcut */
    float u = r / rcut;
    float env = (1.0f - u*u);
    env = env * env;  /* (1 - u^2)^2 */

    float denv_dr = 0.0f;
    if (dRnl_dr && r < rcut) {
        /* d/dr [(1-u^2)^2] = 2*(1-u^2)*(-2*u/rcut) = -4*u*(1-u^2)/rcut */
        denv_dr = -4.0f * u * (1.0f - u*u) / rcut;
    }

    /* Apply envelope to all polynomials */
    for (int n = 0; n < N_cheb; n++) {
        if (dRnl_dr) {
            /* Product rule: d/dr(env * T_n) = denv_dr * T_n + env * dT_n/dr */
            dRnl_dr[n] = denv_dr * Rnl[n] + env * dRnl_dr[n];
        }
        Rnl[n] = env * Rnl[n];
    }
}

/* ============================================================================
 * Real Spherical Harmonics
 * ============================================================================
 *
 * Real spherical harmonics Y_l^m(theta, phi) in Cartesian form.
 * Uses recursive computation for numerical stability.
 *
 * Convention: Y_l^m indexed as Y[l*l + l + m] for -l <= m <= l
 * Total count for maxl: (maxl+1)^2
 */

void ace_ylm(
    float x, float y, float z, int maxl,
    float* Ylm,
    float* dYlm_dx, float* dYlm_dy, float* dYlm_dz
) {
    float r2 = x*x + y*y + z*z;
    float r = sqrtf(r2);

    if (r < 1e-10f) {
        /* At origin, only Y_0^0 is non-zero */
        int n_ylm = (maxl + 1) * (maxl + 1);
        memset(Ylm, 0, n_ylm * sizeof(float));
        Ylm[0] = 0.28209479177387814f;  /* 1/sqrt(4*pi) */
        if (dYlm_dx) memset(dYlm_dx, 0, n_ylm * sizeof(float));
        if (dYlm_dy) memset(dYlm_dy, 0, n_ylm * sizeof(float));
        if (dYlm_dz) memset(dYlm_dz, 0, n_ylm * sizeof(float));
        return;
    }

    float inv_r = 1.0f / r;
    float xhat = x * inv_r;
    float yhat = y * inv_r;
    float zhat = z * inv_r;

    /* l=0: Y_0^0 = 1/sqrt(4*pi) */
    Ylm[0] = 0.28209479177387814f;
    if (dYlm_dx) dYlm_dx[0] = 0.0f;
    if (dYlm_dy) dYlm_dy[0] = 0.0f;
    if (dYlm_dz) dYlm_dz[0] = 0.0f;

    if (maxl < 1) return;

    /* l=1: Y_1^{-1} = sqrt(3/4pi) * y/r, Y_1^0 = sqrt(3/4pi) * z/r, Y_1^1 = sqrt(3/4pi) * x/r */
    float c1 = 0.4886025119029199f;  /* sqrt(3/4pi) */
    Ylm[1] = c1 * yhat;  /* l=1, m=-1 */
    Ylm[2] = c1 * zhat;  /* l=1, m=0 */
    Ylm[3] = c1 * xhat;  /* l=1, m=1 */

    if (dYlm_dx || dYlm_dy || dYlm_dz) {
        /* d/dx (xhat) = (1 - xhat^2) / r, etc. */
        float inv_r2 = inv_r * inv_r;
        if (dYlm_dx) {
            dYlm_dx[1] = c1 * (-xhat * yhat) * inv_r;
            dYlm_dx[2] = c1 * (-xhat * zhat) * inv_r;
            dYlm_dx[3] = c1 * (1.0f - xhat*xhat) * inv_r;
        }
        if (dYlm_dy) {
            dYlm_dy[1] = c1 * (1.0f - yhat*yhat) * inv_r;
            dYlm_dy[2] = c1 * (-yhat * zhat) * inv_r;
            dYlm_dy[3] = c1 * (-yhat * xhat) * inv_r;
        }
        if (dYlm_dz) {
            dYlm_dz[1] = c1 * (-zhat * yhat) * inv_r;
            dYlm_dz[2] = c1 * (1.0f - zhat*zhat) * inv_r;
            dYlm_dz[3] = c1 * (-zhat * xhat) * inv_r;
        }
    }

    if (maxl < 2) return;

    /* l=2: Use standard formulas */
    /* Y_2^{-2} = sqrt(15/4pi)/2 * xy/r^2 */
    /* Y_2^{-1} = sqrt(15/4pi)/2 * yz/r^2 */
    /* Y_2^0 = sqrt(5/16pi) * (3z^2 - r^2)/r^2 = sqrt(5/16pi) * (3*zhat^2 - 1) */
    /* Y_2^1 = sqrt(15/4pi)/2 * xz/r^2 */
    /* Y_2^2 = sqrt(15/16pi) * (x^2 - y^2)/r^2 */

    float c2a = 1.0925484305920792f;  /* sqrt(15/4pi) */
    float c2b = 0.31539156525252005f; /* sqrt(5/16pi) */
    float c2c = 0.5462742152960396f;  /* sqrt(15/16pi) */

    Ylm[4] = c2a * xhat * yhat;                      /* l=2, m=-2 */
    Ylm[5] = c2a * yhat * zhat;                      /* l=2, m=-1 */
    Ylm[6] = c2b * (3.0f * zhat*zhat - 1.0f);       /* l=2, m=0 */
    Ylm[7] = c2a * xhat * zhat;                      /* l=2, m=1 */
    Ylm[8] = c2c * (xhat*xhat - yhat*yhat);         /* l=2, m=2 */

    /* Derivatives for l=2 (more complex, involving product rule) */
    if (dYlm_dx || dYlm_dy || dYlm_dz) {
        /* These derivatives are tedious but straightforward */
        /* d/dx (xhat * yhat) = yhat * d(xhat)/dx + xhat * d(yhat)/dx */
        /* where d(xhat)/dx = (1 - xhat^2)/r, d(yhat)/dx = -xhat*yhat/r */

        float dx_xhat = (1.0f - xhat*xhat) * inv_r;
        float dy_xhat = -xhat * yhat * inv_r;
        float dz_xhat = -xhat * zhat * inv_r;

        float dx_yhat = -yhat * xhat * inv_r;
        float dy_yhat = (1.0f - yhat*yhat) * inv_r;
        float dz_yhat = -yhat * zhat * inv_r;

        float dx_zhat = -zhat * xhat * inv_r;
        float dy_zhat = -zhat * yhat * inv_r;
        float dz_zhat = (1.0f - zhat*zhat) * inv_r;

        if (dYlm_dx) {
            dYlm_dx[4] = c2a * (dx_xhat * yhat + xhat * dx_yhat);
            dYlm_dx[5] = c2a * (dx_yhat * zhat + yhat * dx_zhat);
            dYlm_dx[6] = c2b * 6.0f * zhat * dx_zhat;
            dYlm_dx[7] = c2a * (dx_xhat * zhat + xhat * dx_zhat);
            dYlm_dx[8] = c2c * 2.0f * (xhat * dx_xhat - yhat * dx_yhat);
        }
        if (dYlm_dy) {
            dYlm_dy[4] = c2a * (dy_xhat * yhat + xhat * dy_yhat);
            dYlm_dy[5] = c2a * (dy_yhat * zhat + yhat * dy_zhat);
            dYlm_dy[6] = c2b * 6.0f * zhat * dy_zhat;
            dYlm_dy[7] = c2a * (dy_xhat * zhat + xhat * dy_zhat);
            dYlm_dy[8] = c2c * 2.0f * (xhat * dy_xhat - yhat * dy_yhat);
        }
        if (dYlm_dz) {
            dYlm_dz[4] = c2a * (dz_xhat * yhat + xhat * dz_yhat);
            dYlm_dz[5] = c2a * (dz_yhat * zhat + yhat * dz_zhat);
            dYlm_dz[6] = c2b * 6.0f * zhat * dz_zhat;
            dYlm_dz[7] = c2a * (dz_xhat * zhat + xhat * dz_zhat);
            dYlm_dz[8] = c2c * 2.0f * (xhat * dz_xhat - yhat * dz_yhat);
        }
    }

    /* For l > 2, would use full recursive algorithm */
    /* TODO: Implement general l recursion if needed */
}

/* ============================================================================
 * Chain Rule: Embedding Gradients → Atomic Forces
 * ============================================================================
 *
 * Given dE/dRnl and dE/dYlm (from compiled ACE backward pass), compute forces:
 *
 *   F_i = -sum_j [ (dE/dRnl_ij) * (dRnl/dr_ij) * rhat_ij
 *                + (dE/dYlm_ij) dot (dYlm/dr_ij) ]
 *
 * This is NOT ACE - it's just chain rule on Chebyshev and Ylm functions.
 */

void ace_grad_to_forces(
    const float* dRnl_3, const float* dYlm_3,
    const float* rij,
    const int* edge_i, const int* edge_j,
    int nedges, int natoms, int maxneigs,
    int nRnl, int nYlm, float rcut,
    float* forces
) {
    /* Initialize forces to zero */
    memset(forces, 0, natoms * 3 * sizeof(float));

    /* Temporary buffers for embedding derivatives */
    float* Rnl = (float*)malloc(nRnl * sizeof(float));
    float* dRnl_dr = (float*)malloc(nRnl * sizeof(float));
    float* Ylm = (float*)malloc(nYlm * sizeof(float));
    float* dYlm_dx = (float*)malloc(nYlm * sizeof(float));
    float* dYlm_dy = (float*)malloc(nYlm * sizeof(float));
    float* dYlm_dz = (float*)malloc(nYlm * sizeof(float));

    int maxl = (int)(sqrtf((float)nYlm) + 0.5f) - 1;

    /* Track neighbor index for each atom */
    int* neig_count = (int*)calloc(natoms, sizeof(int));

    /* Process each edge (i,j) */
    for (int e = 0; e < nedges; e++) {
        int i = edge_i[e];
        int j = edge_j[e];

        /* Get displacement vector r_ij = r_j - r_i */
        float rx = rij[e * 3 + 0];
        float ry = rij[e * 3 + 1];
        float rz = rij[e * 3 + 2];
        float r = sqrtf(rx*rx + ry*ry + rz*rz);

        if (r < 1e-10f || r > rcut) continue;

        float rhat_x = rx / r;
        float rhat_y = ry / r;
        float rhat_z = rz / r;

        /* Compute Rnl and dRnl/dr */
        ace_chebyshev(r, rcut, nRnl, Rnl, dRnl_dr);

        /* Compute Ylm and dYlm/d(x,y,z) */
        ace_ylm(rx, ry, rz, maxl, Ylm, dYlm_dx, dYlm_dy, dYlm_dz);

        /* Get neighbor index for this edge */
        int neig_idx = neig_count[i];
        neig_count[i]++;

        /* Accumulate force contribution */
        float fx = 0.0f, fy = 0.0f, fz = 0.0f;

        /* Radial contribution: dE/dRnl * dRnl/dr * rhat */
        for (int n = 0; n < nRnl; n++) {
            /* dRnl_3 is [maxneigs, natoms, nRnl] in row-major */
            int idx = neig_idx * (natoms * nRnl) + i * nRnl + n;
            float dE_dRnl = dRnl_3[idx];
            float contrib = dE_dRnl * dRnl_dr[n];
            fx += contrib * rhat_x;
            fy += contrib * rhat_y;
            fz += contrib * rhat_z;
        }

        /* Angular contribution: dE/dYlm * dYlm/dr */
        for (int m = 0; m < nYlm; m++) {
            /* dYlm_3 is [maxneigs, natoms, nYlm] in row-major */
            int idx = neig_idx * (natoms * nYlm) + i * nYlm + m;
            float dE_dYlm = dYlm_3[idx];
            fx += dE_dYlm * dYlm_dx[m];
            fy += dE_dYlm * dYlm_dy[m];
            fz += dE_dYlm * dYlm_dz[m];
        }

        /* F_i gets -contribution, F_j gets +contribution (Newton's 3rd law) */
        forces[i * 3 + 0] -= fx;
        forces[i * 3 + 1] -= fy;
        forces[i * 3 + 2] -= fz;
        forces[j * 3 + 0] += fx;
        forces[j * 3 + 1] += fy;
        forces[j * 3 + 2] += fz;
    }

    free(Rnl);
    free(dRnl_dr);
    free(Ylm);
    free(dYlm_dx);
    free(dYlm_dy);
    free(dYlm_dz);
    free(neig_count);
}
