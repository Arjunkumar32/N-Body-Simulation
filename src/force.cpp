/*
 * force.cpp
 * Direct O(N^2) pairwise gravitational force computation.
 * Parallelised with OpenMP — each thread owns a stripe of rows.
 * A softening factor epsilon prevents singularities when bodies pass close.
 *
 *   F_ij = G * m_i * m_j / (|r_ij|^2 + eps^2)^(3/2) * r_ij_hat
 */

#include "nbody.h"
#include <vector>
#include <omp.h>

// Reset all forces to zero
void clearForces(std::vector<Body>& bodies) {
    int n = static_cast<int>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].force = {0.0, 0.0, 0.0};
        bodies[i].acc   = {0.0, 0.0, 0.0};
    }
}

/*
 * computeForcesDirect
 * Uses Newton's 3rd law: compute F_ij once, apply ±F to both bodies.
 * The outer loop is parallelised; each thread gets a private partial force
 * array which is then reduced into the shared array.
 */
void computeForcesDirect(std::vector<Body>& bodies) {
    int n = static_cast<int>(bodies.size());
    clearForces(bodies);

    // Thread-private force accumulator (avoids race conditions)
    std::vector<std::vector<Vec3>> threadForces(
        omp_get_max_threads(),
        std::vector<Vec3>(n, {0.0, 0.0, 0.0})
    );

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& localF = threadForces[tid];

        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < n - 1; ++i) {
            for (int j = i + 1; j < n; ++j) {
                Vec3 r   = bodies[j].pos - bodies[i].pos;
                double r2 = r.norm2() + SOFTENING * SOFTENING;
                double r3 = r2 * std::sqrt(r2);           // |r|^3 with softening

                double fmag = G * bodies[i].mass * bodies[j].mass / r3;

                Vec3 fvec = r * fmag;
                localF[i] += fvec;
                localF[j] += (fvec * -1.0);
            }
        }
    } // end parallel region

    // Parallel reduction: combine all thread-local forces
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        Vec3 total = {0.0, 0.0, 0.0};
        for (int t = 0; t < omp_get_max_threads(); ++t) {
            total += threadForces[t][i];
        }
        bodies[i].force = total;
        bodies[i].acc   = total / bodies[i].mass;  // a = F/m
    }
}
