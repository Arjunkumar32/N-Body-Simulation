/*
 * integrator.cpp
 * Leapfrog / Velocity-Verlet time integrator (kick-drift-kick).
 *
 * Algorithm (symplectic – conserves energy well over long runs):
 *   1. KICK:  v(t + dt/2) = v(t)     + a(t)       * dt/2
 *   2. DRIFT: x(t + dt)   = x(t)     + v(t+dt/2)  * dt
 *   3. Recompute forces -> a(t+dt)
 *   4. KICK:  v(t + dt)   = v(t+dt/2) + a(t+dt)   * dt/2
 *
 * The caller is responsible for supplying the force function.
 */

#include "nbody.h"
#include <vector>
#include <omp.h>

// Half-step velocity update (kick)
static void kick(std::vector<Body>& bodies, double halfdt) {
    int n = static_cast<int>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].vel += bodies[i].acc * halfdt;
    }
}

// Full position update (drift)
static void drift(std::vector<Body>& bodies, double dt) {
    int n = static_cast<int>(bodies.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        bodies[i].pos += bodies[i].vel * dt;
    }
}

/*
 * leapfrogStep
 * Performs one full Kick-Drift-Kick leapfrog step.
 * forceFunc: callable (std::vector<Body>&) that fills body.acc for all bodies.
 */
void leapfrogStep(std::vector<Body>& bodies,
                  double dt,
                  void (*forceFunc)(std::vector<Body>&))
{
    double halfdt = dt * 0.5;
    kick(bodies, halfdt);     // v += a * dt/2
    drift(bodies, dt);        // x += v * dt
    forceFunc(bodies);        // recompute accelerations at new positions
    kick(bodies, halfdt);     // v += a * dt/2
}
