/*
 * main.cpp  —  N-Body Gravitational Simulation
 *
 * Usage:
 *   nbody_direct   [n_bodies] [n_steps]     (direct O(N^2), default 100, 500)
 *   nbody_bh       [n_bodies] [n_steps]     (Barnes-Hut  O(N log N))
 *
 * Outputs:
 *   output/trajectories.csv   — step,id,x,y,z,vx,vy,vz
 *   output/energy.csv         — step,KE,PE,total
 *
 * Initial conditions: random disk galaxy
 */

#define _USE_MATH_DEFINES
#include "nbody.h"
#include <vector>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <omp.h>

// ─── Forward declarations ──────────────────────────────────────────────────
void computeForcesDirect   (std::vector<Body>& bodies);
void computeForcesBarnesHut(std::vector<Body>& bodies);
void leapfrogStep(std::vector<Body>& bodies, double dt,
                  void (*forceFunc)(std::vector<Body>&));

// ─── Energy ────────────────────────────────────────────────────────────────
struct Energy { double KE, PE, total; };

Energy computeEnergy(const std::vector<Body>& bodies) {
    int n = static_cast<int>(bodies.size());
    double KE = 0.0, PE = 0.0;

    #pragma omp parallel for reduction(+:KE) schedule(static)
    for (int i = 0; i < n; ++i)
        KE += 0.5 * bodies[i].mass * bodies[i].vel.norm2();

    #pragma omp parallel for reduction(+:PE) schedule(dynamic, 4)
    for (int i = 0; i < n - 1; ++i)
        for (int j = i + 1; j < n; ++j) {
            Vec3 r = bodies[j].pos - bodies[i].pos;
            double dist = std::sqrt(r.norm2() + SOFTENING * SOFTENING);
            PE -= G * bodies[i].mass * bodies[j].mass / dist;
        }

    return {KE, PE, KE + PE};
}

// ─── Initial Conditions — Disk Galaxy (natural units) ──────────────────────
std::vector<Body> initGalaxyDisk(int n, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> angleDist(0.0, 2.0 * M_PI);
    std::uniform_real_distribution<double> radiusDist(1.0, 5.0);
    std::normal_distribution<double>       zDist(0.0, 0.1);
    std::uniform_real_distribution<double> massDist(0.5, 2.0);   // natural units

    std::vector<Body> bodies(n);
    // Central massive body (galaxy core) — mass = N so it dominates
    bodies[0].mass   = static_cast<double>(n);
    bodies[0].pos    = {0.0, 0.0, 0.0};
    bodies[0].vel    = {0.0, 0.0, 0.0};
    bodies[0].label  = "Core";

    for (int i = 1; i < n; ++i) {
        double r     = radiusDist(rng);
        double phi   = angleDist(rng);
        double z     = zDist(rng);
        double m     = massDist(rng);

        // Circular orbit: v_c = sqrt(G * M_core / r), G=1
        double vcirc = std::sqrt(G * bodies[0].mass / r) * 0.9; // slight ellipticity

        bodies[i].mass  = m;
        bodies[i].pos   = { r * std::cos(phi), r * std::sin(phi), z };
        bodies[i].vel   = { -vcirc * std::sin(phi),
                             vcirc * std::cos(phi),
                             0.0 };
        bodies[i].label = "B" + std::to_string(i);
    }
    return bodies;
}


// ─── I/O ───────────────────────────────────────────────────────────────────
void writeHeader(std::ofstream& traj, std::ofstream& enrg) {
    traj << "step,id,x,y,z,vx,vy,vz\n";
    enrg << "step,KE,PE,total\n";
}

void writeStep(std::ofstream& traj, std::ofstream& enrg,
               const std::vector<Body>& bodies, const Energy& e, int step)
{
    for (int i = 0; i < (int)bodies.size(); ++i) {
        const auto& b = bodies[i];
        traj << step << ',' << i << ','
             << std::setprecision(9)
             << b.pos.x << ',' << b.pos.y << ',' << b.pos.z << ','
             << b.vel.x << ',' << b.vel.y << ',' << b.vel.z << '\n';
    }
    enrg << step << ',' << e.KE << ',' << e.PE << ',' << e.total << '\n';
}

// ─── Progress Bar ──────────────────────────────────────────────────────────
void printProgress(int step, int total, double elapsed) {
    int pct   = (step * 100) / total;
    int width = 40;
    int filled = (step * width) / total;
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) std::cout << (i < filled ? '=' : ' ');
    std::cout << "] " << std::setw(3) << pct << "%  step " 
              << std::setw(5) << step << "/" << total
              << "  t=" << std::fixed << std::setprecision(1) << elapsed << "s"
              << std::flush;
}

// ─── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // ── Determine mode (direct / Barnes-Hut) ──
    bool useBarnesHut = false;
    std::string exeName = argc > 0 ? argv[0] : "";
    if (exeName.find("bh") != std::string::npos ||
        exeName.find("BH") != std::string::npos)
        useBarnesHut = true;

    int nBodies = N_BODIES;
    int nSteps  = N_STEPS;
    if (argc >= 2) nBodies = std::atoi(argv[1]);
    if (argc >= 3) nSteps  = std::atoi(argv[2]);
    nBodies = std::max(2, nBodies);
    nSteps  = std::max(10, nSteps);

    auto forceFunc = useBarnesHut ? computeForcesBarnesHut : computeForcesDirect;
    std::string method = useBarnesHut ? "Barnes-Hut O(N log N)" : "Direct O(N^2)";

    std::cout << "=== N-Body Gravitational Simulation ===\n"
              << "  Bodies  : " << nBodies << "\n"
              << "  Steps   : " << nSteps  << "\n"
              << "  dt      : " << DT      << " yr\n"
              << "  Method  : " << method  << "\n"
              << "  Threads : " << omp_get_max_threads() << "\n"
              << "=======================================\n";

    // Initialise
    auto bodies = initGalaxyDisk(nBodies);
    forceFunc(bodies);   // compute initial accelerations

    // Method-specific output directory
    std::string outDir = useBarnesHut ? "output/bh" : "output/direct";
    // Create directory (system call — works on Windows & Linux)
    std::string mkdirCmd = "mkdir \"" + outDir + "\" 2>nul || mkdir -p \"" + outDir + "\" 2>/dev/null";
    (void)std::system(mkdirCmd.c_str());

    // Open output files
    std::ofstream trajFile(outDir + "/trajectories.csv");
    std::ofstream enrgFile(outDir + "/energy.csv");
    writeHeader(trajFile, enrgFile);

    // Initial state
    Energy E0 = computeEnergy(bodies);
    writeStep(trajFile, enrgFile, bodies, E0, 0);

    auto wallStart = std::chrono::steady_clock::now();

    // ── Main simulation loop ──
    for (int step = 1; step <= nSteps; ++step) {
        leapfrogStep(bodies, DT, forceFunc);

        Energy E = computeEnergy(bodies);
        writeStep(trajFile, enrgFile, bodies, E, step);

        // Print progress every 10 steps
        if (step % 10 == 0 || step == nSteps) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - wallStart).count();
            printProgress(step, nSteps, elapsed);
            if (step == nSteps) {
                double drift = std::abs((E.total - E0.total) / E0.total) * 100.0;
                std::cout << "\n  Energy drift: " << std::fixed
                          << std::setprecision(4) << drift << " %\n";
            }
        }
    }

    auto wallEnd = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(wallEnd - wallStart).count();
    std::cout << "Simulation complete in " << std::fixed
              << std::setprecision(2) << total << " s\n"
              << "Output -> output/trajectories.csv, output/energy.csv\n";

    return 0;
}
