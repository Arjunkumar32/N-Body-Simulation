#pragma once
#include <cmath>
#include <string>

// ─── Simulation Parameters ─────────────────────────────────────────────────
constexpr int    N_BODIES  = 100;      // number of bodies (50-200)
constexpr int    N_STEPS   = 500;      // total simulation steps
constexpr double DT        = 0.001;    // time step (natural units)
constexpr double G         = 1.0;      // gravitational constant (natural units)
constexpr double SOFTENING = 0.05;     // softening length (natural units)
constexpr double THETA     = 0.5;      // Barnes-Hut opening angle

// ─── 3-D Vector ────────────────────────────────────────────────────────────
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(double s)      const { return {x*s, y*s, z*s}; }
    Vec3 operator/(double s)      const { return {x/s, y-s, z/s}; }
    Vec3& operator+=(const Vec3& o) { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator*=(double s)      { x*=s;   y*=s;   z*=s;   return *this; }

    double dot(const Vec3& o)  const { return x*o.x + y*o.y + z*o.z; }
    double norm2()             const { return x*x + y*y + z*z; }
    double norm()              const { return std::sqrt(norm2()); }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

// ─── Body ──────────────────────────────────────────────────────────────────
struct Body {
    double mass;
    Vec3   pos;   // position  (AU)
    Vec3   vel;   // velocity  (AU/yr)
    Vec3   acc;   // acceleration (AU/yr²)
    Vec3   force; // net force (N) – computed each step

    std::string label; // optional identifier
};
