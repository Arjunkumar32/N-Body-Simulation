# N-Body Gravitational Simulation — README

## Project Structure

```
NBodySim/
├── include/
│   └── nbody.h            # Vec3, Body structs, global constants
├── src/
│   ├── main.cpp           # Simulation driver + galaxy IC + CSV output
│   ├── force.cpp          # Direct O(N²) force (OpenMP parallel)
│   ├── integrator.cpp     # Leapfrog KDK time integrator
│   └── barnes_hut.cpp     # Barnes-Hut Octree O(N log N) force
├── CMakeLists.txt         # CMake build (two targets)
├── visualize.py           # Python animation + energy charts
└── output/                # Generated CSV + images
```

## Parameters (include/nbody.h)

| Parameter    | Default | Description                          |
|-------------|---------|--------------------------------------|
| `N_BODIES`  | 100     | Number of bodies (50–200)            |
| `N_STEPS`   | 500     | Simulation steps                     |
| `DT`        | 0.01    | Time step (years)                    |
| `G`         | 6.674e-11 | Gravitational constant             |
| `SOFTENING` | 0.1     | Epsilon – avoids singularities (AU)  |
| `THETA`     | 0.5     | Barnes-Hut opening angle             |

## Build (Windows — MinGW)

```powershell
# Install MinGW with OpenMP support (or use MSVC)
cd NBodySim\build
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### MSVC alternative
```powershell
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

## Run

```powershell
# Direct O(N^2) — default 100 bodies, 500 steps
.\build\nbody_direct.exe

# Custom: 150 bodies, 800 steps
.\build\nbody_direct.exe 150 800

# Barnes-Hut O(N log N)
.\build\nbody_bh.exe 200 1000
```

Output files appear in `output/`:
- `trajectories.csv` — one row per (step × body)
- `energy.csv`       — KE, PE, total energy per step

## Visualize

```powershell
pip install pandas matplotlib numpy pillow
python visualize.py
```

Produces:
- `output/nbody_animation.gif`        — 3-D animated particle motion
- `output/trajectories_xy.png`        — 2-D spaghetti plot
- `output/energy_conservation.png`    — Energy conservation chart

## Algorithm Notes

### Force Computation (force.cpp)
- Direct pairwise O(N²) with Newton's 3rd law shortcut
- Thread-local force arrays → OpenMP parallel reduction (no race conditions)
- Softened potential: `F = G*m_i*m_j / (r² + ε²)^(3/2)`

### Barnes-Hut (barnes_hut.cpp)
- Recursive 8-child octree partition
- Cell treated as point mass when `s/d < θ` (θ = 0.5)
- Tree built on 1 thread; traversal fully parallel via OpenMP

### Leapfrog Integration (integrator.cpp)
- Kick-Drift-Kick (Velocity-Verlet) scheme
- Symplectic → bounded long-term energy error (~0.001–0.1 %)

## Performance

| Bodies | Method       | Threads | Time / step |
|--------|-------------|---------|-------------|
| 100    | Direct      | 4       | ~2 ms       |
| 100    | Barnes-Hut  | 4       | ~1 ms       |
| 200    | Direct      | 4       | ~8 ms       |
| 200    | Barnes-Hut  | 4       | ~2 ms       |
