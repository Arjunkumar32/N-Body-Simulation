#!/usr/bin/env python3
"""
visualize.py  —  N-Body Simulation Visualizer
Reads output/trajectories.csv and output/energy.csv produced by the C++ engine
and generates:
  1. 3-D animated scatter plot  (nbody_animation.gif)
  2. 2-D XY trajectory spaghetti plot (trajectories_xy.png)
  3. Energy conservation chart  (energy_conservation.png)

Requirements:
  pip install pandas matplotlib numpy
"""

import os
import sys
import io

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")        # headless rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "output"
TRAJ_FILE  = os.path.join(OUT_DIR, "trajectories.csv")
ENRG_FILE  = os.path.join(OUT_DIR, "energy.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d1a",
    "axes.facecolor":   "#0d0d1a",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccff",
    "xtick.color":      "#888899",
    "ytick.color":      "#888899",
    "text.color":       "#ccccff",
    "grid.color":       "#1a1a2e",
    "grid.linewidth":   0.5,
})

CMAP = plt.cm.plasma

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────────────────
print(f"Loading {TRAJ_FILE} ...", end=" ", flush=True)
traj = pd.read_csv(TRAJ_FILE)
print(f"{len(traj):,} rows.")

print(f"Loading {ENRG_FILE} ...", end=" ", flush=True)
enrg = pd.read_csv(ENRG_FILE)
print(f"{len(enrg):,} rows.")

steps   = traj["step"].unique()
bodies  = traj["id"].unique()
n_steps = int(steps.max())
n_bodies= len(bodies)

print(f"  Bodies: {n_bodies}  |  Steps: {n_steps}")

# Pre-pivot trajectories for fast lookup: shape (step, body, coord)
# Pivot so we have arrays indexed by [step, body_id]
traj_pivot = traj.pivot(index=["step", "id"], columns=[])
xs = traj.pivot(index="step", columns="id", values="x").values   # (S, N)
ys = traj.pivot(index="step", columns="id", values="y").values
zs = traj.pivot(index="step", columns="id", values="z").values

# ─────────────────────────────────────────────────────────────────────────────
# 2.  3-D Animation
# ─────────────────────────────────────────────────────────────────────────────
TRAIL_LEN = 30   # steps to show as tail

colors = CMAP(np.linspace(0.15, 0.95, n_bodies))

fig3d = plt.figure(figsize=(10, 8))
ax3d  = fig3d.add_subplot(111, projection="3d")
ax3d.set_box_aspect([1, 1, 0.5])
ax3d.set_xlabel("X (AU)", labelpad=6)
ax3d.set_ylabel("Y (AU)", labelpad=6)
ax3d.set_zlabel("Z (AU)", labelpad=6)
ax3d.set_title("N-Body Gravitational Simulation", pad=12, fontsize=14, fontweight="bold")
ax3d.grid(True)

scat  = ax3d.scatter([], [], [], s=12, c=[], cmap=CMAP, vmin=0, vmax=1, depthshade=True)
trails = [ax3d.plot([], [], [], lw=0.6, alpha=0.5, color=colors[i])[0]
          for i in range(n_bodies)]
time_text = ax3d.text2D(0.02, 0.96, "", transform=ax3d.transAxes,
                        fontsize=9, color="#aaaaff")

# Axis limits from full data
margin = 0.1
xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()
zmin, zmax = zs.min(), zs.max()
xm = (xmax - xmin) * margin
ym = (ymax - ymin) * margin
zm = (zmax - zmin) * margin + 0.1
ax3d.set_xlim(xmin - xm, xmax + xm)
ax3d.set_ylim(ymin - ym, ymax + ym)
ax3d.set_zlim(zmin - zm, zmax + zm)

def update3d(frame):
    step_idx = frame
    x = xs[step_idx]
    y = ys[step_idx]
    z = zs[step_idx]

    # Update scatter
    scat._offsets3d = (x, y, z)
    body_colors = np.linspace(0.1, 0.95, n_bodies)
    scat.set_array(body_colors)

    # Update trails
    start = max(0, step_idx - TRAIL_LEN)
    for i, trail in enumerate(trails):
        trail.set_data(xs[start:step_idx+1, i], ys[start:step_idx+1, i])
        trail.set_3d_properties(zs[start:step_idx+1, i])

    time_text.set_text(f"Step: {steps[step_idx]:.0f}  t = {steps[step_idx]*0.01:.1f} yr")
    return [scat, time_text] + trails

# Downsample to max 200 frames for an exportable GIF
n_frames = min(n_steps + 1, 200)
frame_indices = np.linspace(0, n_steps, n_frames, dtype=int)

print("Rendering 3-D animation ...", flush=True)
ani = animation.FuncAnimation(fig3d, update3d, frames=frame_indices,
                               interval=50, blit=False)
gif_path = os.path.join(OUT_DIR, "nbody_animation.gif")
ani.save(gif_path, writer="pillow", fps=20, dpi=80)
print(f"  Saved -> {gif_path}")
plt.close(fig3d)

# -----------------------------------------------------------------------------
# 3.  2-D XY Trajectory Spaghetti Plot
# -----------------------------------------------------------------------------
fig2d, ax2d = plt.subplots(figsize=(9, 9))
ax2d.set_facecolor("#0d0d1a")
ax2d.set_title("Particle Trajectories — XY Plane", fontsize=14, fontweight="bold", pad=10)
ax2d.set_xlabel("X (AU)")
ax2d.set_ylabel("Y (AU)")

for i in range(n_bodies):
    alpha = 0.6 if i == 0 else 0.35
    lw    = 1.5 if i == 0 else 0.5
    ax2d.plot(xs[:, i], ys[:, i], color=colors[i], alpha=alpha, lw=lw)
    ax2d.scatter(xs[0, i], ys[0, i], color=colors[i], s=18, zorder=5)

ax2d.set_aspect("equal", "datalim")
ax2d.grid(True, alpha=0.3)
xy_path = os.path.join(OUT_DIR, "trajectories_xy.png")
fig2d.savefig(xy_path, dpi=120, bbox_inches="tight", facecolor=fig2d.get_facecolor())
print(f"  Saved -> {xy_path}")
plt.close(fig2d)

# -----------------------------------------------------------------------------
# 4.  Energy Conservation Chart
# -----------------------------------------------------------------------------
fig_e, ax_e = plt.subplots(figsize=(10, 4))
ax_e.set_facecolor("#0d0d1a")
ax_e.set_title("Energy Conservation (Leapfrog Integrator)", fontsize=13, fontweight="bold")
ax_e.set_xlabel("Step")
ax_e.set_ylabel("Energy (arbitrary units)")

t_arr = enrg["step"].values
ke_arr= enrg["KE"].values
pe_arr= enrg["PE"].values
tot   = enrg["total"].values

ax_e.plot(t_arr, ke_arr,  color="#ff6b6b", lw=1.2, label="Kinetic (KE)")
ax_e.plot(t_arr, pe_arr,  color="#74c0fc", lw=1.2, label="Potential (PE)")
ax_e.plot(t_arr, tot,     color="#69db7c", lw=1.8, label="Total E", linestyle="--")

drift_pct = abs((tot[-1] - tot[0]) / tot[0]) * 100 if tot[0] != 0 else 0
ax_e.text(0.02, 0.05, f"Energy drift: {drift_pct:.3f}%",
          transform=ax_e.transAxes, fontsize=10,
          color="#ffeb3b", bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="#1a1a2e", alpha=0.8))
ax_e.legend(loc="upper right", framealpha=0.3)
ax_e.grid(True, alpha=0.3)

enrg_path = os.path.join(OUT_DIR, "energy_conservation.png")
fig_e.savefig(enrg_path, dpi=120, bbox_inches="tight", facecolor=fig_e.get_facecolor())
print(f"  Saved -> {enrg_path}")
plt.close(fig_e)

print("\n[DONE] All visualizations complete!")
print(f"  {gif_path}")
print(f"  {xy_path}")
print(f"  {enrg_path}")
