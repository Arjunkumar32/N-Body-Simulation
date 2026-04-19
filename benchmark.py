#!/usr/bin/env python3
"""
benchmark.py — N-Body Performance Benchmark

Runs both the Direct O(N^2) and Barnes-Hut O(N log N) implementations
over a range of particle counts and plots the execution time.
"""

import os
import sys
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────────
N_VALUES = [50, 100, 200, 400, 800, 1600]
STEPS = 500

DIRECT_EXE = os.path.join(".", "build", "nbody_direct.exe")
BH_EXE = os.path.join(".", "build", "nbody_bh.exe")
OUT_DIR = "output"

# Prevent unicode errors on stdout parsing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Execution ────────────────────────────────────────────────────────────
def run_sim(exe, n, steps):
    print(f"Running {os.path.basename(exe)} (N={n}, steps={steps})... ", end="", flush=True)
    t0 = time.time()
    result = subprocess.run([exe, str(n), str(steps)], capture_output=True, text=True)
    t1 = time.time()
    
    if result.returncode != 0:
        print("FAILED")
        print(result.stderr)
        return None
    
    elapsed = t1 - t0
    print(f"{elapsed:.2f} s")
    return elapsed

# ── Main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DIRECT_EXE) or not os.path.exists(BH_EXE):
        print("Error: Executables not found. Please build the project first.")
        sys.exit(1)
        
    os.makedirs(OUT_DIR, exist_ok=True)
    
    times_direct = []
    times_bh = []
    
    print(f"=== N-Body Benchmark (Steps = {STEPS}) ===")
    for n in N_VALUES:
        print(f"\n--- Testing N = {n} ---")
        t_dir = run_sim(DIRECT_EXE, n, STEPS)
        t_bh = run_sim(BH_EXE, n, STEPS)
        
        times_direct.append(t_dir if t_dir is not None else 0)
        times_bh.append(t_bh if t_bh is not None else 0)
        
    # ── Plotting ──────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")
    
    ax.plot(N_VALUES, times_direct, marker="o", color="#ff6b6b", lw=2, label="Direct O(N²)")
    ax.plot(N_VALUES, times_bh, marker="s", color="#74c0fc", lw=2, label="Barnes-Hut O(N log N)")
    
    # Optional: Plot theoretical O(N^2) reference line
    # Scale it to match the first data point
    ref_n2 = [(times_direct[1] / (N_VALUES[1]**2)) * n**2 for n in N_VALUES]
    ax.plot(N_VALUES, ref_n2, color="#ff6b6b", alpha=0.3, ls="--", label="Ref: O(N²)")

    ax.set_title(f"Performance Comparison (Steps={STEPS})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Number of Bodies (N)", fontsize=12)
    ax.set_ylabel("Execution Time (seconds)", fontsize=12)
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left")
    
    out_path = os.path.join(OUT_DIR, "benchmark.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n✓ Benchmark complete. Plot saved to {out_path}")
