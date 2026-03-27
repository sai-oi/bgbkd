#!/usr/bin/env python3
"""
scripts/build_pack_topology.py
══════════════════════════════════════════════════════════════
Build A_pack and edge list from pack layout specification.
Run once before first use.

Usage:
    python scripts/build_pack_topology.py

Output:
    data/pack_topology/A_pack_96cell.npy    — adjacency matrix (96×96)
    data/pack_topology/edges_96cell.json    — directed edge list
    data/eis_calibration/alpha_prior_96cell.npy  — initial α prior (96,)

Expected:
    A_pack shape: (96, 96)  |E|=167
    d_c = 96 + 167 = 263
    ✓ Topology saved.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
import sys
import json
import numpy as np
from pathlib import Path

# Make bgbkd importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


def build_topology(
    n_cells:          int   = 96,
    cells_per_module: int   = 8,
    lambda_e:         float = 0.6,
    lambda_th:        float = 0.4,
    A_threshold:      float = 0.05,
) -> tuple[np.ndarray, list]:
    """Build A_pack and edge list for an n_cells series pack."""
    A = np.zeros((n_cells, n_cells))

    def same_module(i, j):
        return (i // cells_per_module) == (j // cells_per_module)

    for i in range(n_cells - 1):
        G_th = 0.8 if same_module(i, i + 1) else 0.3
        G_e  = 1.0
        w = lambda_e * G_e + lambda_th * G_th
        A[i][i + 1] = w
        A[i + 1][i] = w

    for i in range(n_cells - 2):
        if same_module(i, i + 2):
            A[i][i + 2] += lambda_th * 0.2
            A[i + 2][i] += lambda_th * 0.2

    np.fill_diagonal(A, 0.0)

    edges = [
        (i, j) for i in range(n_cells) for j in range(n_cells)
        if j > i and A[i][j] > A_threshold
    ]
    return A, edges


def build_alpha_prior(
    n_cells:          int,
    alpha_prior_mean: float = 0.32,
    seed:             int   = 42,
) -> np.ndarray:
    """
    Build per-cell α prior using impedance spread.

    Cells with higher SEI resistance (weaker cells) have lower
    plating threshold: α_prior_i = mean − 0.15·(ρ_SEI_i − mean_ρ_SEI).
    """
    rng = np.random.default_rng(seed)
    R_SEI_spread = 1.0 + rng.normal(0, 0.08, n_cells)
    R_SEI_spread = np.clip(R_SEI_spread, 0.80, 1.25)
    # Cell 47: weakest cell in use-case scenario
    if n_cells > 48:
        R_SEI_spread[47] = 1.22
        R_SEI_spread[48] = 1.12

    rho_sei = (R_SEI_spread - 1.0) / 0.25   # normalised spread → [0,1]
    alpha = np.clip(
        alpha_prior_mean - 0.15 * (rho_sei - rho_sei.mean()),
        0.18, 0.50,
    )
    return alpha


def main():
    n_cells = 96

    print("Building pack topology...")
    A_pack, edges = build_topology(n_cells=n_cells)

    out_topo = Path("data/pack_topology")
    out_topo.mkdir(parents=True, exist_ok=True)
    np.save(out_topo / "A_pack_96cell.npy", A_pack)
    with open(out_topo / "edges_96cell.json", "w") as f:
        json.dump(edges, f)

    print(f"A_pack shape: {A_pack.shape}  |E|={len(edges)}")
    print(f"d_c = {n_cells} + {len(edges)} = {n_cells + len(edges)}")

    print("\nBuilding α prior...")
    alpha_prior = build_alpha_prior(n_cells=n_cells)

    out_cal = Path("data/eis_calibration")
    out_cal.mkdir(parents=True, exist_ok=True)
    np.save(out_cal / "alpha_prior_96cell.npy", alpha_prior)

    print(f"α_prior: mean={alpha_prior.mean():.3f}  "
          f"min={alpha_prior.min():.3f}  max={alpha_prior.max():.3f}")
    if n_cells > 48:
        print(f"Cell 47 α_prior = {alpha_prior[47]:.3f}  "
              f"(target ≈ 0.287; prior initialised above true threshold)")

    print("\n✓ Topology saved.")
    print(f"  data/pack_topology/A_pack_96cell.npy")
    print(f"  data/pack_topology/edges_96cell.json")
    print(f"  data/eis_calibration/alpha_prior_96cell.npy")


if __name__ == "__main__":
    main()
