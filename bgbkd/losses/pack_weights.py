"""
bgbkd/losses/pack_weights.py
══════════════════════════════════════════════════════════════
Pack thermal-electrical centrality weights for the separatrix loss.

Interior cells (sharing more thermal and electrical neighbours) receive
higher centrality weight w_i, consistent with the observation that they
are more susceptible to thermal runaway propagation.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np


def build_pack_adjacency(
    n_cells:          int,
    cells_per_module: int   = 8,
    lambda_e:         float = 0.6,
    lambda_th:        float = 0.4,
    G_th_same:        float = 0.8,
    G_th_cross:       float = 0.3,
    G_th_skip:        float = 0.2,
    G_e:              float = 1.0,
) -> np.ndarray:
    """
    Build pack thermal-electrical adjacency matrix A_pack.

        A_pack[i][j] = λ_e·G_e[i][j] + λ_th·G_th[i][j]

    Connectivity:
      Nearest-neighbour: series bus-bar + thermal face contact.
        Same module  → G_th = G_th_same
        Cross module → G_th = G_th_cross
      Skip-one within module: thermal skip-coupling (G_th_skip).

    Args:
        n_cells:           Total cell count.
        cells_per_module:  Cells per module (layout: 0-based).
        lambda_e, lambda_th: Electrical / thermal weight in A_pack.
        G_th_same:  Thermal conductance, same-module neighbours.
        G_th_cross: Thermal conductance, cross-module boundary.
        G_th_skip:  Thermal conductance, same-module skip-one.
        G_e:        Normalised series bus-bar conductance.

    Returns:
        A ∈ ℝ^{n×n}, symmetric, zero diagonal.
    """
    A = np.zeros((n_cells, n_cells))

    def same_module(i: int, j: int) -> bool:
        return (i // cells_per_module) == (j // cells_per_module)

    for i in range(n_cells - 1):
        g_th = G_th_same if same_module(i, i + 1) else G_th_cross
        w = lambda_e * G_e + lambda_th * g_th
        A[i][i + 1] = w
        A[i + 1][i] = w

    for i in range(n_cells - 2):
        if same_module(i, i + 2):
            A[i][i + 2] += lambda_th * G_th_skip
            A[i + 2][i] += lambda_th * G_th_skip

    np.fill_diagonal(A, 0.0)
    return A


def compute_centrality_weights(A_pack: np.ndarray) -> np.ndarray:
    """
    Compute thermal-electrical centrality weight for each cell.

        w_i = Σ_j A_pack[i][j] / Σ_k Σ_j A_pack[k][j]

    Interior cells — sharing more neighbours — receive larger w_i.

    Args:
        A_pack: Pack adjacency matrix (n × n).

    Returns:
        w ∈ ℝ^n, normalised to sum to 1.
    """
    row_sums = A_pack.sum(axis=1)
    total = row_sums.sum() + 1e-12
    return row_sums / total


def build_edge_list(
    A_pack:    np.ndarray,
    threshold: float = 0.05,
) -> list[tuple[int, int]]:
    """
    Extract directed edge list from adjacency matrix.
    Only upper-triangle entries (j > i) above threshold are kept.

    Args:
        A_pack:    Pack adjacency matrix (n × n).
        threshold: Minimum edge weight to include.

    Returns:
        List of (i, j) tuples with A_pack[i][j] > threshold.
    """
    n = A_pack.shape[0]
    return [
        (i, j)
        for i in range(n)
        for j in range(n)
        if j > i and A_pack[i][j] > threshold
    ]
