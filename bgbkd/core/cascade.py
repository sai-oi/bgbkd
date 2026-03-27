"""
bgbkd/core/cascade.py
══════════════════════════════════════════════════════════════
Two-hop cascade score ξ^(2)_ij and PackFragility scoring.

    ξ^(2)_ij = |K[φ⁴_i, Φ̂^→_ij]| × |K[Φ̂^→_ij, φ⁴_j]| / ‖K‖²_F

hop1 = K[φ⁴_i, Φ̂^→_ij] — initiation rate:
    Plating precursor intensifying at cell i → EIS-gated stress edge activates.

hop2 = K[Φ̂^→_ij, φ⁴_j] — propagation-to-completion rate:
    Active stress edge (i→j) → cell j approaches its runaway threshold.

Stage-specific mitigation dispatch:
    hop1 dominant  → derate C-rate at cell i
    hop2 dominant  → pre-activate cooling at cell j; prepare isolation
    Both large     → derate full string; emergency cooling
    Negative sign  → edge is a heat-sink path (monitor only)

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .observables import BGBKDObservables, psi_score


# ── Cascade report dataclass ──────────────────────────────────────


@dataclass
class CascadeReport:
    """Two-hop cascade result for a single directed edge (i→j)."""
    edge:       tuple[int, int]
    xi_twohop:  float           # ξ^(2)_ij ≥ 0
    hop1_init:  float           # |K[φ⁴_i, Φ̂^→_ij]| / ‖K‖_F
    hop2_comp:  float           # |K[Φ̂^→_ij, φ⁴_j]| / ‖K‖_F
    mitigation: str             # recommended BMS action


# ── Two-hop cascade detector ──────────────────────────────────────


class TwoHopCascadeDetector:
    """
    Compute ξ^(2)_ij for all pack edges from the fitted Koopman operator K.

    The full ξ^(2) matrix is computed in a single pass from K — one matrix
    index read per edge per hop. Total cost: O(|E|) after K is available.
    """

    def __init__(self, obs: BGBKDObservables):
        self.obs = obs

    def compute(
        self,
        K: np.ndarray,
    ) -> dict[tuple[int, int], CascadeReport]:
        """
        Compute cascade scores for all edges.

        Args:
            K: Fitted Koopman operator (d_c × d_c).

        Returns:
            Dict: edge (i,j) → CascadeReport.
        """
        K_norm = float(np.linalg.norm(K, "fro")) + 1e-12
        reports: dict[tuple[int, int], CascadeReport] = {}

        # Pre-build index arrays for vectorised path
        phi4_i_idx = np.array([self.obs.phi4_index(i) for (i, _) in self.obs.edges])
        phi4_j_idx = np.array([self.obs.phi4_index(j) for (_, j) in self.obs.edges])
        edge_idx   = np.array([self.obs.edge_index(k)  for k in range(self.obs.n_edges)])

        hop1_vec = np.abs(K[phi4_i_idx, edge_idx]) / K_norm
        hop2_vec = np.abs(K[edge_idx,   phi4_j_idx]) / K_norm
        xi_vec   = hop1_vec * hop2_vec

        for k, (i, j) in enumerate(self.obs.edges):
            xi   = float(xi_vec[k])
            hop1 = float(hop1_vec[k])
            hop2 = float(hop2_vec[k])
            mit  = _mitigation_label(xi, hop1, hop2)
            reports[(i, j)] = CascadeReport(
                edge=(i, j),
                xi_twohop=xi,
                hop1_init=hop1,
                hop2_comp=hop2,
                mitigation=mit,
            )

        return reports

    def top_edges(
        self,
        K: np.ndarray,
        n: int = 5,
    ) -> list[tuple[tuple[int, int], CascadeReport]]:
        """Return the n highest-ξ edges."""
        reports = self.compute(K)
        return sorted(reports.items(), key=lambda x: x[1].xi_twohop, reverse=True)[:n]


def _mitigation_label(xi: float, hop1: float, hop2: float) -> str:
    if xi < 1e-8:
        return "monitor"
    total = hop1 + hop2 + 1e-12
    ratio = hop1 / total
    if ratio > 0.65:
        return "derate-c-rate-cell-i"
    elif ratio < 0.35:
        return "cool-isolate-cell-j"
    else:
        return "derate-full-module"


# ── Pack fragility scorer ─────────────────────────────────────────


class PackFragility:
    """
    Compute per-cell fragility ψ_i and system fragility Ψ_sys.

        ψ_i   = |φ⁴_i(u_i; α̂_i)| / φ_max
        Ψ_sys = max(ψ_i) + γ · ξ_max

    Ψ_sys > Θ = 0.40 triggers the SCAL alert (Constitutional Mandate CM-2).
    """

    def __init__(
        self,
        obs:     BGBKDObservables,
        detector:TwoHopCascadeDetector,
        phi_max: float = 0.05,
        gamma:   float = 0.40,
    ):
        self.obs      = obs
        self.detector = detector
        self.phi_max  = phi_max
        self.gamma    = gamma

    def score(
        self,
        u_vec:       np.ndarray,
        alpha_est:   np.ndarray,
        K:           Optional[np.ndarray] = None,
    ) -> dict:
        """
        Compute pack fragility report.

        Args:
            u_vec:     Pack state (n_cells,).
            alpha_est: Per-cell threshold estimates (n_cells,).
            K:         Koopman operator (d_c × d_c), or None.

        Returns:
            Dict with keys: psi, psi_sys, xi_max, cascade_map, scal_alert.
        """
        psi = np.array([
            psi_score(float(u_vec[i]), float(alpha_est[i]), self.phi_max)
            for i in range(self.obs.n_cells)
        ])
        psi_max = float(psi.max())
        xi_max  = 0.0
        cascade_map: dict = {}

        if K is not None:
            cascade_map = self.detector.compute(K)
            xi_max = max(
                (r.xi_twohop for r in cascade_map.values()), default=0.0
            )

        psi_sys = psi_max + self.gamma * xi_max

        return dict(
            psi         = psi,
            psi_max     = psi_max,
            psi_sys     = psi_sys,
            xi_max      = xi_max,
            cascade_map = cascade_map,
            scal_alert  = psi_sys > 0.40,
        )
