"""
bgbkd/core/observables.py
══════════════════════════════════════════════════════════════
BGBKD observable map Ψ: ℝ^N → ℝ^{d_c}

Cascade subspace (real-time BMS alerting):
  d_c = N_cells + |E_pack|

Per-cell terms (column indices 0..N−1):
  φ⁴_i = u_i(1−u_i)(u_i−α_i)   plating/runaway tipping kernel

Edge terms (column indices N..N+|E|−1):
  Φ̂^→_ij = u_i(1−u_i) · Γ_ij · (u_j − α̂_ij)
            EIS-gated directed stress propagation kernel

Hard attractor guarantees (structural, not approximate):
  φ⁴_i = 0  at u_i ∈ {0,1}   ∀ α_i
  Φ̂^→_ij = 0  at u_i ∈ {0,1}  ∀ Γ_ij, α̂_ij

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from ..eis.state import EISState

# ── Backbone BKD functions ────────────────────────────────────────


def phi3(u: float) -> float:
    """
    Saturation mask φ³_i = u_i(1−u_i).
    Hard zeros at both attractors: φ³(0) = φ³(1) = 0.
    Maximum at u = 0.5: φ³ = 0.25.
    """
    return u * (1.0 - u)


def phi4(u: float, alpha: float) -> float:
    """
    Plating/runaway tipping kernel φ⁴_i = u_i(1−u_i)(u_i−α_i).
    Hard zeros at both attractors: φ⁴(0) = φ⁴(1) = 0.
    This is the backbone interpretable fragility observable.
    """
    return u * (1.0 - u) * (u - alpha)


def psi_score(u: float, alpha: float, phi_max: float = 0.05) -> float:
    """
    Per-cell fragility score ψ_i ∈ [0,1].

        ψ_i = |φ⁴_i(u_i; α̂_i)| / φ_max

    ψ_i = 0: cell at an attractor (u_i ≈ 0 or 1), structurally safe.
    ψ_i = 1: cell at maximum saddle proximity; imminent tipping.
    """
    return min(1.0, abs(phi4(u, alpha)) / (phi_max + 1e-12))


# ── EIS-augmented edge observable ─────────────────────────────────


def gamma_ij(
    eis_i:  EISState,
    eis_j:  EISState,
    kappa:  float = 0.8,
    Z_ref:  float = 0.3,
) -> float:
    """
    EIS coupling gate Γ_ij = 1 + κ·|ΔZ_ij(f_thermal)|/Z_ref.

    Larger differential impedance → stronger propagation amplification.
    Physical: impedance gradient at thermal frequencies signals current
    redistribution before temperature sensors register the gradient.
    """
    delta_R = abs(eis_i.R_SEI - eis_j.R_SEI)
    delta_W = abs(eis_i.W_trans - eis_j.W_trans)
    delta_Z = np.sqrt(delta_R ** 2 + delta_W ** 2)
    return 1.0 + kappa * delta_Z / (Z_ref + 1e-12)


def alpha_hat_ij(
    alpha_j: float,
    eis_i:   EISState,
    eis_j:   EISState,
    gamma_T: float = 0.04,
    gamma_R: float = 0.025,
    dT_ref:  float = 10.0,
) -> float:
    """
    Wake-adjusted plating threshold at cell j from cell i.

        α̂_ij = α_j − γ_T·(T_i−T_j)/ΔT_ref − γ_R·ΔR_SEI/R_ref

    Both thermal gradient and resistance redistribution lower j's
    effective plating threshold, making it more vulnerable.
    """
    dT = (eis_i.T_cell - eis_j.T_cell) / (dT_ref + 1e-9)
    dR = eis_i.R_SEI - eis_j.R_SEI
    return float(np.clip(alpha_j - gamma_T * dT - gamma_R * dR, 0.08, 0.80))


def phi_hat_ij(
    u_i:     float,
    u_j:     float,
    eis_i:   EISState,
    eis_j:   EISState,
    alpha_j: float,
    kappa:   float = 0.8,
    Z_ref:   float = 0.3,
    gamma_T: float = 0.04,
    gamma_R: float = 0.025,
    dT_ref:  float = 10.0,
) -> float:
    """
    Full EIS-augmented directed edge observable.

        Φ̂^→_ij = u_i(1−u_i) · Γ_ij · (u_j − α̂_ij)

    Properties:
      P1 — Hard zeros: Φ̂^→_ij = 0 at u_i ∈ {0,1} regardless of Γ_ij, α̂_ij.
      P2 — Directionality: Φ̂^→_ij ≠ Φ̂^→_ji (thermal gradient asymmetry).
      P3 — Amplification: hotter/higher-R cell i → larger Φ̂^→_ij.
      P5 — Classical limit: Γ_ij→1, α̂_ij→α_j ⟹ classical GBKD edge.
    """
    mask = phi3(u_i)                              # = 0 at u_i ∈ {0,1}
    G    = gamma_ij(eis_i, eis_j, kappa, Z_ref)
    a_ij = alpha_hat_ij(alpha_j, eis_i, eis_j, gamma_T, gamma_R, dT_ref)
    return mask * G * (u_j - a_ij)


# ── Observable class ──────────────────────────────────────────────


class BGBKDObservables:
    """
    BGBKD observable map — cascade subspace Ψ: ℝ^N → ℝ^{d_c}.

    Args:
        n_cells:          Number of cells in the pack.
        edges:            Directed edge list [(i,j), ...].
        alpha_i:          Per-cell tipping threshold estimates (n,).
        alpha_ij:         Per-edge stability margins {(i,j): float}.
        cascade_subspace: If True, use cascade subspace only (d_c = N+|E|).
                          If False, use full 5N+|E| dictionary.
        calibration:      EIS calibration dict (optional, passed to Γ).
        eis_cfg:          EIS config dict (channel weights, etc.).
        edge_cfg:         Edge observable config (kappa, Z_ref, γ_T, γ_R).
    """

    def __init__(
        self,
        n_cells:          int,
        edges:            list[tuple[int, int]],
        alpha_i:          np.ndarray,
        alpha_ij:         dict[tuple[int, int], float],
        cascade_subspace: bool = True,
        calibration:      Optional[dict] = None,
        eis_cfg:          Optional[dict] = None,
        edge_cfg:         Optional[dict] = None,
    ):
        self.n_cells          = n_cells
        self.edges            = edges
        self.n_edges          = len(edges)
        self.alpha_i          = alpha_i.copy().astype(float)
        self.alpha_ij         = dict(alpha_ij)
        self.cascade_subspace = cascade_subspace

        # Edge observable parameters
        ec = edge_cfg or {}
        self.kappa   = float(ec.get("kappa",   0.8))
        self.Z_ref   = float(ec.get("Z_ref",   0.3))
        self.gamma_T = float(ec.get("gamma_T", 0.04))
        self.gamma_R = float(ec.get("gamma_R", 0.025))
        self.dT_ref  = float(ec.get("dT_ref_C", 10.0))

        # Fragility
        eis = eis_cfg or {}
        self.phi_max = 0.05

    @property
    def dim(self) -> int:
        """Observable dimension d_c = N_cells + |E_pack| (cascade subspace)."""
        return self.n_cells + self.n_edges

    def lift(
        self,
        u_vec:      np.ndarray,
        eis_states: list[EISState],
    ) -> np.ndarray:
        """
        Lift pack state to cascade subspace Ψ ∈ ℝ^{d_c}.

        Column layout:
          [0..N−1]       φ⁴_i : per-cell tipping kernels
          [N..N+|E|−1]   Φ̂^→_ij : directed edge observables

        Args:
            u_vec:      Pack state vector (n_cells,) ∈ (0,1)^N.
            eis_states: List of n_cells EISState objects.

        Returns:
            Ψ ∈ ℝ^{d_c}.
        """
        phi4_vec = np.array([
            phi4(float(u_vec[i]), float(self.alpha_i[i]))
            for i in range(self.n_cells)
        ])

        edge_obs = np.array([
            phi_hat_ij(
                float(u_vec[i]), float(u_vec[j]),
                eis_states[i], eis_states[j],
                float(self.alpha_ij.get((i, j), self.alpha_i[j])),
                kappa=self.kappa, Z_ref=self.Z_ref,
                gamma_T=self.gamma_T, gamma_R=self.gamma_R,
                dT_ref=self.dT_ref,
            )
            for (i, j) in self.edges
        ])

        return np.concatenate([phi4_vec, edge_obs])

    def phi4_index(self, cell_i: int) -> int:
        """Column index of φ⁴_i in the cascade subspace."""
        return cell_i

    def edge_index(self, edge_k: int) -> int:
        """Column index of the k-th edge observable Φ̂^→_ij."""
        return self.n_cells + edge_k
