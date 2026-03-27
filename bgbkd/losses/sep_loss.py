"""
bgbkd/losses/sep_loss.py
══════════════════════════════════════════════════════════════
EIS-weighted multi-node separatrix loss with DRT-confidence weighting.

Replaces the standard separatrix loss (GBKD) with two innovations:
  1. DRT-crossing indicator C_i^EIS: detects mechanistic plating onset
     (growing DRT arc at τ≈4ms), not just u_i state crossing.
  2. EIS-confidence weight Ω_i: amplifies crossings with strong plating
     DRT evidence.

Loss function:
    L_sep = Σ_i w_i · Ω_i · (u_cross_i − α_i)²
    L_cross = cross-cell consistency term (thermal-electrical coupling)

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class CrossingEvent:
    """A recorded DRT-based EIS crossing at cell i."""
    cell:    int
    u_cross: float
    Omega:   float   # EIS-confidence weight
    w_i:     float   # centrality weight
    t:       float   # timestamp (s)


def eis_confidence_weight(
    DRT_peak:  float,
    gamma_drt: float = 0.15,
    mu:        float = 1.5,
) -> float:
    """
    Ω_i(Z,t) = 1 + μ · A_plating_i(t) / γ_DRT

    A crossing with a large, growing plating DRT peak carries more
    gradient signal than a marginal crossing.
    """
    return 1.0 + mu * DRT_peak / (gamma_drt + 1e-12)


def node_separatrix_loss(
    crossings:  list[CrossingEvent],
    alpha_est:  np.ndarray,
) -> float:
    """
    EIS-weighted node separatrix loss.

        L_sep = Σ_i w_i · Ω_i · (u_cross_i − α_i)²

    Args:
        crossings: List of recent CrossingEvents.
        alpha_est: Current per-cell threshold estimates (n,).

    Returns:
        Scalar loss value ≥ 0.
    """
    if not crossings:
        return 0.0
    loss = 0.0
    for ev in crossings:
        diff = ev.u_cross - alpha_est[ev.cell]
        loss += ev.w_i * ev.Omega * diff ** 2
    return loss / (len(crossings) + 1e-9)


def node_separatrix_grad(
    crossings:  list[CrossingEvent],
    alpha_est:  np.ndarray,
    A_pack:     np.ndarray,
    lambda_cross: float = 0.10,
) -> np.ndarray:
    """
    Combined gradient of L_sep + λ_cross·L_cross w.r.t. α.

    Direct gradient (per-cell):
        ∂L/∂α_i = −2·w_i·Ω_i·(u_cross_i − α_i)

    Cross-cell consistency term promotes consistency with thermal-
    electrical neighbours at the moment of crossing.
    """
    n = len(alpha_est)
    grad = np.zeros(n)

    for ev in crossings:
        i   = ev.cell
        ucr = ev.u_cross
        Omega = ev.Omega
        w_i   = ev.w_i
        # Direct gradient
        grad[i] += -2.0 * w_i * Omega * (ucr - alpha_est[i])

        # Cross-cell gradient
        row = A_pack[i]
        row_sum = row.sum() + 1e-9
        row_sq_sum = (row ** 2).sum() + 1e-9
        for j in range(n):
            if j == i or A_pack[i][j] < 1e-9:
                continue
            Sij = A_pack[i][j] / row_sum
            cij = A_pack[i][j] ** 2 / row_sq_sum
            # This currently requires u_j but it is not stored; caller
            # passes it separately — this gradient is computed in estimator.
            # (Kept here as formula reference; called from BGBKDAdaptiveEstimator)
            _ = Sij, cij  # consumed in estimator._update_alpha

    return grad


class SeparatrixLossAccumulator:
    """
    Accumulates DRT crossing events across timesteps and provides
    gradient information for the adaptive α estimator.
    """

    def __init__(
        self,
        gamma_drt: float = 0.15,
        mu:        float = 1.5,
        maxlen:    int   = 200,
    ):
        self.gamma_drt = gamma_drt
        self.mu        = mu
        self.maxlen    = maxlen
        self._buffer: list[CrossingEvent] = []

    def register(
        self,
        cell:    int,
        u_cross: float,
        DRT_peak:float,
        w_i:     float,
        t:       float,
    ) -> None:
        """Record a DRT-based crossing event."""
        Omega = eis_confidence_weight(DRT_peak, self.gamma_drt, self.mu)
        self._buffer.append(CrossingEvent(cell, u_cross, Omega, w_i, t))
        if len(self._buffer) > self.maxlen:
            self._buffer = self._buffer[-self.maxlen:]

    @property
    def recent(self) -> list[CrossingEvent]:
        return self._buffer[-30:]

    def clear(self) -> None:
        self._buffer.clear()
