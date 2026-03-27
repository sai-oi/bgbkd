"""
bgbkd/eis/state.py
══════════════════════════════════════════════════════════════
EISState dataclass and EIS-to-u_i bistable state mapping.

Three mechanistically orthogonal EIS channels:
  ρ_SEI   — SEI resistance growth        (1 kHz,   f_HF)
  ρ_θ     — Plating precursor phase       (100 Hz,  f_MF)
  ρ_W     — Ion transport deviation       (10 mHz,  f_LF)

Composite: s_i = w_SEI·ρ_SEI + w_θ·ρ_θ + w_W·ρ_W
Bistable state: u_i = 1 − σ_β(s_i − s̄)

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Default channel weights (w_SEI + w_θ + w_W = 1)
W_SEI:   float = 0.25
W_THETA: float = 0.55
W_W:     float = 0.20
BETA:    float = 8.0   # sigmoid sharpness
S_BAR:   float = 0.5   # centering: midpoint of [0,1]

# Hard attractor clip bounds — prevents numerical u=0 or u=1
U_MIN: float = 0.003
U_MAX: float = 0.997


@dataclass
class EISState:
    """
    Per-cell EIS-derived state variables at one timestep.

    All normalised channels are in [0, 1]:
      R_SEI    — ρ_SEI: normalised SEI resistance (high-frequency channel)
      theta    — ρ_θ:   normalised plating-precursor phase (mid-frequency channel)
      W_trans  — ρ_W:   normalised transport deviation (low-frequency channel)
      DRT_peak — amplitude of the plating DRT arc at τ ≈ 1–10 ms
      T_cell   — cell surface temperature (°C)
    """
    R_SEI:    float    # ρ_SEI ∈ [0,1]
    theta:    float    # ρ_θ   ∈ [0,1]
    W_trans:  float    # ρ_W   ∈ [0,1]
    DRT_peak: float    # DRT plating peak amplitude ≥ 0
    T_cell:   float    # cell surface temperature (°C)

    def composite_stress(
        self,
        w_sei:   float = W_SEI,
        w_theta: float = W_THETA,
        w_w:     float = W_W,
    ) -> float:
        """s_i = w_SEI·ρ_SEI + w_θ·ρ_θ + w_W·ρ_W ∈ [0,1]."""
        return w_sei * self.R_SEI + w_theta * self.theta + w_w * self.W_trans


def eis_to_u(
    eis: EISState,
    w_sei:   float = W_SEI,
    w_theta: float = W_THETA,
    w_w:     float = W_W,
    beta:    float = BETA,
    s_bar:   float = S_BAR,
) -> float:
    """
    Map EIS composite stress to BGBKD bistable state variable u_i ∈ (0,1).

        s_i = w_SEI·ρ_SEI + w_θ·ρ_θ + w_W·ρ_W
        u_i = 1 − σ_β(s_i − s̄)

    Attractor mapping:
      s_i → 0 (fresh/healthy)  ⟹  u_i → 1  (stable high-state)
      s_i → 1 (degraded)       ⟹  u_i → 0  (degraded attractor)
      s_i = s̄ (saddle region)  ⟹  u_i = 0.5

    Returns value clipped to (U_MIN, U_MAX) to prevent exact attractor
    numerics while preserving the structural zero of the saturation mask φ³.
    """
    s_i = eis.composite_stress(w_sei, w_theta, w_w)
    u_i = 1.0 - 1.0 / (1.0 + np.exp(-beta * (s_i - s_bar)))
    return float(np.clip(u_i, U_MIN, U_MAX))
