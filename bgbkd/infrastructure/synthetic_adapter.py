"""
bgbkd/infrastructure/synthetic_adapter.py
══════════════════════════════════════════════════════════════
Synthetic EIS adapter — plausible EIS data for testing and demo
without physical BQ40Z80 hardware.

Generates EIS states that reproduce the 96-cell fast-charge scenario
described in bgbkd_usecase_narrative.md:
  - 5°C ambient cold start
  - 8% cell-to-cell impedance spread
  - Cell 47: 22% excess SEI resistance (weakest cell)
  - Cell 48: 12% excess (thermally adjacent)

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from ..eis.state import EISState

N_CELLS   = 96
N_MODULES = 12
CELLS_PER_MODULE = 8


def _module_of(cell_i: int) -> int:
    return cell_i // CELLS_PER_MODULE


def _build_r_sei_spread(n_cells: int = N_CELLS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spread = 1.0 + rng.normal(0, 0.08, n_cells)
    spread = np.clip(spread, 0.80, 1.25)
    spread[47] = 1.22   # weakest cell
    spread[48] = 1.12   # adjacent
    return spread


_R_SEI_SPREAD = _build_r_sei_spread()


def simulate_eis(
    cell_idx:   int,
    t_s:        float,
    C_rate:     float,
    T_ambient:  float = 5.0,
    R_SEI_mult: float = 1.0,
    rng:        np.random.Generator | None = None,
) -> EISState:
    """
    Simulate EIS features for a single cell at time t under C_rate.

    Models:
      R_SEI:   Logarithmic SEI growth with cycling.
      theta:   Exponential plating kinetics above C_plating threshold
               (Butler-Volmer positive feedback).
      DRT_peak: Plating arc amplitude; appears when theta > 0.30.
      W_trans:  Grows with both SEI and plating.
      T_cell:   Self-heating + plating exotherm above ambient.

    Args:
        cell_idx:   Cell index (0-based).
        t_s:        Simulation time in seconds.
        C_rate:     Local C-rate (A/Ah).
        T_ambient:  Ambient temperature (°C).
        R_SEI_mult: Cell-level SEI resistance multiplier (≥1.0 for aged).
        rng:        Random number generator for noise (reproducible).

    Returns:
        EISState with all channels normalised to [0,1] (except T_cell).
    """
    if rng is None:
        rng = np.random.default_rng(cell_idx)

    # C-rate threshold for plating at 5°C (reduced vs 25°C baseline)
    C_plating = 1.8 / R_SEI_mult

    # SEI growth: slow logarithmic increase
    R_SEI_norm = float(np.clip(
        (R_SEI_mult - 1.0) + 0.005 * np.log1p(t_s / 300.0),
        0.0, 1.0,
    ))

    # Plating precursor: exponential above C_plating (BV positive feedback)
    plating_drive = max(0.0, (C_rate - C_plating) / (C_plating + 1e-9))
    theta_norm = float(np.clip(
        0.05 * t_s / 300.0 + 0.80 * (1.0 - np.exp(-plating_drive * t_s / 90.0)),
        0.0, 1.0,
    ))

    # DRT plating peak (appears when theta > 0.30)
    DRT_peak = max(0.0, theta_norm - 0.30) * 2.5 + rng.normal(0, 0.03)

    # Transport deviation: grows with both SEI and plating
    W_norm = float(np.clip(0.3 * R_SEI_norm + 0.5 * theta_norm, 0.0, 1.0))

    # Cell temperature
    Q_joule = C_rate ** 2 * R_SEI_mult * 0.02
    Q_plat  = plating_drive * 0.05
    T_cell  = T_ambient + (Q_joule + Q_plat) * min(t_s, 300.0)
    T_cell  = float(np.clip(T_cell + rng.normal(0, 0.2), T_ambient, 80.0))

    return EISState(
        R_SEI    = float(np.clip(R_SEI_norm + rng.normal(0, 0.01), 0.0, 1.0)),
        theta    = float(np.clip(theta_norm + rng.normal(0, 0.02), 0.0, 1.0)),
        W_trans  = float(np.clip(W_norm + rng.normal(0, 0.01), 0.0, 1.0)),
        DRT_peak = max(0.0, DRT_peak),
        T_cell   = T_cell,
    )


def simulate_pack(
    t_s:           float,
    n_cells:       int     = N_CELLS,
    T_ambient:     float   = 5.0,
    C_rate_global: float | None = None,
    crate_mult:    np.ndarray | None = None,
    rngs:          list | None = None,
) -> list[EISState]:
    """
    Simulate EIS for all cells in the pack at time t_s.

    Args:
        t_s:           Simulation time (seconds).
        n_cells:       Number of cells.
        T_ambient:     Ambient temperature (°C).
        C_rate_global: Global C-rate override (overrides schedule).
        crate_mult:    Per-cell C-rate multipliers (n_cells,); default ones.
        rngs:          Per-cell RNGs; created from seeds if None.

    Returns:
        List of n_cells EISState objects.
    """
    if crate_mult is None:
        crate_mult = np.ones(n_cells)
    if rngs is None:
        rngs = [np.random.default_rng(i + 100) for i in range(n_cells)]

    def _crate(t: float, cell_i: int) -> float:
        if C_rate_global is not None:
            return C_rate_global
        if t < 60:
            base = 3.0
        elif t < 180:
            base = 2.0
        else:
            base = 1.5
        return float(np.clip(base * (0.95 + 0.05 * _R_SEI_SPREAD[cell_i]), 0.1, 4.0))

    return [
        simulate_eis(
            cell_idx   = i,
            t_s        = t_s,
            C_rate     = _crate(t_s, i) * crate_mult[i],
            T_ambient  = T_ambient,
            R_SEI_mult = _R_SEI_SPREAD[i] if i < len(_R_SEI_SPREAD) else 1.0,
            rng        = rngs[i],
        )
        for i in range(n_cells)
    ]
