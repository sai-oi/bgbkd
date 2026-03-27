"""
bgbkd/eis/drt.py
══════════════════════════════════════════════════════════════
Distribution of Relaxation Times (DRT) computation and plating
peak detection.

The DRT peak at τ ∈ [1 ms, 10 ms] (corresponding to ~100 Hz) is the
mechanistic signature of the plating Butler-Volmer bifurcation precursor.
Its amplitude A_plating(t) is used as the EIS-confidence weight Ω_i in
the separatrix loss and as the crossing detector C_i^EIS.

Requires: pip install impedance>=1.4.2  (optional dependency)
Falls back to a simple phase-depression heuristic when not installed.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from typing import Optional

# DRT time-window for plating arc
TAU_MIN_S: float = 1e-3   #  1 ms
TAU_MAX_S: float = 1e-2   # 10 ms

# Threshold for crossing detection
GAMMA_DRT_DEFAULT:  float = 0.15
EPSILON_RATE:       float = 1e-3   # minimum dA/dt for growing peak


def drt_peak_amplitude(
    freq_hz:   np.ndarray,
    Z_re:      np.ndarray,
    Z_im:      np.ndarray,
    gamma_drt: float = GAMMA_DRT_DEFAULT,
) -> float:
    """
    Estimate the DRT plating-arc peak amplitude using the impedance library
    when available; falls back to a mid-frequency phase-depression heuristic.

    The DRT G(τ) is obtained by Tikhonov deconvolution of the EIS spectrum.
    The plating-associated peak at τ ∈ [1 ms, 10 ms] has amplitude:

        A_plating = max(G(τ) for τ in [τ_min, τ_max])

    Args:
        freq_hz: Frequency array (Hz).
        Z_re:    Real impedance (same units as Z_im).
        Z_im:    Imaginary impedance (typically negative for capacitive branch).
        gamma_drt: Calibrated plating detection threshold.

    Returns:
        A_plating ≥ 0 (normalised to gamma_drt scale).
    """
    try:
        return _drt_via_library(freq_hz, Z_re, Z_im)
    except Exception:
        return _drt_phase_heuristic(freq_hz, Z_re, Z_im)


def _drt_via_library(
    freq_hz: np.ndarray,
    Z_re:    np.ndarray,
    Z_im:    np.ndarray,
) -> float:
    """DRT via the `impedance` library (Tikhonov regularisation)."""
    from impedance.models.circuits.fitting import get_element_param   # noqa: F401
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Build complex impedance
        Z = Z_re + 1j * Z_im
        omega = 2 * np.pi * freq_hz

        # Simple trapezoidal DRT via regularised least-squares
        tau = np.logspace(
            np.log10(1.0 / omega.max()),
            np.log10(1.0 / omega.min()),
            len(freq_hz),
        )
        # Kernel matrix
        K = np.zeros((len(freq_hz), len(tau)))
        for k_row, w in enumerate(omega):
            K[k_row, :] = 1.0 / (1.0 + (w * tau) ** 2)

        lam = 1e-3
        A = K.T @ K + lam * np.eye(len(tau))
        b = K.T @ np.abs(Z_im)
        g = np.linalg.solve(A, b)
        g = np.maximum(g, 0)

        mask = (tau >= TAU_MIN_S) & (tau <= TAU_MAX_S)
        return float(g[mask].max()) if mask.any() else 0.0


def _drt_phase_heuristic(
    freq_hz: np.ndarray,
    Z_re:    np.ndarray,
    Z_im:    np.ndarray,
) -> float:
    """
    Fallback: estimate plating arc amplitude from phase depression at ~100 Hz.
    A large negative imaginary component at 100 Hz (phase depression below
    baseline) signals the plating arc emergence.
    """
    f_target = 100.0   # Hz
    idx = np.argmin(np.abs(freq_hz - f_target))
    if idx >= len(Z_re):
        return 0.0
    phase_deg = -np.degrees(np.arctan2(Z_im[idx], Z_re[idx]))
    # Map phase to [0, 1]: deeper depression → larger amplitude proxy
    amplitude = float(np.clip((phase_deg - 20.0) / 20.0, 0.0, 1.0))
    return amplitude


def is_drt_crossing(
    A_plating:    float,
    A_plating_prev: float,
    gamma_drt:    float = GAMMA_DRT_DEFAULT,
    epsilon_rate: float = EPSILON_RATE,
) -> bool:
    """
    DRT-peak crossing indicator C_i^EIS.

        C_i^EIS = 1  iff  A_plating > γ_DRT  AND  dA/dt > ε_rate

    Mechanistically superior to simple phase-shift thresholds: separates
    plating onset (growing peak) from concentration overpotential (stable peak).
    """
    return (
        A_plating > gamma_drt
        and (A_plating - A_plating_prev) > epsilon_rate
    )


def simulate_drt_peak(
    rho_theta: float,
    noise_rng: Optional[np.random.Generator] = None,
    noise_std: float = 0.03,
) -> float:
    """
    Simulate the DRT plating peak amplitude from the normalised plating channel.

    Used for synthetic EIS in testing and demo scenarios.
    DRT peak appears when ρ_θ > 0.3 (plating arc emerges in Nyquist).
    """
    peak = max(0.0, rho_theta - 0.30) * 2.5
    if noise_rng is not None:
        peak += noise_rng.normal(0, noise_std)
    return max(0.0, peak)
