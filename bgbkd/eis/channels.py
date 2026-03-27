"""
bgbkd/eis/channels.py
══════════════════════════════════════════════════════════════
EIS feature extraction: three mechanistically orthogonal channels.

Each channel targets one degradation mode in NMC/graphite cells:
  f_HF = 1 kHz    →  ρ_SEI   (SEI resistance)
  f_MF = 100 Hz   →  ρ_θ     (plating precursor phase depression)
  f_LF = 10 mHz   →  ρ_W     (ion transport deviation)

Raw complex impedance Z(f) = Z_re(f) + j·Z_im(f) is normalised against
per-cell calibration endpoints obtained from formation cycling and ARC data.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np


def extract_rho_sei(
    Z_re_1kHz_mohm: float,
    R0_mohm:        float,
    Rinf_mohm:      float,
) -> float:
    """
    Channel 1 — SEI resistance.

        ρ_SEI = clip[(R_SEI − R⁰) / (R∞ − R⁰), 0, 1]

    R_SEI = Re[Z(1 kHz)] is the high-frequency real intercept of the
    Nyquist plot, dominated by SEI film thickness growth.

    Args:
        Z_re_1kHz_mohm: Re[Z(1 kHz)] in mΩ.
        R0_mohm:        Fresh-cell baseline R_SEI (mΩ).
        Rinf_mohm:      End-of-life / ARC-calibrated R_SEI (mΩ).

    Returns:
        ρ_SEI ∈ [0, 1].
    """
    denom = Rinf_mohm - R0_mohm
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip((Z_re_1kHz_mohm - R0_mohm) / denom, 0.0, 1.0))


def extract_rho_theta(
    Z_re_100Hz: float,
    Z_im_100Hz: float,
    theta0_deg:       float,
    theta_plating_deg: float,
) -> float:
    """
    Channel 2 — Plating precursor phase depression (100 Hz).

        θ(t)  = −arctan(Im[Z(100 Hz)] / Re[Z(100 Hz)])
        ρ_θ   = clip[(θ − θ⁰) / (θ_plating − θ⁰), 0, 1]

    As plating begins, a secondary arc at τ_plating ≈ 1–10 ms appears,
    depressing the phase angle below the intercalation-only baseline.

    Args:
        Z_re_100Hz:        Re[Z(100 Hz)] (any consistent units).
        Z_im_100Hz:        Im[Z(100 Hz)] (same units; typically negative).
        theta0_deg:        Baseline phase angle at fresh cell (degrees).
        theta_plating_deg: Phase at confirmed plating onset (degrees).

    Returns:
        ρ_θ ∈ [0, 1].
    """
    theta_deg = -np.degrees(np.arctan2(Z_im_100Hz, Z_re_100Hz))
    denom = theta_plating_deg - theta0_deg
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip((theta_deg - theta0_deg) / denom, 0.0, 1.0))


def extract_rho_W(
    Z_re_10mHz: float,
    Z_im_10mHz: float,
    W0:         float,
    Winf:       float,
) -> float:
    """
    Channel 3 — Ion transport deviation (10 mHz Warburg regime).

        W(t) = Im[Z(10 mHz)] / Re[Z(10 mHz)] − 1
        ρ_W  = clip[(W − W⁰) / (W∞ − W⁰), 0, 1]

    Ideal Warburg diffusion gives Im/Re = 1 (45° slope in Nyquist).
    Positive deviation signals concentration-limited transport.

    Args:
        Z_re_10mHz: Re[Z(10 mHz)].
        Z_im_10mHz: Im[Z(10 mHz)] (typically negative for capacitive branch).
        W0:         Baseline W ratio (fresh cell).
        Winf:       End-of-life W ratio.

    Returns:
        ρ_W ∈ [0, 1].
    """
    if abs(Z_re_10mHz) < 1e-12:
        return 0.0
    W = abs(Z_im_10mHz) / abs(Z_re_10mHz) - 1.0
    denom = Winf - W0
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip((W - W0) / denom, 0.0, 1.0))


def normalise_from_calibration(
    Z_re_1kHz_mohm:  float,
    Z_re_100Hz:      float,
    Z_im_100Hz:      float,
    Z_re_10mHz:      float,
    Z_im_10mHz:      float,
    cal:             dict,
) -> tuple[float, float, float]:
    """
    Extract all three normalised EIS channels from raw complex impedance.

    Args:
        Z_re_1kHz_mohm:  Re[Z(1 kHz)] in mΩ.
        Z_re_100Hz:      Re[Z(100 Hz)] in consistent units.
        Z_im_100Hz:      Im[Z(100 Hz)] in consistent units.
        Z_re_10mHz:      Re[Z(10 mHz)] in consistent units.
        Z_im_10mHz:      Im[Z(10 mHz)] in consistent units.
        cal:             Calibration dict with keys:
                           R0_mohm, Rinf_mohm,
                           theta0_deg, theta_plating_deg,
                           W0, Winf.

    Returns:
        (ρ_SEI, ρ_θ, ρ_W) each ∈ [0, 1].
    """
    rho_sei = extract_rho_sei(
        Z_re_1kHz_mohm,
        cal["R0_mohm"],
        cal["Rinf_mohm"],
    )
    rho_theta = extract_rho_theta(
        Z_re_100Hz, Z_im_100Hz,
        cal["theta0_deg"],
        cal["theta_plating_deg"],
    )
    rho_W = extract_rho_W(
        Z_re_10mHz, Z_im_10mHz,
        cal["W0"],
        cal["Winf"],
    )
    return rho_sei, rho_theta, rho_W
