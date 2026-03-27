"""bgbkd.eis — EIS state variables, channel extraction, DRT detection."""
from .state import EISState, eis_to_u, W_SEI, W_THETA, W_W, BETA, U_MIN, U_MAX
from .channels import (
    extract_rho_sei, extract_rho_theta, extract_rho_W,
    normalise_from_calibration,
)
from .drt import (
    drt_peak_amplitude, is_drt_crossing, simulate_drt_peak,
    GAMMA_DRT_DEFAULT, TAU_MIN_S, TAU_MAX_S,
)

__all__ = [
    "EISState", "eis_to_u",
    "W_SEI", "W_THETA", "W_W", "BETA", "U_MIN", "U_MAX",
    "extract_rho_sei", "extract_rho_theta", "extract_rho_W",
    "normalise_from_calibration",
    "drt_peak_amplitude", "is_drt_crossing", "simulate_drt_peak",
    "GAMMA_DRT_DEFAULT", "TAU_MIN_S", "TAU_MAX_S",
]
