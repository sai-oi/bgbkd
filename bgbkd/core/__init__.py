"""bgbkd.core — Observables, EDMD solver, cascade detector, estimator."""
from .observables import (
    phi3, phi4, psi_score,
    gamma_ij, alpha_hat_ij, phi_hat_ij,
    BGBKDObservables,
)
from .edmd import BGBKDEDMDSolver, matrix_power_safe
from .cascade import CascadeReport, TwoHopCascadeDetector, PackFragility
from .estimator import BGBKDAdaptiveEstimator

__all__ = [
    "phi3", "phi4", "psi_score",
    "gamma_ij", "alpha_hat_ij", "phi_hat_ij",
    "BGBKDObservables",
    "BGBKDEDMDSolver", "matrix_power_safe",
    "CascadeReport", "TwoHopCascadeDetector", "PackFragility",
    "BGBKDAdaptiveEstimator",
]
