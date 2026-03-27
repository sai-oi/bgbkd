"""
bgbkd/core/edmd.py
══════════════════════════════════════════════════════════════
Rolling EDMD (Extended Dynamic Mode Decomposition) solver.

Fits the Koopman operator K* from a rolling window of observable
snapshots via Tikhonov-regularised least squares:

    K* = (ΨX^T ΨX + λI)^{-1} ΨX^T ΨY

where ΨX = snapshots[:-1], ΨY = snapshots[1:].

k-step prediction: Ψ(t+k) ≈ Ψ(t) · K^k  (via matrix exponentiation)
Cost: O(d_c² log k) — compared to O(k·DFN) for physics-based models.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from numpy.linalg import solve, eigvals


class BGBKDEDMDSolver:
    """
    Tikhonov-regularised EDMD solver for the BGBKD Koopman operator.

    Args:
        lam: Tikhonov regularisation parameter λ (default 1e-4).
             Increase to 1e-3 if the window is too small (W < 3·d_c).
    """

    def __init__(self, lam: float = 1e-4):
        self.lam = lam
        self.K:  Optional[np.ndarray] = None    # Koopman operator
        self._d: Optional[int]        = None    # observable dimension

    def fit(self, Psi: np.ndarray) -> "BGBKDEDMDSolver":
        """
        Fit K* from snapshot matrix Ψ ∈ ℝ^{T × d_c}.

        Requires T ≥ 2. For reliable estimation T ≥ 3·d_c (Tikhonov
        regularisation stabilises for T ≥ d_c).

        Args:
            Psi: Snapshot matrix, rows are consecutive observable
                 evaluations Ψ(t), Ψ(t+1), ..., Ψ(t+T−1).

        Returns:
            self (for chaining).
        """
        if Psi.shape[0] < 2:
            return self

        PsiX = Psi[:-1]   # (T-1, d)
        PsiY = Psi[1:]    # (T-1, d)
        d = PsiX.shape[1]

        A = PsiX.T @ PsiX + self.lam * np.eye(d)
        self.K = solve(A, PsiX.T @ PsiY)
        self._d = d
        self._eigvals_cache = None   # invalidate cached eigenvalues
        return self

    def predict(self, psi_now: np.ndarray, k: int = 1) -> np.ndarray:
        """
        k-step observable prediction via K^k.

        Uses repeated squaring: cost O(d² log k).

        Args:
            psi_now: Current observable Ψ(t) ∈ ℝ^{d_c}.
            k:       Prediction horizon (steps).

        Returns:
            Ψ(t+k) ∈ ℝ^{d_c}.
        """
        if self.K is None:
            return psi_now.copy()
        Kk = matrix_power_safe(self.K, k)
        return psi_now @ Kk

    def koopman_eigenvalues(self) -> np.ndarray:
        """
        Eigenvalues of the fitted K matrix.

        The dominant eigenvalue λ_dom = max|λ| signals critical slowing:
          λ_dom > 1.0 → the cascade subspace is no longer dissipative;
                        perturbations amplify rather than decay.

        Cached: only recomputed after fit() is called, not every BMS cycle.
        """
        if self.K is None:
            return np.array([0.0])
        if not hasattr(self, "_eigvals_cache") or self._eigvals_cache is None:
            self._eigvals_cache = eigvals(self.K)
        return self._eigvals_cache

    @property
    def lambda_dom(self) -> float:
        """Dominant Koopman eigenvalue magnitude (cached between re-fits)."""
        return float(np.max(np.abs(self.koopman_eigenvalues())))

    @property
    def is_fitted(self) -> bool:
        return self.K is not None


def matrix_power_safe(M: np.ndarray, k: int) -> np.ndarray:
    """
    Compute M^k via repeated squaring. Clips spectral radius to prevent
    exponential blow-up during transient calibration windows.

    For k ≤ 30 and d_c = 263: 5 matrix multiplications (log2(30) ≈ 5).
    """
    if k <= 0:
        return np.eye(M.shape[0])
    if k == 1:
        return M.copy()

    # Stabilise: scale if spectral radius > 1.05
    lam_max = np.max(np.abs(eigvals(M)))
    if lam_max > 1.05:
        M = M / lam_max * 1.02   # rescale to just above unit circle

    result = np.eye(M.shape[0])
    base   = M.copy()
    while k > 0:
        if k % 2 == 1:
            result = result @ base
        base = base @ base
        k //= 2
    return result
