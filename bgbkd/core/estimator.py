"""
bgbkd/core/estimator.py
══════════════════════════════════════════════════════════════
BGBKDAdaptiveEstimator — online α estimation and EDMD rolling update.

Orchestrates the full BGBKD update cycle at BMS loop rate:
  1. Observable lift (cascade subspace snapshot)
  2. EIS crossing detection (DRT-based)
  3. α gradient update (EIS-weighted separatrix loss)
  4. EDMD K re-fit (every update_interval seconds, rolling window)

Constitutional Mandates enforced:
  CM-3: α_est ∈ [alpha_min, alpha_max] at all times.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import deque
from typing import Optional
import numpy as np

from .observables import BGBKDObservables
from .edmd import BGBKDEDMDSolver
from ..eis.state import EISState
from ..losses.sep_loss import SeparatrixLossAccumulator


class BGBKDAdaptiveEstimator:
    """
    Online BGBKD estimator.  Maintains:
      - Rolling snapshot buffer for EDMD K re-fit.
      - EIS-weighted separatrix crossing buffer for α updates.
      - Per-cell alpha_est: converges toward true plating threshold.

    Args:
        obs:             BGBKDObservables instance (shared with detector).
        A_pack:          Pack thermal-electrical adjacency matrix (n × n).
        eta_i:           Per-cell α gradient step.
        eta_ij:          Per-edge α gradient step.
        delta:           Finite-difference step for gradient estimation.
        window:          EDMD rolling window length (number of snapshots).
        update_interval: Seconds between K re-fits.
        DRT_threshold:   γ_DRT — DRT peak crossing detection threshold.
        DRT_mu:          μ — EIS-confidence amplifier in Ω_i.
        alpha_min:       Hard lower bound on α_est.
        alpha_max:       Hard upper bound on α_est.
        lam:             Tikhonov regularisation for EDMD.
        alpha_prior:     Initial α estimates (n,).
    """

    def __init__(
        self,
        obs:              BGBKDObservables,
        A_pack:           np.ndarray,
        eta_i:            float = 0.03,
        eta_ij:           float = 0.02,
        delta:            float = 0.015,
        window:           int   = 60,
        update_interval:  float = 10.0,
        DRT_threshold:    float = 0.15,
        DRT_mu:           float = 1.5,
        alpha_min:        float = 0.10,
        alpha_max:        float = 0.75,
        lam:              float = 1e-4,
        alpha_prior:      Optional[np.ndarray] = None,
    ):
        self.obs     = obs
        self.A_pack  = A_pack
        self.eta_i   = eta_i
        self.eta_ij  = eta_ij
        self.delta   = delta
        self.window  = window
        self.update_interval = update_interval
        self.DRT_threshold   = DRT_threshold
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # Initialise α estimates
        n = obs.n_cells
        if alpha_prior is not None:
            self.alpha_est = np.clip(alpha_prior.copy().astype(float),
                                     alpha_min, alpha_max)
        else:
            self.alpha_est = np.full(n, 0.35)

        self.solver  = BGBKDEDMDSolver(lam=lam)
        self._snap   = deque(maxlen=window + 5)
        self._loss   = SeparatrixLossAccumulator(
            gamma_drt=DRT_threshold, mu=DRT_mu, maxlen=200
        )
        self._t_last_edmd = -1e9
        self._step = 0

        # Public state
        self.K: Optional[np.ndarray] = None

    @property
    def SCAL_THETA(self) -> float:
        return 0.40

    def update(
        self,
        t:          float,
        u_vec:      np.ndarray,
        eis_states: list[EISState],
    ) -> dict:
        """
        Process one BMS cycle.

        Args:
            t:          Simulation time (seconds).
            u_vec:      Pack state (n_cells,).
            eis_states: List of n_cells EISState objects.

        Returns:
            Report dict with fragility and cascade information.
        """
        self._step += 1

        # Update obs with current alpha
        self.obs.alpha_i = self.alpha_est.copy()

        # 1. Observable lift
        snap = self.obs.lift(u_vec, eis_states)
        self._snap.append(snap)

        # 2. DRT crossing detection → populate loss accumulator
        from ..losses.pack_weights import compute_centrality_weights
        W_cent = compute_centrality_weights(self.A_pack)

        for i in range(self.obs.n_cells):
            if eis_states[i].DRT_peak > self.DRT_threshold:
                self._loss.register(
                    cell=i,
                    u_cross=float(u_vec[i]),
                    DRT_peak=float(eis_states[i].DRT_peak),
                    w_i=float(W_cent[i]),
                    t=t,
                )

        # 3. α gradient update
        if self._loss.recent:
            self._update_alpha(u_vec)

        # 4. EDMD re-fit
        if (t - self._t_last_edmd >= self.update_interval
                and len(self._snap) >= 15):
            Psi = np.array(list(self._snap))
            self.solver.fit(Psi)
            self.K = self.solver.K
            self._t_last_edmd = t

        return self._build_report(t, u_vec, eis_states)

    def _update_alpha(self, u_vec: np.ndarray) -> None:
        """EIS-weighted separatrix crossing gradient step for all cells."""
        n    = self.obs.n_cells
        grad = np.zeros(n)
        A    = self.A_pack
        A_thresh = 0.05

        for ev in self._loss.recent:
            i   = ev.cell
            ucr = ev.u_cross
            Omega = ev.Omega
            w_i   = ev.w_i

            # Direct per-cell gradient
            grad[i] += -2.0 * w_i * Omega * (ucr - self.alpha_est[i])

            # Cross-cell consistency (thermal-electrical coupling)
            row_sum = A[i].sum() + 1e-9
            row_sq  = (A[i] ** 2).sum() + 1e-9
            for j in range(n):
                if j == i or A[i][j] < A_thresh:
                    continue
                Sij = A[i][j] / row_sum
                cij = A[i][j] ** 2 / row_sq
                residual = (float(u_vec[j]) - self.alpha_est[j]) - Sij * (ucr - self.alpha_est[i])
                grad[j] += -2.0 * 0.10 * cij * residual
                grad[i] +=  2.0 * 0.10 * cij * Sij * residual

        self.alpha_est = np.clip(
            self.alpha_est - self.eta_i * grad,
            self.alpha_min, self.alpha_max,
        )

    def _build_report(
        self,
        t:          float,
        u_vec:      np.ndarray,
        eis_states: list[EISState],
    ) -> dict:
        """Build the per-cycle BGBKD report dict (vectorised path)."""
        from .cascade import TwoHopCascadeDetector

        # Vectorised ψ_i — avoids Python loop overhead
        u   = np.asarray(u_vec, dtype=float)
        a   = self.alpha_est
        phi4_vec = u * (1.0 - u) * (u - a)
        psi = np.minimum(1.0, np.abs(phi4_vec) / 0.05)

        psi_max = float(psi.max())
        xi_max  = 0.0
        top_edges: list = []
        lam_dom = 0.0

        if self.K is not None:
            # Cache detector on first use; invalidate when obs changes
            if not hasattr(self, "_detector"):
                self._detector = TwoHopCascadeDetector(self.obs)
            cascade_map = self._detector.compute(self.K)
            xi_max = max((v.xi_twohop for v in cascade_map.values()), default=0.0)
            top_edges = sorted(
                cascade_map.items(),
                key=lambda x: x[1].xi_twohop, reverse=True,
            )[:5]
            top_edges = [(e, dict(
                xi=r.xi_twohop, hop1=r.hop1_init, hop2=r.hop2_comp,
                mitigation=r.mitigation,
            )) for e, r in top_edges]

            # lambda_dom: only recompute when K was just re-fit
            lam_dom = self.solver.lambda_dom

        psi_sys = psi_max + 0.40 * xi_max

        report = dict(
            t=t,
            psi=psi,
            psi_max=psi_max,
            psi_sys=psi_sys,
            xi_max=xi_max,
            lam_dom=lam_dom,
            alpha_est=self.alpha_est.copy(),
            top_edges=top_edges,
            scal_alert=psi_sys > self.SCAL_THETA,
            cell_47_u=float(u_vec[47]) if len(u_vec) > 47 else 0.0,
            cell_47_psi=float(psi[47]) if len(psi) > 47 else 0.0,
            cell_48_psi=float(psi[48]) if len(psi) > 48 else 0.0,
            cell_39_psi=float(psi[39]) if len(psi) > 39 else 0.0,
            T_47=float(eis_states[47].T_cell) if len(eis_states) > 47 else 0.0,
            DRT_47=float(eis_states[47].DRT_peak) if len(eis_states) > 47 else 0.0,
        )
        return report
