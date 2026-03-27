"""
bgbkd/application/bms_pipeline.py
══════════════════════════════════════════════════════════════
BGBKD BMS Pipeline — main integration loop.

Orchestrates:
  1. EIS acquisition from all cells  (per BMS cycle)
  2. EIS → u_i mapping               (EISState → BGBKD state)
  3. BGBKD observable lift           (cascade subspace)
  4. Adaptive α estimation           (rolling EDMD + separatrix gradient)
  5. Two-hop cascade scoring         (K matrix → ξ^(2))
  6. Fragility scoring               (ψ_i, Ψ_sys)
  7. SCAL alert and mitigation dispatch

SAI-OI Constitutional Mandates enforced:
  CM-1: Ψ_sys computed on every EIS acquisition cycle.
  CM-2: SCAL alert raised before cascade completes (ξ-based, not voltage-based).
  CM-3: α estimates within [alpha_min, alpha_max] at all times.
  CM-4: Mitigation dispatch latency ≤ 1 BMS cycle.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("bms.bgbkd")


class BMSPipeline:
    """
    BGBKD-augmented BMS pipeline for a 96-cell fast-charge EV pack.

    Args:
        config_path:   Path to YAML config file.
        synthetic_eis: Use synthetic EIS data (True) or BQ40Z80 hardware (False).
        bms_rate_hz:   BMS loop frequency (Hz).
    """

    def __init__(
        self,
        config_path:   str   = "config/bgbkd_96cell_nmc.yaml",
        synthetic_eis: bool  = True,
        bms_rate_hz:   float = 1.0,
    ):
        logger.info("Initialising BGBKD BMS pipeline...")

        from ..infrastructure.config_loader import load_bgbkd
        from ..infrastructure.bq40z80_adapter import BQ40Z80Adapter, BQ40Z80Config

        bgbkd = load_bgbkd(config_path)
        self.estimator      = bgbkd["estimator"]
        self.detector       = bgbkd["detector"]
        self.fragility      = bgbkd["fragility"]
        self.scal_threshold = bgbkd["scal_threshold"]
        self.n_cells        = bgbkd["n_cells"]
        self.config         = bgbkd["config_yaml"]

        self.eis_adapter = BQ40Z80Adapter(
            cfg       = BQ40Z80Config(),
            synthetic = synthetic_eis,
        )
        self.cal = self._load_calibration(config_path)

        self._step        = 0
        self._scal_active = False
        self._last_report: Optional[dict] = None
        self._cycle_s     = 1.0 / bms_rate_hz

        logger.info(
            f"BGBKD pipeline ready. n={self.n_cells} cells, "
            f"SCAL Θ={self.scal_threshold}"
        )

    def _load_calibration(self, config_path: str) -> dict:
        import yaml, json
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        with open(cfg["eis"]["calibration_file"]) as f:
            return json.load(f)

    def step(self) -> dict:
        """
        Execute one BMS cycle.

        Returns:
            Report dict with keys:
              psi, psi_sys, scal_alert, top_edges,
              alpha_est, lambda_dom, eis_states, u_vec,
              inference_ms.
        """
        t0 = time.perf_counter()
        self._step += 1

        # 1. EIS acquisition
        eis_states = [
            self.eis_adapter.read_eis(i, self.cal)
            for i in range(self.n_cells)
        ]

        # 2. Map EIS → u_i
        from ..eis.state import eis_to_u
        u_vec = np.array([eis_to_u(e) for e in eis_states])

        # 3–5. BGBKD update (lift, EDMD, cascade, α)
        t_s = self._step * self._cycle_s
        report = self.estimator.update(t_s, u_vec, eis_states)

        # 6. SCAL alerting (CM-2)
        scal_now = report["scal_alert"]
        if scal_now and not self._scal_active:
            logger.critical(
                f"SCAL ALERT step={self._step} "
                f"Ψ_sys={report['psi_sys']:.4f} > Θ={self.scal_threshold}"
            )
            for (i, j), r in report["top_edges"][:3]:
                logger.critical(
                    f"  CASCADE: Cell{i}→Cell{j}  "
                    f"ξ={r['xi']:.2e}  {r['mitigation']}"
                )
        self._scal_active = scal_now

        dt_ms = (time.perf_counter() - t0) * 1000
        report["eis_states"]   = eis_states
        report["u_vec"]        = u_vec
        report["inference_ms"] = dt_ms

        self._last_report = report
        return report

    def run(
        self,
        n_steps:    Optional[int]   = None,
        max_time_s: Optional[float] = None,
    ) -> None:
        """
        Run the BMS pipeline loop.

        Args:
            n_steps:    Stop after this many steps (None = run indefinitely).
            max_time_s: Stop after this many wall-clock seconds.
        """
        logger.info(f"Starting BGBKD pipeline. Rate={1/self._cycle_s:.0f} Hz")
        step = 0
        t_start = time.time()

        while True:
            t_cycle = time.time()
            report  = self.step()

            if step % 30 == 0:
                logger.info(
                    f"step={step:6d}  Ψ_sys={report['psi_sys']:.4f}  "
                    f"ψ_max={report['psi_max']:.4f}  "
                    f"SCAL={'ON' if report['scal_alert'] else 'off'}  "
                    f"inference={report['inference_ms']:.2f}ms"
                )

            step += 1
            if n_steps and step >= n_steps:
                break
            if max_time_s and (time.time() - t_start) >= max_time_s:
                break

            elapsed = time.time() - t_cycle
            sleep_s = max(0.0, self._cycle_s - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        logger.info(f"Pipeline stopped at step {step}.")
