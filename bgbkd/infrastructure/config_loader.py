"""
bgbkd/infrastructure/config_loader.py
══════════════════════════════════════════════════════════════
Load BGBKD runtime configuration from YAML and instantiate
all domain objects.

Returns a single dict (the "bgbkd bundle") consumed by BMSPipeline
and the demo scripts.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Optional

# ── Public entry point ────────────────────────────────────────────


def load_bgbkd(config_path: str = "config/bgbkd_96cell_nmc.yaml") -> dict:
    """
    Load config and instantiate all BGBKD objects.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dict with keys:
          obs, solver, estimator, detector, fragility,
          scal_threshold, n_cells, edges, A_pack, config_yaml.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("pip install pyyaml") from exc

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from ..core.observables import BGBKDObservables
    from ..core.edmd import BGBKDEDMDSolver
    from ..core.cascade import TwoHopCascadeDetector, PackFragility
    from ..core.estimator import BGBKDAdaptiveEstimator

    # ── Pack topology ─────────────────────────────────────────────
    A_pack = np.load(cfg["pack"]["topology_file"])
    with open(cfg["pack"]["edges_file"]) as f:
        edges = [tuple(e) for e in json.load(f)]
    n = cfg["pack"]["n_cells"]

    # ── Calibration ───────────────────────────────────────────────
    with open(cfg["eis"]["calibration_file"]) as f:
        cal = json.load(f)

    alpha_prior = np.load(cfg["eis"]["alpha_prior_file"])

    # ── Edge thresholds ───────────────────────────────────────────
    alpha_ij = {
        (i, j): float(np.clip(
            alpha_prior[j] + 0.15 * A_pack[i][j], 0.10, 0.75
        ))
        for (i, j) in edges
    }

    # ── Observables ───────────────────────────────────────────────
    obs = BGBKDObservables(
        n_cells          = n,
        edges            = edges,
        alpha_i          = alpha_prior.copy(),
        alpha_ij         = alpha_ij,
        cascade_subspace = True,
        calibration      = cal,
        eis_cfg          = cfg.get("eis", {}),
        edge_cfg         = cfg.get("edge_observable", {}),
    )

    solver   = BGBKDEDMDSolver(lam=cfg["edmd"]["lam"])
    detector = TwoHopCascadeDetector(obs)

    frag_cfg  = cfg.get("fragility", {})
    fragility = PackFragility(
        obs,
        detector,
        phi_max = frag_cfg.get("phi_max", 0.05),
        gamma   = frag_cfg.get("gamma",   0.40),
    )

    ada_cfg   = cfg.get("adaptive_alpha", {})
    estimator = BGBKDAdaptiveEstimator(
        obs             = obs,
        A_pack          = A_pack,
        eta_i           = ada_cfg.get("eta_i",         0.03),
        eta_ij          = ada_cfg.get("eta_ij",        0.02),
        delta           = ada_cfg.get("delta",         0.015),
        window          = cfg["edmd"].get("window",    60),
        update_interval = cfg["edmd"].get("update_interval_s", 10.0),
        DRT_threshold   = ada_cfg.get("DRT_threshold", 0.15),
        DRT_mu          = ada_cfg.get("DRT_mu",        1.5),
        alpha_min       = ada_cfg.get("alpha_min",     0.10),
        alpha_max       = ada_cfg.get("alpha_max",     0.75),
        lam             = cfg["edmd"].get("lam",       1e-4),
        alpha_prior     = alpha_prior,
    )

    print(f"BGBKD loaded: n={n}  |E|={len(edges)}  d_c={obs.dim}")

    return dict(
        obs            = obs,
        solver         = solver,
        estimator      = estimator,
        detector       = detector,
        fragility      = fragility,
        scal_threshold = cfg["scal"]["threshold"],
        n_cells        = n,
        edges          = edges,
        A_pack         = A_pack,
        config_yaml    = cfg,
    )
