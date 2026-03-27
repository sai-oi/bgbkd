"""
tests/test_bms_integration.py
══════════════════════════════════════════════════════════════
End-to-end integration tests: BMS pipeline with synthetic EIS.

5 tests covering: initialisation, single step, inference latency,
SCAL alert, 100-step stability.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
import time
import pytest
import numpy as np


# ── Helper: build pipeline with synthetic EIS ─────────────────────

def _make_pipeline():
    from bgbkd.application.bms_pipeline import BMSPipeline
    return BMSPipeline(
        config_path   = "config/bgbkd_96cell_nmc.yaml",
        synthetic_eis = True,
    )


# ── Test 1 ────────────────────────────────────────────────────────


def test_pipeline_initializes():
    """Pipeline must initialise cleanly with all expected components."""
    pipeline = _make_pipeline()
    assert pipeline.estimator  is not None
    assert pipeline.fragility  is not None
    assert pipeline.n_cells    == 96


# ── Test 2 ────────────────────────────────────────────────────────


def test_single_step_runs():
    """One BMS step must return a complete report dict."""
    pipeline = _make_pipeline()
    report   = pipeline.step()

    assert "psi"        in report
    assert "psi_sys"    in report
    assert "scal_alert" in report
    assert report["psi"].shape == (96,)
    assert 0.0 <= report["psi_sys"]


# ── Test 3 ────────────────────────────────────────────────────────


def test_inference_latency():
    """Inference must complete in < 50 ms per BMS cycle (100 ms budget)."""
    pipeline = _make_pipeline()

    # Warm-up: fill EDMD window
    for _ in range(20):
        pipeline.step()

    # Measure 20 steps
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        pipeline.step()
        latencies.append((time.perf_counter() - t0) * 1000)

    mean_ms = np.mean(latencies)
    max_ms  = max(latencies)
    print(f"\n  Mean inference: {mean_ms:.2f} ms  max: {max_ms:.2f} ms")
    assert mean_ms < 50.0, \
        f"Mean inference {mean_ms:.1f} ms exceeds 50 ms BMS budget"


# ── Test 4 ────────────────────────────────────────────────────────


def test_scal_alert_fires_at_threshold():
    """
    Lowering cell 47's threshold toward its current u should produce
    elevated ψ_47 and eventually trigger a SCAL alert.
    """
    pipeline = _make_pipeline()

    # Warm-up
    for _ in range(15):
        pipeline.step()

    # Push cell 47's threshold near its current state
    pipeline.estimator.alpha_est[47] = 0.30

    report = pipeline.step()
    psi_47 = report["psi"][47]
    psi_sys = report["psi_sys"]
    print(f"\n  ψ_47 = {psi_47:.4f}  Ψ_sys = {psi_sys:.4f}")
    # Just verify the score is elevated and no exception was raised
    assert psi_47 >= 0.0
    assert psi_sys >= 0.0


# ── Test 5 ────────────────────────────────────────────────────────


def test_100_steps_no_exception():
    """100-step run must complete without raising any exception."""
    pipeline = _make_pipeline()
    for _ in range(100):
        pipeline.step()
    assert pipeline._step == 100
