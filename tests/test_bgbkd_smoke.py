"""
tests/test_bgbkd_smoke.py
══════════════════════════════════════════════════════════════
BGBKD domain smoke tests.

9 tests covering: EIS mapping, observable lift, hard attractor
guarantees, EDMD fit, two-hop cascade, fragility bounds, simulation.

All 9 must pass before proceeding to BMS integration (Stage 3).

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
import pytest
import numpy as np

from bgbkd import (
    EISState, eis_to_u, phi4, phi3, psi_score,
    BGBKDObservables, BGBKDEDMDSolver,
    TwoHopCascadeDetector, BGBKDAdaptiveEstimator,
    PackFragility, simulate_eis, simulate_pack,
)

N_TEST = 8   # small 8-cell pack for fast tests


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def small_pack():
    """8-cell series pack topology."""
    n = N_TEST
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i][i + 1] = A[i + 1][i] = 0.8
    np.fill_diagonal(A, 0.0)
    edges      = [(i, i + 1) for i in range(n - 1)]
    alpha_init = np.full(n, 0.35)
    alpha_ij   = {(i, i + 1): 0.35 for i in range(n - 1)}
    return dict(n=n, A=A, edges=edges,
                alpha_init=alpha_init, alpha_ij=alpha_ij)


# ── Test 1 ────────────────────────────────────────────────────────


def test_eis_state_valid(small_pack):
    """eis_to_u must return u ∈ (0,1) for a typical EIS measurement."""
    eis = EISState(R_SEI=0.3, theta=0.4, W_trans=0.2, DRT_peak=0.1, T_cell=25.0)
    u = eis_to_u(eis)
    assert 0.0 < u < 1.0, f"u must be in open interval (0,1), got {u}"


# ── Test 2 ────────────────────────────────────────────────────────


def test_phi4_zeros_at_attractors(small_pack):
    """φ⁴_i must be exactly zero at both attractors regardless of α."""
    for alpha in [0.20, 0.35, 0.50]:
        assert abs(phi4(0.0, alpha)) < 1e-12, \
            f"φ⁴(0, {alpha}) = {phi4(0.0,alpha)} ≠ 0"
        assert abs(phi4(1.0, alpha)) < 1e-12, \
            f"φ⁴(1, {alpha}) = {phi4(1.0,alpha)} ≠ 0"


# ── Test 3 ────────────────────────────────────────────────────────


def test_phi3_saturation_mask(small_pack):
    """φ³_i = 0 at both attractors; φ³(0.5) = 0.25 (maximum)."""
    assert abs(phi3(0.0)) < 1e-12,  f"φ³(0) = {phi3(0.0)}"
    assert abs(phi3(1.0)) < 1e-12,  f"φ³(1) = {phi3(1.0)}"
    assert abs(phi3(0.5) - 0.25) < 1e-10, f"φ³(0.5) = {phi3(0.5)}"


# ── Test 4 ────────────────────────────────────────────────────────


def test_observable_lift_shape(small_pack):
    """Lifted observable must have dimension d_c = n + |E|."""
    p   = small_pack
    obs = BGBKDObservables(
        n_cells=p["n"], edges=p["edges"],
        alpha_i=p["alpha_init"], alpha_ij=p["alpha_ij"],
        cascade_subspace=True,
    )
    eis = [EISState(0.2, 0.3, 0.1, 0.0, 25.0)] * p["n"]
    u   = np.array([eis_to_u(e) for e in eis])
    psi = obs.lift(u, eis)
    d_c = p["n"] + len(p["edges"])
    assert psi.shape == (d_c,), f"Expected ({d_c},), got {psi.shape}"


# ── Test 5 ────────────────────────────────────────────────────────


def test_edge_observable_hard_zeros(small_pack):
    """Φ̂^→_ij = 0 when u_i = 0 or u_i = 1 (structural attractor protection)."""
    p   = small_pack
    obs = BGBKDObservables(
        n_cells=p["n"], edges=p["edges"],
        alpha_i=p["alpha_init"], alpha_ij=p["alpha_ij"],
        cascade_subspace=True,
    )
    eis = [EISState(0.5, 0.5, 0.5, 0.2, 30.0)] * p["n"]

    for u_i_val in [0.0, 1.0]:
        u = np.full(p["n"], 0.5)
        u[0] = u_i_val
        psi = obs.lift(u, eis)
        edge_col = p["n"] + 0   # edge (0,1) is first edge; column = n+0
        assert abs(psi[edge_col]) < 1e-10, \
            f"Φ̂^→_01 ≠ 0 at u_0={u_i_val}: psi[{edge_col}]={psi[edge_col]}"


# ── Test 6 ────────────────────────────────────────────────────────


def test_edmd_fit(small_pack):
    """EDMD solver must return finite K of correct shape from 40 snapshots."""
    p   = small_pack
    obs = BGBKDObservables(
        n_cells=p["n"], edges=p["edges"],
        alpha_i=p["alpha_init"], alpha_ij=p["alpha_ij"],
        cascade_subspace=True,
    )
    rng = np.random.default_rng(42)
    T   = 40
    d_c = p["n"] + len(p["edges"])
    u   = np.full(p["n"], 0.6)
    eis = [EISState(0.2, 0.3, 0.1, 0.0, 25.0)] * p["n"]
    Psi = np.zeros((T, d_c))
    for t in range(T):
        Psi[t] = obs.lift(u, eis)
        u = np.clip(u + rng.normal(0, 0.02, p["n"]), 0.05, 0.95)

    solver = BGBKDEDMDSolver(lam=1e-4).fit(Psi)
    assert solver.K is not None
    assert solver.K.shape == (d_c, d_c)
    assert np.all(np.isfinite(solver.K)), "K contains NaN or Inf"


# ── Test 7 ────────────────────────────────────────────────────────


def test_two_hop_cascade(small_pack):
    """Two-hop scores must be non-negative for every edge."""
    p    = small_pack
    obs  = BGBKDObservables(
        n_cells=p["n"], edges=p["edges"],
        alpha_i=p["alpha_init"], alpha_ij=p["alpha_ij"],
        cascade_subspace=True,
    )
    rng  = np.random.default_rng(7)
    d_c  = p["n"] + len(p["edges"])
    eis  = [EISState(0.2, 0.3, 0.1, 0.0, 25.0)] * p["n"]
    Psi  = np.array([
        obs.lift(
            np.clip(np.full(p["n"], 0.55) + rng.normal(0, 0.05, p["n"]), 0.05, 0.95),
            eis,
        )
        for _ in range(35)
    ])
    solver   = BGBKDEDMDSolver(lam=1e-4).fit(Psi)
    detector = TwoHopCascadeDetector(obs)
    reports  = detector.compute(solver.K)

    assert len(reports) == len(p["edges"]), \
        f"Expected {len(p['edges'])} reports, got {len(reports)}"
    for r in reports.values():
        assert r.xi_twohop  >= 0, f"xi_twohop < 0: {r.xi_twohop}"
        assert r.hop1_init  >= 0, f"hop1_init < 0: {r.hop1_init}"
        assert r.hop2_comp  >= 0, f"hop2_comp < 0: {r.hop2_comp}"


# ── Test 8 ────────────────────────────────────────────────────────


def test_psi_score_bounds(small_pack):
    """ψ_i must be ∈ [0,1] for any u ∈ (0,1)."""
    for u in [0.05, 0.25, 0.35, 0.50, 0.65, 0.75, 0.95]:
        score = psi_score(u, alpha=0.35)
        assert 0.0 <= score <= 1.0, \
            f"ψ out of [0,1] at u={u}: {score}"


# ── Test 9 ────────────────────────────────────────────────────────


def test_simulate_eis_runs(small_pack):
    """simulate_eis must return a valid EISState with channels in range."""
    eis = simulate_eis(
        cell_idx=0, t_s=120.0, C_rate=2.0,
        T_ambient=5.0, R_SEI_mult=1.1,
    )
    assert 0.0 <= eis.R_SEI  <= 1.0, f"R_SEI={eis.R_SEI}"
    assert 0.0 <= eis.theta  <= 1.0, f"theta={eis.theta}"
    assert 0.0 <= eis.W_trans <= 1.0, f"W_trans={eis.W_trans}"
    assert eis.DRT_peak >= 0.0,       f"DRT_peak={eis.DRT_peak}"
    assert eis.T_cell   >= 5.0,       f"T_cell={eis.T_cell} below ambient"
