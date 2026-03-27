# BGBKD — Battery Graph Bistable Koopman Dictionary

**Pack-Level Thermal Runaway Cascade Forecasting — 226 Seconds Before Voltage Trigger**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![SAI-OI](https://img.shields.io/badge/SAI--OI-ROIS-22d3ee.svg)](https://saicommand.com)


---

## What It Does

BGBKD fits a Koopman operator to per-cell EIS (Electrochemical Impedance Spectroscopy) data and produces, at BMS loop rate, a complete pack-level cascade propagation map:

- **ψ_i** — per-cell plating/runaway fragility score; proximity to tipping threshold
- **ξ^(2)_ij** — two-hop cascade score for every cell-to-cell thermal-electrical edge; decomposed into initiation (cell i → derate) and completion (cell j → pre-cool/isolate)
- **λ_dom** — dominant Koopman eigenvalue; critical slowing down signal
- **Ψ_sys** — system-level alert trigger

**96-cell NMC/graphite pack, 150 kW fast charge, 5°C:**

| Signal | Time | Lead Over Voltage Trigger |
|--------|------|--------------------------|
| BGBKD structural fragility (ψ > 0.35) | t = 0 s | **226 seconds** |
| BGBKD SCAL alert (Ψ_sys > 0.40) | t = 3 s | 223 seconds |
| BGBKD cascade map active | t = 14 s | 212 seconds |
| Classical voltage plateau (2.5V) | t = 237 s | — |
| Classical gas sensor | Not triggered | — |

Full 30-second pack cascade forecast (all 96 cells, all 167 edges): **1.9 ms on ARM Cortex-M7**.

---

## The Core Idea

Lithium plating onset is a **fold bifurcation** in Butler-Volmer kinetics at the plating overpotential — the same saddle-node structure at the heart of GBKD. BGBKD maps three mechanistically orthogonal EIS channels into a single degradation stress metric:

```
s_i = 0.25·ρ_SEI  +  0.55·ρ_θ  +  0.20·ρ_W
      └─ 1 kHz ─┘   └─ 100 Hz ─┘  └─ 10 mHz ─┘
      SEI growth    plating arc    transport dev.

u_i = 1 − σ_β(s_i − s̄)         ∈ [0, 1]
```

The 100 Hz channel (phase depression from plating DRT arc) dominates because it directly tracks the Butler-Volmer bifurcation precursor — not a proxy, the mechanistic source.

The EIS-augmented edge observable:

```
Φ̂^→_ij = u_i(1−u_i) · Γ_ij · (u_j − α̂_ij)
```

is **structurally zero at both attractors** regardless of differential impedance Γ_ij or edge threshold α̂_ij. Γ_ij amplifies the edge when differential impedance at thermal frequencies signals current redistribution between cells — earlier than temperature sensors register the gradient.

---

## Quickstart

```bash
git clone https://github.com/sai-oi/bgbkd
cd bgbkd
pip install -r requirements.txt
python -m bgbkd.demo.ev_fastcharge
```

```python
from bgbkd import BGBKDConfig, BGBKDEstimator
from bgbkd.eis import EISState, eis_to_u

# Load pack configuration
cfg = BGBKDConfig.from_yaml("config/bgbkd_96cell_nmc.yaml")

# Initialize estimator
estimator = BGBKDEstimator(cfg)

# BMS loop — 1 Hz
for eis_measurements in eis_stream:        # list of 96 EISState objects
    u_vec   = [eis_to_u(e) for e in eis_measurements]
    report  = estimator.update(t, u_vec, eis_measurements)

    if report["scal_alert"]:
        print(f"SCAL ALERT — Ψ_sys={report['psi_sys']:.3f}")
        for (i, j), r in report["top_edges"][:3]:
            print(f"  Cell {i}→{j}:  ξ={r.xi:.2e}  → {r.mitigation}")
```

---

## Installation

**Requirements:** Python 3.10+, numpy ≥ 1.24, scipy ≥ 1.11

```bash
pip install numpy scipy pandas
pip install -e .
```

**Optional (for DRT computation):**
```bash
pip install impedance>=1.4.2
```

**Build pack topology:**
```bash
python scripts/build_pack_topology.py --n_cells 96 --n_modules 12
# → data/pack_topology/A_pack_96cell.npy
# → data/pack_topology/edges_96cell.json
```

**Run smoke tests:**
```bash
pytest tests/ -v
# 9 tests, ~2 seconds
# Includes: hard zero guarantee, EIS mapping, EDMD fit, cascade detector
```

---

## Repository Structure

```
bgbkd/
├── bgbkd/
│   ├── core/
│   │   ├── observables.py       EIS composite u_i, BKD dictionary, Φ̂^→_ij
│   │   ├── edmd.py              Rolling EDMD solver
│   │   ├── estimator.py         Online α estimator, EIS-weighted loss
│   │   └── cascade.py           Two-hop cascade score, mitigation dispatch
│   ├── eis/
│   │   ├── state.py             EISState dataclass, eis_to_u mapping
│   │   ├── channels.py          R_SEI, θ_plating, W_transport extraction
│   │   └── drt.py               DRT plating peak detection (requires impedance)
│   ├── losses/
│   │   ├── sep_loss.py          EIS-weighted separatrix loss with Ω_i confidence
│   │   └── pack_weights.py      Thermal-electrical centrality weights
│   ├── infrastructure/
│   │   ├── bq40z80_adapter.py   TI BQ40Z80 EIS chip interface
│   │   └── synthetic_adapter.py Synthetic EIS for testing
│   ├── application/
│   │   └── bms_pipeline.py      Full BMS integration loop
│   └── demo/
│       ├── ev_fastcharge.py     96-cell fast-charge simulation
│       └── cascade_replay.py    Replay from EIS CSV data
├── config/
│   └── bgbkd_96cell_nmc.yaml
├── data/
│   └── eis_calibration/         (populate from your EIS baseline measurements)
├── tests/
└── scripts/
```

---

## Key Results

### 96-Cell EV Pack Simulation (150 kW, 5°C Cold Start)

| Signal | Time | Lead Over Voltage |
|--------|------|------------------|
| BGBKD structural fragility | t = 0 s | **226 s** |
| BGBKD cascade map | t = 14 s | 212 s |
| Thermal detection (ΔT > 3°C) | t = 11 s | 215 s |
| Voltage plateau trigger | t = 237 s | — |

### Computational Performance

| Computation | d_c | Operations | Hardware | Latency |
|-------------|-----|-----------|----------|---------|
| Per-cycle inference | 263 | 69,169 | Cortex-M7 | < 1 ms |
| 30s full-pack forecast (K^30) | 263 | 415,014 | Cortex-M7 | **1.9 ms** |
| DFN equivalent (96 cells) | — | 51,840,000 | GPU | > 500 ms |

BGBKD is **125× faster** than DFN for a 30-second pack-level forecast.

---

## The Two-Hop Cascade Score

```python
detector = TwoHopCascadeDetector(obs)
cascade_map = detector.compute(K)

# Get top cascade pathways
for (i, j), r in sorted(cascade_map.items(),
                         key=lambda x: x[1].xi, reverse=True)[:5]:
    print(f"Cell {i}→{j}:  "
          f"ξ={r.xi:.2e}  hop1={r.hop1:.2e}  hop2={r.hop2:.2e}  "
          f"→ {r.mitigation}")
```

| hop1 | hop2 | Interpretation | BMS Action |
|------|------|----------------|------------|
| Large | Small | Cell i approaching plating threshold | Reduce C-rate at cell i |
| Small | Large | Thermal/electrical stress reaching cell j | Pre-activate cooling at j; prepare isolation relay |
| Both large | — | Imminent cascade | Derate string; emergency cooling |

---

## EIS Calibration

BGBKD requires per-cell calibration endpoints for the three EIS channels. Minimum required:

```json
{
  "chemistry": "NMC622/graphite",
  "format": "21700",
  "R0_mohm": 18.0,
  "Rinf_mohm": 85.0,
  "theta0_deg": -28.5,
  "theta_plating_deg": -41.2,
  "W0": 1.02,
  "Winf": 1.85,
  "alpha_prior_mean": 0.32
}
```

For NMC/graphite cells, reference values are available from the published EIS literature [Schindler et al. 2016, Illig et al. 2012]. For other chemistries, ARC-calibrated endpoints are recommended.

---

## Hard Guarantees

**Attractor protection.** `δΨ_i = φ³_i · h_θ(Ψ_BKD_i, z_EIS) = 0` at `u_i ∈ {0, 1}` unconditionally — regardless of EIS feature values, temperature, impedance, or learned MLP weights. Proof in Section 5 of the paper.

**Φ̂^→_ij attractor zeros.** `Φ̂^→_ij = u_i(1−u_i) · Γ_ij · (u_j − α̂_ij) = 0` at `u_i ∈ {0, 1}` for all values of Γ_ij and α̂_ij.

**φ⁴_i interpretability.** Fragility score `ψ_i = |φ⁴_i(u_i; α̂_i)| / φ_max` is always the backbone tipping kernel. The MLP corrector improves prediction accuracy for unmodeled effects (electrolyte decomposition, electrode cracking) but never contaminates ψ_i.

---

## Known Limitations

- **NMC/graphite calibrated.** LFP and NCA chemistries require recalibrated EIS channel bounds and DRT threshold γ_DRT.
- **Fixed wind direction (wake analogue: fixed pack topology).** Cell swelling changes thermal contact resistance over lifetime. Online A_pack estimation from impedance cross-correlation is future work.
- **Linear thermal correction α̂_ij.** Valid for ΔT < 20°C between adjacent cells. Under rapid thermal runaway (> 10°C/s), a nonlinear kernel is needed.
- **Single timescale.** BGBKD models slow degradation dynamics. Fast transients (mechanical cell deformation, external short circuit) require a separate fast-timescale Koopman extension.

---

## Citation

```bibtex
@article{bgbkd2025,
  title   = {BGBKD: A Battery Graph Bistable Koopman Dictionary for
             Real-Time Pack-Level Thermal Runaway Cascade Forecasting
             from EIS Data},
  author  = {[Authors]},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```

---

## Related Repositories

| Repository | Domain | Key Signal |
|-----------|--------|-----------|
| [sai-oi/gbkd](https://github.com/sai-oi/gbkd) | Power grid | AUC 0.9948, IEEE 39-bus |
| [sai-oi/fgbkd](https://github.com/sai-oi/fgbkd) | Financial markets | 5-day lead on March 2020 crash |
| [sai-oi/wgbkd](https://github.com/sai-oi/wgbkd) | Wind farms | Stall cascade pathway forecasting |
| [sai-oi/scgbkd_loop](https://github.com/sai-oi/scgbkd_loop) | Supply chain | A Self-Recovery, Self-Optimizing, and Cyber-Physical Supply-Chain Framework |

---

## License

MIT. See [LICENSE](LICENSE).

*[SAI-OI / ROIS](https://saicommand.com) · koopman.bgbkd*
