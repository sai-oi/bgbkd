"""
bgbkd/infrastructure/bq40z80_adapter.py
══════════════════════════════════════════════════════════════
Interface to Texas Instruments BQ40Z80 EIS-capable fuel gauge.

BQ40Z80 supports on-chip PRBS EIS at up to 10 points per sweep.
Communication: SMBus / I²C at 100 kHz.

Falls back to synthetic mode if SMBus hardware is unavailable,
so integration tests run without physical hardware.

Register map from TI datasheet SLUSBY5.

SAI-OI / ROIS · koopman.bgbkd
Conception & direction: Mene · Formulation: Claude
══════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..eis.state import EISState


@dataclass
class BQ40Z80Config:
    """I²C / SMBus configuration for one BQ40Z80 cell monitor."""
    i2c_bus:     int   = 1      # Linux I²C bus index (e.g. /dev/i2c-1)
    i2c_addr:    int   = 0x16   # BQ40Z80 default SMBus address
    n_freqs:     int   = 10     # EIS frequency sweep points
    prbs_amp_ma: float = 50.0   # PRBS perturbation amplitude (mA)


class BQ40Z80Adapter:
    """
    Reads EIS spectra from a TI BQ40Z80 via SMBus and maps to EISState.

    Usage::

        adapter = BQ40Z80Adapter(cfg=BQ40Z80Config(), synthetic=False)
        eis_cell0 = adapter.read_eis(cell_idx=0, cal=calibration_dict)

    In synthetic mode the adapter generates plausible EIS values using
    the same model as SyntheticEISAdapter, enabling full integration
    tests without hardware.
    """

    # Register addresses (SLUSBY5 §7.6)
    REG_EIS_REAL = 0x74   # Real impedance output  (16-bit, mΩ × 100)
    REG_EIS_IMAG = 0x76   # Imaginary impedance     (16-bit, mΩ × 100)
    REG_TEMP_INT = 0x08   # Internal temperature    (0.1 K units)

    def __init__(
        self,
        cfg:       BQ40Z80Config = BQ40Z80Config(),
        synthetic: bool          = False,
    ):
        self.cfg       = cfg
        self.synthetic = synthetic
        self._bus      = None

        if not synthetic:
            try:
                import smbus2
                self._bus = smbus2.SMBus(cfg.i2c_bus)
            except (ImportError, OSError) as exc:
                print(f"BQ40Z80: hardware unavailable ({exc}); using synthetic mode.")
                self.synthetic = True

    def read_eis(self, cell_idx: int, cal: dict) -> EISState:
        """
        Read EIS spectrum for one cell and map to EISState.

        Args:
            cell_idx: Cell index (for logging / addressing in multi-cell system).
            cal:      Calibration dict with keys:
                        R0_mohm, Rinf_mohm,
                        theta0_deg, theta_plating_deg,
                        W0, Winf.

        Returns:
            EISState with normalised channels.
        """
        if self.synthetic:
            return self._synthetic_eis(cell_idx, cal)
        return self._hardware_eis(cell_idx, cal)

    # ── Hardware path ──────────────────────────────────────────────

    def _hardware_eis(self, cell_idx: int, cal: dict) -> EISState:
        """Read from BQ40Z80 registers (single-frequency per call)."""
        try:
            raw_re = self._bus.read_word_data(self.cfg.i2c_addr, self.REG_EIS_REAL)
            raw_im = self._bus.read_word_data(self.cfg.i2c_addr, self.REG_EIS_IMAG)
            Z_re_mohm = raw_re / 100.0
            Z_im_mohm = raw_im / 100.0

            R0    = cal["R0_mohm"]
            Rinf  = cal["Rinf_mohm"]
            denom = Rinf - R0
            R_SEI_norm = float(np.clip((Z_re_mohm - R0) / (denom + 1e-9), 0, 1))

            theta_raw = -np.degrees(np.arctan2(Z_im_mohm, Z_re_mohm))
            th0   = cal["theta0_deg"]
            th_pl = cal["theta_plating_deg"]
            theta_norm = float(np.clip(
                (theta_raw - th0) / (th_pl - th0 + 1e-9), 0, 1
            ))

            DRT_peak = max(0.0, theta_norm - 0.30) * 2.5

            raw_temp = self._bus.read_word_data(self.cfg.i2c_addr, self.REG_TEMP_INT)
            T_cell   = raw_temp / 10.0 - 273.15

            return EISState(
                R_SEI    = R_SEI_norm,
                theta    = theta_norm,
                W_trans  = float(R_SEI_norm * 0.3 + theta_norm * 0.5),
                DRT_peak = DRT_peak,
                T_cell   = T_cell,
            )

        except Exception as exc:
            print(f"BQ40Z80 read error (cell {cell_idx}): {exc}")
            return EISState(R_SEI=0.0, theta=0.0, W_trans=0.0, DRT_peak=0.0, T_cell=25.0)

    # ── Synthetic path ─────────────────────────────────────────────

    def _synthetic_eis(self, cell_idx: int, cal: dict) -> EISState:
        """Plausible synthetic EIS for integration testing."""
        rng = np.random.default_rng(cell_idx + int(np.random.rand() * 1000))
        base = 0.15 + cell_idx * 0.003
        return EISState(
            R_SEI    = float(np.clip(base + rng.normal(0, 0.01), 0, 1)),
            theta    = float(np.clip(base * 0.8 + rng.normal(0, 0.02), 0, 1)),
            W_trans  = float(np.clip(base * 0.5 + rng.normal(0, 0.01), 0, 1)),
            DRT_peak = float(max(0, base - 0.20 + rng.normal(0, 0.03))),
            T_cell   = float(25.0 + cell_idx * 0.1 + rng.normal(0, 0.2)),
        )

    def close(self) -> None:
        """Release the I²C bus."""
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
