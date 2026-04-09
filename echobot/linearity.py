"""System linearity diagnostics: power / pulse duration / bandwidth sweeps.

The manuscript's linearity analysis (main-text Figure 6, Table 6) covers three
orthogonal parameter axes:

1. **Transmit power** — does the received peak level scale linearly with the
   commanded power setting? EchoBot ``scale`` values 0/3/5 dB vs EK80 60/90/
   120 W. Linearity is measured as the slope of peak-level vs nominal-level on
   a dB-dB plot (ideal slope = 1).

2. **Pulse duration** — does TS(f) spectral *shape* remain invariant to pulse
   length? The B1 (1.0 ms) group is compared against the A1 baseline (0.5 ms)
   by Pearson r between their mean-subtracted TS(f) spectra.

3. **Bandwidth** — does TS(f) spectral shape remain invariant to chirp
   bandwidth? The C1 (100–140 kHz) group is compared against A1 (90–150 kHz)
   over the 100–140 kHz overlap band, which is the correct normalization
   window for a bandwidth-sensitivity comparison (full-band normalization would
   introduce a spurious ~12 dB offset in the EK80 panel of Figure 6 because
   the two bands sample different parts of the chirp rolloff).

All three routines accept processed-data containers and return a small
summary dataclass so they can be aggregated into a single linearity table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.signal import hilbert

from . import config
from .compare import cross_instrument_r

__all__ = [
    "PowerLinearityResult",
    "ShapeInvarianceResult",
    "power_linearity",
    "duration_shape_r",
    "bandwidth_shape_r",
]


# ---------------------------------------------------------------------------
# Power linearity
# ---------------------------------------------------------------------------

@dataclass
class PowerLinearityResult:
    """Per-instrument slope of peak-level vs nominal-level on a dB-dB plot."""

    instrument: str
    nominal_db: np.ndarray      # nominal power in dB relative to the low setting
    measured_db: np.ndarray     # measured peak level in dB (arbitrary reference)
    slope: float                # linear-regression slope (ideal = 1)
    intercept: float
    rms_residual_db: float      # std of (measured - (slope*nominal + intercept))
    points: list[dict] = field(default_factory=list)


def _peak_target_level_db(all_mf: np.ndarray, target_gate: tuple[int, int]) -> float:
    """Return the median-across-pings peak-envelope level in dB inside a gate."""
    j_lo, j_hi = target_gate
    gated = all_mf[:, j_lo:j_hi]
    env = np.abs(gated) if np.iscomplexobj(gated) else np.abs(hilbert(gated, axis=1))
    peaks = env.max(axis=1)
    return float(20.0 * np.log10(np.median(peaks)))


def power_linearity(
    power_series: Sequence[dict],
    instrument: str,
) -> PowerLinearityResult:
    """Fit peak-level vs nominal-power (dB) across a power sweep.

    Parameters
    ----------
    power_series : sequence of dict
        Each entry must have ``'nominal_db'`` (dB relative to lowest level,
        e.g. 0/4.77/7.78 for EK80 60/90/120 W; 0/3/5 for EchoBot scale 0/3/5)
        and ``'processed'`` which is an ``EchoBotProcessed`` or ``EK80Processed``.
    instrument : str
        Label, e.g. ``"EchoBot"`` or ``"EK80"``.

    Returns
    -------
    PowerLinearityResult
    """
    nominal = []
    measured = []
    points = []
    for entry in power_series:
        proc = entry["processed"]
        # Access target_gate and all-MF/pc output generically
        gate = getattr(proc, "target_gate")
        mf = getattr(proc, "all_pc", None)  # EK80 first
        if mf is None:
            mf = getattr(proc, "all_mf")     # EchoBot fallback
        level_db = _peak_target_level_db(mf, gate)
        nominal.append(float(entry["nominal_db"]))
        measured.append(level_db)
        points.append({"nominal_db": entry["nominal_db"], "measured_db": level_db})

    nominal_arr = np.asarray(nominal)
    measured_arr = np.asarray(measured)

    # Linear regression via np.polyfit
    slope, intercept = np.polyfit(nominal_arr, measured_arr, 1)
    residuals = measured_arr - (slope * nominal_arr + intercept)
    rms = float(np.sqrt(np.mean(residuals**2)))

    return PowerLinearityResult(
        instrument=instrument,
        nominal_db=nominal_arr,
        measured_db=measured_arr,
        slope=float(slope),
        intercept=float(intercept),
        rms_residual_db=rms,
        points=points,
    )


# ---------------------------------------------------------------------------
# Spectral-shape invariance (pulse duration and bandwidth)
# ---------------------------------------------------------------------------

@dataclass
class ShapeInvarianceResult:
    """Spectral-shape Pearson r between a reference and test TS(f) configuration."""

    axis: str                   # "duration" or "bandwidth"
    reference_cid: str          # e.g. "A1"
    test_cid: str               # e.g. "B1" or "C1"
    instrument: str             # "EchoBot" or "EK80"
    pearson_r: float
    rms_residual_db: float      # RMS of (test - reference) after mean subtraction
    n_common: int               # number of common-grid samples used


def _self_compare(
    tsf_ref_db: np.ndarray,
    f_ref_hz: np.ndarray,
    tsf_test_db: np.ndarray,
    f_test_hz: np.ndarray,
    f_lo_hz: float,
    f_hi_hz: float,
) -> tuple[float, float, int]:
    """Within-instrument comparison: same logic as ``cross_instrument_r`` but
    we reuse that helper by treating the reference as "EB" and test as "EK"."""
    res = cross_instrument_r(
        tsf_ref_db,
        f_ref_hz,
        tsf_test_db,
        f_test_hz,
        f_lo_hz=f_lo_hz,
        f_hi_hz=f_hi_hz,
    )
    residual = res.ek_norm - res.eb_norm
    rms = float(np.sqrt(np.mean(residual**2)))
    return res.pearson_r, rms, len(res.f_common)


def duration_shape_r(
    reference_proc,
    test_proc,
    instrument: str,
    reference_cid: str = "A1",
    test_cid: str = "B1",
    f_lo_hz: float = config.F_BAND_LO_HZ,
    f_hi_hz: float = config.F_BAND_HI_HZ,
) -> ShapeInvarianceResult:
    """TS(f) shape invariance across pulse duration (A1 vs B1).

    Both groups use the full 90–150 kHz chirp bandwidth, so normalization is
    over the full band.

    Parameters
    ----------
    reference_proc, test_proc :
        ``EchoBotProcessed`` or ``EK80Processed`` for the reference (A1) and
        test (B1) runs of the same instrument.
    instrument : str
    reference_cid, test_cid : str
    f_lo_hz, f_hi_hz : float
    """
    ref_f, ref_t, test_f, test_t = _tsf_accessors(reference_proc, test_proc)
    r, rms, n = _self_compare(ref_t, ref_f, test_t, test_f, f_lo_hz, f_hi_hz)
    return ShapeInvarianceResult(
        axis="duration",
        reference_cid=reference_cid,
        test_cid=test_cid,
        instrument=instrument,
        pearson_r=r,
        rms_residual_db=rms,
        n_common=n,
    )


def bandwidth_shape_r(
    reference_proc,
    test_proc,
    instrument: str,
    reference_cid: str = "A1",
    test_cid: str = "C1",
    f_lo_hz: float = config.F_OVERLAP_LO_HZ,
    f_hi_hz: float = config.F_OVERLAP_HI_HZ,
) -> ShapeInvarianceResult:
    """TS(f) shape invariance across chirp bandwidth (A1 vs C1).

    **Important:** normalization *must* be over the 100–140 kHz overlap band.
    The C1 group uses a 100–140 kHz chirp, so normalizing over the full 90–150
    kHz band would subtract different means from the two curves and introduce
    a spurious offset. This is the fix applied to Figure 6 in v3 of the
    manuscript.
    """
    ref_f, ref_t, test_f, test_t = _tsf_accessors(reference_proc, test_proc)
    r, rms, n = _self_compare(ref_t, ref_f, test_t, test_f, f_lo_hz, f_hi_hz)
    return ShapeInvarianceResult(
        axis="bandwidth",
        reference_cid=reference_cid,
        test_cid=test_cid,
        instrument=instrument,
        pearson_r=r,
        rms_residual_db=rms,
        n_common=n,
    )


def _tsf_accessors(reference_proc, test_proc):
    """Return (f_ref, tsf_ref, f_test, tsf_test) handling EB vs EK containers."""
    # EchoBotProcessed has f_hz and tsf_db; EK80Processed has f_band_sorted and tsf_calibrated_db
    def _get(p):
        if hasattr(p, "f_band_sorted"):  # EK80Processed
            return p.f_band_sorted, p.tsf_calibrated_db
        return p.f_hz, p.tsf_db  # EchoBotProcessed
    f_ref, t_ref = _get(reference_proc)
    f_test, t_test = _get(test_proc)
    return f_ref, t_ref, f_test, t_test


def summarize_linearity(
    power_results: list[PowerLinearityResult] | None = None,
    shape_results: list[ShapeInvarianceResult] | None = None,
) -> pd.DataFrame:
    """Aggregate linearity results into a single summary DataFrame.

    Row schema:
    ``['axis', 'instrument', 'reference', 'test', 'metric', 'value', 'unit']``
    """
    rows = []
    for pr in power_results or []:
        rows.append(
            dict(
                axis="power",
                instrument=pr.instrument,
                reference="",
                test="",
                metric="slope",
                value=pr.slope,
                unit="dB/dB",
            )
        )
        rows.append(
            dict(
                axis="power",
                instrument=pr.instrument,
                reference="",
                test="",
                metric="rms_residual",
                value=pr.rms_residual_db,
                unit="dB",
            )
        )
    for sr in shape_results or []:
        rows.append(
            dict(
                axis=sr.axis,
                instrument=sr.instrument,
                reference=sr.reference_cid,
                test=sr.test_cid,
                metric="pearson_r",
                value=sr.pearson_r,
                unit="",
            )
        )
        rows.append(
            dict(
                axis=sr.axis,
                instrument=sr.instrument,
                reference=sr.reference_cid,
                test=sr.test_cid,
                metric="rms_residual",
                value=sr.rms_residual_db,
                unit="dB",
            )
        )
    return pd.DataFrame(rows)
