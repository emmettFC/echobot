"""SoundTrap hydrophone processing.

Supports two use cases documented in the manuscript:

1. **Passive monitoring of sonar transmissions during test runs** — the
   hydrophone sits off to the side while the sphere occupies the target
   position. Produces the pressure-time-series + spectrogram figures in
   Supplementary Figure S1.

2. **Hydrophone-as-target runs** — the hydrophone is *at* the target position
   (replacing the sphere), so every recorded ping is the direct transmit
   waveform. Used for two things:
   - Visual confirmation of the transmit pulse envelope (Supplementary
     Figure S2).
   - Building the **EchoBot dB ↔ EK80 Watt receive-level mapping**: by
     measuring the received peak/RMS SPL at the same hydrophone position for
     each EchoBot ``scale`` setting and each EK80 ``transmit_power`` setting,
     the relative source-level relationship between the two instruments can be
     calibrated directly. This is the only independent check on the ~37 dB
     EB uncalibrated → EK80 calibrated offset reported in Section 3.3 and
     Figure S11 of the manuscript.

Functions here are deliberately small and pure; plotting lives in
``plotting.plot_hydro_runs``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert

from .io import (
    HydrophoneFileInfo,
    find_hydrophone_file,
    load_hydrophone_segment,
)

__all__ = [
    "PingSPLResult",
    "DBWattMapping",
    "bandpass_sonar_band",
    "analyze_pings_spl",
    "map_db_to_watts",
]


# ---------------------------------------------------------------------------
# Ping-level SPL extraction
# ---------------------------------------------------------------------------

@dataclass
class PingSPLResult:
    """Per-ping SPL measurements from a hydrophone recording.

    Attributes
    ----------
    label : str
        Free-form identifier (e.g. ``"EB scale=3"`` or ``"EK80 60 W"``).
    n_pings : int
    peak_spl_db : np.ndarray
        Per-ping peak SPL (dB re 1 µPa, 0-to-peak).
    rms_spl_db : np.ndarray
        Per-ping RMS SPL over the T90 energy window.
    ambient_rms_db : float
        RMS SPL of the inter-ping quiet intervals (ambient tank noise).
    mean_peak_db : float
    mean_rms_db : float
    """

    label: str
    n_pings: int
    peak_spl_db: np.ndarray
    rms_spl_db: np.ndarray
    ambient_rms_db: float
    mean_peak_db: float
    mean_rms_db: float


def bandpass_sonar_band(
    y: np.ndarray,
    fs: float,
    lo_hz: float = 80e3,
    hi_hz: float = 170e3,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass for the 80–170 kHz sonar band (zero-phase)."""
    nyq = fs / 2.0
    b, a = butter(order, [lo_hz / nyq, hi_hz / nyq], btype="band")
    return filtfilt(b, a, y)


def analyze_pings_spl(
    y_calibrated: np.ndarray,
    fs: float,
    label: str,
    min_ping_sep_s: float = 0.3,
    win_half_s: float = 0.002,
) -> PingSPLResult:
    """Detect pings in a calibrated hydrophone segment and compute per-ping SPL.

    Procedure:
    1. Bandpass to the 80–170 kHz sonar band.
    2. Compute the Hilbert envelope; smooth with a 5 ms moving average.
    3. Find peaks above the 60th-percentile envelope level separated by at
       least ``min_ping_sep_s``.
    4. For each peak, take a ±``win_half_s`` window, compute:
       - Peak SPL = ``20·log₁₀(max|y|)``
       - T90 RMS SPL = ``20·log₁₀(rms(y in the 5%–95% cumulative-energy window))``
    5. Ambient RMS from the inter-ping quiet intervals.

    Parameters
    ----------
    y_calibrated : np.ndarray
        Calibrated pressure in µPa (1-D).
    fs : float
        Sample rate (Hz).
    label : str
        Free-form label stored on the result for later aggregation.
    min_ping_sep_s : float, default 0.3
        Minimum allowed separation between detected pings.
    win_half_s : float, default 0.002
        Half-window size (s) around each detected peak for SPL integration.

    Returns
    -------
    PingSPLResult
    """
    y_bp = bandpass_sonar_band(y_calibrated, fs)
    env = np.abs(hilbert(y_bp))

    # 5 ms smoothing
    win_sm = max(1, int(0.005 * fs))
    env_sm = np.convolve(env, np.ones(win_sm) / win_sm, mode="same")

    thresh = np.percentile(env_sm, 60)
    min_dist = int(min_ping_sep_s * fs)
    peaks, _ = find_peaks(env_sm, height=thresh, distance=min_dist)

    if len(peaks) == 0:
        # Fallback to 40th percentile if nothing comes through
        thresh = np.percentile(env_sm, 40)
        peaks, _ = find_peaks(env_sm, height=thresh, distance=min_dist)

    win_half = int(win_half_s * fs)
    peak_spl = []
    rms_spl = []
    for pk in peaks:
        lo = max(0, pk - win_half)
        hi = min(len(y_bp), pk + win_half)
        seg = y_bp[lo:hi]
        if len(seg) < 10:
            continue
        pk_amp = float(np.max(np.abs(seg)))
        if pk_amp > 0:
            peak_spl.append(20.0 * np.log10(pk_amp))
        # T90 RMS: cumulative energy in [5%, 95%] window
        cum_e = np.cumsum(np.abs(seg)) / fs
        if cum_e[-1] > 0:
            in_t90 = (cum_e > 0.05 * cum_e[-1]) & (cum_e < 0.95 * cum_e[-1])
            t90_idx = np.where(in_t90)[0]
            if len(t90_idx) > 2:
                rms_val = float(np.sqrt(np.mean(seg[t90_idx] ** 2)))
                if rms_val > 0:
                    rms_spl.append(20.0 * np.log10(rms_val))

    # Ambient from inter-ping quiet intervals
    noise_samples: list[float] = []
    for k in range(len(peaks) - 1):
        q_lo = peaks[k] + 3 * win_half
        q_hi = peaks[k + 1] - 3 * win_half
        if q_hi - q_lo > fs * 0.05:  # at least 50 ms of quiet
            noise_samples.append(float(np.sqrt(np.mean(y_bp[q_lo:q_hi] ** 2))))
    ambient_rms_db = (
        float(20.0 * np.log10(np.mean(noise_samples)))
        if noise_samples
        else float("nan")
    )

    peak_arr = np.asarray(peak_spl)
    rms_arr = np.asarray(rms_spl)
    return PingSPLResult(
        label=label,
        n_pings=len(peak_arr),
        peak_spl_db=peak_arr,
        rms_spl_db=rms_arr,
        ambient_rms_db=ambient_rms_db,
        mean_peak_db=float(peak_arr.mean()) if len(peak_arr) else float("nan"),
        mean_rms_db=float(rms_arr.mean()) if len(rms_arr) else float("nan"),
    )


# ---------------------------------------------------------------------------
# dB ↔ Watt mapping from hydrophone-as-target runs
# ---------------------------------------------------------------------------

@dataclass
class DBWattMapping:
    """Linear fit of EchoBot source level (dB) vs EK80 transmit power (dBW).

    Both instruments are positioned identically; the hydrophone sits at the
    target range, so transmission loss cancels out and Δ(received SPL) =
    Δ(source level). The mapping is therefore a direct dB ↔ dBW translation:

    .. code-block::

        SPL_EB = slope_eb · scale_eb_dB + intercept_eb
        SPL_EK = slope_ek · 10·log₁₀(Ptx_W) + intercept_ek

    From the two fits the EchoBot ``scale`` setting at which EchoBot matches
    EK80 60 W (the A1 baseline setting) can be computed and cross-checked
    against the ~37 dB offset in Figure S11.
    """

    eb_slope: float
    eb_intercept: float
    ek_slope: float
    ek_intercept: float
    eb_points: list[dict]
    ek_points: list[dict]
    eb_rms_residual_db: float
    ek_rms_residual_db: float
    # Equivalent EchoBot scale setting for EK80 60 W reference
    eb_scale_for_ek60w_db: float


def map_db_to_watts(
    eb_series: list[PingSPLResult],
    eb_scales_db: list[float],
    ek_series: list[PingSPLResult],
    ek_powers_w: list[float],
    ek_ref_w: float = 60.0,
) -> DBWattMapping:
    """Fit EB and EK receive levels separately and derive the cross-mapping.

    Parameters
    ----------
    eb_series : list of PingSPLResult
        Hydrophone-as-target recordings for each EchoBot ``scale`` setting,
        in the same order as ``eb_scales_db``. Each result provides
        ``mean_peak_db``.
    eb_scales_db : list of float
        EchoBot ``scale`` values in dB (e.g. ``[0, 3, 5]`` for P1-lo/mid/hi).
    ek_series : list of PingSPLResult
        Hydrophone-as-target recordings for each EK80 transmit power, in the
        same order as ``ek_powers_w``.
    ek_powers_w : list of float
        EK80 transmit powers in watts (e.g. ``[60, 90, 120]``).
    ek_ref_w : float
        Reference EK80 power for which an equivalent EchoBot scale is solved.

    Returns
    -------
    DBWattMapping
    """
    eb_nominal = np.asarray(eb_scales_db, dtype=float)
    eb_measured = np.asarray([r.mean_peak_db for r in eb_series], dtype=float)

    ek_nominal = 10.0 * np.log10(np.asarray(ek_powers_w, dtype=float))
    ek_measured = np.asarray([r.mean_peak_db for r in ek_series], dtype=float)

    eb_slope, eb_intercept = np.polyfit(eb_nominal, eb_measured, 1)
    ek_slope, ek_intercept = np.polyfit(ek_nominal, ek_measured, 1)

    eb_res = eb_measured - (eb_slope * eb_nominal + eb_intercept)
    ek_res = ek_measured - (ek_slope * ek_nominal + ek_intercept)

    # Solve for EB scale that matches EK at ek_ref_w:
    # slope_eb · scale + intercept_eb = slope_ek · 10·log10(ek_ref_w) + intercept_ek
    ek_ref_db = 10.0 * np.log10(ek_ref_w)
    target_db = ek_slope * ek_ref_db + ek_intercept
    eb_scale_for_ref = (target_db - eb_intercept) / eb_slope

    return DBWattMapping(
        eb_slope=float(eb_slope),
        eb_intercept=float(eb_intercept),
        ek_slope=float(ek_slope),
        ek_intercept=float(ek_intercept),
        eb_points=[
            {"scale_db": s, "measured_db": m, "label": r.label}
            for s, m, r in zip(eb_nominal.tolist(), eb_measured.tolist(), eb_series)
        ],
        ek_points=[
            {"power_w": w, "nominal_db": n, "measured_db": m, "label": r.label}
            for w, n, m, r in zip(
                ek_powers_w, ek_nominal.tolist(), ek_measured.tolist(), ek_series
            )
        ],
        eb_rms_residual_db=float(np.sqrt(np.mean(eb_res**2))),
        ek_rms_residual_db=float(np.sqrt(np.mean(ek_res**2))),
        eb_scale_for_ek60w_db=float(eb_scale_for_ref),
    )


# ---------------------------------------------------------------------------
# Convenience: load + analyze a run from its UTC window
# ---------------------------------------------------------------------------

def analyze_run_from_index(
    label: str,
    start_utc: datetime,
    end_utc: datetime,
    hydro_index: list[HydrophoneFileInfo],
) -> PingSPLResult | None:
    """Look up the wav file covering ``[start_utc, end_utc]``, load, and analyze.

    Returns ``None`` if no wav file covers the requested window (useful for
    runs whose hydrophone .wav was not archived).
    """
    wav_path, wav_start = find_hydrophone_file(start_utc, hydro_index)
    if wav_path is None or wav_start is None:
        return None
    _, y, fs = load_hydrophone_segment(wav_path, wav_start, start_utc, end_utc)
    if len(y) < 256:
        return None
    return analyze_pings_spl(y, fs, label)
