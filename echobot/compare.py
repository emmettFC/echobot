"""Cross-instrument TS(f) comparison.

Provides:
- ``normalize_tsf``: mean-subtraction over a user-specified band (default
  100–140 kHz, the overlap region that includes the C1 narrow-bandwidth group).
- ``cross_instrument_r``: Pearson correlation of two TS(f) spectra on a common
  frequency grid, for use in Tables 3, S1, and Figures 4, 5 of the manuscript.
- ``per_group_comparison``: convenience wrapper that takes a list of groups and
  processed instrument containers and returns a per-group DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

__all__ = [
    "ComparisonResult",
    "normalize_tsf",
    "cross_instrument_r",
    "per_group_comparison",
]


@dataclass
class ComparisonResult:
    """Output of a single EchoBot vs EK80 TS(f) comparison.

    Attributes
    ----------
    f_common : np.ndarray
        Common frequency grid used for both spectra (Hz).
    eb_norm : np.ndarray
        Mean-subtracted EchoBot TS(f) on ``f_common``.
    ek_norm : np.ndarray
        Mean-subtracted EK80 TS(f) on ``f_common``.
    pearson_r : float
        Pearson correlation between the normalized spectra on ``f_common``.
    """

    f_common: np.ndarray
    eb_norm: np.ndarray
    ek_norm: np.ndarray
    pearson_r: float

    def __repr__(self) -> str:
        return f"ComparisonResult(r={self.pearson_r:.4f}, n_freq={len(self.f_common)})"


def normalize_tsf(
    tsf_db: np.ndarray,
    f_hz: np.ndarray,
    f_lo_hz: float = config.F_OVERLAP_LO_HZ,
    f_hi_hz: float = config.F_OVERLAP_HI_HZ,
) -> np.ndarray:
    """Mean-subtract a TS(f) spectrum over a given frequency band.

    This is the normalization used for every cross-instrument comparison in the
    main text. The mean is computed over the in-band values only; the full
    spectrum is returned with the same mean subtracted, so out-of-band values
    retain their original shape relative to the in-band mean.

    Parameters
    ----------
    tsf_db : np.ndarray
        TS(f) in dB.
    f_hz : np.ndarray
        Frequency axis in Hz.
    f_lo_hz, f_hi_hz : float
        Overlap band for computing the mean. Defaults to 100–140 kHz so the
        narrow-bandwidth C1 group (100–140 kHz) is handled correctly.

    Returns
    -------
    np.ndarray
        ``tsf_db - mean(tsf_db[in band])`` with the same shape as the input.
    """
    mask = (f_hz >= f_lo_hz) & (f_hz <= f_hi_hz)
    if not np.any(mask):
        raise ValueError(
            f"No frequency samples in normalization band {f_lo_hz/1e3:.0f}-{f_hi_hz/1e3:.0f} kHz"
        )
    return tsf_db - float(np.mean(tsf_db[mask]))


def cross_instrument_r(
    eb_tsf_db: np.ndarray,
    eb_f_hz: np.ndarray,
    ek_tsf_db: np.ndarray,
    ek_f_hz: np.ndarray,
    f_lo_hz: float = config.F_OVERLAP_LO_HZ,
    f_hi_hz: float = config.F_OVERLAP_HI_HZ,
    n_common: int = 500,
) -> ComparisonResult:
    """Interpolate two TS(f) spectra onto a common grid, normalize, and correlate.

    Both spectra are interpolated onto a linear ``n_common``-point grid over
    ``[f_lo_hz, f_hi_hz]``, mean-subtracted (over that same band), and fed to
    ``numpy.corrcoef``.

    Parameters
    ----------
    eb_tsf_db, eb_f_hz : np.ndarray
        EchoBot TS(f) and its frequency axis.
    ek_tsf_db, ek_f_hz : np.ndarray
        EK80 TS(f) and its frequency axis. The EK80 axis may be unsorted (it
        comes out of the aliasing-aware mapping already sorted but we don't
        assume that); we sort here.
    f_lo_hz, f_hi_hz : float
        Overlap band.
    n_common : int
        Number of samples in the common grid.

    Returns
    -------
    ComparisonResult
    """
    f_common = np.linspace(f_lo_hz, f_hi_hz, n_common)

    # EchoBot: already sorted ascending; restrict to band and interpolate
    eb_mask = (eb_f_hz >= f_lo_hz) & (eb_f_hz <= f_hi_hz)
    eb_interp = np.interp(f_common, eb_f_hz[eb_mask], eb_tsf_db[eb_mask])

    # EK80: ensure sorted
    order = np.argsort(ek_f_hz)
    ek_f_sorted = ek_f_hz[order]
    ek_t_sorted = ek_tsf_db[order]
    ek_mask = (ek_f_sorted >= f_lo_hz) & (ek_f_sorted <= f_hi_hz)
    ek_interp = np.interp(f_common, ek_f_sorted[ek_mask], ek_t_sorted[ek_mask])

    eb_norm = eb_interp - np.mean(eb_interp)
    ek_norm = ek_interp - np.mean(ek_interp)

    r = float(np.corrcoef(eb_norm, ek_norm)[0, 1])

    return ComparisonResult(f_common=f_common, eb_norm=eb_norm, ek_norm=ek_norm, pearson_r=r)


def per_group_comparison(
    groups: list[dict],
    eb_proc: dict,
    ek_proc: dict,
    f_lo_hz: float = config.F_OVERLAP_LO_HZ,
    f_hi_hz: float = config.F_OVERLAP_HI_HZ,
) -> pd.DataFrame:
    """Compute cross-instrument Pearson r for a list of comparison groups.

    Parameters
    ----------
    groups : list[dict]
        Each dict must have ``cid``, ``name``, ``eb_down_idx``, and ``ek_idx``
        keys. Optional ``eb_up_idx`` for groups where the upsweep was recorded.
    eb_proc : dict
        Mapping ``{file_idx: EchoBotProcessed}``.
    ek_proc : dict
        Mapping ``{file_idx: EK80Processed}``.
    f_lo_hz, f_hi_hz : float
        Overlap band for the comparison.

    Returns
    -------
    pandas.DataFrame
        One row per group × sweep (downsweep/upsweep), with columns
        ``['cid', 'name', 'sweep', 'pearson_r', 'n_eb_pings', 'n_ek_pings']``.
    """
    rows = []
    for g in groups:
        ek = ek_proc.get(g["ek_idx"])
        if ek is None:
            continue
        ek_f = ek.f_band_sorted
        ek_t = ek.tsf_calibrated_db

        for sweep_key, sweep_label in [("eb_down_idx", "down"), ("eb_up_idx", "up")]:
            idx = g.get(sweep_key)
            if idx is None:
                continue
            eb = eb_proc.get(idx)
            if eb is None:
                continue
            res = cross_instrument_r(
                eb.tsf_db, eb.f_hz, ek_t, ek_f, f_lo_hz=f_lo_hz, f_hi_hz=f_hi_hz
            )
            rows.append(
                {
                    "cid": g["cid"],
                    "name": g.get("name", ""),
                    "sweep": sweep_label,
                    "pearson_r": res.pearson_r,
                    "n_eb_pings": eb.run.n_pings,
                    "n_ek_pings": ek.meta.n_pings,
                }
            )
    return pd.DataFrame(rows)
