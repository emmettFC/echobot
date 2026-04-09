"""Shared plotting helpers.

All plotting functions take matplotlib ``ax``/``fig`` arguments and return them
so notebooks can compose sub-figures freely. None of them call ``plt.show()`` —
that's the caller's job.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import firwin, hilbert, lfilter

from . import config
from .io import (
    HydrophoneFileInfo,
    find_hydrophone_file,
    load_hydrophone_segment,
)

__all__ = [
    "ek500_colormap",
    "EB_COLOR",
    "EK_COLOR",
    "plot_tsf_overlay",
    "plot_baseline_comparison",
    "plot_snr_histograms",
    "plot_ek80_validation",
    "plot_absolute_tsf",
    "plot_hydro_runs",
]

# Manuscript color convention
EB_COLOR = "#2166ac"
EK_COLOR = "#b2182b"


def ek500_colormap() -> mpl.colors.ListedColormap:
    """Simrad EK500 palette for echogram displays.

    Twelve discrete colors from black (−90 dB) through white/brown (−30 dB).
    Matches the convention used in the manuscript Figure 4 echogram panel and
    in commercial echosounder software.
    """
    colors = [
        (1.0, 1.0, 1.0),        # white — below −90
        (0.6235, 0.6235, 0.6235),
        (0.3725, 0.3725, 0.3725),
        (0.0, 0.0, 1.0),        # blue
        (0.0, 0.0, 0.5),
        (0.0, 0.7490, 0.0),     # green
        (0.0, 0.5, 0.0),
        (1.0, 1.0, 0.0),        # yellow
        (1.0, 0.5, 0.0),        # orange
        (1.0, 0.0, 0.7490),     # magenta
        (1.0, 0.0, 0.0),        # red
        (0.6509, 0.3255, 0.2353),  # brown
    ]
    return mpl.colors.ListedColormap(colors, name="ek500")


# ---------------------------------------------------------------------------
# TS(f) overlays
# ---------------------------------------------------------------------------

def plot_tsf_overlay(
    eb_f_hz: np.ndarray,
    eb_tsf_db: np.ndarray,
    ek_f_hz: np.ndarray,
    ek_tsf_db: np.ndarray,
    normalized: bool = True,
    f_lo_hz: float = config.F_OVERLAP_LO_HZ,
    f_hi_hz: float = config.F_OVERLAP_HI_HZ,
    ax: plt.Axes | None = None,
    eb_label: str = "EchoBot",
    ek_label: str = "EK80",
) -> plt.Axes:
    """Overlay an EchoBot and an EK80 TS(f) spectrum on a single axis.

    Parameters
    ----------
    eb_f_hz, eb_tsf_db : np.ndarray
    ek_f_hz, ek_tsf_db : np.ndarray
    normalized : bool
        If True, mean-subtract each spectrum over the overlap band so only
        spectral shape is compared (same convention as manuscript figs).
    f_lo_hz, f_hi_hz : float
        Band used for normalization and x-axis limits.
    ax : matplotlib Axes, optional
    eb_label, ek_label : str
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    if normalized:
        from .compare import normalize_tsf

        eb_plot = normalize_tsf(eb_tsf_db, eb_f_hz, f_lo_hz, f_hi_hz)
        ek_plot = normalize_tsf(ek_tsf_db, ek_f_hz, f_lo_hz, f_hi_hz)
        y_label = "TS(f) − mean(TS) over overlap band (dB)"
    else:
        eb_plot, ek_plot = eb_tsf_db, ek_tsf_db
        y_label = "TS(f) (dB)"

    # Mask to band for display
    eb_mask = (eb_f_hz >= f_lo_hz) & (eb_f_hz <= f_hi_hz)
    ek_mask = (ek_f_hz >= f_lo_hz) & (ek_f_hz <= f_hi_hz)

    ax.plot(eb_f_hz[eb_mask] / 1e3, eb_plot[eb_mask], color=EB_COLOR, lw=1.6, label=eb_label)
    ax.plot(ek_f_hz[ek_mask] / 1e3, ek_plot[ek_mask], color=EK_COLOR, lw=1.6, label=ek_label)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax


# ---------------------------------------------------------------------------
# Baseline comparison (Figure 4 equivalent)
# ---------------------------------------------------------------------------

def plot_baseline_comparison(
    eb_processed,
    ek_processed,
    title: str = "A1 Baseline",
) -> plt.Figure:
    """Two-panel figure: (a) MF envelope vs range, (b) normalized TS(f) overlay.

    Equivalent to main-text Figure 4 of the manuscript for a given group.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

    # --- (a) Envelope vs range ---
    eb_env = np.abs(hilbert(eb_processed.all_mf[0]))
    ek_env = np.abs(ek_processed.all_pc[0])

    eb_env_db = 20 * np.log10(eb_env / eb_env.max() + 1e-30)
    ek_env_db = 20 * np.log10(ek_env / ek_env.max() + 1e-30)

    ax1.plot(eb_processed.r_mf, eb_env_db, color=EB_COLOR, lw=1.2, label="EchoBot")
    ax1.plot(ek_processed.r_ek, ek_env_db, color=EK_COLOR, lw=1.2, label="EK80")
    ax1.axvspan(
        eb_processed.target_center_m - config.GATE_HALF_M,
        eb_processed.target_center_m + config.GATE_HALF_M,
        alpha=0.1,
        color=EB_COLOR,
    )
    ax1.axvspan(
        ek_processed.target_center_m - config.GATE_HALF_M,
        ek_processed.target_center_m + config.GATE_HALF_M,
        alpha=0.1,
        color=EK_COLOR,
    )
    ax1.set_xlim(0.5, 3.5)
    ax1.set_ylim(-60, 2)
    ax1.set_xlabel("Range (m)")
    ax1.set_ylabel("Normalized envelope (dB)")
    ax1.set_title("(a) Matched-filter envelope")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- (b) Normalized TS(f) overlay ---
    plot_tsf_overlay(
        eb_processed.f_hz,
        eb_processed.tsf_db,
        ek_processed.f_band_sorted,
        ek_processed.tsf_calibrated_db,
        normalized=True,
        ax=ax2,
    )
    ax2.set_title("(b) Normalized TS(f)")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SNR histograms
# ---------------------------------------------------------------------------

def plot_snr_histograms(
    snr_results: dict,
    method_label: str,
    ax: plt.Axes | None = None,
    bins: int = 30,
) -> plt.Axes:
    """Plot per-group SNR histograms on one axis.

    Parameters
    ----------
    snr_results : dict
        Mapping ``{group_label: SNRResult}``.
    method_label : str
        For the axis title, e.g. "empty-region method".
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for label, res in snr_results.items():
        ax.hist(res.per_ping_db, bins=bins, alpha=0.5, label=label)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Ping count")
    ax.set_title(f"Per-ping SNR — {method_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


# ---------------------------------------------------------------------------
# Supplementary figures (S10 + S11)
# ---------------------------------------------------------------------------

def plot_ek80_validation(
    df_ts120: pd.DataFrame,
    df_tsf_detailed: pd.DataFrame,
    per_ping_tsf: dict,
    f_band_sorted: np.ndarray,
    mask_ek: np.ndarray | slice | None = None,
) -> plt.Figure:
    """Manuscript Figure S10: pipeline vs EK80 desktop software validation.

    Parameters
    ----------
    df_ts120 : pandas.DataFrame
        Must have columns ``ping_index``, ``pipeline_cal_TS120``,
        ``ek80_software_TS_120kHz_dB``, ``stationary``.
    df_tsf_detailed : pandas.DataFrame
        Per-ping × per-frequency DataFrame with columns
        ``['ping', 'freq_actual', 'pipeline_cal', 'software', 'is_null']``.
    per_ping_tsf : dict
        Mapping ``{ping_index: calibrated TSF array}`` for the 5 detailed pings.
    f_band_sorted : np.ndarray
        Frequency axis for ``per_ping_tsf`` values (Hz).
    mask_ek : optional boolean mask into ``f_band_sorted``; default uses all.
    """
    from matplotlib.gridspec import GridSpec

    ping_indices_5 = sorted(per_ping_tsf.keys())
    if mask_ek is None:
        mask_ek = slice(None)

    fig = plt.figure(figsize=(14, 9.5))
    gs = GridSpec(
        2, len(ping_indices_5),
        figure=fig,
        height_ratios=[1.0, 1.1],
        hspace=0.55,
        wspace=0.12,
        top=0.93,
        bottom=0.08,
        left=0.07,
        right=0.98,
    )

    # Panel (a): TS@120 vs ping
    ax_a = fig.add_subplot(gs[0, :])
    pings = df_ts120["ping_index"].values
    ax_a.plot(
        pings, df_ts120["pipeline_cal_TS120"].values,
        "o--", color=EK_COLOR, markersize=7, lw=1.2, label="Pipeline (calibrated)",
    )
    ax_a.plot(
        pings, df_ts120["ek80_software_TS_120kHz_dB"].values,
        "o--", color="black", markersize=7, lw=1.2, label="EK80 desktop software",
    )
    stat_mask = (df_ts120["stationary"] == "Y").values
    if stat_mask.any():
        ax_a.axvspan(
            pings[stat_mask].min(), pings[stat_mask].max(),
            alpha=0.08, color="green", label="Stationary period",
        )
    mean_resid = float(
        (df_ts120["pipeline_cal_TS120"] - df_ts120["ek80_software_TS_120kHz_dB"]).mean()
    )
    ax_a.set_xlabel("Ping index")
    ax_a.set_ylabel("TS @ 120 kHz (dB)")
    ax_a.set_title(
        f"(a) Calibrated TS @ 120 kHz: pipeline vs. EK80 software  "
        f"(n = {len(df_ts120)}; mean residual = {mean_resid:+.2f} dB)",
        fontsize=11,
    )
    ax_a.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax_a.grid(True, alpha=0.3)

    # Sub-title for row 2
    bbox_row1 = gs[1, :].get_position(fig)
    fig.text(
        0.5,
        bbox_row1.y1 + 0.04,
        "(b) Per-ping TS(f): pipeline vs. EK80 software  "
        "(hollow 135 kHz markers = deep null, excluded)",
        ha="center",
        fontsize=11,
    )

    # Panel (b): per-ping sub-panels
    y_all = np.concatenate(
        [df_tsf_detailed["pipeline_cal"].values, df_tsf_detailed["software"].values]
    )
    ymin, ymax = y_all.min() - 1.5, y_all.max() + 1.5

    for i, p in enumerate(ping_indices_5):
        ax = fig.add_subplot(gs[1, i])
        sub = df_tsf_detailed[df_tsf_detailed["ping"] == p].sort_values("freq_actual")
        null_row = sub[sub["is_null"]]
        good = sub[~sub["is_null"]]

        ax.plot(
            good["freq_actual"].values, good["pipeline_cal"].values,
            "o--", color=EK_COLOR, markersize=6, lw=1.1, label="Pipeline",
        )
        ax.plot(
            good["freq_actual"].values, good["software"].values,
            "o--", color="black", markersize=6, lw=1.1, label="Software",
        )
        if len(null_row):
            ax.plot(
                null_row["freq_actual"].values, null_row["pipeline_cal"].values,
                "o", mfc="white", mec=EK_COLOR, markersize=7,
            )
            ax.plot(
                null_row["freq_actual"].values, null_row["software"].values,
                "o", mfc="white", mec="black", markersize=7,
            )
        ax.set_title(f"Ping {p}", fontsize=10, pad=4)
        ax.set_xlabel("Freq (kHz)", fontsize=9)
        if i == 0:
            ax.set_ylabel("TS (dB)", fontsize=10)
            ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
        else:
            ax.set_yticklabels([])
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(85, 155)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    return fig


def plot_absolute_tsf(
    eb_f_hz: np.ndarray,
    eb_tsf_db: np.ndarray,
    ek_f_hz: np.ndarray,
    ek_tsf_db: np.ndarray,
    noaa_f_hz: np.ndarray,
    noaa_tsf_db: np.ndarray,
    f_lo_hz: float = config.F_BAND_LO_HZ,
    f_hi_hz: float = config.F_BAND_HI_HZ,
    title: str = "Absolute TS(f) Comparison",
) -> plt.Figure:
    """Manuscript Figure S11: three-curve absolute TS(f) overlay.

    EchoBot (uncalibrated), EK80 (calibrated), NOAA theoretical sphere TS(f)
    on the same absolute dB axis. Annotates the EB–EK offset at 120 kHz.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    eb_mask = (eb_f_hz >= f_lo_hz) & (eb_f_hz <= f_hi_hz)
    ek_mask = (ek_f_hz >= f_lo_hz) & (ek_f_hz <= f_hi_hz)

    ax.plot(eb_f_hz[eb_mask] / 1e3, eb_tsf_db[eb_mask], color=EB_COLOR, lw=1.8, label="EchoBot (uncalibrated)")
    ax.plot(ek_f_hz[ek_mask] / 1e3, ek_tsf_db[ek_mask], color=EK_COLOR, lw=1.8, label="EK80 (calibrated)")
    ax.plot(noaa_f_hz / 1e3, noaa_tsf_db, "k--", lw=1.5, label="NOAA theoretical")

    eb_120 = float(np.interp(120e3, eb_f_hz[eb_mask], eb_tsf_db[eb_mask]))
    ek_120 = float(np.interp(120e3, ek_f_hz[ek_mask], ek_tsf_db[ek_mask]))
    ax.annotate("", xy=(120, ek_120), xytext=(120, eb_120),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(121, (eb_120 + ek_120) / 2, f"{eb_120 - ek_120:.1f} dB",
            fontsize=9, color="gray", va="center")

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("TS (dB)")
    ax.set_xlim(f_lo_hz / 1e3, f_hi_hz / 1e3)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hydrophone pressure/spectrogram panels (Figures S1, S2)
# ---------------------------------------------------------------------------

def plot_hydro_runs(
    runs: list[tuple[str, datetime, datetime]],
    suptitle: str,
    hydro_index: list[HydrophoneFileInfo],
    savepath: str | Path | None = None,
) -> plt.Figure:
    """Pressure time series + spectrogram stack for a list of sonar runs.

    Each run becomes two stacked panels (pressure on top, spectrogram below).
    If a run's UTC window is not covered by any hydrophone file in
    ``hydro_index``, its panels display "No hydrophone file found" and the
    function continues without raising.
    """
    n_runs = len(runs)
    fig, axes = plt.subplots(
        2 * n_runs, 1, figsize=(18, 3.5 * 2 * n_runs), squeeze=False
    )

    for ri, (label, q_start, q_end) in enumerate(runs):
        ax_pres = axes[2 * ri, 0]
        ax_spec = axes[2 * ri + 1, 0]

        wav_path, wav_start = find_hydrophone_file(q_start, hydro_index)
        if wav_path is None:
            ax_pres.text(
                0.5, 0.5, "No hydrophone file found",
                transform=ax_pres.transAxes, ha="center", va="center",
            )
            ax_pres.set_title(label, fontsize=10)
            ax_spec.set_visible(False)
            continue

        _, y_h, fs_h = load_hydrophone_segment(wav_path, wav_start, q_start, q_end)
        if len(y_h) < 256:
            ax_pres.text(
                0.5, 0.5, "Segment too short",
                transform=ax_pres.transAxes, ha="center", va="center",
            )
            ax_pres.set_title(label, fontsize=10)
            ax_spec.set_visible(False)
            continue

        # Find middle ping for a 5-second zoom window
        nyq_h = fs_h / 2
        bp_h = firwin(201, [80e3 / nyq_h, 170e3 / nyq_h], pass_zero=False)
        y_bp = lfilter(bp_h, 1.0, y_h)
        env_sm = np.abs(hilbert(y_bp))
        win_smooth = int(0.005 * fs_h)
        env_sm = np.convolve(env_sm, np.ones(win_smooth) / win_smooth, mode="same")
        thr = np.percentile(env_sm, 95)
        min_dist = int(0.3 * fs_h)
        peaks = []
        above = env_sm > thr
        k = 0
        while k < len(above):
            if above[k]:
                j_end = k
                while j_end < len(above) and above[j_end]:
                    j_end += 1
                peaks.append(k + int(np.argmax(env_sm[k:j_end])))
                k = j_end + min_dist
            else:
                k += 1

        center_s = (peaks[len(peaks) // 2] / fs_h) if peaks else (len(y_h) / fs_h / 2)
        half_win = 2.5
        t_lo = max(0, center_s - half_win)
        t_hi = min(len(y_h) / fs_h, center_s + half_win)
        z0 = int(t_lo * fs_h)
        z1 = int(t_hi * fs_h)
        seg_zoom = y_h[z0:z1]
        t_zoom = np.arange(len(seg_zoom)) / fs_h + t_lo

        ax_pres.plot(t_zoom, seg_zoom, "k", lw=0.3)
        ax_pres.set_title(label, fontsize=10, fontweight="bold")
        ax_pres.set_ylabel("Pressure (µPa)")
        ax_pres.grid(True, alpha=0.3)
        ax_pres.set_xlim(t_lo, t_hi)

        nfft_h = max(64, min(1024, len(seg_zoom) // 8))
        ax_spec.specgram(
            seg_zoom, NFFT=nfft_h, Fs=fs_h, noverlap=nfft_h // 2, cmap="viridis"
        )
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        ax_spec.set_xlim(0, t_hi - t_lo)

    fig.suptitle(suptitle, fontsize=13, y=1.01)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig
