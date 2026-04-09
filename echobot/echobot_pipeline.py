"""EchoBot signal processing pipeline.

End-to-end: raw ``.mat`` data → uncalibrated TS(f) spectrum for a stationary
target. Implements steps 1–6 of Section 2.4 of the manuscript:

    1. Data loading (via ``io.load_echobot_mat``)
    2. Transmit reference construction (zero-padded chirp)
    3. Bandpass filtering (FIR LP 175 kHz + HP 80 kHz)
    4. Matched filtering (cross-correlation per sector, sum across 3 sectors)
    5. Range gating (symmetric window around autodetected target peak)
    6. TS(f) computation (Hanning window + FFT + |TX|² deconvolution + coherent avg)

The output is uncalibrated: the absolute dB level depends on the unknown
EchoBot transmit/receive electronics gain constant. For relative cross-
instrument comparisons, normalize with ``compare.normalize_tsf``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import correlate, firwin, hilbert, lfilter

from . import config
from .io import EchoBotRun

__all__ = [
    "EchoBotProcessed",
    "bp_filter",
    "build_tx_reference",
    "matched_filter_ping",
    "matched_filter_run",
    "autodetect_target_center",
    "tsf_from_mf",
    "process_echobot_run",
]


# ---------------------------------------------------------------------------
# Data class for the fully-processed output
# ---------------------------------------------------------------------------

@dataclass
class EchoBotProcessed:
    """Everything a downstream routine needs from a processed EchoBot run.

    Attributes
    ----------
    run : EchoBotRun
        The source run (handy for introspection — range axis, chirp, etc.).
    r_mf : np.ndarray
        Range axis in metres for the matched-filter output (one-way × 2 /c).
    all_mf : np.ndarray
        Matched-filter output, shape ``(n_pings, n_samples)``. Real-valued
        (3-sector sum). Use ``np.abs(scipy.signal.hilbert(...))`` for envelope.
    target_center_m : float
        Autodetected target range in metres (peak of mean matched-filter envelope).
    floor_center_m : float
        Autodetected tank-floor range in metres.
    target_gate : tuple[int, int]
        ``(j_lo, j_hi)`` sample indices for the ±gate_half target window.
    noise_gate : tuple[int, int]
        Sample indices for an inter-region noise gate (between target and floor).
    f_hz : np.ndarray
        Frequency axis for the TS(f) spectrum (length = ``Nfft``).
    tsf_db : np.ndarray
        Coherently-averaged uncalibrated TS(f) in dB, shape ``(Nfft,)``.
    band_mask : np.ndarray
        Boolean mask selecting the 90–150 kHz (or user-specified) analysis band.
    extra : dict
        Reserved for downstream routines to attach derived quantities.
    """

    run: EchoBotRun
    r_mf: np.ndarray
    all_mf: np.ndarray
    target_center_m: float
    floor_center_m: float
    target_gate: tuple[int, int]
    noise_gate: tuple[int, int]
    f_hz: np.ndarray
    tsf_db: np.ndarray
    band_mask: np.ndarray
    extra: dict[str, Any]


# ---------------------------------------------------------------------------
# Step 3: bandpass filter
# ---------------------------------------------------------------------------

def bp_filter(
    x: np.ndarray,
    fs: float,
    lp_cutoff_hz: float = config.EB_BP_LP_HZ,
    hp_cutoff_hz: float = config.EB_BP_HP_HZ,
    n_taps: int = config.EB_BP_NTAPS,
) -> np.ndarray:
    """Apply an FIR bandpass (LP + HP cascade) to a 1-D real signal.

    Default cutoffs (LP 175 kHz, HP 80 kHz) are wider than the original MATLAB
    implementation's (LP 150 kHz, HP 90 kHz) so the full chirp energy — including
    spectral rolloff at the band edges — passes through unattenuated. Final band
    selection is delegated to the matched filter in Step 4. See
    ``docs/PIPELINE.md`` Section 2.4 Step 3 for the full justification.
    """
    lp = firwin(n_taps, lp_cutoff_hz, fs=fs)
    hp = firwin(n_taps, hp_cutoff_hz, fs=fs, pass_zero=False)
    return lfilter(hp, 1.0, lfilter(lp, 1.0, x))


# ---------------------------------------------------------------------------
# Step 2: transmit reference construction
# ---------------------------------------------------------------------------

def build_tx_reference(
    s_chirp: np.ndarray,
    t_pre: float,
    t_post: float,
    fs: float,
    pad_to: int | None = None,
) -> np.ndarray:
    """Construct the zero-padded transmit reference ``[zeros(T_pre), s_chirp, zeros(T_post)]``.

    This matches the MATLAB ``Bob_txrx_init.m`` convention (lines 73–77). If
    ``pad_to`` is given and is longer than the natural length, additional trailing
    zeros are added so the reference has the same length as the recorded data.
    """
    n_pre = int(round(t_pre * fs))
    n_post = int(round(t_post * fs))
    tx_ref = np.concatenate([np.zeros(n_pre), s_chirp, np.zeros(n_post)])
    if pad_to is not None and len(tx_ref) < pad_to:
        tx_ref = np.concatenate([tx_ref, np.zeros(pad_to - len(tx_ref))])
    return tx_ref


# ---------------------------------------------------------------------------
# Step 4: matched filter
# ---------------------------------------------------------------------------

def matched_filter_ping(
    data: np.ndarray,
    tx_ref: np.ndarray,
    ping_idx: int,
    fs: float,
    n_channels_to_sum: int = 3,
) -> np.ndarray:
    """Bandpass + matched-filter one ping, summing across the first N channels.

    Parameters
    ----------
    data : np.ndarray
        Raw sample array with shape ``(n_samples, n_channels, n_pings)``.
    tx_ref : np.ndarray
        Zero-padded transmit reference from ``build_tx_reference``.
    ping_idx : int
        Index into the ping dimension.
    fs : float
        Sample rate (Hz).
    n_channels_to_sum : int, default 3
        Number of receive channels (sectors) to sum after matched filtering.
        Matches the MATLAB pipeline (3 active sectors).

    Returns
    -------
    np.ndarray
        Real-valued matched-filter output of shape ``(n_samples,)``.
    """
    ns = data.shape[0]
    nref = len(tx_ref)
    c_sum = np.zeros(ns)
    for ch in range(min(n_channels_to_sum, data.shape[1])):
        filtered = bp_filter(data[:, ch, ping_idx].astype(float), fs)
        cc = correlate(filtered, tx_ref, mode="full")
        # Take the causal lag range aligned with the data (length ns starting
        # at lag zero); this mirrors the MATLAB ``cc(n_ref:n_ref+Ns-1)`` slice.
        c_sum += cc[nref - 1 : nref - 1 + ns]
    return c_sum


def matched_filter_run(run: EchoBotRun) -> np.ndarray:
    """Apply the matched filter to every ping in a run.

    Returns
    -------
    np.ndarray
        Matched-filter output with shape ``(n_pings, n_samples)``.
    """
    tx_ref = build_tx_reference(
        run.s_chirp, run.t_pre, run.t_post, run.fs, pad_to=run.n_samples
    )
    out = np.zeros((run.n_pings, run.n_samples))
    for p in range(run.n_pings):
        out[p] = matched_filter_ping(run.data, tx_ref, p, run.fs)
    return out


# ---------------------------------------------------------------------------
# Step 5: range gating & target autodetection
# ---------------------------------------------------------------------------

def autodetect_target_center(
    all_mf: np.ndarray,
    r_mf: np.ndarray,
    search_lo_m: float = config.TARGET_SEARCH_LO_M,
    search_hi_m: float = config.TARGET_SEARCH_HI_M,
) -> float:
    """Return the range (m) of the envelope peak inside the target search window.

    Uses the first ping only for autodetection (the sphere is stationary, so any
    ping gives the same peak to sub-millimetre precision).
    """
    env0 = np.abs(hilbert(all_mf[0]))
    i_lo = np.searchsorted(r_mf, search_lo_m)
    i_hi = np.searchsorted(r_mf, search_hi_m, side="right")
    return float(r_mf[i_lo + int(np.argmax(env0[i_lo:i_hi]))])


def autodetect_floor_center(
    all_mf: np.ndarray,
    r_mf: np.ndarray,
    search_lo_m: float = config.FLOOR_SEARCH_LO_M,
    search_hi_m: float = config.FLOOR_SEARCH_HI_M,
) -> float:
    """Return the range (m) of the tank-floor envelope peak."""
    env0 = np.abs(hilbert(all_mf[0]))
    i_lo = np.searchsorted(r_mf, search_lo_m)
    i_hi = np.searchsorted(r_mf, search_hi_m, side="right")
    return float(r_mf[i_lo + int(np.argmax(env0[i_lo:i_hi]))])


def gate_indices(
    r_mf: np.ndarray,
    center_m: float,
    half_m: float = config.GATE_HALF_M,
) -> tuple[int, int]:
    """Return ``(j_lo, j_hi)`` sample indices for a symmetric ±half_m gate."""
    j_lo = int(np.searchsorted(r_mf, center_m - half_m))
    j_hi = int(np.searchsorted(r_mf, center_m + half_m, side="right"))
    return j_lo, j_hi


# ---------------------------------------------------------------------------
# Step 6: TS(f) computation
# ---------------------------------------------------------------------------

def tsf_from_mf(
    all_mf: np.ndarray,
    target_gate: tuple[int, int],
    s_chirp: np.ndarray,
    fs: float,
    nfft: int = config.NFFT,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute coherently-averaged uncalibrated TS(f) from matched-filter output.

    The spectral deconvolution uses ``|FFT{s_chirp}|²`` as the denominator (Demer
    et al. 2017 Eq. 11) and a Hanning window on the gated matched-filter output
    before the FFT.

    Parameters
    ----------
    all_mf : np.ndarray
        Matched-filter output, shape ``(n_pings, n_samples)``.
    target_gate : tuple[int, int]
        ``(j_lo, j_hi)`` sample indices of the range gate.
    s_chirp : np.ndarray
        Transmit chirp (1-D, raw waveform from the run header).
    fs : float
        Sample rate (Hz).
    nfft : int
        FFT length.

    Returns
    -------
    f_hz : np.ndarray
        Frequency axis in Hz, length ``nfft``.
    tsf_db : np.ndarray
        Uncalibrated TS(f) in dB, length ``nfft``.
    """
    j_lo, j_hi = target_gate
    gate_len = j_hi - j_lo
    if gate_len <= 0:
        raise ValueError(f"Empty gate: indices {target_gate}")

    # Coherent average across pings
    spec_acc = np.zeros(nfft, dtype=complex)
    for p in range(all_mf.shape[0]):
        windowed = all_mf[p, j_lo:j_hi] * np.hanning(gate_len)
        spec_acc += np.fft.fft(windowed, n=nfft)
    spec_avg = spec_acc / all_mf.shape[0]

    # Deconvolve by |TX|²
    tx_fft = np.fft.fft(s_chirp, n=nfft)
    tx_pow = np.maximum(np.abs(tx_fft) ** 2, np.max(np.abs(tx_fft) ** 2) * 1e-10)
    tsf_lin = np.abs(spec_avg / tx_pow) + 1e-30
    tsf_db = 20.0 * np.log10(tsf_lin)

    f_hz = np.arange(nfft) * (fs / nfft)
    return f_hz, tsf_db


# ---------------------------------------------------------------------------
# One-call top-level
# ---------------------------------------------------------------------------

def process_echobot_run(
    run: EchoBotRun,
    c_sound: float = config.C_DEFAULT,
    gate_half_m: float = config.GATE_HALF_M,
    nfft: int = config.NFFT,
    f_lo_hz: float = config.F_BAND_LO_HZ,
    f_hi_hz: float = config.F_BAND_HI_HZ,
    target_center_override_m: float | None = None,
) -> EchoBotProcessed:
    """Run the full EchoBot pipeline on one ``EchoBotRun``.

    Parameters
    ----------
    run : EchoBotRun
        Loaded from ``io.load_echobot_mat``.
    c_sound : float
        Sound speed in m/s used to build the range axis. Default pulls from
        ``config.C_DEFAULT``; pass a per-session CTD value for best accuracy.
    gate_half_m : float
        Half-width of the target range gate.
    nfft : int
        FFT length for TS(f) computation.
    f_lo_hz, f_hi_hz : float
        Analysis band; used only for the ``band_mask`` output.
    target_center_override_m : float, optional
        Skip autodetection and use this range as the gate center. Useful for
        the fixed-center sensitivity analyses in supplementary Section S4.

    Returns
    -------
    EchoBotProcessed
    """
    # Range axis (one-way × 2/c per sample)
    r_mf = 0.5 * c_sound * np.arange(run.n_samples) / run.fs

    # Matched filter every ping
    all_mf = matched_filter_run(run)

    # Autodetect target + floor (or use override)
    if target_center_override_m is not None:
        target_center_m = float(target_center_override_m)
    else:
        target_center_m = autodetect_target_center(all_mf, r_mf)
    floor_center_m = autodetect_floor_center(all_mf, r_mf)

    # Gates
    target_gate = gate_indices(r_mf, target_center_m, gate_half_m)
    # Noise gate: between target+0.15 m and floor-0.6 m (fallback to pre-target
    # window if the interval collapses, e.g. for no-target runs)
    n_lo = target_center_m + gate_half_m + 0.15
    n_hi = floor_center_m - 0.6
    if n_hi - n_lo < 0.2:
        n_lo = config.TARGET_SEARCH_LO_M
        n_hi = target_center_m - gate_half_m - 0.1
    noise_gate = gate_indices(r_mf, 0.5 * (n_lo + n_hi), 0.5 * (n_hi - n_lo))

    # TS(f)
    f_hz, tsf_db = tsf_from_mf(all_mf, target_gate, run.s_chirp, run.fs, nfft)
    band_mask = (f_hz >= f_lo_hz) & (f_hz <= f_hi_hz)

    return EchoBotProcessed(
        run=run,
        r_mf=r_mf,
        all_mf=all_mf,
        target_center_m=target_center_m,
        floor_center_m=floor_center_m,
        target_gate=target_gate,
        noise_gate=noise_gate,
        f_hz=f_hz,
        tsf_db=tsf_db,
        band_mask=band_mask,
        extra={},
    )
