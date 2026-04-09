"""EK80 signal processing pipeline: raw ``.raw`` → calibrated absolute TS(f).

This module implements steps 1–8 of Section 2.5 of the manuscript. Steps 1–5
(pulse compression) are a faithful reimplementation of echopype's
``ek80_complex`` module so that the outputs are bit-for-bit comparable to
echopype's internal calibration. Steps 6–7 add the aliasing-aware frequency
mapping and |TX|² deconvolution needed for per-frequency TS(f) spectra (which
echopype does not expose directly). Step 8 is the full sonar equation — the
``compute_absolute_tsf`` function — which mirrors echopype's
``_cal_complex_samples`` and implements Demer et al. (2017) §1.2.4.4 Eq. 13.

Every term in ``compute_absolute_tsf`` is defined and traced in
``docs/SONAR_EQUATION.md``; the provenance for the entire pipeline is in
``docs/PIPELINE.md`` Section 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import fftconvolve as sig_convolve

from . import config
from .io import EK80Metadata, get_ek80_metadata

try:
    from echopype.calibrate.ek80_complex import get_filter_coeff
except ImportError:  # pragma: no cover
    get_filter_coeff = None  # type: ignore


__all__ = [
    "EK80Processed",
    "build_chirp_replica",
    "filter_decimate_chirp",
    "alias_aware_freq_map",
    "pulse_compress_run",
    "compute_absolute_tsf",
    "process_ek80_ed",
]


# ---------------------------------------------------------------------------
# Processed-data container
# ---------------------------------------------------------------------------

@dataclass
class EK80Processed:
    """Everything a downstream routine needs from a processed EK80 file."""

    meta: EK80Metadata
    r_ek: np.ndarray
    tx_filt: np.ndarray
    norm_fac: float
    mf_rep: np.ndarray
    band_idx_sorted: np.ndarray
    f_band_sorted: np.ndarray
    tx_power_band: np.ndarray
    all_pc: np.ndarray
    target_center_m: float
    target_gate: tuple[int, int]
    f_hz_full: np.ndarray  # f_band_sorted (same as above, named for symmetry with EB)
    tsf_calibrated_db: np.ndarray  # calibrated TS(f) via the sonar equation
    extra: dict[str, Any]


# ---------------------------------------------------------------------------
# Step 2 + 3: chirp replica construction + WBT/PC filter chain
# ---------------------------------------------------------------------------

def build_chirp_replica(meta: EK80Metadata) -> np.ndarray:
    """Construct the EK80 transmit chirp replica with Hanning edge tapers.

    Replicates ``echopype.calibrate.ek80_complex.tapered_chirp``. The chirp is
    generated at the 1500 kHz receiver sampling rate before the WBT/PC filter
    chain decimates it to the baseband rate.
    """
    f_start = meta.f_start_hz
    f_stop = meta.f_stop_hz
    t_dur = meta.t_dur_s
    slope = meta.slope
    fs_rx = meta.fs_rx_hz

    n_rf = int(np.floor(t_dur * fs_rx))
    t_rf = np.arange(n_rf) / fs_rx
    y_rf = np.cos(np.pi * (f_stop - f_start) / t_dur * t_rf**2 + 2 * np.pi * f_start * t_rf)

    # Hanning edge taper of total length 2 × slope × n_rf
    l_taper = int(np.round(t_dur * fs_rx * slope * 2.0))
    if l_taper > 1:
        w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(l_taper) / (l_taper - 1)))
        half = l_taper // 2
        y_rf[:half] *= w[:half]
        w2 = w[half:]
        if len(w2):
            y_rf[-len(w2):] *= w2[: len(y_rf[-len(w2):])]

    y_rf /= np.max(np.abs(y_rf))
    return y_rf


def filter_decimate_chirp(
    y_rf: np.ndarray,
    vendor_specific,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Apply the WBT + PC filter cascade and return (tx_filt, norm_fac, mf_rep).

    Parameters
    ----------
    y_rf : np.ndarray
        Raw chirp replica at ``fs_rx`` (1500 kHz).
    vendor_specific : xarray.Dataset
        The ``Vendor_specific`` group from an echopype ``EchoData``.

    Returns
    -------
    tx_filt : np.ndarray
        Chirp after WBT (→ /8) + PC (→ /2) filter chain; real-valued coefficients
        but may have complex phase through the filter taps.
    norm_fac : float
        ``||tx_filt||²`` — used both to normalize pulse-compressed output and
        (added back as ``20·log₁₀(norm_fac)``) to undo that normalization in the
        sonar equation.
    mf_rep : np.ndarray
        ``np.flipud(np.conj(tx_filt))`` — the matched-filter replica used by
        ``pulse_compress_run``.
    """
    if get_filter_coeff is None:  # pragma: no cover
        raise ImportError("echopype is required for filter_decimate_chirp")

    coeff = get_filter_coeff(vendor_specific)
    ch_id = list(coeff.keys())[0]
    wbt_fil = coeff[ch_id]["wbt_fil"]
    pc_fil = coeff[ch_id]["pc_fil"]
    wbt_dec = int(coeff[ch_id]["wbt_decifac"])
    pc_dec = int(coeff[ch_id]["pc_decifac"])

    after_wbt = sig_convolve(y_rf, wbt_fil)[::wbt_dec]
    tx_filt = sig_convolve(after_wbt, pc_fil)[::pc_dec]

    norm_fac = float(np.linalg.norm(tx_filt) ** 2)
    mf_rep = np.flipud(np.conj(tx_filt))
    return tx_filt, norm_fac, mf_rep


# ---------------------------------------------------------------------------
# Step 6: aliasing-aware frequency mapping
# ---------------------------------------------------------------------------

def alias_aware_freq_map(
    nfft: int,
    fs_bb: float,
    f_start: float,
    f_stop: float,
    margin_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Map baseband FFT bins back to their true RF frequencies.

    The EK80 baseband sample rate (~93.75 kHz) is below the chirp bandwidth
    (60 kHz), so the 90–150 kHz band aliases into baseband. For each FFT bin,
    we search across integer folds ``k = -10 … +10`` for an ``f = f_bb + k·fs``
    that lies within ``[f_start·(1-margin), f_stop·(1+margin)]``.

    Returns
    -------
    band_idx_sorted : np.ndarray
        Indices into the baseband FFT bin array, sorted by ascending RF frequency.
    f_band_sorted : np.ndarray
        Corresponding RF frequencies in Hz (same length as ``band_idx_sorted``).
    """
    f_bb = np.fft.fftfreq(nfft, d=1.0 / fs_bb)
    f_actual = np.full(nfft, np.nan)
    f_lo_bound = f_start * (1.0 - margin_frac)
    f_hi_bound = f_stop * (1.0 + margin_frac)

    for i, fb in enumerate(f_bb):
        for k in range(-10, 11):
            ft = fb + k * fs_bb
            if f_lo_bound <= ft <= f_hi_bound:
                f_actual[i] = ft
                break

    band_idx = np.where(~np.isnan(f_actual))[0]
    order = np.argsort(f_actual[band_idx])
    band_idx_sorted = band_idx[order]
    f_band_sorted = f_actual[band_idx_sorted]
    return band_idx_sorted, f_band_sorted


# ---------------------------------------------------------------------------
# Step 4 + 5: pulse compression per ping
# ---------------------------------------------------------------------------

def pulse_compress_run(
    ed,
    mf_rep: np.ndarray,
    norm_fac: float,
) -> np.ndarray:
    """Pulse-compress every ping in an EK80 file.

    Mean-averages the I/Q across transducer beams (elements) *before* pulse
    compression, then convolves with the matched-filter replica and divides by
    ``norm_fac``. This matches echopype's ``compress_pulse`` + ``get_norm_fac``.

    Parameters
    ----------
    ed : echopype.EchoData
        Opened EK80 raw file.
    mf_rep : np.ndarray
        Matched-filter replica from ``filter_decimate_chirp``.
    norm_fac : float
        Pulse-compression normalization factor.

    Returns
    -------
    np.ndarray
        Complex pulse-compressed output, shape ``(n_pings, n_samples)``.
    """
    beam = ed["Sonar/Beam_group1"]
    n_pings = beam.sizes["ping_time"]
    n_samp = beam.sizes["range_sample"]
    bs_r = beam["backscatter_r"].values[0]
    bs_i = beam["backscatter_i"].values[0]

    all_pc = np.zeros((n_pings, n_samp), dtype=complex)
    for p in range(n_pings):
        iq_raw = bs_r[p] + 1j * np.nan_to_num(bs_i[p], nan=0.0)
        raw_iq = np.nanmean(iq_raw, axis=1)  # mean across beams
        pc = sig_convolve(raw_iq, mf_rep, mode="full")[len(mf_rep) - 1:][:n_samp]
        all_pc[p] = pc / norm_fac
    return all_pc


# ---------------------------------------------------------------------------
# Step 8: full sonar equation (Demer et al. 2017 Eq. 13)
# ---------------------------------------------------------------------------

def compute_absolute_tsf(
    spec_raw: np.ndarray,
    tx_power_band: np.ndarray,
    band_idx_sorted: np.ndarray,
    f_band_sorted: np.ndarray,
    r_center_m: float,
    meta: EK80Metadata,
    norm_fac: float,
    c_sound: float = config.C_DEFAULT,
    alpha_db_per_m: float = config.ALPHA_DEFAULT_DB_PER_M,
    z_et_ohm: float = config.Z_ET_OHM,
    n_ref: int = config.N_REF,
) -> np.ndarray:
    """Convert a gated-spectrum FFT output into absolute TS(f) in dB.

    This is the full sonar equation matching echopype's ``_cal_complex_samples``
    (Demer et al. 2017 §1.2.4.4 Eq. 13):

    .. code-block::

        TS(f) = 10·log₁₀(|H(f)|²)                         (1)  power spectrum
              + 20·log₁₀(norm_fac)                        (2)  undo PC norm
              + Z_dB                                      (3)  impedance + beam
              + 40·log₁₀(R)                               (4)  spherical spreading
              + 2·α·R                                     (5)  absorption
              − 10·log₁₀(λ²·Ptx / (16π²))                 (6)  source term
              − 2·G                                       (7)  gain correction

    where the impedance/beam factor is

    .. code-block::

        Z_dB = 10·log₁₀(N_beams/N_ref)
             + 20·log₁₀(|z_er + z_et| / z_er)
             − 10·log₁₀(z_et)

    Every parameter is documented in ``docs/SONAR_EQUATION.md`` along with a
    worked example at 120 kHz that sums to −37.14 dB for the A1 baseline.

    Parameters
    ----------
    spec_raw : np.ndarray
        Coherently-averaged FFT of ``all_pc[gate] * hanning`` at ``nfft`` bins
        (length ``nfft``, complex).
    tx_power_band : np.ndarray
        ``|FFT{tx_filt}|²`` evaluated at ``band_idx_sorted`` with an ``1e-10``
        floor to prevent divide-by-zero.
    band_idx_sorted : np.ndarray
        Indices of in-band FFT bins, sorted by ascending RF frequency (from
        ``alias_aware_freq_map``).
    f_band_sorted : np.ndarray
        Corresponding RF frequencies (Hz).
    r_center_m : float
        Target range in metres (gate center).
    meta : EK80Metadata
        Scalar parameters read from the ``.raw`` file.
    norm_fac : float
        Pulse-compression normalization factor from ``filter_decimate_chirp``.
    c_sound : float
        Sound speed (m/s).
    alpha_db_per_m : float
        Absorption coefficient (dB/m).
    z_et_ohm : float
        Transducer impedance.
    n_ref : int
        Impedance/beam normalization denominator (echopype convention = 8).

    Returns
    -------
    np.ndarray
        Calibrated TS(f) in dB at each frequency in ``f_band_sorted``.
    """
    # Raw deconvolved spectrum at in-band bins
    h = spec_raw[band_idx_sorted] / tx_power_band
    ts_db = 10.0 * np.log10(np.abs(h) ** 2 + 1e-30)         # (1)

    ts_db += 20.0 * np.log10(norm_fac)                       # (2)

    # (3) Impedance + beam scaling
    z_db = (
        10.0 * np.log10(meta.n_beams / n_ref)
        + 20.0 * np.log10(np.abs(meta.impedance_transceiver_ohm + z_et_ohm) / meta.impedance_transceiver_ohm)
        - 10.0 * np.log10(z_et_ohm)
    )
    ts_db += z_db

    ts_db += 40.0 * np.log10(max(r_center_m, 0.01))          # (4)
    ts_db += 2.0 * alpha_db_per_m * r_center_m               # (5)

    lam_f = c_sound / f_band_sorted
    ts_db -= 10.0 * np.log10(lam_f**2 * meta.transmit_power_w / (16.0 * np.pi**2))  # (6)

    ts_db -= 2.0 * meta.gain_correction_db                   # (7)

    return ts_db


# ---------------------------------------------------------------------------
# Top-level convenience wrapper
# ---------------------------------------------------------------------------

def process_ek80_ed(
    ed,
    c_sound: float = config.C_DEFAULT,
    gate_half_m: float = config.GATE_HALF_M,
    nfft: int = config.NFFT,
    f_lo_hz: float = config.F_BAND_LO_HZ,
    f_hi_hz: float = config.F_BAND_HI_HZ,
    target_center_override_m: float | None = None,
    alpha_db_per_m: float = config.ALPHA_DEFAULT_DB_PER_M,
) -> EK80Processed:
    """End-to-end EK80 pipeline from an opened ``EchoData`` to calibrated TS(f)."""
    meta = get_ek80_metadata(ed)
    r_ek = 0.5 * c_sound * np.arange(meta.n_samples) * meta.sample_interval_s

    # Steps 2–3: chirp + filter chain
    y_rf = build_chirp_replica(meta)
    tx_filt, norm_fac, mf_rep = filter_decimate_chirp(y_rf, ed["Vendor_specific"])

    # Step 6: aliasing-aware frequency mapping
    band_idx_sorted, f_band_sorted = alias_aware_freq_map(
        nfft, meta.fs_bb_hz, meta.f_start_hz, meta.f_stop_hz
    )
    tx_fft = np.fft.fft(tx_filt, n=nfft)
    tx_power_band = np.maximum(
        np.abs(tx_fft[band_idx_sorted]) ** 2,
        np.max(np.abs(tx_fft[band_idx_sorted]) ** 2) * 1e-10,
    )

    # Step 4: pulse compress every ping
    all_pc = pulse_compress_run(ed, mf_rep, norm_fac)

    # Step 5: range gate around autodetected target (mean envelope of ping 0)
    if target_center_override_m is not None:
        target_center_m = float(target_center_override_m)
    else:
        env0 = np.abs(all_pc[0])
        # Pulse-duration-aware lower bound (exclude transmit leakthrough)
        pulse_range_m = c_sound * meta.t_dur_s / 2.0
        search_lo = max(config.TARGET_SEARCH_LO_M, pulse_range_m + 0.4)
        i_lo = int(np.searchsorted(r_ek, search_lo))
        i_hi = int(np.searchsorted(r_ek, config.TARGET_SEARCH_HI_M, side="right"))
        target_center_m = float(r_ek[i_lo + int(np.argmax(env0[i_lo:i_hi]))])

    j_lo = int(np.searchsorted(r_ek, target_center_m - gate_half_m))
    j_hi = int(np.searchsorted(r_ek, target_center_m + gate_half_m, side="right"))
    gate_len = j_hi - j_lo

    # Step 7: coherent-average FFT of gated pulse-compressed output
    spec_acc = np.zeros(nfft, dtype=complex)
    for p in range(meta.n_pings):
        spec_acc += np.fft.fft(all_pc[p, j_lo:j_hi] * np.hanning(gate_len), n=nfft)
    spec_avg = spec_acc / meta.n_pings

    # Step 8: absolute TS(f) via the full sonar equation
    tsf_calibrated_db = compute_absolute_tsf(
        spec_avg,
        tx_power_band,
        band_idx_sorted,
        f_band_sorted,
        target_center_m,
        meta,
        norm_fac,
        c_sound=c_sound,
        alpha_db_per_m=alpha_db_per_m,
    )

    return EK80Processed(
        meta=meta,
        r_ek=r_ek,
        tx_filt=tx_filt,
        norm_fac=norm_fac,
        mf_rep=mf_rep,
        band_idx_sorted=band_idx_sorted,
        f_band_sorted=f_band_sorted,
        tx_power_band=tx_power_band,
        all_pc=all_pc,
        target_center_m=target_center_m,
        target_gate=(j_lo, j_hi),
        f_hz_full=f_band_sorted,
        tsf_calibrated_db=tsf_calibrated_db,
        extra={},
    )
