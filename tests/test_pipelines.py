"""Smoke tests for the echobot package.

These tests exercise the pipeline functions against *synthetic* inputs so the
test suite runs in a few seconds without any of the large raw data from the
Zenodo archive. Real-data end-to-end reproduction is done in
``notebooks/02_manuscript_analysis.ipynb`` and
``scripts/generate_manuscript_figures.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from echobot import (
    compare,
    config,
    echobot_pipeline,
    ek80_pipeline,
    linearity,
    snr,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_chirp():
    """Generate a linear FM chirp sampled at 2 MHz (EchoBot sample rate)."""
    fs = 2_000_000.0
    t_dur = 5e-4  # 0.5 ms
    n = int(fs * t_dur)
    t = np.arange(n) / fs
    f0, f1 = 90e3, 150e3
    chirp = np.cos(np.pi * (f1 - f0) / t_dur * t**2 + 2 * np.pi * f0 * t)
    # Apply Hanning taper to mimic the real transmit pulse
    w = np.hanning(n)
    chirp *= w
    return fs, chirp


@pytest.fixture
def synthetic_eb_mf(synthetic_chirp):
    """Build a synthetic matched-filter-output-shaped array with a clean target peak.

    Three pings × ``n_samples`` samples, with a single sharp peak at 1.4 m.
    Used to exercise the TS(f) and SNR pipelines without needing a real .mat file.
    """
    fs, _ = synthetic_chirp
    c = config.C_DEFAULT
    n_samples = int(3.5 * 2 * fs / c)  # enough for a 3.5 m range axis

    # Target at 1.4 m → sample index
    target_range = 1.4
    j_target = int(2 * target_range * fs / c)

    rng = np.random.default_rng(42)
    all_mf = rng.normal(scale=0.001, size=(3, n_samples))  # low-level noise
    pulse_half = 10
    for p in range(3):
        # Narrow Gaussian target pulse
        g = np.exp(-((np.arange(-pulse_half, pulse_half + 1)) ** 2) / 8.0)
        all_mf[p, j_target - pulse_half : j_target + pulse_half + 1] += g

    r_mf = 0.5 * c * np.arange(n_samples) / fs
    return all_mf, r_mf, fs


# ---------------------------------------------------------------------------
# EchoBot pipeline tests
# ---------------------------------------------------------------------------

def test_bp_filter_runs(synthetic_chirp):
    fs, chirp = synthetic_chirp
    out = echobot_pipeline.bp_filter(chirp, fs)
    assert out.shape == chirp.shape
    assert np.all(np.isfinite(out))


def test_build_tx_reference_zero_padding(synthetic_chirp):
    fs, chirp = synthetic_chirp
    tx = echobot_pipeline.build_tx_reference(chirp, t_pre=1e-4, t_post=1e-4, fs=fs)
    # Pre-pad + chirp + post-pad
    expected_len = int(round(1e-4 * fs)) + len(chirp) + int(round(1e-4 * fs))
    assert len(tx) == expected_len
    # Pre/post regions are exactly zero
    n_pre = int(round(1e-4 * fs))
    assert np.allclose(tx[:n_pre], 0.0)
    assert np.allclose(tx[-n_pre:], 0.0)


def test_autodetect_target_center(synthetic_eb_mf):
    all_mf, r_mf, _ = synthetic_eb_mf
    center = echobot_pipeline.autodetect_target_center(all_mf, r_mf)
    # Synthetic target is at 1.4 m; tolerance set by range-axis resolution
    assert abs(center - 1.4) < 0.01


def test_tsf_from_mf_shape(synthetic_eb_mf, synthetic_chirp):
    all_mf, r_mf, fs = synthetic_eb_mf
    _, chirp = synthetic_chirp
    j_lo = int(np.searchsorted(r_mf, 1.0))
    j_hi = int(np.searchsorted(r_mf, 1.8))
    f_hz, tsf_db = echobot_pipeline.tsf_from_mf(all_mf, (j_lo, j_hi), chirp, fs)
    assert len(f_hz) == config.NFFT
    assert len(tsf_db) == config.NFFT
    assert np.all(np.isfinite(tsf_db))


# ---------------------------------------------------------------------------
# EK80 sonar equation tests
# ---------------------------------------------------------------------------

class _FakeMeta:
    """Stand-in for ``io.EK80Metadata`` with only the fields ``compute_absolute_tsf`` needs."""

    def __init__(
        self,
        n_beams=3,
        impedance_transceiver_ohm=10800.0,
        transmit_power_w=60.0,
        gain_correction_db=18.0,
    ):
        self.n_beams = n_beams
        self.impedance_transceiver_ohm = impedance_transceiver_ohm
        self.transmit_power_w = transmit_power_w
        self.gain_correction_db = gain_correction_db


def test_compute_absolute_tsf_worked_example():
    """Reproduces the A1 worked example from docs/SONAR_EQUATION.md.

    At 120 kHz with the A1 baseline parameters, the raw ``10·log10(|H|²)`` is
    -59.224 dB and the seven calibration terms sum to -37.136 dB.
    """
    meta = _FakeMeta()
    norm_fac = 44.1819  # exact A1 value
    r_center = 1.386
    c = 1484.96
    alpha = 0.04
    nfft = 4096

    # Build a synthetic single-bin "spectrum" that produces 10·log10(|H|²) = -59.224 dB
    # after the deconvolution by tx_power. We set tx_power = 1 at that bin and
    # |spec| = sqrt(10^(-59.224/10)).
    f_target = 120e3
    band_idx_sorted = np.array([0])
    f_band_sorted = np.array([f_target])
    tx_power_band = np.array([1.0])
    target_h_lin = 10 ** (-59.224 / 10.0)
    spec_raw = np.zeros(nfft, dtype=complex)
    spec_raw[0] = np.sqrt(target_h_lin)  # real, magnitude only

    ts_db = ek80_pipeline.compute_absolute_tsf(
        spec_raw,
        tx_power_band,
        band_idx_sorted,
        f_band_sorted,
        r_center,
        meta,
        norm_fac,
        c_sound=c,
        alpha_db_per_m=alpha,
    )

    # Expected from worked example: -37.136 dB (to 2 decimals)
    assert ts_db.shape == (1,)
    assert abs(ts_db[0] - (-37.136)) < 0.05


def test_alias_aware_freq_map_returns_in_band():
    band_idx, f_band = ek80_pipeline.alias_aware_freq_map(
        nfft=4096, fs_bb=93750.0, f_start=90e3, f_stop=150e3
    )
    assert len(band_idx) == len(f_band)
    assert len(band_idx) > 0
    # All returned frequencies should be inside the ±10% margin
    assert f_band.min() >= 90e3 * 0.90
    assert f_band.max() <= 150e3 * 1.10
    # Sorted ascending
    assert np.all(np.diff(f_band) >= 0)


# ---------------------------------------------------------------------------
# SNR tests
# ---------------------------------------------------------------------------

def test_snr_empty_region_on_synthetic_target(synthetic_eb_mf):
    all_mf, r_mf, _ = synthetic_eb_mf
    j_lo = int(np.searchsorted(r_mf, 1.0))
    j_hi = int(np.searchsorted(r_mf, 1.8))
    result = snr.snr_empty_region(all_mf, r_mf, (j_lo, j_hi))
    # Synthetic target is ~1000× the noise floor → SNR > 30 dB
    assert result.mean_db > 30.0
    assert result.method == "empty_region"


def test_snr_no_target_ref_on_synthetic(synthetic_eb_mf):
    all_mf, r_mf, _ = synthetic_eb_mf
    # Use a copy with no target (pure noise) as the reference
    rng = np.random.default_rng(7)
    nt = rng.normal(scale=0.001, size=all_mf.shape)
    j_lo = int(np.searchsorted(r_mf, 1.0))
    j_hi = int(np.searchsorted(r_mf, 1.8))
    result = snr.snr_no_target_ref(all_mf, (j_lo, j_hi), nt)
    assert result.mean_db > 30.0
    assert result.method == "no_target_ref"


# ---------------------------------------------------------------------------
# compare module tests
# ---------------------------------------------------------------------------

def test_normalize_tsf_mean_zero():
    f = np.linspace(90e3, 150e3, 100)
    tsf = np.sin(np.linspace(0, 2 * np.pi, 100)) + 5.0
    out = compare.normalize_tsf(tsf, f, f_lo_hz=100e3, f_hi_hz=140e3)
    in_band = (f >= 100e3) & (f <= 140e3)
    assert abs(out[in_band].mean()) < 1e-10


def test_cross_instrument_r_identical_curves_gives_one():
    f = np.linspace(90e3, 150e3, 500)
    tsf = np.sin(np.linspace(0, 4 * np.pi, 500))
    res = compare.cross_instrument_r(tsf, f, tsf, f)
    assert res.pearson_r == pytest.approx(1.0, abs=1e-12)


def test_cross_instrument_r_anticorrelated_gives_minus_one():
    f = np.linspace(90e3, 150e3, 500)
    tsf_a = np.sin(np.linspace(0, 4 * np.pi, 500))
    tsf_b = -tsf_a
    res = compare.cross_instrument_r(tsf_a, f, tsf_b, f)
    assert res.pearson_r == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# linearity tests
# ---------------------------------------------------------------------------

def test_power_linearity_ideal_slope_is_one():
    """A perfectly linear system should return slope = 1 and zero residual."""

    class _FakeProc:
        def __init__(self, level_db):
            self.target_gate = (0, 100)
            # Build an array whose peak is exactly ``level_db`` (complex so the
            # SNR helper uses ``np.abs`` directly and avoids the hilbert path).
            linear = 10 ** (level_db / 20.0)
            self.all_pc = np.full((1, 100), linear, dtype=complex)

    series = [
        dict(nominal_db=0.0, processed=_FakeProc(0.0)),
        dict(nominal_db=3.0, processed=_FakeProc(3.0)),
        dict(nominal_db=6.0, processed=_FakeProc(6.0)),
    ]
    res = linearity.power_linearity(series, "synthetic")
    assert res.slope == pytest.approx(1.0, abs=1e-6)
    assert res.rms_residual_db == pytest.approx(0.0, abs=1e-6)
