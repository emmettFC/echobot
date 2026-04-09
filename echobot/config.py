"""Physical and processing constants used throughout the EchoBot pipelines.

All constants can be overridden per call by passing an explicit value to any
pipeline function. These defaults reflect the March 2026 CRL tank experiments
documented in the accompanying manuscript.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Environmental / physical
# ---------------------------------------------------------------------------

C_DEFAULT: float = 1484.96
"""Sound speed in m/s.

March 2026 CRL tank, Mackenzie (1981) formula applied to CTD measurements at the
sphere depth. Update per deployment using a fresh CTD cast.
"""

ALPHA_DEFAULT_DB_PER_M: float = 0.04
"""Acoustic absorption coefficient in dB/m (one-way).

Broadband average over 90–150 kHz for tank conditions (~15 °C, ~30 PSU, depth
< 2 m). The two-way error from using a flat value instead of the frequency-
dependent Francois–Garrison absorption is < 0.1 dB over the band at 1.4 m range.
"""

# ---------------------------------------------------------------------------
# EK80 sonar-equation constants (Demer et al. 2017 §1.2.4.4, Eq. 13)
# ---------------------------------------------------------------------------

Z_ET_OHM: float = 75.0
"""Transducer (ET) impedance in ohms.

Hardcoded constant from Demer et al. (2017) §1.2.4.4; used by echopype's internal
``_cal_complex_samples`` as the default. A transducer-specific impedance
measurement would refine this value.
"""

N_REF: int = 8
"""Normalization denominator in the impedance/beam scaling factor.

Echopype ``_cal_complex_samples`` convention: this combines the ``/4`` four-
element split-beam voltage division and the ``/2`` factor arising from the
``mean_beam`` proxy for the element sum in the received-power calculation.
"""

# ---------------------------------------------------------------------------
# TS(f) processing parameters
# ---------------------------------------------------------------------------

NFFT: int = 4096
"""FFT length for TS(f) computation.

Both the EchoBot and EK80 pipelines use the same NFFT so that their spectral
resolutions match across the 90–150 kHz band. 4096 yields ~2 kHz resolution,
sufficient to resolve the 38.1 mm WC sphere's spectral features (notably the
~135 kHz null) without oversampling to no physical benefit.
"""

GATE_HALF_M: float = 0.40
"""Half-width of the range gate around the autodetected target center, in meters.

Selected from the gate-sensitivity analysis in Section 3.9 of the manuscript:
wider gates admit tank-floor reflections; narrower gates degrade cross-instrument
correlation due to the 4.0 cm inter-instrument target peak offset.
"""

F_BAND_LO_HZ: float = 90e3
F_BAND_HI_HZ: float = 150e3
"""Default analysis band (EchoBot standard chirp: 90–150 kHz)."""

F_OVERLAP_LO_HZ: float = 100e3
F_OVERLAP_HI_HZ: float = 140e3
"""Overlap band used when normalizing TS(f) for bandwidth-sensitivity comparisons
that include the C1 group (100–140 kHz). Used by ``compare.normalize_tsf``.
"""

# ---------------------------------------------------------------------------
# EchoBot bandpass filter
# ---------------------------------------------------------------------------

EB_BP_LP_HZ: float = 175e3
EB_BP_HP_HZ: float = 80e3
EB_BP_NTAPS: int = 101
"""EchoBot matched-filter front-end bandpass (FIR, firwin).

Widened vs the original MATLAB implementation (which used LP 150 kHz / HP 90 kHz)
to pass the full chirp energy including spectral rolloff at the band edges; final
band selection is delegated to the matched-filter itself in step 4 of the EB
pipeline. See ``docs/PIPELINE.md`` Section 2.4 Step 3 for the full justification.
"""

# ---------------------------------------------------------------------------
# Target and noise detection
# ---------------------------------------------------------------------------

TARGET_SEARCH_LO_M: float = 0.8
TARGET_SEARCH_HI_M: float = 2.0
"""Range window to search for the target envelope peak."""

FLOOR_SEARCH_LO_M: float = 2.4
FLOOR_SEARCH_HI_M: float = 3.5
"""Range window to search for the tank-floor envelope peak."""

EMPTY_REGION_LO_M: float = 2.1
EMPTY_REGION_HI_M: float = 2.6
"""Within-ping empty-water-column gate used by ``snr.snr_empty_region``.

Located between the target gate (~1.4 m) and the tank floor (~2.9 m) so that it
contains no target and no floor reflection — a pure noise reference usable in any
ping without a separate no-target run.
"""

# ---------------------------------------------------------------------------
# Per-instrument A1 target centers (March 2026 data)
# ---------------------------------------------------------------------------

EB_CENTER_A1_M: float = 1.4259
EK_CENTER_A1_M: float = 1.3860
"""Autodetected per-instrument target centers for the A1 baseline group.

These differ by ~4.0 cm because the two instruments are mounted at slightly
different positions relative to the sphere; the difference is absorbed by
per-instrument gate centering.
"""
