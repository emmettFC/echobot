"""EchoBot broadband echosounder — signal processing and validation package.

This package provides the signal processing, calibration, and cross-instrument
comparison routines used in the accompanying manuscript on the MIT/WHOI EchoBot
broadband echosounder.

Submodules
----------
config
    Physical and processing constants (sound speed, absorption, impedance, etc.).
io
    File loading for EchoBot ``.mat``, EK80 ``.raw``, and SoundTrap ``.wav`` data.
echobot_pipeline
    EchoBot bandpass → matched-filter → range-gate → uncalibrated TS(f).
ek80_pipeline
    EK80 chirp reconstruction → WBT/PC filters → pulse compression → calibrated TS(f)
    via the full sonar equation (``compute_absolute_tsf``).
snr
    Two SNR methods: ``snr_no_target_ref`` (dedicated no-target reference run) and
    ``snr_empty_region`` (within-ping empty-region reference).
compare
    Cross-instrument TS(f) normalization and Pearson correlation.
linearity
    Power / pulse-duration / bandwidth linearity analyses.
hydrophone
    SoundTrap hydrophone processing, including the hydrophone-as-target runs used
    to build the EchoBot dB ↔ EK80 Watt receive-level mapping.
plotting
    Shared plotting helpers (EK500 colormap, TS(f) overlays, SNR histograms, and
    the calibration-validation and absolute-TS(f) figures from the manuscript
    supplementary material).

See ``README.md`` and ``docs/`` for the pipeline provenance, sonar equation
parameter table, and data layout.
"""

__version__ = "1.0.0"

from . import config  # noqa: F401
