"""Two signal-to-noise ratio methods used in the manuscript.

Both methods return per-ping SNR in dB. They differ in where the noise reference
is drawn from:

- ``snr_no_target_ref`` uses the grand-mean matched-filter envelope of a
  dedicated no-target run at the same range as the target gate. This is the
  "gold standard" because the reference noise and the target signal are
  measured under identical conditions.

- ``snr_empty_region`` uses a within-ping empty-water-column range window (by
  default 2.1–2.6 m, between the target and the tank floor) as the noise
  reference. This works for every run — no dedicated no-target data required —
  and was applied to all comparison groups in Supplementary Table S1 of the
  manuscript.

Both functions are intentionally small and pure so they can be composed into
per-group SNR tables from processed ``EchoBotProcessed`` / ``EK80Processed``
containers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

from . import config


__all__ = [
    "SNRResult",
    "peak_env_in_gate",
    "snr_no_target_ref",
    "snr_empty_region",
]


@dataclass
class SNRResult:
    """Per-ping and aggregate SNR values.

    Attributes
    ----------
    per_ping_db : np.ndarray
        SNR for each ping in dB, shape ``(n_pings,)``.
    mean_db : float
    std_db : float
    method : str
        Either ``"no_target_ref"`` or ``"empty_region"``.
    """

    per_ping_db: np.ndarray
    mean_db: float
    std_db: float
    method: str

    def __repr__(self) -> str:
        return f"SNRResult(method={self.method!r}, mean={self.mean_db:.2f} dB, std={self.std_db:.2f} dB, n={len(self.per_ping_db)})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def peak_env_in_gate(
    all_mf: np.ndarray,
    gate: tuple[int, int],
) -> np.ndarray:
    """Peak matched-filter envelope amplitude inside a range gate, per ping.

    Returns
    -------
    np.ndarray
        One non-negative peak value per ping, shape ``(n_pings,)``. If ``all_mf``
        is real (EchoBot matched-filter output), ``scipy.signal.hilbert`` is
        used to form the analytic envelope; if it is complex (EK80 pulse-
        compressed output) the magnitude is taken directly.
    """
    j_lo, j_hi = gate
    gated = all_mf[:, j_lo:j_hi]
    if np.iscomplexobj(gated):
        env = np.abs(gated)
    else:
        env = np.abs(hilbert(gated, axis=1))
    return env.max(axis=1)


def _rms_in_gate(all_mf: np.ndarray, gate: tuple[int, int]) -> np.ndarray:
    """RMS matched-filter envelope inside a range gate, per ping."""
    j_lo, j_hi = gate
    gated = all_mf[:, j_lo:j_hi]
    if np.iscomplexobj(gated):
        env = np.abs(gated)
    else:
        env = np.abs(hilbert(gated, axis=1))
    return np.sqrt(np.mean(env**2, axis=1))


# ---------------------------------------------------------------------------
# Method 1: dedicated no-target reference run
# ---------------------------------------------------------------------------

def snr_no_target_ref(
    target_all_mf: np.ndarray,
    target_gate: tuple[int, int],
    no_target_all_mf: np.ndarray,
) -> SNRResult:
    """Per-ping SNR using a dedicated no-target run as the noise reference.

    The signal is the per-ping peak envelope inside the target gate of the
    *target* run. The noise reference is the grand-mean (across pings and
    samples inside the same target gate) of the envelope from the *no-target*
    run. Using a grand mean makes the reference ping-count-invariant and
    suppresses the ping-to-ping fluctuations in the noise estimate.

    Parameters
    ----------
    target_all_mf : np.ndarray
        Matched-filter output of the target run, shape ``(n_pings, n_samples)``.
        Real for EchoBot, complex for EK80 pulse-compressed output.
    target_gate : tuple[int, int]
        ``(j_lo, j_hi)`` sample indices of the target range gate.
    no_target_all_mf : np.ndarray
        Matched-filter output of the dedicated no-target run, same orientation
        and gate indexing as ``target_all_mf``.

    Returns
    -------
    SNRResult
        Per-ping SNR in dB + summary statistics.
    """
    target_peaks = peak_env_in_gate(target_all_mf, target_gate)

    j_lo, j_hi = target_gate
    nt_gated = no_target_all_mf[:, j_lo:j_hi]
    if np.iscomplexobj(nt_gated):
        nt_env = np.abs(nt_gated)
    else:
        nt_env = np.abs(hilbert(nt_gated, axis=1))
    noise_ref = float(nt_env.mean())  # grand mean across pings × samples

    if noise_ref <= 0:
        raise ValueError("no-target noise reference is zero or negative")

    per_ping_db = 20.0 * np.log10(target_peaks / noise_ref)
    return SNRResult(
        per_ping_db=per_ping_db,
        mean_db=float(per_ping_db.mean()),
        std_db=float(per_ping_db.std()),
        method="no_target_ref",
    )


# ---------------------------------------------------------------------------
# Method 2: within-ping empty-water-column reference
# ---------------------------------------------------------------------------

def snr_empty_region(
    all_mf: np.ndarray,
    r_mf: np.ndarray,
    target_gate: tuple[int, int],
    empty_region_m: tuple[float, float] = (
        config.EMPTY_REGION_LO_M,
        config.EMPTY_REGION_HI_M,
    ),
) -> SNRResult:
    """Per-ping SNR using a within-ping empty-water-column reference.

    For each ping, the signal is the peak envelope inside the target gate, and
    the noise is the RMS envelope inside an empty-water range window located
    between the target and the tank floor (default 2.1–2.6 m). No separate
    no-target run is needed, so this method applies to every comparison group.

    Parameters
    ----------
    all_mf : np.ndarray
        Matched-filter output, shape ``(n_pings, n_samples)``.
    r_mf : np.ndarray
        Range axis (m) for the sample dimension.
    target_gate : tuple[int, int]
        ``(j_lo, j_hi)`` sample indices of the target range gate.
    empty_region_m : tuple[float, float], optional
        Empty-water-column range window in metres.

    Returns
    -------
    SNRResult
        Per-ping SNR in dB + summary statistics.
    """
    target_peaks = peak_env_in_gate(all_mf, target_gate)

    j_nlo = int(np.searchsorted(r_mf, empty_region_m[0]))
    j_nhi = int(np.searchsorted(r_mf, empty_region_m[1], side="right"))
    if j_nhi - j_nlo < 2:
        raise ValueError(
            f"Empty-region gate too narrow: indices {j_nlo}-{j_nhi} for range "
            f"{empty_region_m[0]}-{empty_region_m[1]} m"
        )

    noise_rms = _rms_in_gate(all_mf, (j_nlo, j_nhi))
    # Avoid log(0) on perfectly-zero pings (synthetic data edge case)
    noise_rms = np.where(noise_rms > 0, noise_rms, np.nan)

    per_ping_db = 20.0 * np.log10(target_peaks / noise_rms)
    per_ping_db = per_ping_db[np.isfinite(per_ping_db)]

    return SNRResult(
        per_ping_db=per_ping_db,
        mean_db=float(per_ping_db.mean()),
        std_db=float(per_ping_db.std()),
        method="empty_region",
    )
