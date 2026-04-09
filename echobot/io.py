"""File loading and timestamp parsing for EchoBot, EK80, and hydrophone data.

All loaders return plain Python dictionaries or xarray/echopype objects rather
than custom classes, so downstream code remains explicit about what fields are
used and callers can inspect loaded data easily in a notebook.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.io import wavfile

try:  # echopype is optional at import time; raise a clearer error if absent
    import echopype as ep
except ImportError:  # pragma: no cover
    ep = None  # type: ignore


# ---------------------------------------------------------------------------
# EchoBot .mat loader
# ---------------------------------------------------------------------------

@dataclass
class EchoBotRun:
    """In-memory representation of one EchoBot ``.mat`` acquisition file.

    Attributes
    ----------
    path : Path
        Source ``.mat`` file path.
    data : np.ndarray
        Raw samples with shape ``(n_samples, n_channels, n_pings)``.
    fs : float
        Sampling rate in Hz (always 2 MHz for the standard EchoBot).
    c : float
        Sound speed recorded in the header (m/s). For analysis, prefer the CTD
        value passed explicitly to pipeline functions.
    s_chirp : np.ndarray
        Transmit chirp waveform (1-D).
    t_pre : float
        Pre-chirp zero-pad duration (s).
    t_post : float
        Post-chirp zero-pad duration (s).
    t_now : float
        MATLAB serial datenum at acquisition start.
    start_utc : datetime
        Acquisition start time, converted to UTC assuming EST (UTC-5)
        wall-clock at the recording host.
    n_samples : int
        Number of samples per ping (``data.shape[0]``).
    n_channels : int
        Number of channels / sectors (``data.shape[1]``).
    n_pings : int
        Number of pings in the file (``data.shape[2]``).
    extra : dict
        Any additional header fields not promoted to attributes.
    """

    path: Path
    data: np.ndarray
    fs: float
    c: float
    s_chirp: np.ndarray
    t_pre: float
    t_post: float
    t_now: float
    start_utc: datetime
    n_samples: int
    n_channels: int
    n_pings: int
    extra: dict[str, Any] = field(default_factory=dict)


def load_echobot_mat(path: str | Path) -> EchoBotRun:
    """Load an EchoBot ``.mat`` file and return an ``EchoBotRun``.

    The ``.mat`` layout is defined by the MATLAB operating code
    (``Bob_txrx.m`` saves) — a ``data`` array and a ``header`` struct. This
    function extracts the core header fields and leaves the raw ``data``
    array untouched so downstream pipelines can index it directly.

    Parameters
    ----------
    path : str or Path
        Path to an EchoBot ``.mat`` file.

    Returns
    -------
    EchoBotRun
    """
    path = Path(path)
    raw = sio.loadmat(str(path), squeeze_me=False)
    data = raw["data"]
    hdr = raw["header"][0, 0]

    fs = float(hdr["fs"].flat[0])
    c = float(hdr["c"].flat[0])
    s_chirp = hdr["s_chirp"].flatten().astype(float)
    t_pre = float(hdr["T_pre"].flat[0])
    t_post = float(hdr["T_post"].flat[0])
    t_now = float(hdr["T_now"].flat[0])

    # MATLAB datenum → Python datetime; the host clock was set to EST, so add
    # 4 h to convert to UTC (March 2026 was daylight-saving, UTC-4).
    dt_est = datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(days=t_now - 367)
    start_utc = dt_est + timedelta(hours=4)

    ns, nc, npings = data.shape
    return EchoBotRun(
        path=path,
        data=data,
        fs=fs,
        c=c,
        s_chirp=s_chirp,
        t_pre=t_pre,
        t_post=t_post,
        t_now=t_now,
        start_utc=start_utc,
        n_samples=ns,
        n_channels=nc,
        n_pings=npings,
        extra={},
    )


def eb_start_utc(mat_path: str | Path) -> datetime:
    """Cheap header-only extraction of the EchoBot run start time in UTC.

    Useful for building a timestamp index without loading the full sample array.
    """
    mat_path = Path(mat_path)
    raw = sio.loadmat(str(mat_path), squeeze_me=False)
    t_now = float(raw["header"][0, 0]["T_now"].flat[0])
    dt_est = datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(days=t_now - 367)
    return dt_est + timedelta(hours=4)


def list_echobot_files(directory: str | Path) -> list[Path]:
    """List ``.mat`` files in a directory, sorted chronologically by filename timestamp.

    EchoBot filenames embed a ``T<HHMMSS>`` tag that reflects the save time;
    we sort on that tag rather than on mtime so chronological ordering is
    stable regardless of filesystem metadata.
    """
    files = list(Path(directory).glob("*.mat"))
    return sorted(files, key=lambda f: re.search(r"T(\d{6})", f.name).group(1))


# ---------------------------------------------------------------------------
# EK80 .raw loader
# ---------------------------------------------------------------------------

@dataclass
class EK80Metadata:
    """Parameters extracted from an EK80 ``.raw`` file via echopype.

    Everything here comes directly from the Kongsberg firmware writes and should
    not be altered — they feed the sonar equation in ``ek80_pipeline``.
    """

    path: Path
    n_pings: int
    n_samples: int
    n_beams: int
    sample_interval_s: float
    fs_bb_hz: float
    fs_rx_hz: float
    f_start_hz: float
    f_stop_hz: float
    t_dur_s: float
    slope: float
    transmit_power_w: float
    gain_correction_db: float
    impedance_transceiver_ohm: float
    start_utc: datetime
    end_utc: datetime


def open_ek80_raw(path: str | Path):
    """Open an EK80 ``.raw`` file via ``echopype.open_raw``.

    Returns the echopype ``EchoData`` object. Use ``get_ek80_metadata`` to
    extract the scalar parameters needed by the pipeline.
    """
    if ep is None:  # pragma: no cover
        raise ImportError("echopype is required to open EK80 .raw files")
    return ep.open_raw(str(path), sonar_model="EK80")


def get_ek80_metadata(ed, path: str | Path | None = None) -> EK80Metadata:
    """Extract pipeline-relevant scalar parameters from an echopype ``EchoData``.

    All values come from the ``Sonar/Beam_group1`` and ``Vendor_specific``
    datasets written by the Kongsberg firmware at transmit time.
    """
    beam = ed["Sonar/Beam_group1"]
    vs = ed["Vendor_specific"]

    sample_int = float(beam["sample_interval"].values[0, 0])
    fs_bb = 1.0 / sample_int
    t0 = pd.Timestamp(beam.ping_time.values[0]).to_pydatetime().replace(tzinfo=timezone.utc)
    t1 = pd.Timestamp(beam.ping_time.values[-1]).to_pydatetime().replace(tzinfo=timezone.utc)

    return EK80Metadata(
        path=Path(path) if path is not None else Path(""),
        n_pings=beam.sizes["ping_time"],
        n_samples=beam.sizes["range_sample"],
        n_beams=beam.sizes["beam"],
        sample_interval_s=sample_int,
        fs_bb_hz=fs_bb,
        fs_rx_hz=float(vs["receiver_sampling_frequency"].values[0]),
        f_start_hz=float(beam["transmit_frequency_start"].values[0, 0]),
        f_stop_hz=float(beam["transmit_frequency_stop"].values[0, 0]),
        t_dur_s=float(beam["transmit_duration_nominal"].values[0, 0]),
        slope=float(beam["slope"].values[0, 0]),
        transmit_power_w=float(beam["transmit_power"].values[0, 0]),
        gain_correction_db=float(vs["gain_correction"].values[0, 0]),
        impedance_transceiver_ohm=float(vs["impedance_transceiver"].values[0]),
        start_utc=t0,
        end_utc=t1,
    )


def ek_start_end_utc(raw_path: str | Path) -> tuple[datetime, datetime]:
    """Return ``(start_utc, end_utc)`` from an EK80 ``.raw`` file, opening once."""
    ed = open_ek80_raw(raw_path)
    beam = ed["Sonar/Beam_group1"]
    t0 = pd.Timestamp(beam.ping_time.values[0]).to_pydatetime().replace(tzinfo=timezone.utc)
    t1 = pd.Timestamp(beam.ping_time.values[-1]).to_pydatetime().replace(tzinfo=timezone.utc)
    return t0, t1


def list_ek80_files(directory: str | Path) -> list[Path]:
    """List ``.raw`` files in a directory, sorted by filename (which is timestamp-prefixed)."""
    return sorted(Path(directory).glob("*.raw"))


# ---------------------------------------------------------------------------
# Hydrophone .wav loader (SoundTrap 600)
# ---------------------------------------------------------------------------

HYDRO_SENSITIVITY_DB_DEFAULT = 176.0
"""Default SoundTrap 600 receive sensitivity in dB re 1 µPa / full-scale.

Override per deployment if the instrument has been re-calibrated.
"""


@dataclass
class HydrophoneFileInfo:
    path: Path
    start_utc: datetime
    est_dur_s: float


def build_hydrophone_index(directory: str | Path) -> list[HydrophoneFileInfo]:
    """Build a timestamp index over SoundTrap ``.wav`` files in ``directory``.

    SoundTrap filenames follow ``7817.YYMMDDhhmmss.wav``. Estimated duration is
    computed from file size assuming mono int16 at 384 kHz (768,000 bytes/s).
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.wav"))
    index: list[HydrophoneFileInfo] = []
    for wf in files:
        ts_str = wf.stem.split(".")[1]
        dt = datetime.strptime(ts_str, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        est_dur = wf.stat().st_size / 768000.0
        index.append(HydrophoneFileInfo(path=wf, start_utc=dt, est_dur_s=est_dur))
    return index


def find_hydrophone_file(
    query_utc: datetime,
    index: list[HydrophoneFileInfo],
) -> tuple[Path | None, datetime | None]:
    """Return ``(wav_path, wav_start_utc)`` covering ``query_utc``, or ``(None, None)``."""
    for hi in index:
        end_utc = hi.start_utc + timedelta(seconds=hi.est_dur_s)
        if hi.start_utc <= query_utc <= end_utc:
            return hi.path, hi.start_utc
    return None, None


def load_hydrophone_segment(
    wav_path: str | Path,
    wav_start_utc: datetime,
    query_start: datetime,
    query_end: datetime,
    sensitivity_db: float = HYDRO_SENSITIVITY_DB_DEFAULT,
    gain_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a calibrated hydrophone segment in µPa covering a UTC time window.

    Returns
    -------
    t : np.ndarray
        Time axis (s) relative to segment start.
    y : np.ndarray
        Pressure (µPa).
    fs : float
        Sample rate (Hz).
    """
    fs_h, y_raw = wavfile.read(str(wav_path), mmap=True)
    col0 = y_raw[:, 0] if y_raw.ndim > 1 else y_raw

    total_dur = len(col0) / fs_h
    dt_start = max(0, (query_start - wav_start_utc).total_seconds())
    dt_end = min(total_dur, (query_end - wav_start_utc).total_seconds())
    i0 = int(dt_start * fs_h)
    i1 = min(int(dt_end * fs_h), len(col0))

    seg = col0[i0:i1].astype(np.float64)
    if y_raw.dtype == np.int16:
        seg /= 32768.0
    elif y_raw.dtype == np.int32:
        seg /= 2147483648.0

    sens = -sensitivity_db + gain_db
    seg *= 10.0 ** (-sens / 20.0)

    t = np.arange(len(seg)) / fs_h
    return t, seg, float(fs_h)
