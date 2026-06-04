"""EchoBot DAQ acquisition module — Python port of Bob_txrx.m + Bob_txrx_init.m.

This module replaces the MATLAB-based DAQ acquisition code with a pure Python
implementation using the National Instruments ``nidaqmx`` package. It generates
the transmit chirp waveform, configures the NI USB-6366 DAQ for simultaneous
analog input/output, executes a ping loop, and saves data in the same ``.mat``
format used by the original MATLAB code.

Hardware requirements:
    - NI USB-6366 (or compatible X Series DAQ)
    - NI-DAQmx driver installed (Windows only; Linux not supported for USB X Series)
    - Python ``nidaqmx`` package: ``pip install nidaqmx``

The saved ``.mat`` files are fully compatible with the EchoBot signal processing
pipeline (``echobot.echobot_pipeline``) and the replay application.

Original MATLAB code by K. Ball (WHOI, 2015), modified by M. Al Mursaline (2020).
Python port for EchoBot open-source release.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from scipy.io import savemat
from scipy.signal import tukey


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EchoBotConfig:
    """All user-configurable acquisition parameters.

    Mirrors the header structure from Bob_txrx_init.m. Every field has a
    sensible default matching the standard CRL test configuration.
    """

    # Sampling
    fs: float = 2_000_000.0          # Hz — transmit and receive sample rate
    device: str = "Dev1"              # NI device identifier

    # Channels
    in_channels: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    out_channels: list[int] = field(default_factory=lambda: [0, 1])
    vin_range: float = 5.0            # V — DAQ input voltage range
    vout_range: float = 5.0           # V — DAQ output voltage range

    # Chirp waveform
    fqi: float = 150_000.0            # Hz — initial frequency (downsweep: high→low)
    fqf: float = 90_000.0             # Hz — final frequency
    t_d: float = 5e-4                 # s  — chirp duration (0.5 ms)
    scale_db: float = 3.0             # dB — transmit amplitude scaling
    tukey_alpha: float = 0.1          # Tukey window shape parameter

    # Timing
    t_pre: float = 3.5e-4            # s  — pre-trigger silence
    t_post: float = 0.05             # s  — post-trigger listen period
    delay: float = 1 / 1500          # s  — inter-ping pause

    # Acquisition
    num_pings: int = 100              # pings per recording
    sound_speed: float = 1486.0       # m/s — nominal (overridden by CTD if available)

    # File saving
    target_type: str = "cyl_bis_rgh"
    target_radius: float = 1 * 0.0254 * 0.5  # m
    save_dir: Optional[str] = None    # defaults to ./YYYY-MM-DD/
    file_prefix: Optional[str] = None


# ---------------------------------------------------------------------------
# Waveform generation
# ---------------------------------------------------------------------------

def generate_chirp(cfg: EchoBotConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the transmit waveform arrays.

    Returns
    -------
    tx_data : np.ndarray
        Full transmit waveform (pre-silence + chirp + post-silence).
    tx_switch : np.ndarray
        T/R switch control signal (0 during TX, 2.5 V during RX).
    s_chirp : np.ndarray
        The chirp waveform alone (for saving in the header).
    """
    t_chirp = np.arange(0, cfg.t_d + 1 / cfg.fs, 1 / cfg.fs)

    # Linear FM chirp: cos(pi * (fqf - fqi) / t_d * t^2 + 2*pi*fqi*t - pi/2)
    # The -pi/2 phase matches MATLAB's chirp(..., -90)
    phase = np.pi * (cfg.fqf - cfg.fqi) / cfg.t_d * t_chirp ** 2 + 2 * np.pi * cfg.fqi * t_chirp
    s_chirp = np.cos(phase - np.pi / 2)

    # Apply Tukey window
    s_chirp = tukey(len(s_chirp), alpha=cfg.tukey_alpha) * s_chirp

    # Apply amplitude scaling
    scale_linear = 10 ** (cfg.scale_db / 10)
    s_chirp = scale_linear * (s_chirp / np.max(np.abs(s_chirp)))

    # Build full transmit waveform: [pre-silence | chirp | post-silence]
    n_pre = int(round(cfg.t_pre * cfg.fs))
    n_post = int(round(cfg.t_post * cfg.fs))

    tx_data = np.concatenate([
        np.zeros(n_pre),
        s_chirp,
        np.zeros(n_post),
    ])

    # T/R switch: 0 during pre+chirp, 2.5 V during post (listen period)
    tx_switch = np.concatenate([
        np.zeros(n_pre),
        np.zeros(len(s_chirp)),
        2.5 * np.ones(n_post),
    ])

    return tx_data, tx_switch, s_chirp


# ---------------------------------------------------------------------------
# DAQ session
# ---------------------------------------------------------------------------

def run_acquisition(
    cfg: EchoBotConfig,
    on_ping: Optional[Callable[[int, int, np.ndarray], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """Execute a full acquisition session.

    Parameters
    ----------
    cfg : EchoBotConfig
        Acquisition configuration.
    on_ping : callable, optional
        Callback ``on_ping(ping_idx, total_pings, ping_data)`` called after
        each ping completes. ``ping_data`` has shape ``(n_samples, n_channels)``.
        Used by the Flask app to stream live data.
    stop_flag : callable, optional
        If provided, checked before each ping. Return ``True`` to abort early.

    Returns
    -------
    dict
        ``{"data": np.ndarray, "header": dict, "filepath": str | None}``
        where ``data`` has shape ``(n_samples, n_channels, n_pings)``.
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    except ImportError:
        raise ImportError(
            "nidaqmx package not installed. Install with: pip install nidaqmx\n"
            "Also requires NI-DAQmx driver (Windows only for USB X Series)."
        )

    tx_data, tx_switch, s_chirp = generate_chirp(cfg)
    n_samples = len(tx_data)

    # Build header (matches MATLAB .mat header structure)
    header = {
        "T_now": datetime.now().isoformat(),
        "fs": cfg.fs,
        "delay": cfg.delay,
        "scale": cfg.scale_db,
        "Vin_Range": cfg.vin_range,
        "Vout_Range": cfg.vout_range,
        "capture_duration": 1.0,
        "in_chan_vec": np.array(cfg.in_channels),
        "out_chan_vec": np.array(cfg.out_channels),
        "fqi": cfg.fqi,
        "fqf": cfg.fqf,
        "T_d": cfg.t_d,
        "T_pre": cfg.t_pre,
        "T_post": cfg.t_post,
        "s_chirp": s_chirp,
        "c": cfg.sound_speed,
        "a": cfg.target_radius,
        "num_pings": cfg.num_pings,
    }

    # Allocate data array: (samples, channels, pings)
    n_in = len(cfg.in_channels)
    data = np.zeros((n_samples, n_in, cfg.num_pings))

    # Combine TX waveform + switch signal for the two output channels
    tx_out = np.column_stack([tx_data, tx_switch])  # (n_samples, 2)

    # Configure and run DAQ
    with nidaqmx.Task("echobot_ai") as ai_task, nidaqmx.Task("echobot_ao") as ao_task:

        # Analog input channels
        for ch in cfg.in_channels:
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{cfg.device}/ai{ch}",
                min_val=-cfg.vin_range,
                max_val=cfg.vin_range,
            )

        # Analog output channels
        for ch in cfg.out_channels:
            ao_task.ao_channels.add_ao_voltage_chan(
                f"{cfg.device}/ao{ch}",
                min_val=-cfg.vout_range,
                max_val=cfg.vout_range,
            )

        # Configure timing — finite samples, clocked at fs
        ai_task.timing.cfg_samp_clk_timing(
            rate=cfg.fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )
        ao_task.timing.cfg_samp_clk_timing(
            rate=cfg.fs,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples,
        )

        # Synchronize: AI starts on AO start trigger
        ai_task.triggers.start_trigger.cfg_dig_edge_start_trig(
            f"/{cfg.device}/ao/StartTrigger"
        )

        # Ping loop
        for p in range(cfg.num_pings):
            if stop_flag and stop_flag():
                print(f"Acquisition stopped at ping {p}/{cfg.num_pings}")
                data = data[:, :, :p]  # truncate
                header["num_pings"] = p
                break

            t0 = time.perf_counter()

            # Write TX waveform
            ao_task.write(tx_out.T.tolist(), auto_start=False)

            # Start AI first (it waits for trigger), then AO (triggers both)
            ai_task.start()
            ao_task.start()

            # Read received data
            rx = ai_task.read(
                number_of_samples_per_channel=n_samples,
            )
            data[:, :, p] = np.array(rx).T

            # Wait for output to finish, then stop both
            ao_task.wait_until_done(timeout=10.0)
            ao_task.stop()
            ai_task.stop()

            elapsed = time.perf_counter() - t0
            print(f"Ping {p + 1}/{cfg.num_pings} ({elapsed:.3f}s)")

            if on_ping:
                on_ping(p, cfg.num_pings, data[:, :, p])

            # Inter-ping delay
            time.sleep(cfg.delay)

    # Save to .mat
    filepath = _save_mat(cfg, data, header)

    return {"data": data, "header": header, "filepath": filepath}


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------

def _save_mat(cfg: EchoBotConfig, data: np.ndarray, header: dict) -> Optional[str]:
    """Save data + header in the same .mat format as the MATLAB code."""
    now = datetime.now()

    if cfg.save_dir:
        save_dir = Path(cfg.save_dir)
    else:
        save_dir = Path.cwd() / now.strftime("%d-%b-%Y")

    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg.file_prefix or f"back{cfg.target_type}{cfg.target_radius}"
    timestamp = now.strftime("T%H%M%S")
    filename = f"{prefix}_{timestamp}_{header['num_pings']:03d}.mat"
    filepath = save_dir / filename

    savemat(str(filepath), {"data": data, "header": header})
    print(f"Saved: {filepath}")
    return str(filepath)


# ---------------------------------------------------------------------------
# Simulated acquisition (for testing without hardware)
# ---------------------------------------------------------------------------

def run_simulated(
    cfg: EchoBotConfig,
    on_ping: Optional[Callable[[int, int, np.ndarray], None]] = None,
    stop_flag: Optional[Callable[[], bool]] = None,
) -> dict:
    """Simulated acquisition for testing without DAQ hardware.

    Generates synthetic data with a target echo at ~1.4 m range and noise,
    using the same waveform and timing parameters as the real acquisition.
    """
    tx_data, tx_switch, s_chirp = generate_chirp(cfg)
    n_samples = len(tx_data)
    n_in = len(cfg.in_channels)
    data = np.zeros((n_samples, n_in, cfg.num_pings))

    # Simulate a target at ~1.4 m
    target_range = 1.4  # m
    target_delay_samples = int(2 * target_range / cfg.sound_speed * cfg.fs)
    n_pre = int(round(cfg.t_pre * cfg.fs))

    header = {
        "T_now": datetime.now().isoformat(),
        "fs": cfg.fs, "delay": cfg.delay, "scale": cfg.scale_db,
        "Vin_Range": cfg.vin_range, "Vout_Range": cfg.vout_range,
        "capture_duration": 1.0,
        "in_chan_vec": np.array(cfg.in_channels),
        "out_chan_vec": np.array(cfg.out_channels),
        "fqi": cfg.fqi, "fqf": cfg.fqf, "T_d": cfg.t_d,
        "T_pre": cfg.t_pre, "T_post": cfg.t_post,
        "s_chirp": s_chirp, "c": cfg.sound_speed,
        "a": cfg.target_radius, "num_pings": cfg.num_pings,
    }

    for p in range(cfg.num_pings):
        if stop_flag and stop_flag():
            data = data[:, :, :p]
            header["num_pings"] = p
            break

        # Simulated echo: attenuated, delayed chirp + noise
        echo_start = n_pre + target_delay_samples
        echo_len = len(s_chirp)
        noise = np.random.randn(n_samples) * 0.001

        for ch in range(min(n_in, 3)):  # 3 sector channels
            sig = noise.copy()
            if echo_start + echo_len < n_samples:
                attenuation = 0.01 / (target_range ** 2)  # 1/r^2
                sig[echo_start:echo_start + echo_len] += attenuation * s_chirp
            data[:, ch, p] = sig

        if on_ping:
            on_ping(p, cfg.num_pings, data[:, :, p])

        time.sleep(0.05)  # simulate ping interval

    filepath = _save_mat(cfg, data, header)
    return {"data": data, "header": header, "filepath": filepath}
