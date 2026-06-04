# EchoBot DAQ Acquisition — Python Port

## Overview

`echobot_daq.py` is a pure Python replacement for the MATLAB-based DAQ acquisition
code (`Bob_txrx.m` + `Bob_txrx_init.m`). It uses the National Instruments `nidaqmx`
Python package to control the NI USB-6366 DAQ, eliminating the MATLAB license
dependency.

## Architecture

```
Bob_txrx_init.m  →  EchoBotConfig (dataclass)
                     generate_chirp() (waveform generation)

Bob_txrx.m       →  run_acquisition() (DAQ session + ping loop)
                     run_simulated() (testing without hardware)

Flask app         →  /api/run/start, /api/run/stop, /api/run/status
                     (RUN mode in the browser UI)
```

## MATLAB → Python Function Mapping

| MATLAB | Python (`nidaqmx`) | Purpose |
|---|---|---|
| `daq.createSession('ni')` | `nidaqmx.Task()` | Create DAQ session |
| `addAnalogInputChannel(s, 'Dev1', ch, 'Voltage')` | `task.ai_channels.add_ai_voltage_chan(...)` | Add input channel |
| `addAnalogOutputChannel(s, 'Dev1', ch, 'Voltage')` | `task.ao_channels.add_ao_voltage_chan(...)` | Add output channel |
| `set(ai_ch, 'Range', [-5 5])` | `min_val=-5, max_val=5` in channel constructor | Set voltage range |
| `aio_s.Rate = fs` | `task.timing.cfg_samp_clk_timing(rate=fs)` | Set sample rate |
| `queueOutputData(s, [tx_data tx_switch])` | `task.write(data)` | Queue transmit waveform |
| `startForeground(s)` | `task.start()` + `task.read()` | Execute TX/RX |
| `chirp(t, fqi, T_d, fqf, 'linear', -90)` | `np.cos(phase - pi/2)` with linear FM | Generate chirp |
| `tukeywin(n, 0.1)` | `scipy.signal.tukey(n, alpha=0.1)` | Window function |
| `save('file.mat', 'data', 'header')` | `scipy.io.savemat(...)` | Save data |

## Key Differences from MATLAB Code

1. **Separate AI/AO tasks** — Python `nidaqmx` uses separate Task objects for
   input and output (vs MATLAB's single session). Synchronization is achieved
   via a digital trigger: AI waits for the AO start trigger.

2. **No `startForeground` equivalent** — replaced by `task.start()` + `task.read()`
   for input and `task.write()` + `task.start()` for output.

3. **Simulated mode** — `run_simulated()` generates synthetic data with a target
   echo at 1.4 m range for testing without hardware. The Flask app defaults to
   simulated mode for safety.

## Flask App Integration

The RUN mode in the browser UI calls three API endpoints:

- `POST /api/run/start` — Start acquisition (body: `{num_pings, scale_db, simulate, ...}`)
- `POST /api/run/stop` — Stop acquisition early
- `GET /api/run/status` — Poll progress (`{active, ping, total, latest_file}`)

When RUN mode is selected, the Play button changes to "Acquire." Clicking it
prompts for the number of pings and whether to simulate. During acquisition,
the button shows "Stop" and the ping counter updates in real time. When complete,
the saved `.mat` file is automatically loaded into REPLAY mode.

## Hardware Requirements

- NI USB-6366 (or compatible NI X Series DAQ)
- NI-DAQmx driver (Windows only for USB X Series devices)
- Python packages: `nidaqmx`, `numpy`, `scipy`

**Linux note:** NI does not provide Linux drivers for USB X Series devices.
Running the DAQ acquisition requires Windows. The replay/analysis functionality
works on any platform.

## Configuration

All parameters are set via `EchoBotConfig`:

```python
from echobot_daq import EchoBotConfig, run_acquisition

cfg = EchoBotConfig(
    fs=2_000_000,           # 2 MHz sample rate
    fqi=150_000,            # 150 kHz start (downsweep)
    fqf=90_000,             # 90 kHz end
    t_d=5e-4,               # 0.5 ms chirp duration
    scale_db=3.0,           # transmit amplitude
    num_pings=100,          # pings per recording
    device="Dev1",          # NI device ID
)

result = run_acquisition(cfg)
# result["data"].shape = (n_samples, 5, 100)
# result["filepath"] = path to saved .mat file
```

## Output Format

Saved `.mat` files have the same structure as the original MATLAB code:

- `data`: `(n_samples, n_channels, n_pings)` — raw voltage array
- `header`: struct with fields `fs`, `fqi`, `fqf`, `T_d`, `T_pre`, `T_post`,
  `s_chirp`, `c`, `scale`, `num_pings`, etc.

These files are directly compatible with:
- The `echobot` Python processing library
- The replay/visualization application
- The original MATLAB replay scripts
