"""EchoBot Replay – Flask app.

Replays .mat data files in the browser with:
  - Main echogram panel (depth × ping, fills sequentially)
  - Radial target-localization plot with history
  - Real-time amplitude envelope panel
  - TS(f) spectral panel (per-ping + running mean)
  - Transducer info / mode selector / depth-range dropdown
  - File browser (WATCH mode) / CTD import (REPLAY mode)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from flask import Flask, jsonify, render_template, request, send_from_directory
from scipy.signal import correlate, firwin, hilbert, lfilter

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent / "work" / "data" / "echobot"
DEFAULT_FILE = str(
    DATA_ROOT / "0311-CRL-tests" / "backcyl_bis_rgh0.01271_T141226_100.mat"
)

# Cache the loaded run so we don't reload on every ping request
_cache = {}


# ---------------------------------------------------------------------------
# Signal processing (mirrors MATLAB replay + Python pipeline)
# ---------------------------------------------------------------------------

def bp_filter(x, fs, lp_hz=175e3, hp_hz=80e3, n_taps=101):
    lp = firwin(n_taps, lp_hz, fs=fs)
    hp = firwin(n_taps, hp_hz, fs=fs, pass_zero=False)
    return lfilter(hp, 1.0, lfilter(lp, 1.0, x))


def load_run(mat_path):
    """Load a .mat file and pre-compute constants."""
    mat_path = str(mat_path)
    if mat_path in _cache:
        return _cache[mat_path]

    raw = sio.loadmat(mat_path, squeeze_me=False)
    data = raw["data"]
    hdr = raw["header"][0, 0]

    fs = float(hdr["fs"].flat[0])
    c = float(hdr["c"].flat[0])
    s_chirp = hdr["s_chirp"].flatten().astype(float)
    t_pre = float(hdr["T_pre"].flat[0])
    t_post = float(hdr["T_post"].flat[0])
    fqi = float(hdr["fqi"].flat[0])
    fqf = float(hdr["fqf"].flat[0])
    t_d = float(hdr["T_d"].flat[0])
    scale = float(hdr["scale"].flat[0])

    # Transducer info from filename
    fname = Path(mat_path).stem
    target_radius = float(hdr["a"].flat[0])

    ns, nc, npings = data.shape

    # Build TX reference
    n_pre = int(round(t_pre * fs))
    n_post = int(round(t_post * fs))
    tx_ref = np.concatenate([np.zeros(n_pre), s_chirp, np.zeros(n_post)])
    if len(tx_ref) < ns:
        tx_ref = np.concatenate([tx_ref, np.zeros(ns - len(tx_ref))])

    # Range axis
    r_mf = 0.5 * c * np.arange(ns) / fs

    # Frequency axis for TS(f)
    nfft = 4096
    f_hz = np.arange(nfft) * (fs / nfft)
    f_lo = min(fqi, fqf)
    f_hi = max(fqi, fqf)
    band_mask = (f_hz >= f_lo) & (f_hz <= f_hi)
    f_band_khz = f_hz[band_mask] / 1e3

    # Sector channel indices — detect by RMS (some files have 5 channels
    # where ch3,ch4 are active sectors; others have signal on ch0,ch1,ch2)
    if nc >= 5:
        rms = [np.sqrt(np.mean(data[:, ch, 0].astype(float) ** 2)) for ch in range(nc)]
        # If ch3,ch4 have substantial signal (>10% of ch0), use operate_echobot layout
        if rms[3] > 0.1 * rms[0] and rms[4] > 0.1 * rms[0]:
            ch_sec = [0, 3, 4]
        else:
            # Older recordings: 3 sector signals on ch0, ch1, ch2
            ch_sec = [0, 1, 2]
    elif nc == 3:
        ch_sec = [0, 1, 2]
    else:
        ch_sec = [0, min(1, nc - 1), min(2, nc - 1)]

    # Chirp type
    if fqi < fqf:
        chirp_type = "Upsweep"
    else:
        chirp_type = "Downsweep"

    # Default calibration offset: +37 dB electronics gain measured vs EK80
    cal_offset_db = 37.0

    run = {
        "data": data,
        "fs": fs,
        "c": c,
        "s_chirp": s_chirp,
        "tx_ref": tx_ref,
        "t_pre": t_pre,
        "t_post": t_post,
        "t_d": t_d,
        "fqi": fqi,
        "fqf": fqf,
        "scale": scale,
        "target_radius": target_radius,
        "ns": ns,
        "nc": nc,
        "npings": npings,
        "r_mf": r_mf,
        "nfft": nfft,
        "f_hz": f_hz,
        "f_band_khz": f_band_khz,
        "band_mask": band_mask,
        "ch_sec": ch_sec,
        "chirp_type": chirp_type,
        "cal_offset_db": cal_offset_db,
        "fname": fname,
        "path": mat_path,
        # Running TS(f) accumulator
        "tsf_sum": np.zeros(int(band_mask.sum())),
        "tsf_count": 0,
        # Target location history
        "loc_a1": [],
        "loc_a2": [],
    }
    _cache[mat_path] = run
    return run


def process_ping(run, ping_idx):
    """Process one ping: MF envelope, target localization, TS(f)."""
    data = run["data"]
    fs = run["fs"]
    c = run["c"]
    tx_ref = run["tx_ref"]
    ns = run["ns"]
    ch_sec = run["ch_sec"]
    nfft = run["nfft"]
    band_mask = run["band_mask"]
    r_mf = run["r_mf"]

    nref = len(tx_ref)

    # Matched filter per sector
    mf_channels = []
    for ch in ch_sec:
        filtered = bp_filter(data[:, ch, ping_idx].astype(float), fs)
        cc = correlate(filtered, tx_ref, mode="full")
        mf_channels.append(cc[nref - 1: nref - 1 + ns])

    c1, c2, c3 = mf_channels
    mf_sum = c1 + c2 + c3

    # Envelopes
    env1 = np.abs(hilbert(c1))
    env2 = np.abs(hilbert(c2))
    env3 = np.abs(hilbert(c3))
    env_sum = env1 + env2 + env3
    env_db = 20.0 * np.log10(env_sum + 1e-30)

    # --- Target localization ---
    chirp_range = c * run["t_d"] * 0.5
    phase_k = 25.12
    loc_range_win = 0.15  # m

    valid_mask = (r_mf > chirp_range) & (r_mf <= 5.0)
    valid_idx = np.where(valid_mask)[0]

    a1_deg = None
    a2_deg = None
    target_range = None

    if len(valid_idx) > 0:
        pk_rel = np.argmax(env_sum[valid_idx])
        pk_idx = valid_idx[pk_rel]
        target_range = float(r_mf[pk_idx])

        half_win_n = int(round((loc_range_win / (c * 0.5)) * fs))
        idx_lo = max(0, pk_idx - half_win_n)
        idx_hi = min(len(c1), pk_idx + half_win_n)

        C1 = hilbert(c1[idx_lo:idx_hi])
        C2 = hilbert(c2[idx_lo:idx_hi])
        C3 = hilbert(c3[idx_lo:idx_hi])
        id_pk = np.argmax(np.abs(C1 + C2 + C3))

        C12 = C1[id_pk] * np.conj(C2[id_pk])
        C32 = C3[id_pk] * np.conj(C2[id_pk])
        a1_ph = (np.angle(C12) + np.angle(C32)) / np.sqrt(3)
        a2_ph = np.angle(C32) - np.angle(C12)

        a1_deg = float(np.degrees(np.arcsin(np.clip(a1_ph / phase_k, -1, 1))))
        a2_deg = float(np.degrees(np.arcsin(np.clip(a2_ph / phase_k, -1, 1))))

        run["loc_a1"].append(a1_deg)
        run["loc_a2"].append(a2_deg)

    # --- TS(f) ---
    # Auto-detect target peak in search window
    targ_search = (r_mf >= 0.8) & (r_mf <= 2.0)
    targ_idx = np.where(targ_search)[0]
    env_mf = np.abs(hilbert(mf_sum))

    tsf_this = None
    tsf_mean = None
    targ_range_tsf = None

    if len(targ_idx) > 0:
        pk_t = np.argmax(env_mf[targ_idx])
        targ_pk_idx = targ_idx[pk_t]
        targ_range_tsf = float(r_mf[targ_pk_idx])

        gate_half_m = 0.40
        gate_half_n = int(round((gate_half_m / (c * 0.5)) * fs))
        gt_lo = max(0, targ_pk_idx - gate_half_n)
        gt_hi = min(len(mf_sum), targ_pk_idx + gate_half_n)

        mf_targ = mf_sum[gt_lo:gt_hi]
        windowed = mf_targ * np.hanning(len(mf_targ))

        fft_targ = np.fft.fft(windowed, n=nfft)
        tx_fft = np.fft.fft(run["s_chirp"], n=nfft)
        tx_pow = np.maximum(np.abs(tx_fft) ** 2, np.max(np.abs(tx_fft) ** 2) * 1e-10)

        tsf_full = 20.0 * np.log10(np.abs(fft_targ / tx_pow) + 1e-30)
        # Apply calibration offset (subtract electronics gain to get calibrated TS)
        tsf_full -= run["cal_offset_db"]
        tsf_this = tsf_full[band_mask].tolist()

        run["tsf_count"] += 1
        run["tsf_sum"] += np.array(tsf_this)
        tsf_mean = (run["tsf_sum"] / run["tsf_count"]).tolist()

    # Downsample for transfer: keep every Nth sample for display
    max_display_pts = 2000
    step = max(1, ns // max_display_pts)
    r_disp = r_mf[::step].tolist()
    env_disp = env_db[::step].tolist()

    # Sector raw voltage (just sector 1 for amplitude panel)
    raw_v = data[:, ch_sec[0], ping_idx].astype(float)
    raw_disp = raw_v[::step].tolist()

    return {
        "ping_idx": int(ping_idx),
        "range": r_disp,
        "envelope_db": env_disp,
        "raw_amplitude": raw_disp,
        "a1_deg": a1_deg,
        "a2_deg": a2_deg,
        "target_range": target_range,
        "loc_history_a1": run["loc_a1"][-200:],
        "loc_history_a2": run["loc_a2"][-200:],
        "tsf_this": tsf_this,
        "tsf_mean": tsf_mean,
        "tsf_range": targ_range_tsf,
        "f_band_khz": run["f_band_khz"].tolist(),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/file_info", methods=["POST"])
def file_info():
    """Load a file and return metadata (no ping processing)."""
    body = request.get_json(force=True)
    mat_path = body.get("path", DEFAULT_FILE)

    # Clear cache for fresh load (reset accumulators)
    if mat_path in _cache:
        del _cache[mat_path]

    run = load_run(mat_path)
    return jsonify({
        "fname": run["fname"],
        "npings": run["npings"],
        "ns": run["ns"],
        "fs": run["fs"],
        "c": run["c"],
        "fqi": run["fqi"],
        "fqf": run["fqf"],
        "t_d": run["t_d"],
        "scale": run["scale"],
        "chirp_type": run["chirp_type"],
        "target_radius": run["target_radius"],
        "nc": run["nc"],
        "range_max": float(run["r_mf"][-1]),
        "transducer": "Kongsberg ES120-18CDK",
        "cal_offset_db": run["cal_offset_db"],
        "path": mat_path,
    })


@app.route("/api/ping", methods=["POST"])
def get_ping():
    """Process and return one ping."""
    body = request.get_json(force=True)
    mat_path = body.get("path", DEFAULT_FILE)
    ping_idx = int(body.get("ping_idx", 0))

    run = load_run(mat_path)
    if ping_idx < 0 or ping_idx >= run["npings"]:
        return jsonify({"error": f"ping_idx out of range [0, {run['npings']-1})"}), 400

    result = process_ping(run, ping_idx)
    return jsonify(result)


@app.route("/api/set_cal_offset", methods=["POST"])
def set_cal_offset():
    """Set calibration offset for current file."""
    body = request.get_json(force=True)
    mat_path = body.get("path", DEFAULT_FILE)
    offset = float(body.get("cal_offset_db", 37.0))
    if mat_path in _cache:
        _cache[mat_path]["cal_offset_db"] = offset
        # Reset TS(f) accumulator since offset changed
        _cache[mat_path]["tsf_sum"] = np.zeros_like(_cache[mat_path]["tsf_sum"])
        _cache[mat_path]["tsf_count"] = 0
    return jsonify({"cal_offset_db": offset})


@app.route("/api/browse", methods=["GET"])
def browse_files():
    """List .mat files available for replay."""
    files = []
    for dirpath, dirnames, filenames in os.walk(str(DATA_ROOT)):
        for fn in sorted(filenames):
            if fn.endswith(".mat"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, str(DATA_ROOT))
                files.append({"path": full, "name": rel})
    return jsonify({"files": files, "default": DEFAULT_FILE})


@app.route("/api/list_all_mat", methods=["GET"])
def list_all_mat():
    """List all .mat files on the system starting from the echobot root."""
    root = request.args.get("root", str(DATA_ROOT))
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in sorted(filenames):
            if fn.endswith(".mat"):
                full = os.path.join(dirpath, fn)
                files.append(full)
    return jsonify({"files": files})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"EchoBot Replay starting on http://localhost:{port}")
    print(f"Default data file: {DEFAULT_FILE}")
    app.run(debug=True, port=port, host="0.0.0.0")
