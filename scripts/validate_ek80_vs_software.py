#!/usr/bin/env python
"""Validate the EK80 calibration pipeline against manual Simrad EK80 software reads.

Loads the bundled validation CSVs (TS@120 kHz across 15 pings, TS(f) at 10
frequencies across 5 pings), runs the calibrated EK80 pipeline on the same raw
file, interpolates the pipeline output at each manual read point, and prints
the residual table.

Example
-------

    python scripts/validate_ek80_vs_software.py \\
        --raw-file data/EK80/0311-CRL-tests/prod-D20260311-T182413.raw

Expected residuals (reproducing the manuscript Supplementary Figure S10):

    TS@120 residual (15 pings, mean):   -0.54 dB
    TS(f) residual (5p × 9f excl null): +0.02 dB
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from echobot import io, ek80_pipeline, config
from echobot.ek80_pipeline import compute_absolute_tsf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-file",
        type=Path,
        default=Path("data/EK80/0311-CRL-tests/prod-D20260311-T182413.raw"),
    )
    parser.add_argument(
        "--val-ts120",
        type=Path,
        default=Path("data/validation/validation_0311_ping_ts120.csv"),
    )
    parser.add_argument(
        "--val-tsf",
        type=Path,
        default=Path("data/validation/validation_0311_tsf_detailed.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.raw_file.exists():
        print(f"ERROR: raw file not found: {args.raw_file}")
        return 1
    if not args.val_ts120.exists() or not args.val_tsf.exists():
        print("ERROR: validation CSVs not found")
        return 1

    print(f"Loading {args.raw_file.name}...")
    ed = io.open_ek80_raw(args.raw_file)
    proc = ek80_pipeline.process_ek80_ed(ed)
    print(f"  {proc.meta.n_pings} pings, Ptx={proc.meta.transmit_power_w} W, G={proc.meta.gain_correction_db} dB")
    print(f"  target_center={proc.target_center_m:.4f} m, norm_fac={proc.norm_fac:.3f}")

    mask_ek = (proc.f_band_sorted >= 90e3) & (proc.f_band_sorted <= 150e3)
    j_lo, j_hi = proc.target_gate
    gate_len = j_hi - j_lo

    def ping_tsf(p: int) -> np.ndarray:
        spec = np.fft.fft(proc.all_pc[p, j_lo:j_hi] * np.hanning(gate_len), n=config.NFFT)
        return compute_absolute_tsf(
            spec,
            proc.tx_power_band,
            proc.band_idx_sorted,
            proc.f_band_sorted,
            proc.target_center_m,
            proc.meta,
            proc.norm_fac,
        )

    # ---- TS@120 comparison ----
    df_ts120 = pd.read_csv(args.val_ts120)
    pipe_vals = []
    for _, row in df_ts120.iterrows():
        p = int(row["ping_index"])
        tsf_cal = ping_tsf(p)
        pipe_vals.append(
            float(np.interp(120e3, proc.f_band_sorted[mask_ek], tsf_cal[mask_ek]))
        )
    df_ts120["pipeline_cal_TS120"] = pipe_vals
    df_ts120["residual"] = df_ts120["pipeline_cal_TS120"] - df_ts120["ek80_software_TS_120kHz_dB"]

    print()
    print("=" * 70)
    print("TS @ 120 kHz comparison (pipeline vs EK80 desktop software)")
    print("=" * 70)
    print(f"  n pings:        {len(df_ts120)}")
    print(f"  pipeline mean:  {df_ts120['pipeline_cal_TS120'].mean():+.3f} dB")
    print(f"  software mean:  {df_ts120['ek80_software_TS_120kHz_dB'].mean():+.3f} dB")
    print(f"  mean residual:  {df_ts120['residual'].mean():+.3f} dB")
    print(f"  std residual:   {df_ts120['residual'].std():.4f} dB")

    # ---- TS(f) detailed comparison ----
    df_tsf = pd.read_csv(args.val_tsf)
    freq_labels = [90, 95, 100, 110, 120, 125, 130, 135, 140, 150]
    freq_actual = [90, 95, 100, 110, 120, 125, 130, 135, 140, 149.9]

    rows = []
    for p in sorted(df_tsf["ping_index"].unique()):
        tsf_cal = ping_tsf(int(p))
        sub = df_tsf[df_tsf["ping_index"] == p].sort_values("freq_kHz").reset_index(drop=True)
        for lbl, f_true in zip(freq_labels, freq_actual):
            sw_val = float(sub[sub["freq_kHz"] == lbl]["ek80_software_TS_dB"].values[0])
            cal_val = float(np.interp(f_true * 1e3, proc.f_band_sorted[mask_ek], tsf_cal[mask_ek]))
            rows.append(
                dict(
                    ping=int(p),
                    freq_label=lbl,
                    freq_actual=f_true,
                    pipeline=cal_val,
                    software=sw_val,
                    residual=cal_val - sw_val,
                    is_null=(lbl == 135),
                )
            )
    df_tsf_r = pd.DataFrame(rows)
    df_no_null = df_tsf_r[~df_tsf_r["is_null"]]

    print()
    print("=" * 70)
    print("TS(f) detailed comparison (excluding 135 kHz null)")
    print("=" * 70)
    print(f"  n pings:        {df_tsf_r['ping'].nunique()}")
    print(f"  n freq:         9 (excl null)")
    print(f"  mean residual:  {df_no_null['residual'].mean():+.3f} dB")
    print(f"  std residual:   {df_no_null['residual'].std():.3f} dB")

    print()
    print("Per-frequency mean residual:")
    mean_by_f = df_no_null.groupby("freq_label")["residual"].agg(["mean", "std"])
    for f, row in mean_by_f.iterrows():
        print(f"  {f:>6} kHz: {row['mean']:+6.2f} ± {row['std']:4.2f} dB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
