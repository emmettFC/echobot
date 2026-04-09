#!/usr/bin/env python
"""Headless reproduction of the manuscript figures.

Runs the same analysis as ``notebooks/02_manuscript_analysis.ipynb`` from the
command line, writing every figure to ``--output-dir`` and every summary
statistic to ``<output-dir>/../stats``.

Example
-------

    python scripts/generate_manuscript_figures.py \\
        --echobot-dir data/echobot/0311-CRL-tests \\
        --ek80-dir    data/EK80/0311-CRL-tests \\
        --noaa-file   data/NOAA-381-WC-TSf.xlsx \\
        --output-dir  output/figures

All arguments are optional; defaults match the repository layout.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from echobot import (
    io,
    echobot_pipeline,
    ek80_pipeline,
    snr,
    compare,
    linearity,
    plotting,
    config,
)


COMPARISONS = [
    dict(cid="A1",     name="Baseline",              eb_down_idx=0,  eb_up_idx=7,  ek_idx=0),
    dict(cid="A1-dup", name="Baseline (duplicate)",  eb_down_idx=1,  eb_up_idx=None, ek_idx=0),
    dict(cid="A1-rep", name="Baseline (late repeat)",eb_down_idx=13, eb_up_idx=12, ek_idx=0),
    dict(cid="B1",     name="Tx Duration 1.0 ms",    eb_down_idx=5,  eb_up_idx=10, ek_idx=3),
    dict(cid="C1",     name="Bandwidth 100-140",     eb_down_idx=6,  eb_up_idx=11, ek_idx=4),
    dict(cid="D1",     name="No Target",             eb_down_idx=14, eb_up_idx=15, ek_idx=7),
    dict(cid="P1-lo",  name="Low power",             eb_down_idx=3,  eb_up_idx=8,  ek_idx=0),
    dict(cid="P1-mid", name="Medium power",          eb_down_idx=13, eb_up_idx=12, ek_idx=1),
    dict(cid="P1-hi",  name="High power",            eb_down_idx=4,  eb_up_idx=9,  ek_idx=2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echobot-dir", type=Path, default=Path("data/echobot/0311-CRL-tests"))
    parser.add_argument("--ek80-dir",    type=Path, default=Path("data/EK80/0311-CRL-tests"))
    parser.add_argument("--noaa-file",   type=Path, default=Path("data/NOAA-381-WC-TSf.xlsx"))
    parser.add_argument("--val-ts120",   type=Path, default=Path("data/validation/validation_0311_ping_ts120.csv"))
    parser.add_argument("--val-tsf",     type=Path, default=Path("data/validation/validation_0311_tsf_detailed.csv"))
    parser.add_argument("--output-dir",  type=Path, default=Path("output/figures"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_fig = args.output_dir
    out_stats = args.output_dir.parent / "stats"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_stats.mkdir(parents=True, exist_ok=True)

    # Verify inputs
    for label, path in [
        ("echobot-dir", args.echobot_dir),
        ("ek80-dir", args.ek80_dir),
        ("noaa-file", args.noaa_file),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            return 1

    eb_files = io.list_echobot_files(args.echobot_dir)
    ek_files = io.list_ek80_files(args.ek80_dir)

    eb_needed = set()
    ek_needed = set()
    for g in COMPARISONS:
        ek_needed.add(g["ek_idx"])
        eb_needed.add(g["eb_down_idx"])
        if g["eb_up_idx"] is not None:
            eb_needed.add(g["eb_up_idx"])

    print(f"Processing {len(eb_needed)} EchoBot files and {len(ek_needed)} EK80 files...")

    eb_proc = {}
    for idx in sorted(eb_needed):
        print(f"  EB #{idx+1}: {eb_files[idx].name}")
        run = io.load_echobot_mat(eb_files[idx])
        eb_proc[idx] = echobot_pipeline.process_echobot_run(run)

    ek_proc = {}
    for idx in sorted(ek_needed):
        print(f"  EK80 #{idx+1}: {ek_files[idx].name}")
        ed = io.open_ek80_raw(ek_files[idx])
        ek_proc[idx] = ek80_pipeline.process_ek80_ed(ed)

    # --- Cross-instrument comparison ---
    df_r = compare.per_group_comparison(COMPARISONS, eb_proc, ek_proc, f_lo_hz=100e3, f_hi_hz=140e3)
    df_r.to_csv(out_stats / "cross_instrument_r.csv", index=False)
    print(f"Wrote {out_stats / 'cross_instrument_r.csv'}")

    # --- Figure 4 ---
    a1 = next(g for g in COMPARISONS if g["cid"] == "A1")
    fig = plotting.plot_baseline_comparison(
        eb_proc[a1["eb_down_idx"]], ek_proc[a1["ek_idx"]], title="A1 Baseline"
    )
    fig.savefig(out_fig / "fig04_baseline_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig / 'fig04_baseline_comparison.png'}")

    # --- Figure 5 ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
    for ax, g in zip(axes.flat, COMPARISONS):
        ek = ek_proc.get(g["ek_idx"])
        eb = eb_proc.get(g["eb_down_idx"])
        if ek is None or eb is None:
            ax.set_visible(False)
            continue
        plotting.plot_tsf_overlay(
            eb.f_hz, eb.tsf_db, ek.f_band_sorted, ek.tsf_calibrated_db, ax=ax
        )
        r_row = df_r[(df_r["cid"] == g["cid"]) & (df_r["sweep"] == "down")]
        r_val = float(r_row["pearson_r"].iloc[0]) if len(r_row) else float("nan")
        ax.set_title(f"{g['cid']} (r={r_val:.3f})", fontsize=10)
        ax.legend().set_visible(False)
    fig.tight_layout()
    fig.savefig(out_fig / "fig05_spectral_preservation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig / 'fig05_spectral_preservation.png'}")

    # --- Linearity (summary CSV only from headless) ---
    eb_power = linearity.power_linearity(
        [
            dict(nominal_db=0, processed=eb_proc[3]),
            dict(nominal_db=3, processed=eb_proc[13]),
            dict(nominal_db=5, processed=eb_proc[4]),
        ],
        "EchoBot",
    )
    ek_power = linearity.power_linearity(
        [
            dict(nominal_db=0.0, processed=ek_proc[0]),
            dict(nominal_db=10 * np.log10(90 / 60.0), processed=ek_proc[1]),
            dict(nominal_db=10 * np.log10(120 / 60.0), processed=ek_proc[2]),
        ],
        "EK80",
    )
    eb_dur = linearity.duration_shape_r(eb_proc[0], eb_proc[5], "EchoBot")
    ek_dur = linearity.duration_shape_r(ek_proc[0], ek_proc[3], "EK80")
    eb_bw = linearity.bandwidth_shape_r(eb_proc[0], eb_proc[6], "EchoBot")
    ek_bw = linearity.bandwidth_shape_r(ek_proc[0], ek_proc[4], "EK80")
    lin_summary = linearity.summarize_linearity(
        power_results=[eb_power, ek_power],
        shape_results=[eb_dur, ek_dur, eb_bw, ek_bw],
    )
    lin_summary.to_csv(out_stats / "linearity_summary.csv", index=False)
    print(f"Wrote {out_stats / 'linearity_summary.csv'}")

    # --- SNR table ---
    d1 = next(g for g in COMPARISONS if g["cid"] == "D1")
    eb_nt = eb_proc[d1["eb_down_idx"]]
    ek_nt = ek_proc[d1["ek_idx"]]
    snr_rows = []
    for g in COMPARISONS:
        if g["cid"] == "D1":
            continue
        eb = eb_proc.get(g["eb_down_idx"])
        ek = ek_proc.get(g["ek_idx"])
        if eb is None or ek is None:
            continue
        eb_er = snr.snr_empty_region(eb.all_mf, eb.r_mf, eb.target_gate)
        ek_er = snr.snr_empty_region(ek.all_pc, ek.r_ek, ek.target_gate)
        eb_nt_res = snr.snr_no_target_ref(eb.all_mf, eb.target_gate, eb_nt.all_mf)
        ek_nt_res = snr.snr_no_target_ref(ek.all_pc, ek.target_gate, ek_nt.all_pc)
        snr_rows.append(
            dict(
                cid=g["cid"],
                eb_empty_mean=eb_er.mean_db,
                eb_empty_std=eb_er.std_db,
                ek_empty_mean=ek_er.mean_db,
                ek_empty_std=ek_er.std_db,
                eb_ntref_mean=eb_nt_res.mean_db,
                ek_ntref_mean=ek_nt_res.mean_db,
            )
        )
    pd.DataFrame(snr_rows).to_csv(out_stats / "snr_per_group.csv", index=False)
    print(f"Wrote {out_stats / 'snr_per_group.csv'}")

    # --- Figure S11: absolute TS(f) ---
    noaa = pd.read_excel(args.noaa_file)
    noaa.columns = [c.strip() for c in noaa.columns]
    a1_eb = eb_proc[0]
    a1_ek = ek_proc[0]
    fig = plotting.plot_absolute_tsf(
        a1_eb.f_hz, a1_eb.tsf_db,
        a1_ek.f_band_sorted, a1_ek.tsf_calibrated_db,
        noaa["Frequency"].values * 1e3, noaa["dB (*-1)"].values,
    )
    fig.savefig(out_fig / "supp_absolute_tsf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_fig / 'supp_absolute_tsf.png'}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
