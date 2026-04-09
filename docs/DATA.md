# Data Layout and Availability

This repository contains only the small reference assets needed to run the
pipeline documentation and validation examples. The raw acquisition data
(EchoBot `.mat`, EK80 `.raw`, SoundTrap `.wav`, CTD `.nc`) are large (~10–15 GB
total for the March 2026 CRL experiments) and are distributed separately via
Zenodo.

## What ships in this repo

```
data/
├── README.md                                    # this file
├── NOAA-381-WC-TSf.xlsx                         # NOAA theoretical sphere TS(f), 90-150 kHz
└── validation/
    ├── validation_0311_ping_ts120.csv           # 15 pings × TS@120 kHz manual reads
    └── validation_0311_tsf_detailed.csv         # 5 pings × 10 frequencies manual reads
```

- **`NOAA-381-WC-TSf.xlsx`** — Theoretical TS(f) for a 38.1 mm tungsten carbide
  calibration sphere (6% cobalt binder), computed with the NOAA Standard Sphere
  Target Strength Calculator at 1 kHz resolution across 90–150 kHz. Used as
  the ground truth for the absolute TS(f) comparison in Supplementary
  Figure S11 of the manuscript.

- **`validation/validation_0311_ping_ts120.csv`** — Manual TS values at
  120 kHz read from the Simrad EK80 desktop software for 15 pings of the A1
  baseline recording (`prod-D20260311-T182413.raw`). Columns:
  `ping_index`, `ping_timestamp`, `pipeline_TS_120kHz_dB`,
  `ek80_software_TS_120kHz_dB`, `stationary`. See Supplementary Figure S10a.

- **`validation/validation_0311_tsf_detailed.csv`** — Manual TS values at 10
  frequencies (90, 95, 100, 110, 120, 125, 130, 135, 140, 149.9 kHz) for 5
  pings of the same recording. Columns: `ping_index`, `ping_timestamp`,
  `freq_kHz`, `pipeline_TS_dB`, `ek80_software_TS_dB`, `stationary`. See
  Supplementary Figure S10b.

## What does NOT ship (and how to get it)

The full raw-data archive will be published on Zenodo with a persistent DOI.
**The DOI will be added here once the record is minted; until then, contact
the lead author (see [AUTHORS.md](../AUTHORS.md)) for access.**

Expected Zenodo record contents:

| Subdirectory | Contents | Approximate size |
|---|---|---|
| `echobot/0311-CRL-tests/` | 18 EchoBot `.mat` files (A1, A1-dup, A1-rep, B1, C1, D1, P1-lo/mid/hi + hydrophone-as-target runs) | ~3.5 GB |
| `EK80/0311-CRL-tests/` | 9 EK80 `.raw` files + `.idx`, `.xml` metadata | ~300 MB |
| `hydrophone/0311-CRL-tests/` | 4 SoundTrap `.wav` files + `.sud` originals + `.log.xml` | ~8 GB |
| `ctd/060633_20260311_1409.nc` | Tank CTD cast | < 1 MB |
| `validation/*.csv` | Manual software validation CSVs (also bundled here) | < 10 KB |
| `NOAA-381-WC-TSf.xlsx` | NOAA theoretical sphere TS(f) (also bundled here) | < 20 KB |
| `README.md` | Dataset description, instrument settings, sphere spec, tank geometry, file-to-group mapping | — |

## Where to put the data once you have it

After downloading the Zenodo archive, extract it into the repository's
`data/` directory so the final layout matches:

```
data/
├── NOAA-381-WC-TSf.xlsx                         # already bundled
├── validation/                                  # already bundled
│   ├── validation_0311_ping_ts120.csv
│   └── validation_0311_tsf_detailed.csv
├── echobot/0311-CRL-tests/                      # from Zenodo
│   └── backcyl_bis_rgh0.01271_*.mat
├── EK80/0311-CRL-tests/                         # from Zenodo
│   └── prod-D20260311-*.raw
├── hydrophone/0311-CRL-tests/                   # from Zenodo
│   ├── 7817.*.wav
│   └── 7817.*.sud
└── ctd/                                         # from Zenodo
    └── 060633_20260311_1409.nc
```

The `data/echobot/`, `data/EK80/`, `data/hydrophone/`, and `data/ctd/`
directories are excluded from git via [`.gitignore`](../.gitignore) so raw data
will never accidentally be committed.

## Comparison group mapping

The manuscript's nine comparison groups correspond to the following raw files
(indices into the chronologically sorted EchoBot `.mat` list and the EK80
`.raw` list):

| Group | Description | EB down | EB up | EK80 |
|---|---|---|---|---|
| A1 | Baseline (0.5 ms, 90–150 kHz, sphere) | 1 | 8 | 1 |
| A1-dup | Baseline (duplicate) | 2 | — | 1 |
| A1-rep | Baseline (late repeat) | 14 | 13 | 1 |
| B1 | Tx Duration (1.0 ms) | 6 | 11 | 4 |
| C1 | Bandwidth (100–140 kHz) | 7 | 12 | 5 |
| D1 | No Target | 15 | 16 | 8 |
| P1-lo | Low power (EB scale=0, EK80 60 W) | 4 | 9 | 1 |
| P1-mid | Medium power (EB scale=3, EK80 90 W) | 14 | 13 | 2 |
| P1-hi | High power (EB scale=5, EK80 120 W) | 5 | 10 | 3 |

These mappings are encoded in
[`notebooks/02_manuscript_analysis.ipynb`](../notebooks/02_manuscript_analysis.ipynb)
and passed to `echobot.compare.per_group_comparison`.

## Hydrophone-as-target runs

Two dedicated EchoBot runs (#17, #18) and one EK80 run (#9) used the
hydrophone itself as the target (replacing the calibration sphere). These
recordings feed the EchoBot dB ↔ EK80 Watt receive-level mapping implemented
in [`echobot.hydrophone`](../echobot/hydrophone.py) and the corresponding
supplementary figure (S2).
