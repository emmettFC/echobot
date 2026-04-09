# `data/` directory

See [`docs/DATA.md`](../docs/DATA.md) for the full data layout, Zenodo record
description, and comparison-group-to-raw-file mapping.

**Bundled in this repo:**

- `NOAA-381-WC-TSf.xlsx` — NOAA theoretical TS(f) for a 38.1 mm WC sphere
- `validation/validation_0311_ping_ts120.csv` — manual EK80 software reads at 120 kHz
- `validation/validation_0311_tsf_detailed.csv` — manual EK80 software reads across 10 frequencies

**Expected but not committed (obtain from Zenodo — DOI TBD):**

- `echobot/0311-CRL-tests/*.mat`
- `EK80/0311-CRL-tests/*.raw`
- `hydrophone/0311-CRL-tests/*.wav` (and `.sud` originals)
- `ctd/060633_20260311_1409.nc`

These subdirectories are excluded via `.gitignore`.
