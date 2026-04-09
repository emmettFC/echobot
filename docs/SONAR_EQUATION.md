# EK80 Absolute TS(f) Sonar Equation — Parameter Reference

This document is a standalone reference for every parameter in the TS(f)
sonar equation used by [`echobot.ek80_pipeline.compute_absolute_tsf`](../echobot/ek80_pipeline.py).
It is extracted verbatim from the "Validation: EK80 Pipeline vs EK80 Software —
March 2026 Data" section of [`PIPELINE.md`](PIPELINE.md) so that readers who
need to audit the equation don't have to wade through the full provenance doc.

---

## Equation

The calibrated TS(f) for an EK80 pulse-compressed spectrum is computed via the
full sonar equation (Demer et al. 2017 §1.2.4.4 Eq. 13), matching echopype's
internal `_cal_complex_samples`:

```
TS(f) = 10*log10(|H(f)|^2)                        (1)  received-power spectrum
      + 20*log10(norm_fac)                        (2)  undo PC normalization
      + Z_dB                                      (3)  impedance + beam scaling
      + 40*log10(R)                               (4)  two-way spherical spreading
      + 2*alpha*R                                 (5)  two-way absorption
      - 10*log10(lam^2 * Ptx / (16*pi^2))         (6)  source term
      - 2*G                                       (7)  scalar gain correction
```

where the impedance/beam scaling factor is

```
Z_dB = 10*log10(N_beams / N_ref) + 20*log10(|z_er + z_et| / z_er) - 10*log10(z_et)
```

with `N_ref = 8` (the echopype `_cal_complex_samples` convention: `/4` from
4-element split-beam voltage division × `/2` from the `mean_beam` proxy for
the element sum).

---

## Parameter table

Every symbol in the equation, its physical meaning, units, source of its value
in the pipeline, and a concrete A1 baseline value for reference (EK80 file
`prod-D20260311-T182413.raw`, range gate centered at 1.386 m, f = 120 kHz
unless noted).

| Symbol | Description | Units | Source | A1 value | Notes |
|---|---|---|---|---|---|
| `f` | Analysis frequency | Hz | Aliasing-aware mapping of the baseband FFT bins back to RF (see `alias_aware_freq_map` in [`ek80_pipeline.py`](../echobot/ek80_pipeline.py)) | 90–150 kHz sweep | Equation evaluated at every in-band FFT bin |
| `H(f)` | Deconvolved complex spectrum | dimensionless | `H = FFT{pc[gate]·hann} / |TX_filt|²` | Complex array | `pc` is the mean-beam pulse-compressed I/Q already divided by `norm_fac`; `TX_filt = FFT{tx_filt}` |
| `\|H(f)\|²` | Received power spectrum | dimensionless (V²-proxy) | Magnitude-squared of `H(f)` | Complex → power | Equation term (1) |
| `norm_fac` | Pulse-compression norm factor | dimensionless | `np.linalg.norm(tx_filt)²` — mirrors echopype `get_norm_fac()`; Demer et al. (2017) §1.2.4.1 Eq. 5 | **44.18** (20·log₁₀ = +32.90 dB) | Depends on WBT/PC filter chain + chirp parameters |
| `tx_filt` | Decimated, filtered transmit replica | complex samples | `y_rf` → WBT filter → /8 dec → PC filter → /2 dec; Demer et al. (2017) §1.2.3, Eqs. 2–3 | 96 samples (A1) | Raw chirp `y_rf` = 767 samples at 1500 kHz → 102 after WBT+/8 → 96 after PC+/2; filter coefficients from `.raw` vendor metadata |
| `N_beams` | Number of transducer elements in `Beam_group1` | count | `.raw` file: `Sonar/Beam_group1` beam dimension | **3** | Simrad ES120-7C split-beam; firmware-reported |
| `N_ref` | Normalization denominator in `Z_dB` | count | Hardcoded constant = 8 in echopype `_cal_complex_samples` | **8** | `/4` split-beam + `/2` voltage-division factor |
| `z_er` | Transceiver (WBT) receive impedance | Ω | `.raw` file: `Vendor_specific/impedance_transceiver` | **10 800 Ω** | Firmware-reported; varies slightly by WBT channel |
| `z_et` | Transducer (ET) impedance | Ω | **Hardcoded constant = 75 Ω** in `compute_absolute_tsf` | **75 Ω** | Demer et al. (2017) §1.2.4.4; nominal ES-series default |
| `Z_dB` | Combined impedance + beam scaling | dB | `10·log₁₀(N_beams/N_ref) + 20·log₁₀(\|z_er+z_et\|/z_er) − 10·log₁₀(z_et)` | **−22.95 dB** | Equation term (3); −4.26 + 0.060 − 18.75 |
| `R` | Target range (gate center) | m | Autodetected envelope peak, per-instrument | **1.386 m** | Two-way TVG and absorption; 4.0 cm offset from EB center |
| `α` (alpha) | Seawater absorption coefficient | dB/m | Hardcoded constant = 0.04 dB/m | **0.04 dB/m** | Nominal for ~120 kHz in ~15 °C brackish tank water |
| `c` | Sound speed | m/s | Per-session CTD measurement (Mackenzie 1981) | **1484.96 m/s** | Used for `λ = c/f` and range axis |
| `λ` (lam) | Acoustic wavelength | m | Computed: `λ = c/f` | **12.375 mm** at 120 kHz | Varies 9.90–16.50 mm across 90–150 kHz |
| `Ptx` | Transmit electrical power | W | `.raw` file: `Sonar/Beam_group1/transmit_power` | **60 W** | Firmware-reported per-ping |
| `G` | Scalar gain (center-frequency) | dB | `.raw` file: `Vendor_specific/gain_correction` | **18.0 dB** | Single-frequency value; applied as `−2G` |
| `40·log₁₀(R)` | Two-way spherical spreading | dB | Computed from `R` | **+5.67 dB** | Equation term (4) |
| `2·α·R` | Two-way absorption loss | dB | Computed from `α`, `R` | **+0.111 dB** | Equation term (5) |
| `10·log₁₀(λ²·Ptx/(16π²))` | Source level component | dB | Computed from `λ`, `Ptx` | **−42.35 dB** at 120 kHz | Equation term (6); inverts Demer Eq. 13 source-pressure term |
| `2·G` | Two-way gain correction | dB | Computed: `2 × G_correction` | **36.0 dB** | Equation term (7); subtracted |

### Source categories used in the table

- **Firmware-reported** — written by the Kongsberg EK80 transceiver to the
  `.raw` file at transmit time; not a discretionary choice.
- **Hardcoded constant** — set as a Python literal in
  [`echobot/config.py`](../echobot/config.py) or
  [`echobot/ek80_pipeline.py`](../echobot/ek80_pipeline.py); not drawn from data.
- **Autodetected** — computed from the data via an automated algorithm (not hand-chosen).
- **Computed** — derived from other firmware/constants via the equations above.

### Independent constants and their rationale

| Constant | Value | Rationale |
|---|---|---|
| `z_et` | 75 Ω | Nominal ES-series transducer impedance; Demer et al. (2017) §1.2.4.4; echopype default. A transducer-specific impedance measurement would refine this. |
| `N_ref` | 8 | Echopype `_cal_complex_samples` convention: `/4` four-element split-beam voltage division + `/2` mean-beam proxy. |
| `α` | 0.04 dB/m | Broadband average absorption over 90–150 kHz for tank conditions (~15 °C, ~30 PSU, ~1 m depth). Two-way error from using a flat value vs Francois–Garrison is < 0.1 dB over the band at 1.4 m range. |
| `c` | 1484.96 m/s | Per-session CTD measurement; updated per run from the tank CTD cast. |

---

## Worked example: A1 baseline TS at 120 kHz

Concrete term-by-term evaluation of the sonar equation using the parameter
values in the table above. The raw deconvolved power `10·log₁₀(|H(120 kHz)|²)`
is −59.22 dB (measured from the A1 127-ping coherent average), and each
calibration term is added in turn:

| Term | Contribution (dB) |
|---|---|
| (1) `10·log₁₀(\|H\|²)`                          | **−59.224** |
| (2) `20·log₁₀(norm_fac)`                       | +32.905 |
| (3) `Z_dB`                                     | −22.950 |
| (4) `40·log₁₀(R)` = `40·log₁₀(1.386)`          | +5.671 |
| (5) `2·α·R` = `2·(0.04)·(1.386)`               | +0.111 |
| (6) `−10·log₁₀(λ²·Ptx/(16π²))` at λ=12.375 mm, Ptx=60 W | +42.352 |
| (7) `−2·G` = `−2·(18.0)`                       | −36.000 |
| **TS(120 kHz)**                                | **−37.136** |

This matches the value reported in the EK80-vs-NOAA-theory comparison table
(EK = −37.14 at 120 kHz) and the pipeline-vs-software residual at 120 kHz in
Supplementary Figure S10 (−0.54 dB relative to the EK80 software's −36.49 dB
read). The arithmetic is closed end-to-end; see
[`tests/test_pipelines.py`](../tests/test_pipelines.py) for an automated
reproduction of this worked example.
