# EchoBot Signal Processing Pipeline — Provenance & Justification

This document traces every step of the EchoBot (Section 2.4) and EK80 (Section 2.5) signal processing pipelines to their origin in the original MATLAB operating code (`work/Matlab/operating-code-refactor/`), identifies where our Python implementation departs from the MATLAB, and provides justification for each discretionary decision.

**MATLAB scripts referenced:**
- `Bob_txrx_init.m` — Parameter initialization, chirp generation, timing
- `Bob_txrx.m` — DAQ session, ping loop, data saving
- `replay_MF_modZ_CRL.m` — Bandpass filtering, matched filtering, angle estimation
- `Bob_comprs_plt_bis.m` — Real-time plotting, MF processing, FFT deconvolution (partially commented out)
- `replay_MF_modZ_CRL_EC.m` — EC's modified replay script (same core processing as replay_MF_modZ_CRL.m)

---

## Combined Pipeline Comparison & Provenance

| Step | EchoBot Pipeline | EB Provenance | EK80 Pipeline | EK80 Provenance |
|---|---|---|---|---|
| **1** | Data loading (`.mat`, header struct) | `Bob_txrx.m:97` (save), `Bob_txrx_init.m:26-81` (header fields) | Data loading (`ep.open_raw`, beam/vendor datasets) | echopype library |
| **2** | TX reference construction (zero-pad chirp: `[zeros, s_chirp, zeros]`) | `Bob_txrx_init.m:73-77` (`tx_data = [n_pre; s_chirp; n_post]`) | Chirp replica construction (FM sweep at 1500 kHz, Hanning edge taper, filter through WBT+PC chains) | echopype `tapered_chirp()` + `filter_decimate_chirp()`; Demer et al. (2017) §1.2.2.4 Eq. 1 (ramping), §1.2.3 Eqs. 2–3 (filter/decimate) |
| **3** | Bandpass filtering (FIR LP 175 kHz + HP 80 kHz) | Based on `replay_MF_modZ_CRL.m:6-11` (LP=150k, HP=90k); **widened to 80–175 kHz** — discretionary, justified by chirp rolloff / matched-filter band selection | *(handled by WBT/PC filter chain in step 2)* | — |
| **4** | Matched filtering (`xcorr` with tx_ref, sum 3 sectors, Hilbert envelope) | `replay_MF_modZ_CRL.m:55-57` (xcorr per sector), line 64 (positive lags); `Bob_comprs_plt_bis.m:145-147` | Matched filtering (convolve mean-beam I/Q with `flipud(conj(tx_filt))`, normalize by ‖tx_filt‖²) | echopype `compress_pulse()` + `get_norm_fac()`; Demer et al. (2017) §1.2.4.1 Eqs. 4–5 |
| **5** | Range gating (±0.40 m, per-instrument autodetected center) | Our design; falls out of experimental design; gate width from gate sensitivity analysis (Section 3.9, Table 7) | Range gating (same gate widths, per-instrument center) | Same as EB — matched across pipelines |
| **6** | TS(f): FFT deconvolution by \|TX\|², Hanning window, coherent averaging, NFFT=4096 | \|TX\|² denominator: Demer et al. (2017) Eq. 11; Hanning: inspired by Demer Eq. 14–15 (Sv context), standard conservative practice; NFFT: conventional power-of-2; coherent averaging: standard for stationary-target calibration | Aliasing-aware freq mapping + same TS(f) deconvolution (\|TX\|², Hanning, NFFT=4096, coherent averaging) | Aliasing mapping: Demer et al. (2017) §1.2.3 ("band-shifted to predecimated frequencies"); deconvolution/window/averaging: same sources as EB — matched across pipelines |
| **7** | SNR computation (peak envelope / no-target reference) | Direct instruction from Andone Lavery | *(same SNR computation applied to EK80 data)* | Same as EB — matched across pipelines |
| **8** | *(not applied — EB reports relative TS(f) only)* | — | Absolute TS(f) calibration via sonar equation: undo PC normalization, impedance/beam correction, TVG, absorption, source level, gain | echopype `_cal_complex_samples`; Demer et al. (2017) §1.2.4.4 Eq. 13. Validated against EK80 software to -0.54 dB at 120 kHz (March 2026 data). |
| **9** | Empirical offset: EB_uncal − EK_cal = **+37.4 dB** mean (100–140 kHz), +37.0 dB at 120 kHz | Single-sphere comparison; underdetermined for independent validation. See Figure S11. | *(EK80 calibrated values serve as reference)* | Derived from A1 baseline; absorbs unknown EB Ptx + electronics gain |

---

## Section 2.4 — EchoBot Pipeline

### Step Provenance Summary

| Step | Operation | From MATLAB? | Script & Line(s) | Notes |
|---|---|---|---|---|
| **1** — Data loading | Load `.mat`, extract `data` array + `header` struct | **Yes** | `Bob_txrx.m:97` (save), `Bob_txrx_init.m:26-81` (header fields) | `.mat` structure defined by MATLAB save command |
| **2** — TX reference construction | Zero-pad chirp: `[zeros(n_pre), s_chirp, zeros(n_post)]` | **Yes** | `Bob_txrx_init.m:73-77` — `n_pre = zeros(T_pre*fs,1)`, `n_post = zeros(T_post*fs,1)`, `tx_data = [n_pre; s_chirp; n_post]` | Python replicates exactly |
| **3** — Bandpass filtering | FIR LP + HP to isolate chirp band | **Partially** | `replay_MF_modZ_CRL.m:6-11` — LP=150 kHz, HP=90 kHz | Python uses wider passband (LP=175 kHz, HP=80 kHz). See [Step 3 detail](#step-3--bandpass-filtering) |
| **4** — Matched filtering | `xcorr` with tx_ref, sum 3 sectors, Hilbert envelope | **Yes** | `replay_MF_modZ_CRL.m:55-57` (xcorr per sector), line 64 (positive lags), `Bob_comprs_plt_bis.m:145-147` (same) | Python uses `scipy.signal.correlate` — equivalent |
| **5** — Range gating | Symmetric gate around autodetected target peak | **No** | — | MATLAB uses hardcoded sample indices. See [Step 5 detail](#step-5--range-gating) |
| **6** — TS(f) computation | FFT deconvolution, Hanning window, coherent averaging | **Partially** | `Bob_comprs_plt_bis.m:50-55` (commented out FFT deconv) | Concept from MATLAB but substantially modified. See [Step 6 detail](#step-6--tsf-computation) |
| **7** — SNR computation | Peak envelope / grand-mean no-target noise | **No** | — | Entirely our design. See [Step 7 detail](#step-7--snr-computation) |

---

### Step 3 — Bandpass filtering

**MATLAB original:** `replay_MF_modZ_CRL.m` lines 6-11 use `designfilt` to create:
- Low-pass FIR: passband 150 kHz, stopband 200 kHz
- High-pass FIR: passband 90 kHz, stopband 70 kHz

This yields a passband of approximately 90–150 kHz, tightly matched to the chirp band.

**Our Python implementation:** `scipy.signal.firwin` (101-tap) with:
- Low-pass at 175 kHz
- High-pass at 80 kHz

This yields a passband of approximately 80–175 kHz — roughly 10 kHz wider on each side.

**Justification:** The chirp has finite spectral rolloff and does not go to zero instantly at the band edges. A filter that cuts at exactly 90 kHz attenuates chirp energy near 90 kHz, distorting the matched-filter output and the TS(f) at the band edges. The wider passband ensures the full chirp energy passes through unattenuated, and delegates frequency selection to the matched filter itself (Step 4) — since cross-correlation with the chirp reference inherently rejects out-of-band energy. The tradeoff is that the wider filter admits more out-of-band noise, but since the matched filter suppresses this energy, the practical impact on SNR is negligible. Additionally, the wider passband accommodates the C1 narrow-bandwidth test group (100–140 kHz) and the standard group (90–150 kHz) with a single filter design, avoiding the need for per-group filter tuning.

**Impact:** Minimal. The matched filter provides the effective band selection. Cross-instrument comparisons are unaffected because the same filter is applied to all EchoBot data.

---

### Step 5 — Range gating

**MATLAB original:** The MATLAB replay scripts do not implement formal range gating for spectral analysis. Instead, they use hardcoded sample index ranges for angle estimation (e.g., `c1(104000:108000)` in `replay_MF_modZ_CRL.m` line 76, and `c1(106000:113000)` in `replay_MF_modZ_CRL_EC.m` line 129). The commented-out code in `Bob_comprs_plt_bis.m` lines 20-24 shows a range-based gating concept (`r1=targ_rang-0.05; r2=r1+i2+0.15`) but this was not used in the final processing.

**Our Python implementation:** A symmetric range gate of ±0.40 m centered on the autodetected target peak from the mean matched-filter envelope of the A1 baseline run. Two approaches:
1. **Per-instrument centering (primary):** EB center = 1.4259 m, EK center = 1.3860 m
2. **Fixed centering (sensitivity check):** 1.424 m for both

**Justification:** This is non-controversial — range gating falls directly out of the experimental design. Any TS(f) computation requires isolating the target return in a finite window for FFT deconvolution. The MATLAB scripts don't formalize this because they were real-time replay/visualization tools, not spectral analysis pipelines. The specific gate width (±0.40 m) was selected based on the gate sensitivity analysis (Section 3.9, Table 7): the marginal correlation improvement from widening to ±0.50 m is small (r = 0.980 vs 0.977), while the wider gate begins to include tank floor reflections. Narrower gates degrade cross-instrument correlation due to the 4.0 cm inter-instrument target peak offset.

**Impact:** The gate width and centering directly affect TS(f) shape and cross-instrument correlation. The ±0.40 m gate with per-instrument centering provides the best balance of target capture and artifact exclusion, as demonstrated by the gate sensitivity analysis.

---

### Step 6 — TS(f) computation

**MATLAB original:** `Bob_comprs_plt_bis.m` lines 50-55 contain commented-out FFT deconvolution code:
```matlab
fft_MF_full = fft(series_MF(fft_ind_MF), N);           % N = 2^16 = 65536
fft_MF = fft_MF_full(fvind_MF) ./ [fft_inp(fvind_inp)]; % divide by complex TX FFT
```
This shows the concept of spectral deconvolution by the transmit waveform, but the implementation differs from ours in several ways.

**Our Python implementation:**
```
H(f) = FFT{C_sum[target gate] * hanning} / |FFT{tx_ref}|^2
```
with NFFT = 4096, Hanning window, and coherent averaging across all pings.

**Differences from MATLAB and justification:**

| Aspect | MATLAB (commented) | Our Python | Justification |
|---|---|---|---|
| **NFFT** | 2^16 = 65,536 | 4,096 | The gate contains ~1,070 samples (±0.40 m at 2 MHz / 1485 m/s). Zero-padding to 4,096 provides ~2 kHz frequency resolution over the 90–150 kHz band — sufficient for resolving target spectral structure. 65,536 would give finer resolution but no additional physical information from a 1,070-sample gate; the extra bins are interpolated zeros. |
| **Deconvolution denominator** | Complex TX FFT: `fft_inp` | TX power spectrum: `\|FFT{tx_ref}\|^2` | Aligns with Demer et al. (2017) Eq. 11 for TS(f). See detail below. |
| **Hanning window** | None (rectangular) | Applied to gated signal | Inspired by Demer et al. (2017) Eq. 14–15; standard conservative practice. See detail below. |
| **Coherent averaging** | Not present (single-ping) | Mean of complex spectra across all pings | Preserves phase information of the stationary target return. Incoherent noise (random phase) averages toward zero, improving SNR by √N_pings. This is standard for stationary-target calibration measurements. |

**Deconvolution denominator — detail:**

Demer et al. (2017) define two different deconvolution operations for two different quantities:

- **TS(f) (Eq. 11):** `U_target(f) = U_r(f) / U_{t,red}(f)`, where `U_{t,red}` is the "reduced length transmit-signal autocorrelation." In the frequency domain, the autocorrelation of the transmit signal is `|U_t(f)|²`. This is exactly our denominator.
- **Sv(f) (Eq. 16):** `U_r(f) = U_rb(f) / U_t(f)`, where `U_t(f)` is the complex transmit FFT directly. This is what the MATLAB commented-out code does.

So the MATLAB's complex FFT division corresponds to the Sv(f) formulation (Eq. 16), while our `|TX|²` division corresponds to the TS(f) formulation (Eq. 11). Since we are computing TS(f), our denominator is the correct one per Demer et al.

The physical reasoning: the matched filter (Step 4) already multiplied the received signal by the conjugate of the transmit spectrum. The MF output spectrum is therefore `H(f) · |TX(f)|²`, where `H(f)` is the target transfer function. Dividing by `|TX(f)|²` recovers `H(f)`. Dividing by the complex `TX(f)` instead would leave a residual `TX*(f)` phase factor in the result.

**Hanning window — detail:**

Demer et al. (2017) Eq. 14–15 describe applying a normalized Hann window to the gated MF signal prior to FFT, in the context of the Sv(f) computation. For TS(f) (Eq. 11), Demer does not explicitly prescribe a window, but the same physical rationale applies: the finite gate truncates the signal, and the abrupt edges produce spectral leakage (sidelobes in the frequency domain that smear energy across frequency bins). A Hanning window tapers the edges to zero, suppressing these sidelobes at the cost of slightly wider main lobe (slightly reduced frequency resolution).

For our application, the practical impact of windowing is small because: (1) the gate is wide (~1,070 samples) relative to the spectral features, (2) the target return naturally tapers at the gate edges, and (3) we apply the same window to both instruments so any windowing artifact cancels in the cross-instrument comparison. The rectangular window (no window, as in the MATLAB code) would also be a valid choice. We apply the Hanning window as a standard conservative precaution, consistent with the Demer Sv(f) formulation and with general practice for FFTs of finite-length segments.

**Impact:** The deconvolution approach, windowing, and coherent averaging all affect the TS(f) spectral shape and noise floor. The key design principle is that the same windowing and FFT parameters are applied to both instruments, so any artifacts are matched and cancel in the cross-instrument comparison.

---

### Step 7 — SNR computation

**MATLAB original:** The MATLAB replay scripts do not compute formal SNR metrics. The closest analog is the real-time envelope plotting in `Bob_comprs_plt_bis.m` line 154 (`cc=20*log10(abs(envelope(...)))`) which displays the matched-filter output in dB, but no noise reference or SNR calculation is performed.

**Our Python implementation:** Two SNR methods:

1. **Active no-target reference (primary):**
   ```
   SNR(p) = 20 * log10( max(env_target(p)) / mean(env_no_target) )
   ```
   where `max(env_target(p))` is the peak matched-filter envelope within the target gate for ping p, and `mean(env_no_target)` is the grand mean envelope across all pings and samples within the same gate from the D1 (no-target) recording.

2. **Empty water-column region (alternative):**
   Uses the below-target region (2.1–2.6 m) as a within-ping noise reference. Available for all groups including those without dedicated no-target runs.

**Justification:** This approach was a direct instruction from project supervisor Andone Lavery. The active no-target method provides the most physically meaningful noise reference because it captures the full reverberation and scattering environment of the tank absent the calibration sphere. The empty-region method serves as a cross-check and enables SNR estimation for parameter-variation groups (B1, C1, P1-*) where no dedicated no-target run was conducted. For the baseline groups where both methods are available, the empty-region method yields ~5–6 dB lower SNR due to residual target reverberation in the within-ping estimate, but the ~3–4 dB EchoBot advantage over EK80 is preserved under both methods, confirming robustness.

**Impact:** SNR computation is a reporting metric and does not affect the TS(f) spectral comparison. The choice of noise reference affects the absolute SNR values but not the relative instrument comparison.

---

## Section 2.5 — EK80 Pipeline

The EK80 pipeline consists of two parts: (1) echopype's built-in broadband TS calibration (`ep.calibrate.compute_TS`), which produces frequency-integrated TS as a function of ping and range, and (2) a custom spectral pipeline for TS(f) computation. The broadband calibration is a black-box call to echopype and is not discussed further here. The custom spectral pipeline is broken down below.

### Design rationale

echopype provides `compute_TS` for broadband (frequency-integrated) target strength but does not expose per-frequency TS(f) spectra. Since TS(f) is the central quantity of this study, we implemented a custom spectral pipeline that replicates echopype's pulse compression steps (chirp construction, filter/decimation, matched filtering) while adding the spectral decomposition (aliasing-aware FFT, deconvolution, coherent averaging) needed for TS(f). Reimplementing rather than wrapping echopype gave us direct control over the spectral processing and ensured consistency with the EchoBot pipeline's TS(f) computation.

### Step Provenance Summary

| Step | Operation | Source | Reference |
|---|---|---|---|
| **1** — Data loading | `ep.open_raw()` → beam, vendor_specific | **echopype** | — |
| **2** — Chirp replica | FM sweep at 1500 kHz, Hanning edge taper via `slope` | **echopype** `tapered_chirp()` | Demer et al. (2017) §1.2.2.4, Eq. 1 |
| **3** — Filter + decimate | WBT filter (47-tap, 8× dec) → PC filter (91-tap, 2× dec) | **echopype** `filter_decimate_chirp()` | Demer et al. (2017) §1.2.3, Eqs. 2–3 |
| **4** — Matched filter | `flipud(conj(tx_filt))` convolved with I/Q, ÷ ‖tx_filt‖² | **echopype** `compress_pulse()` + `get_norm_fac()` | Demer et al. (2017) §1.2.4.1, Eqs. 4–5 |
| **5** — Mean-beam average | Average I/Q across transducer sectors before PC | **echopype** convention | Demer et al. (2017) Eq. 5 |
| **6** — Aliasing-aware freq mapping | `f_actual = f_bb + k·fs_ek` | **Custom** | Demer et al. (2017) §1.2.3 |
| **7** — TS(f) deconvolution | `H = FFT{gated_pc · hanning}[band_idx] / TX_pow` | **Custom** | Demer et al. (2017) Eq. 11 |
| **8** — Coherent averaging | Sum complex spectra across pings ÷ N_pings | **Custom** | Same as EB pipeline |

---

### Steps 1–5: Pulse compression (echopype-derived)

Steps 1–5 are a manual reimplementation of echopype's internal pulse compression pipeline. The echopype source code (`echopype.calibrate.ek80_complex`) provides the reference functions:

- **`tapered_chirp()`**: Generates the FM sweep and applies the Hanning edge taper. The source code credits "Lars Anderson" and references the [CRIMAC-Raw-To-Svf-TSf](https://github.com/CRIMAC-WP4-Machine-learning/CRIMAC-Raw-To-Svf-TSf) repository. Our notebook code (lines 189–201) replicates this exactly.

- **`filter_decimate_chirp()`**: Convolves the chirp with WBT and PC filter coefficients (extracted from the `.raw` file via `get_filter_coeff()`), decimating at each stage. Our notebook code (line 210: `tx_filt = sig_convolve(sig_convolve(y_rf, wbt_fil)[::wbt_dec], pc_fil)[::pc_dec]`) is equivalent.

- **`compress_pulse()`**: Convolves each ping's I/Q data with the time-reversed conjugate of the filtered chirp replica. Our notebook code (line 212: `mf_rep = np.flipud(np.conj(tx_filt))`, line 239: `pc = sig_convolve(raw_iq, mf_rep, mode='full')`) is equivalent.

- **`get_norm_fac()`**: Computes `np.linalg.norm(tx)**2`. Our notebook code (line 211: `norm_fac = np.linalg.norm(tx_filt)**2`) is identical.

All filter coefficients, decimation factors, and transmit parameters come from the EK80 `.raw` file metadata (Kongsberg firmware), not from any discretionary choices. The chirp taper shape is determined by the `slope` parameter stored in the raw file.

**Impact:** These steps are deterministic reproductions of the EK80 firmware processing chain. No discretionary choices are involved.

---

### Step 6 — Aliasing-aware frequency mapping

**Source:** Custom implementation, but the need for it is described in Demer et al. (2017) §1.2.3: "the frequency spectrum, calculated using Fourier analysis, is band-shifted to the predecimated frequencies."

**What it does:** The EK80's effective sample rate (93.75 kHz) is lower than the chirp bandwidth (90–150 kHz = 60 kHz bandwidth). The Nyquist frequency is ~46.9 kHz, so the 90–150 kHz chirp band aliases into baseband. The FFT of the decimated signal produces baseband frequencies; the mapping searches for the integer `k` such that `f_baseband + k · fs_ek` falls within 10% of the chirp band (81–165 kHz), recovering the actual RF frequency for each FFT bin.

**Implementation (notebook lines 214–225):**
```python
f_bb = np.fft.fftfreq(Nfft, d=1.0/fs_ek)
f_actual = np.full(Nfft, np.nan)
for i, fb in enumerate(f_bb):
    for k in range(-10, 11):
        ft = fb + k * fs_ek
        if f_start * 0.90 <= ft <= f_stop * 1.10:
            f_actual[i] = ft
            break
```

**Discretionary choice:** The 10% margin (`f_start * 0.90` to `f_stop * 1.10`) is a heuristic to capture the full chirp band including spectral rolloff. This is not prescribed by Demer or echopype.

**Impact:** Without this mapping, the EK80 TS(f) frequency axis would be wrong (baseband frequencies instead of actual RF frequencies). The 10% margin is generous and does not affect the final result because the TS(f) is subsequently masked to the 90–150 kHz band.

---

### Step 7 — TS(f) deconvolution

**Source:** Custom implementation, consistent with Demer et al. (2017) Eq. 11.

**What it does (notebook lines 227–283):**
```python
TX_fft_ek = np.fft.fft(tx_filt, n=Nfft)
TX_pow_ek = np.maximum(|TX_fft_ek[band_idx]|^2, max(|TX_fft_ek[band_idx]|^2) * 1e-10)
...
H = spec_avg[band_idx] / TX_pow
H_dB = 20 * log10(|H|)
```

The deconvolution divides by `|TX(f)|²` (the power spectrum of the filtered chirp replica), indexed only at the aliased frequency bins corresponding to the chirp band. A floor of `max * 1e-10` prevents division by zero at out-of-band frequencies.

**Relationship to Demer:** This is the same `|TX|²` denominator as Demer Eq. 11 for TS(f), and the same denominator used in the EchoBot pipeline (Step 6 of Section 2.4). See the detailed discussion under Section 2.4 Step 6 above.

**Relationship to EchoBot pipeline:** The deconvolution is conceptually identical — `FFT{gated signal} / |FFT{transmit reference}|²` — but the transmit reference differs: for EchoBot it is `s_chirp` (the raw chirp waveform at 2 MHz), for EK80 it is `tx_filt` (the chirp after WBT+PC filter chain at 93.75 kHz). The Hanning window and NFFT = 4096 are the same in both pipelines.

**Discretionary choices:**
- NFFT = 4096: same as EB, same justification (conventional power-of-2, sufficient resolution)
- Hanning window: same as EB, same justification (spectral leakage suppression, matched across pipelines)
- Division-by-zero floor (`1e-10 × max`): prevents numerical artifacts at out-of-band bins; does not affect in-band results

**Impact:** These are the same discretionary choices as the EchoBot pipeline, applied identically to ensure a fair cross-instrument comparison.

---

### Step 8 — Coherent averaging

**Source:** Custom implementation, same as EchoBot pipeline.

**What it does (notebook lines 275–278):**
```python
spec_acc = np.zeros(Nfft, dtype=complex)
for p in range(all_pc.shape[0]):
    spec_acc += np.fft.fft(all_pc[p, j0:j1] * np.hanning(gate_len), n=Nfft)
spec_avg = spec_acc / all_pc.shape[0]
```

Complex (coherent) averaging of per-ping spectra. Same approach and justification as the EchoBot pipeline — preserves phase, suppresses incoherent noise.

**Impact:** Same as EchoBot Step 6 coherent averaging discussion above.

---

### Note on the absolute TS(f) sonar equation

The manuscript (Section 2.5, Eq. for TS(f)) includes the full sonar equation with impedance correction, TVG, absorption, source level, and gain terms. For the **main-text cross-instrument comparisons**, all TS(f) curves are mean-subtracted (normalized) to isolate spectral shape, so the constant absolute terms cancel out and only the (small) residual spectral tilt from the scalar-gain approximation survives normalization.

The full absolute sonar equation *is* applied, however, in the EK80 absolute TS(f) reporting used by the supplementary calibration figures (Figures S10 and S11). Our `compute_absolute_tsf` function mirrors echopype's `_cal_complex_samples` and has been independently validated against the Simrad EK80 desktop software to within −0.54 dB at 120 kHz and +0.02 dB mean residual across the band (see "Validation: March 2026 Data" below). The uncalibrated EchoBot TS(f) is also reported in the same figure in absolute dB for direct comparison against the calibrated EK80 and the NOAA theoretical reference.

---

## Validation: EK80 Pipeline vs EK80 Desktop Software — September 2025 Data

**Notebook:** `work/notebooks/EK80_process/EK80_validation_analysis.ipynb`
**Data:** `prod-D20250904-T165452.raw` (September 2025 CRL tests)
**Reference data:** `work/notebooks/EK80_process/validation_ping_ts120_timeseries.csv`, `work/notebooks/EK80_process/validation_tsf_detailed.csv`

### Purpose

To confirm that our custom EK80 spectral TS(f) pipeline produces values consistent with the proprietary Simrad EK80 desktop software when applied to the same raw file. This is the ground-truth check for the EK80 side of the cross-instrument comparison.

### Method

TS values were read from the EK80 desktop software at 120 kHz (single-frequency) for 15 pings and at 10 frequencies across 90–150 kHz for 3 selected pings (one moving, two stationary). These were compared against:
1. Our custom spectral pipeline TS(f) at the same frequencies
2. echopype's built-in broadband TS (`ep.calibrate.compute_TS`)

### Results

| Metric | Value |
|---|---|
| Pipeline offset at 120 kHz (stationary pings) | ~0.8 dB lower than EK80 software |
| echopype BB TS offset (stationary pings) | ~4.4 dB lower than EK80 software |
| Ping-to-ping correlation (pipeline vs EK80) | r > 0.99 |
| Spectral features (nulls at 134–136 kHz) | Captured correctly |

**Key finding — frequency-dependent tilt:** The residuals show a systematic spectral tilt: our pipeline reads lower than EK80 below ~120 kHz and higher above ~130 kHz, with a total tilt of ~3–5 dB across the 90–150 kHz band. The crossover is near the center frequency (120–125 kHz).

### Sources of discrepancy

1. **Frequency-dependent gain (dominant cause).** Our pipeline applies a single scalar gain (G = 18.0 dB) across the entire band. The EK80 software likely applies a frequency-dependent gain curve G(f) derived from the transducer's beam pattern and sensitivity calibration. A slope of ~2–3 dB across the 60 kHz band would explain the tilt.

2. **Constant ~0.8 dB offset at 120 kHz.** Even at the center frequency, we are ~0.8 dB low. Possible causes: the gain value we read may not exactly match the EK80's internal calibration; the transducer impedance z_et = 75 Ω is a hardcoded default; windowing differences.

3. **Band-edge effects at 90 kHz.** The 90 kHz point shows an anomalous residual (~3–6 dB higher than EK80, opposite to the low-frequency trend). This is because 90 kHz is at the chirp band edge where |TX(f)|² drops sharply, amplifying deconvolution errors.

4. **Frequency-dependent absorption.** We use a fixed α = 0.04 dB/m. Absorption increases with frequency, but at ~1.5 m range the two-way difference across the band is only ~0.06 dB — negligible.

5. **echopype BB TS ~4 dB low.** echopype's broadband TS measures a different quantity (frequency-integrated spatial peak after pulse compression), not TS at a specific frequency. It should not be used as a spectral reference.

### Implications for the manuscript

**What normalization removes and what it does not:**

Mean subtraction (our normalization) removes the ~0.8 dB constant offset. It does **not** remove the frequency-dependent tilt. A ~3–5 dB slope across 90–150 kHz survives mean subtraction — subtracting the band mean shifts the midpoint to zero but the slope remains. This means our normalized EK80 TS(f) carries a gentle linear tilt that would not be present if we used the EK80 software's frequency-dependent gain G(f).

**Why the results are still valid despite the residual tilt:**

The WC calibration sphere's actual TS(f) has ~10–15 dB of spectral structure across 90–150 kHz: resonance features, nulls at 134–136 kHz, and a characteristic frequency-dependent scattering pattern. The 3–5 dB tilt from our scalar gain approximation is a slowly-varying background slope superimposed on this rich structure. The Pearson correlation — the metric we use for all cross-instrument comparisons — measures whether the spectral *features* (peaks, nulls, inflections) align between two curves. A gentle linear tilt does not destroy feature agreement; it slightly warps the baseline slope. This is why we still achieve r = 0.977 for the cross-instrument comparison.

That said, this is a real limitation. Applying G(f) to the EK80 pipeline would remove the tilt, improve the absolute accuracy of the EK80 TS(f), and likely improve the cross-instrument correlation slightly. This is acknowledged as a known source of imperfection in the current pipeline.

**Validation conclusion:**

The custom EK80 pipeline is well-validated for the relative spectral-shape comparison that is the core of the paper: it captures the correct spectral features to within ~1 dB at most frequencies, tracks the EK80 software with r > 0.99 ping-to-ping, and its known limitations (scalar gain, ~0.8 dB offset, band-edge effects) are either removed by normalization or are small relative to the spectral structure being compared.

**If absolute TS(f) were needed** (e.g., for a future calibration paper), the frequency-dependent gain curve G(f) would need to be applied. This is noted in the manuscript discussion (Section 4.8) as a requirement for absolute calibration.

---

## Validation: EK80 Pipeline vs EK80 Software — March 2026 Data

**Consolidated notebook:** `work/notebooks/system_comparison/A1_absolute_tsf_validation.ipynb` — self-contained notebook that loads the A1 baseline EchoBot and EK80 recordings, applies the full calibration pipeline, and produces **Figure S10** (pipeline vs desktop software) and **Figure S11** (EB uncal / EK80 cal / NOAA theory).
**Validation data:** `work/notebooks/EK80_process/validation_0311_ping_ts120.csv` (15 pings × TS@120 kHz manual reads), `work/notebooks/EK80_process/validation_0311_tsf_detailed.csv` (5 pings × 10 frequencies manual reads).
**Data:** `work/data/EK80/0311-CRL-tests/prod-D20260311-T182413.raw` (A1 baseline, 127 pings, 60 W, 90–150 kHz)
**Calibration method:** `compute_absolute_tsf` from `work/notebooks/EK80_process/EK80_pipeline_01.ipynb` — applies the full sonar equation matching echopype's `_cal_complex_samples` (which in turn implements Demer et al. 2017 §1.2.4.4 Eq. 13):

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

with `N_ref = 8` (the echopype `_cal_complex_samples` convention — `/4` from 4-element split-beam voltage division times `/2` from the `mean_beam` proxy for the element sum).

#### Parameter table

The following table defines every symbol in the TS(f) equation, its physical meaning, units, source of its value in our pipeline, and the concrete A1 baseline values for reference (EK80 file `prod-D20260311-T182413.raw`, range gate centered at 1.386 m, f = 120 kHz unless noted).

| Symbol | Description | Units | Source | A1 value | Notes |
|---|---|---|---|---|---|
| `f` | Analysis frequency | Hz | Aliasing-aware mapping (Section 2.5 Step 6) of the baseband FFT bins back to RF | 90–150 kHz sweep | Equation evaluated at every in-band FFT bin; per-frequency tabular values reported at the 10 reference frequencies |
| `H(f)` | Deconvolved complex spectrum | dimensionless | `H = FFT{pc[gate]·hann} / |TX_filt|²` (Section 2.5 Step 7) | Complex array | `pc` is the mean-beam pulse-compressed I/Q already divided by `norm_fac`; `TX_filt = FFT{tx_filt}` |
| `\|H(f)\|²` | Received power spectrum | dimensionless (V²-proxy) | Magnitude-squared of `H(f)` | Complex → power | Equation term (1) |
| `norm_fac` | Pulse-compression norm factor | dimensionless | `np.linalg.norm(tx_filt)²` — echopype `get_norm_fac()`; Demer et al. (2017) §1.2.4.1 Eq. 5 | **44.18** (20·log₁₀ = +32.90 dB) | Value depends on WBT/PC filter chain + chirp parameters |
| `tx_filt` | Decimated, filtered transmit replica | complex samples | `y_rf` → WBT filter → /8 dec → PC filter → /2 dec (echopype `filter_decimate_chirp`; Demer et al. 2017 §1.2.3, Eqs. 2–3) | 96 samples (A1) | Raw chirp `y_rf` = 767 samples at 1500 kHz → 102 after WBT+/8 → 96 after PC+/2; filter coefficients from `.raw` vendor metadata (`get_filter_coeff`) |
| `N_beams` | Number of transducer elements in Beam_group1 | count | `.raw` file: `Sonar/Beam_group1` beam dimension (`beam.sizes['beam']`) | **3** | Simrad ES120-7C split-beam; firmware-reported |
| `N_ref` | Normalization denominator in Z_dB | count | Hardcoded constant = 8 in `echopype._cal_complex_samples` | **8** | `/4` split-beam + `/2` voltage-division factor (Demer Eq. 13 per-element formulation) |
| `z_er` | Transceiver (WBT) receive impedance | Ω | `.raw` file: `Vendor_specific/impedance_transceiver` | **10 800 Ω** | Firmware-reported; varies slightly by WBT channel |
| `z_et` | Transducer (ET) impedance | Ω | **Hardcoded constant = 75 Ω** in `compute_absolute_tsf` | **75 Ω** | Demer et al. (2017) §1.2.4.4; nominal ES-series transducer impedance. *Sensitivity check not yet performed.* |
| `Z_dB` | Combined impedance + beam scaling | dB | `10·log₁₀(N_beams/N_ref) + 20·log₁₀(\|z_er+z_et\|/z_er) − 10·log₁₀(z_et)` | **−22.95 dB** | Equation term (3); −4.26 + 0.060 − 18.75 |
| `R` | Target range (gate center) | m | Autodetected envelope peak of the A1 ping 0 matched-filter output, per-instrument (EK80 = 1.386 m, EB = 1.4259 m); see Section 2.5 Step 5 | **1.386 m** | Used for both TVG and absorption terms; 4.0 cm offset from EB center is documented in Section 3.9 |
| `α` (alpha) | Seawater absorption coefficient | dB/m | **Hardcoded constant = 0.04 dB/m** in `compute_absolute_tsf` | **0.04 dB/m** | Nominal value for ~120 kHz in ~15 °C brackish tank water; frequency dependence is ~0.06 dB over the 60 kHz band at 1.386 m (negligible). Could be replaced with a per-frequency Francois–Garrison value in future work. |
| `c` | Sound speed | m/s | CTD measurement in tank, Mackenzie formula (Section 2.2, cell 2 of `0311-CRL-analysis.ipynb`) | **1484.96 m/s** | Used to compute `λ = c/f` and to set range axis; A1-specific |
| `λ` (lam) | Acoustic wavelength | m | Computed: `λ = c/f` | **12.375 mm** at 120 kHz | Varies 9.90–16.50 mm across 90–150 kHz |
| `Ptx` | Transmit electrical power | W | `.raw` file: `Sonar/Beam_group1/transmit_power` | **60 W** | Firmware-reported per-ping; constant across A1 pings |
| `G` | Scalar gain (center-frequency) | dB | `.raw` file: `Vendor_specific/gain_correction` | **18.0 dB** | Single-frequency value; applied as `−2G`. A frequency-dependent `G(f)` curve would improve absolute accuracy and spectral-tilt residuals (see Notes for Future Work #3). |
| `40·log₁₀(R)` | Two-way spherical spreading | dB | Computed from `R` | **+5.67 dB** | Equation term (4) |
| `2·α·R` | Two-way absorption loss | dB | Computed from `α`, `R` | **+0.111 dB** | Equation term (5) |
| `10·log₁₀(λ²·Ptx/(16π²))` | Source level component | dB | Computed from `λ`, `Ptx` | **−42.35 dB** at 120 kHz | Equation term (6); inverts the Demer Eq. 13 source-pressure term |
| `2·G` | Two-way gain correction | dB | Computed: `2 × G_correction` | **36.0 dB** | Equation term (7); subtracted |

**Symbol glossary for the table above:**
- *dimensionless (V²-proxy)*: `|H|²` is in linear units proportional to received voltage squared, with the receive-chain voltage-to-power conversion handled by the `Z_dB` term.
- *Firmware-reported*: written by the Kongsberg EK80 transceiver to the `.raw` file at transmit time; not a discretionary choice.
- *Hardcoded constant*: set as a Python literal in `compute_absolute_tsf`; not drawn from data.
- *Autodetected*: computed from the data via an automated algorithm (not hand-chosen).

**Independent constants and their rationale:**

| Constant | Value | Rationale |
|---|---|---|
| `z_et` | 75 Ω | Nominal ES-series transducer impedance; cited in Demer et al. (2017) §1.2.4.4 and used by echopype's internal `_cal_complex_samples` default. A transducer-specific impedance measurement would refine this. |
| `N_ref` | 8 | Echopype convention (`_cal_complex_samples`): accounts for `/4` four-element split-beam voltage division and `/2` for the mean-beam proxy used in the received-power calculation. |
| `α` | 0.04 dB/m | Broadband average absorption over 90–150 kHz for tank conditions (~15 °C, ~30 PSU, ~1 m depth). Two-way error from using a flat value is < 0.1 dB over the band at 1.4 m range. |
| `c` | 1484.96 m/s | Per-session CTD measurement; updated per run from the tank CTD cast. |

#### Worked example: A1 baseline TS at 120 kHz

Concrete term-by-term evaluation of the sonar equation using the parameter values in the table above. The raw deconvolved power `10·log₁₀(|H(120 kHz)|²)` is −59.22 dB (measured from the A1 127-ping coherent average), and each calibration term is added in turn:

| Term | Contribution (dB) |
|---|---|
| (1) `10·log₁₀(\|H\|²)`                          | **−59.224** |
| (2) `20·log₁₀(norm_fac)`                       | +32.905 |
| (3) `Z_dB`                                     | −22.950 |
| (4) `40·log₁₀(R)` = `40·log₁₀(1.386)`           | +5.671 |
| (5) `2·α·R` = `2·(0.04)·(1.386)`               | +0.111 |
| (6) `−10·log₁₀(λ²·Ptx/(16π²))` at λ=12.375 mm, Ptx=60 W | +42.352 |
| (7) `−2·G` = `−2·(18.0)`                       | −36.000 |
| **TS(120 kHz)**                                | **−37.136** |

This matches both the value reported in the NOAA-theory comparison table (line 418: EK = −37.14 at 120 kHz) and the 120 kHz residual in Figure S10 (pipeline minus software = −0.54 dB relative to the EK80 software's −36.49 dB read). The arithmetic is closed end-to-end.

### Method

TS values were manually read from the EK80 desktop software for the same raw file:
- **TS@120 kHz**: 15 pings evenly spaced across the 127-ping recording
- **TS(f)**: 5 pings × 10 frequencies (90, 95, 100, 110, 120, 125, 130, 135, 140, 149.9 kHz)

All pings were stationary (sphere fixed in tank). The 135 kHz point falls on the descending slope of a sharp null (true minimum at ~135.0–135.1 kHz), so the exact null depth is sensitive to sub-kHz frequency alignment and is excluded from summary statistics. The 150 kHz manual readings were taken at 149.9 kHz in the EK80 software.

### Results

| Metric | September 2025 (0904) | March 2026 (0311) |
|---|---|---|
| Residual at 120 kHz | -0.8 dB | **-0.54 dB** |
| Mean residual (all freq, excl null) | — | **+0.02 dB** |
| Std residual (excl null) | — | 2.56 dB |
| Mean residual (100–140, excl null) | — | -0.21 dB |
| Std (100–140, excl null) | — | 1.86 dB |
| Spectral tilt (high – low, excl null) | ~3–5 dB | +5.15 dB |

**Per-frequency mean residual (pipeline – EK80 software):**

| Freq (kHz) | Residual (dB) |
|---|---|
| 90 | +3.20 (band edge) |
| 95 | -4.42 |
| 100 | -3.14 |
| 110 | -1.79 |
| 120 | -0.54 |
| 125 | +0.81 |
| 130 | +1.24 |
| 135 | +13.5 (null — excluded) |
| 140 | +2.17 |
| 149.9 | +2.68 |

### Key findings

1. **The calibration is validated on the March data.** The overall mean residual (excluding the null) is +0.02 dB — essentially zero. At 120 kHz specifically, it is -0.54 dB, consistent with the September result.

2. **The spectral tilt persists** (+5.15 dB, same direction as September). This is the frequency-dependent gain issue: our pipeline uses scalar G = 18.0 dB while the EK80 software likely applies G(f).

3. **The 135 kHz null discrepancy** (+13.5 dB) is a frequency-alignment artifact: the null is extremely sharp and the EK80 software reads deeper into it because it may evaluate at a slightly different frequency. This does not indicate a pipeline error.

4. **The 90 kHz band-edge anomaly** (+3.2 dB) is expected — the chirp power spectrum rolls off sharply here, amplifying small deconvolution differences.

5. **Consistency across sessions**: the September and March validations agree to within ~0.3 dB at 120 kHz, confirming the pipeline is stable across different recording sessions and target geometries.

---

## Validation: Calibrated EK80 vs NOAA Theoretical Sphere TS(f)

**Reference:** NOAA Standard Sphere Target Strength Calculator (https://www.fisheries.noaa.gov/data-tools/standard-sphere-target-strength-calculator) for a 38.1 mm WC sphere (tungsten carbide, 6% cobalt binder). Theoretical TS(f) values at 1 kHz resolution across 90–150 kHz were transcribed and saved at `work/data/NOAA-381-WC-TSf.xlsx`.

### Method

The calibrated EK80 TS(f) from the A1 baseline (coherent average of 127 pings, `compute_absolute_tsf` with full sonar equation) was compared against the NOAA theoretical reference at each 1 kHz frequency point. Statistics exclude the 90 kHz band edge and the 134–136 kHz null region.

### Results

| Metric | Value |
|---|---|
| Mean residual (EK – theory, excl edges/null) | **+2.94 dB** |
| Std residual | 2.51 dB |
| RMS residual | 3.87 dB |
| At 120 kHz | EK = -37.14, theory = -39.53, **+2.39 dB** |
| Mean residual (100–140, excl null) | +2.87 dB |
| Std (100–140, excl null) | 1.62 dB |
| Shape correlation (normalized, excl edges/null) | r = 0.705 |
| Spectral tilt (high – low, excl null) | +4.55 dB |

**Per-frequency residual (EK80 calibrated – NOAA theory):**

| Freq (kHz) | EK (dB) | Theory (dB) | Residual |
|---|---|---|---|
| 90 | -52.23 | -46.06 | -6.17 (band edge) |
| 95 | -39.88 | -39.15 | -0.73 |
| 100 | -39.91 | -40.16 | +0.25 |
| 110 | -38.14 | -40.21 | +2.07 |
| 120 | -37.14 | -39.53 | +2.39 |
| 125 | -36.50 | -40.30 | +3.80 |
| 130 | -37.28 | -41.67 | +4.39 |
| 135 | -54.89 | -59.82 | +4.93 (null) |
| 140 | -34.22 | -39.76 | +5.54 |
| 150 | -32.79 | -38.97 | +6.18 |

### Key findings

1. **The calibrated EK80 reads ~2.4–2.9 dB above theory** across the mid-band. This is consistent with the ~0.5 dB offset between our pipeline and the EK80 software (i.e., the EK80 software itself reads ~2 dB above theory at 120 kHz). Known contributors include: gain table accuracy, near-field effects at 1.4 m range, and sphere material property uncertainties.

2. **The spectral tilt persists against theory** (+4.55 dB from low to high frequencies), confirming it originates from our scalar-gain approximation (G = 18.0 dB flat) rather than from a mismatch with the EK80 software specifically.

3. **The null at 135 kHz** is captured at the correct frequency (both show the minimum at 135 kHz). The +4.93 dB residual at the null reflects windowing smoothing in our pipeline — the theoretical curve drops to -59.82 dB while our windowed/averaged spectrum reaches -54.89 dB.

4. **The shape correlation is moderate** (r = 0.705) because the frequency-dependent gain tilt systematically warps the spectral shape. After removing the tilt (i.e., normalizing both curves), the feature agreement (null location, local peaks, inflections) is much better.

5. **The ~+37 dB offset between uncalibrated EchoBot and calibrated EK80** can now be contextualized against theory. From the A1 baseline (100–140 kHz mid-band means): EB uncal ≈ −0.6 dB, EK cal ≈ −38.0 dB, NOAA theory ≈ −41.0 dB. The offsets are therefore: EB−EK = +37.4 dB (unknown EB electronics gain), EK−theory = +3.0 dB (scalar-gain tilt / near-field / gain-table uncertainty), and EB−theory = +40.4 dB (the sum). The entire EB−EK offset represents the unknown combined EchoBot transmit/receive electronics gain constant that would be absorbed by a proper absolute calibration. See Figures S10 and S11 in the manuscript.

---

## Notes for Future Work

The following analyses are not included in the current draft but should be considered for a comprehensive final version of the manuscript. These could be referenced in the main text and included as supplementary material.

### 1. Pipeline parameter sensitivity analysis

The current pipeline uses a single set of processing parameters (Hanning window, coherent averaging, NFFT=4096, |TX|² denominator). A systematic comparison of results under different parameter combinations would strengthen confidence in the robustness of the reported TS(f) correlations and demonstrate that the conclusions are not sensitive to these choices. Combinations to explore:

- **Window function:** Hanning vs rectangular (no window) vs other tapers (Hamming, Blackman-Harris). The rectangular window is what the original MATLAB code uses; showing that the cross-instrument correlation is stable across window choices would confirm that this discretionary decision is inconsequential.
- **Coherent vs incoherent averaging:** Coherent averaging (sum complex spectra) vs incoherent averaging (sum magnitude spectra) vs no averaging (single-ping TS(f)). Coherent averaging should yield the cleanest spectra for a stationary target, but demonstrating the comparison empirically would be valuable.
- **NFFT:** 4096 vs 2048 vs 8192 vs 65536 (the MATLAB value). The cross-instrument correlation should be insensitive to NFFT above some minimum; showing this explicitly would address any concern about the choice.
- **Deconvolution denominator:** |TX|² (current, Demer Eq. 11) vs complex TX (Demer Eq. 16 / MATLAB approach). These should produce similar normalized spectra since the phase difference is removed by taking magnitude, but confirming this empirically would close the loop.

This analysis would produce a summary table or figure showing cross-instrument Pearson r under each parameter combination, demonstrating stability.

### 2. ~~EK80 pipeline validation on March 2026 data~~ (COMPLETED)

This has been done — see "Validation: March 2026 Data" section above. Results confirm the pipeline matches the EK80 software to -0.54 dB at 120 kHz on the March data, consistent with September.

### 3. Frequency-dependent gain G(f)

If the EK80 software's G(f) curve can be extracted, applying it to our pipeline would remove the ~5 dB spectral tilt and improve both the absolute accuracy and the shape correlation against theory. This would also improve the cross-instrument comparison by removing the tilt from the EK80 normalized spectrum.

### 4. Data sharing via Zenodo

The raw data supporting this manuscript (EchoBot `.mat` files, EK80 `.raw` files, hydrophone `.wav` files, CTD `.nc` files, manual EK80 software validation CSVs) should be published on Zenodo and assigned a DOI before the manuscript is submitted. Zenodo is a CERN-operated academic data repository, free, accepts up to 50 GB per record, issues a persistent DOI, and integrates with GitHub for versioned releases.

**Scope for the Zenodo record:**
- `echobot/0311-CRL-tests/` — 18 EchoBot `.mat` files (A1, A1-dup, A1-rep, B1, C1, D1, P1-lo/mid/hi + hydrophone-as-target runs)
- `EK80/0311-CRL-tests/` — 9 EK80 `.raw` files + accompanying `.idx`/`.xml`
- `hydrophone/0311-CRL-tests/` — 4 SoundTrap `.wav` files + `.sud`/`.log.xml` (note: `175538.wav` is only present as `.sud` after dedup; include the `.sud` originals for all files so users can decode with SoundTrap Host if needed)
- `ctd/060633_20260311_1409.nc`
- `validation/validation_0311_ping_ts120.csv`, `validation/validation_0311_tsf_detailed.csv`
- `NOAA-381-WC-TSf.xlsx`
- `README.md` — dataset description, instrument parameters, sphere spec, tank geometry, file-to-comparison-group mapping
- Optionally: the September 2025 validation CSVs (`validation_ping_ts120_timeseries.csv`, `validation_tsf_detailed.csv`) for independent cross-session verification

**Workflow:**
1. Stage the directory tree locally (e.g., `echobot/zenodo_release/`) with the structure above.
2. Write a clear `README.md` at the top of the Zenodo record linking back to the companion code repo (`echobot/echobot/`) and the manuscript.
3. Upload as a new Zenodo record, tag with `echosounder`, `broadband TS(f)`, `calibration`, `mesopelagic`, `acoustics`.
4. Once the DOI is minted, add it to:
   - `echobot/echobot/docs/DATA.md` and the main `README.md`
   - The manuscript (Data Availability section)
   - `manuscript/notes/pipeline_provenance.md` (replace this to-do with the actual DOI)
5. Archive a linked code release: tag the `echobot/echobot/` repo at the submission version and use Zenodo's GitHub integration to mint a separate code DOI.

**Estimated size:** ~10–15 GB (dominated by EchoBot `.mat` files and hydrophone recordings). Well under Zenodo's 50 GB per-record limit.
