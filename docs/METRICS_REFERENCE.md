# esl Metrics Reference

This document defines every built-in metric currently registered by default in `esl`.

Scope:
- 74 built-in metrics (core + extended)
- Mathematical definition (implementation-aligned)
- Plain-English interpretation
- Operational extraction workflow: [`docs/MOMENTS_EXTRACTION.md`](MOMENTS_EXTRACTION.md)

## Notation

- `x_c[n]`: sample `n` in channel `c`
- `C`: number of channels
- `F_k`: frame `k` (windowed sample block)
- `N_k`: number of samples in frame `k`
- `RMS(F_k) = sqrt((1/(N_k*C)) * sum_{n,c} x_{k,c}[n]^2)`
- `dB(v) = 20*log10(max(|v|, eps))`
- `dB_power(p) = 10*log10(max(p, eps))`
- `L_SPL = L_dBFS + (spl_reference_db - dbfs_reference)` when calibration is provided
- `M(f,t)`: magnitude spectrogram
- `P(f,t) = M(f,t)^2`: power spectrogram
- `eps`: small positive constant for numerical stability

All formulas below reflect current `esl` implementation, including metrics labeled as "proxy".

## Rendered Equation Companion

These equations are rendered in generated HTML/PDF docs and each has a plain-English interpretation.

### 1) Frame Energy and RMS

$$
E_k = \sum_{c=1}^{C} \sum_{n=0}^{N_k-1} x_{k,c}[n]^2
$$

$$
\mathrm{RMS}(F_k) = \sqrt{\frac{1}{N_k C}\sum_{c=1}^{C}\sum_{n=0}^{N_k-1} x_{k,c}[n]^2}
$$

Plain English: frame energy is total squared amplitude; RMS is the energy-normalized average amplitude.

### 2) dBFS Mapping

$$
L_{\mathrm{dBFS},k} = 20\log_{10}\!\big(\max(\mathrm{RMS}(F_k), \varepsilon)\big)
$$

Plain English: RMS is mapped to logarithmic decibels relative to full scale.

### 3) Calibrated SPL Mapping

$$
L_{\mathrm{SPL},k} = L_{\mathrm{dBFS},k} + \left(L_{\mathrm{SPL,ref}} - L_{\mathrm{dBFS,ref}}\right)
$$

Plain English: calibrated SPL is a fixed offset from digital dBFS, derived from calibration reference points.

### 4) Equivalent Continuous Level (Leq)

$$
L_{\mathrm{eq}} = 10\log_{10}\!\left(\frac{1}{T}\int_0^T \frac{p^2(t)}{p_0^2}\,dt\right)
$$

Plain English: Leq is the single steady level carrying the same acoustic energy as the varying signal.

### 5) Spectral Centroid

$$
f_c(t) = \frac{\sum_f f \cdot M(f,t)}{\sum_f M(f,t) + \varepsilon}
$$

Plain English: spectral centroid tracks where the spectrum is centered in frequency (perceptual brightness proxy).

### 6) Spectral Flux Novelty

$$
N(t) = \sum_f \max\!\left(M(f,t)-M(f,t-1), 0\right)
$$

Plain English: novelty rises when new spectral energy appears between successive frames.

### 7) NDSI

$$
\mathrm{NDSI} = \frac{E_{\mathrm{bio}} - E_{\mathrm{anthro}}}{E_{\mathrm{bio}} + E_{\mathrm{anthro}} + \varepsilon}
$$

Plain English: NDSI is positive when biological-band energy dominates and negative when anthropogenic-band energy dominates.

### 8) Clarity (C50/C80)

$$
C_T = 10\log_{10}\!\left(\frac{\int_0^T h^2(t)\,dt}{\int_T^\infty h^2(t)\,dt + \varepsilon}\right), \quad T\in\{50\text{ms},80\text{ms}\}
$$

Plain English: clarity compares early arriving energy to late reverberant energy.

### 9) Definition (D50)

$$
D_{50} = \frac{\int_0^{50\text{ms}} h^2(t)\,dt}{\int_0^\infty h^2(t)\,dt + \varepsilon}
$$

Plain English: D50 is the fraction of energy arriving in the first 50 ms.

### 10) Reverberation Time Regression

$$
RT60 \approx -\frac{60}{m}
$$

where \(m\) is slope (dB/s) from linear regression of Schroeder decay in the selected dB window.

Plain English: steeper decay slope implies shorter reverberation time.

## Metric Contract Matrix

This matrix is the stable, ML-facing contract for every built-in metric ID.
Definitions are anchored to the equation tables by category.

| Metric ID | Units | Window/Hop | Streamable | Cal. Dep. | Aggregation semantics | Definition |
|---|---:|---|---:|---:|---|---|
| `acoustic_complexity_index` | ratio | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `acoustic_entropy` | ratio | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `adi` | ratio | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `aei` | ratio | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `ambisonic_diffuseness` | ratio | `0/0` | no | no | First four channels interpreted as FOA WXYZ. | [Spatial](#spatial-ambisonic-metrics) |
| `ambisonic_energy_vector_azimuth_deg` | deg | `0/0` | no | no | First four channels interpreted as FOA WXYZ. | [Spatial](#spatial-ambisonic-metrics) |
| `ambisonic_energy_vector_elevation_deg` | deg | `0/0` | no | no | First four channels interpreted as FOA WXYZ. | [Spatial](#spatial-ambisonic-metrics) |
| `autoencoder_recon_error` | mse | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Anomaly / Novelty](#anomaly-novelty-metrics) |
| `bass_ratio` | ratio | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `bioacoustic_index` | a.u. | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `c50_db` | dB | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `c80_db` | dB | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `calibration_drift_db` | dB | `0/0` | no | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `change_point_confidence` | ratio | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Anomaly / Novelty](#anomaly-novelty-metrics) |
| `clipping_event_count` | count | `0/0` | yes | no | All samples over all channels in one pool. | [Basic](#basic-quality-control-metrics) |
| `clipping_ratio` | ratio | `0/0` | yes | no | All samples over all channels in one pool. | [Basic](#basic-quality-control-metrics) |
| `completeness_ratio` | ratio | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `crest_factor_db` | dB | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `d50` | ratio | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `dc_offset` | linear | `0/0` | yes | no | All samples over all channels in one pool. | [Basic](#basic-quality-control-metrics) |
| `diurnal_coverage_ratio` | ratio | `0/0` | no | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `doa_azimuth_proxy_deg` | deg | `2048/512` | yes | no | First two channels (L/R pair); mono duplicates ch1. | [Spatial](#spatial-ambisonic-metrics) |
| `dropout_ratio` | ratio | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `eco_octave_trends` | dB/s | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `edt_s` | s | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `g_strength_db` | dB | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `iacc` | ratio | `2048/512` | yes | no | First two channels (L/R pair); mono duplicates ch1. | [Spatial](#spatial-ambisonic-metrics) |
| `ild_db` | dB | `2048/512` | yes | no | First two channels (L/R pair); mono duplicates ch1. | [Spatial](#spatial-ambisonic-metrics) |
| `integrated_lufs` | LUFS | `0/0` | no | no | Multichannel loudness combines per-channel powers equally. | [Level & Loudness](#level-loudness-metrics) |
| `interchannel_coherence` | ratio | `2048/512` | yes | no | Mean adjacent-channel correlation per frame. | [Spatial](#spatial-ambisonic-metrics) |
| `ipd_rad` | rad | `2048/512` | yes | no | First two channels (L/R pair); mono duplicates ch1. | [Spatial](#spatial-ambisonic-metrics) |
| `isolation_forest_score` | score | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Anomaly / Novelty](#anomaly-novelty-metrics) |
| `itd_s` | s | `2048/512` | yes | no | First two channels (L/R pair); mono duplicates ch1. | [Spatial](#spatial-ambisonic-metrics) |
| `l10_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `l50_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `l90_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `l95_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `lae_db` | dBA | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `leq_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `lf_ratio` | ratio | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `lfc_ratio` | ratio | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `lmax_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `lmin_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `loudness_range_lu` | LU | `0/0` | no | no | Multichannel loudness combines per-channel powers equally. | [Level & Loudness](#level-loudness-metrics) |
| `lpeak_dbfs` | dBFS | `0/0` | yes | no | All samples over all channels in one pool. | [Level & Loudness](#level-loudness-metrics) |
| `momentary_lufs` | LUFS | `0/0` | no | no | Multichannel loudness combines per-channel powers equally. | [Level & Loudness](#level-loudness-metrics) |
| `ndsi` | ratio | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Ecoacoustics](#ecoacoustics-metrics) |
| `novelty_curve` | a.u. | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Anomaly / Novelty](#anomaly-novelty-metrics) |
| `ocsvm_score` | score | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Anomaly / Novelty](#anomaly-novelty-metrics) |
| `octave_band_level_db` | dB | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Spectral](#spectral-metrics) |
| `peak_dbfs` | dBFS | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `rms_dbfs` | dBFS | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `rt60_s` | s | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `sel_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `short_term_lufs` | LUFS | `0/0` | no | no | Multichannel loudness combines per-channel powers equally. | [Level & Loudness](#level-loudness-metrics) |
| `silence_ratio` | ratio | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `site_comparability_score` | ratio | `0/0` | no | yes | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `snr_db` | dB | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Noise & SNR](#noise-snr-metrics) |
| `spectral_bandwidth_hz` | Hz | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Spectral](#spectral-metrics) |
| `spectral_centroid_hz` | Hz | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Spectral](#spectral-metrics) |
| `spectral_change_detection` | zscore | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Anomaly / Novelty](#anomaly-novelty-metrics) |
| `spectral_flatness` | ratio | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Spectral](#spectral-metrics) |
| `spectral_rolloff_hz` | Hz | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Spectral](#spectral-metrics) |
| `spl_a_db` | dBA | `2048/512` | no | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `spl_c_db` | dBC | `2048/512` | no | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `spl_z_db` | dB | `2048/512` | yes | yes | Frame-level features aggregate channels jointly. | [Level & Loudness](#level-loudness-metrics) |
| `sti_proxy` | ratio | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `t20_s` | s | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `t30_s` | s | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `third_octave_band_level_db` | dB | `2048/512` | no | no | Mono downmix (channel mean) before analysis. | [Spectral](#spectral-metrics) |
| `true_peak_dbfs` | dBFS | `0/0` | no | no | Multichannel loudness combines per-channel powers equally. | [Level & Loudness](#level-loudness-metrics) |
| `ts_ms` | ms | `0/0` | no | no | Mono downmix (channel mean) before analysis. | [Architectural Acoustics](#architectural-acoustics-intelligibility-metrics) |
| `uptime_ratio` | ratio | `2048/512` | yes | no | Frame-level features aggregate channels jointly. | [Basic](#basic-quality-control-metrics) |
| `zero_crossing_rate` | ratio | `2048/512` | yes | no | Mono downmix (channel mean) before analysis. | [Temporal](#temporal-metrics) |

## Streaming Semantics

- `streaming_capable = yes`: metric can be merged from chunk-local computations in `chunk_size` mode.
- `streaming_capable = no`: `esl` falls back to full-file analysis for correctness.
- Architectural, many ecoacoustic, and model-based anomaly metrics are intentionally non-streaming.

## Basic + Quality Control Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `rms_dbfs` | dBFS | `L_k = dB(RMS(F_k))` | Average signal level per frame on a full-scale digital reference. |
| `peak_dbfs` | dBFS | `L_k = dB(max_{n,c} |x_{k,c}[n]|)` | Peak sample level per frame in dBFS. |
| `lpeak_dbfs` | dBFS | `L = dB(max_{n,c} |x_c[n]|)` | Highest sample peak over the entire clip. |
| `clipping_ratio` | ratio | `mean( 1[|x_c[n]| >= 0.999] )` over all samples/channels | Fraction of samples at/near digital clipping. |
| `clipping_event_count` | count | Number of rising edges in clip mask `any_c 1[|x_c[n]|>=0.999]` | Number of distinct clipping bursts (not just clipped sample count). |
| `dc_offset` | linear | `mean_{n,c}(x_c[n])` | Constant bias in waveform baseline. |
| `dropout_ratio` | ratio | Fraction of frames where `RMS(F_k)<1e-4` and `mean(|F_k|)<1e-4` | Share of frames likely affected by data dropouts or sensor failure. |
| `silence_ratio` | ratio | Fraction of frames where `mean(|F_k|)<1e-4` | Share of near-silent frames. |
| `uptime_ratio` | ratio | `1 - dropout_ratio` | Estimated fraction of time sensor/audio path was active. |
| `completeness_ratio` | ratio | `clip(uptime_ratio * finite_sample_ratio, 0, 1)` | Proxy for overall usable data completeness. |
| `diurnal_coverage_ratio` | ratio | Parsed-hour coverage over 24 h, else `clip(duration_s/86400,0,1)` | Proxy for how much of the day-night cycle is represented. |
| `site_comparability_score` | ratio | `clip(0.4*has_cal + 0.2*sr_score + 0.2*layout_score + 0.2*clip_score,0,1)` | Composite score indicating how comparable this file is across sites/campaigns. |

## Level + Loudness Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `crest_factor_db` | dB | `CF_k = dB( peak(F_k) / RMS(F_k) )` | Per-frame transientness or peakiness. |
| `spl_z_db` | dB | `L_k = dB(RMS_Z(F_k))`, calibrated via `dbfs_to_spl` when calibration exists | Z-weighted broadband level track; SPL-calibrated when calibration is provided. |
| `spl_a_db` | dBA | `L_k = dB(RMS_A(F_k))`, then optional calibration offset | A-weighted level track emphasizing human hearing sensitivity. |
| `spl_c_db` | dBC | `L_k = dB(RMS_C(F_k))`, then optional calibration offset | C-weighted level track preserving more low-frequency content. |
| `leq_db` | dB | `Leq = 10*log10( mean_k(10^(L_k/10)) )` from Z-weighted frame levels | Energy-equivalent constant sound level across the full interval. |
| `lmax_db` | dB | `max_k L_k` from Z-weighted frame levels | Maximum short-term level during the clip. |
| `lmin_db` | dB | `min_k L_k` from Z-weighted frame levels | Minimum short-term level during the clip. |
| `l10_db` | dB | `percentile(L_k, 90)` | Level exceeded 10% of the time (high-level indicator). |
| `l50_db` | dB | `percentile(L_k, 50)` | Median level. |
| `l90_db` | dB | `percentile(L_k, 10)` | Level exceeded 90% of the time (noise-floor proxy). |
| `l95_db` | dB | `percentile(L_k, 5)` | Very conservative noise-floor proxy. |
| `sel_db` | dB | `SEL = Leq + 10*log10(duration_s)` from Z-weighted `Leq` | Time-normalized event sound exposure level. |
| `lae_db` | dBA | `LAE = Leq_A + 10*log10(duration_s)` | A-weighted sound exposure level. |
| `integrated_lufs` | LUFS | ITU-style gated loudness: `L_m = -0.691 + 10*log10(E_m)` with absolute gate `>-70 LUFS` and relative gate `>= L_ungated-10` | Program loudness estimate for whole clip with loudness gating. |
| `short_term_lufs` | LUFS | Same loudness equation on 3 s blocks (0.1 s hop) | 3-second loudness trajectory. |
| `momentary_lufs` | LUFS | Same loudness equation on 0.4 s blocks (0.1 s hop) | Fast loudness trajectory for short events/transients. |
| `loudness_range_lu` | LU | `LRA = percentile(L_short,95) - percentile(L_short,10)` after gate `>-70` | Loudness variability range over time. |
| `true_peak_dbfs` | dBFS | `dB(max |resample_poly(x_c, oversample=4)| )` across channels | Inter-sample peak estimate beyond native sample grid. |
| `calibration_drift_db` | dB | `measured_tone_dbfs - dbfs_reference` using optional calibration tone file | Difference between expected and observed calibration reference level. |

## Noise + SNR Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `snr_db` | dB | `percentile(level_dbfs,90) - percentile(level_dbfs,10)` (replicated across frames) | Robust SNR proxy using level percentiles instead of explicit noise-only segments. |

## Spectral Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `spectral_centroid_hz` | Hz | `sum_f f*M(f,t) / sum_f M(f,t)` | Spectral center-of-mass over time; higher values imply brighter spectra. |
| `spectral_bandwidth_hz` | Hz | `sqrt( sum_f ((f-centroid_t)^2 * M(f,t)) / sum_f M(f,t) )` | Spectral spread around centroid. |
| `spectral_flatness` | ratio | `exp(mean_f(log(M(f,t)))) / mean_f(M(f,t))` | Tonal vs noise-like character; high flatness is more noise-like. |
| `spectral_rolloff_hz` | Hz | Smallest `f_r` where cumulative `P(f,t)` reaches `0.85 * total_power_t` | Frequency below which 85% of spectral energy lies. |
| `octave_band_level_db` | dB | For each octave band `b`: `Leq_b = 10*log10(mean_t(P_b(t)))` | Mean level per octave band (ISO-style center bands). |
| `third_octave_band_level_db` | dB | For each 1/3-octave band `b`: `Leq_b = 10*log10(mean_t(P_b(t)))` | Mean level per third-octave band for detailed spectral diagnostics. |

## Temporal Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `zero_crossing_rate` | ratio | `ZCR_k = (1/N_k) * sum_n 1[sign(x[n]) != sign(x[n-1])]` on mono-mixed frame | Rate of waveform polarity changes; tracks noisiness/percussiveness. |

## Ecoacoustics Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `bioacoustic_index` | a.u. | `sum_{f in [2,8]kHz} M(f,t)` per frame | High-frequency biophony activity proxy (birds/insects dominant ranges). |
| `acoustic_complexity_index` | ratio | `ACI_t = sum_f |M(f,t)-M(f,t-1)| / sum_f M(f,t)` | Spectro-temporal activity complexity proxy. |
| `ndsi` | ratio | `(E_bio - E_anthro) / (E_bio + E_anthro + eps)`, with anthro `[1,2]kHz`, bio `[2,11]kHz` (Nyquist-limited) | Balance between biophony and anthropophony. |
| `adi` | ratio | Normalized Shannon entropy over 1 kHz eco bins in `[1,10]kHz`: `-sum_i q_i log(q_i) / log(K)` | Diversity of acoustic energy across eco-frequency bins. |
| `aei` | ratio | Simpson-evenness style proxy: `(1/sum_i q_i^2)/K` | Evenness of acoustic energy distribution across eco bins. |
| `acoustic_entropy` | ratio | `H = Hf * Ht`, where `Hf` is normalized spectral entropy and `Ht` normalized envelope-histogram entropy | Composite entropy of both spectral and temporal structure. |
| `eco_octave_trends` | dB/s | For each octave band `b`, slope `a_b` from linear fit `L_b(t) = a_b*t + b0` | Rate of increase/decrease per octave band over time. |

## Spatial + Ambisonic Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `interchannel_coherence` | ratio | Mean adjacent-channel Pearson correlation per frame | Similarity/coherence among channels. |
| `iacc` | ratio | `max_{|tau|<=1ms} |corr_norm(L,R,tau)|` per frame | Interaural cross-correlation proxy for spaciousness/localization cues. |
| `ild_db` | dB | `20*log10(RMS(L)/RMS(R))` per frame | Interaural level difference proxy. |
| `ipd_rad` | rad | `angle(sum_f L(f) * conj(R(f)))` per frame | Interaural phase difference proxy. |
| `itd_s` | s | `tau_hat` from GCC-PHAT maximizing correlation within `+/-1 ms` | Interaural time delay proxy. |
| `doa_azimuth_proxy_deg` | deg | `az = asin( clip(c*mean_itd/d, -1,1) ) * 180/pi`, with `d=0.2 m` | Stereo baseline direction-of-arrival azimuth proxy. |
| `ambisonic_diffuseness` | ratio | `D = clip(1 - ||i||/E, 0,1)`, `i=[<wx>,<wy>,<wz>]`, `E=<w^2+x^2+y^2+z^2>` | FOA diffuseness proxy from energy/intensity relation. |
| `ambisonic_energy_vector_azimuth_deg` | deg | `atan2(<wy>, <wx>) * 180/pi` | FOA energy-vector azimuth. |
| `ambisonic_energy_vector_elevation_deg` | deg | `atan2(<wz>, sqrt(<wx>^2+<wy>^2)) * 180/pi` | FOA energy-vector elevation. |

## Architectural Acoustics + Intelligibility Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `rt60_s` | s | From Schroeder decay slope `a` on `[-5,-35] dB`: `RT60 = -60/a` | Reverberation time extrapolated from decay slope. |
| `edt_s` | s | From slope `a` on `[0,-10] dB`: `EDT = -60/a` | Early decay time emphasizing initial reverberant impression. |
| `t20_s` | s | From slope `a` on `[-5,-25] dB`: `T20 = -60/a` | T20 reverberation estimate. |
| `t30_s` | s | From slope `a` on `[-5,-35] dB`: `T30 = -60/a` | T30 reverberation estimate. |
| `c50_db` | dB | `10*log10(E_0-50ms / E_50ms-inf)` | Speech clarity metric (early-to-late energy ratio). |
| `c80_db` | dB | `10*log10(E_0-80ms / E_80ms-inf)` | Music clarity metric (early-to-late energy ratio). |
| `d50` | ratio | `E_0-50ms / E_total` | Speech definition ratio. |
| `ts_ms` | ms | `1000 * (sum_n t_n*e_n)/(sum_n e_n)` | Energy center time of impulse response. |
| `g_strength_db` | dB | `10*log10(E_total/eps)` (proxy reference) | Room strength proxy from total IR energy. |
| `lf_ratio` | ratio | `E_lateral(5-80ms) / E_total(5-80ms)` using non-reference channels as lateral proxy | Early lateral energy fraction proxy. |
| `lfc_ratio` | ratio | `LF * mean(|corr(ref, lateral_c)|)` in 5-80 ms | Correlation-weighted lateral fraction proxy. |
| `bass_ratio` | ratio | `(RT125 + RT250)/(RT500 + RT1000)` using bandpass RT estimates | Relative low-frequency reverberance proxy. |
| `sti_proxy` | ratio | `clip( sigmoid((C50-2)/2) * sigmoid((SNR-6)/4) * exp(-max(T30-0.6,0)/1.8), 0,1 )` | Speech intelligibility proxy from clarity, SNR, and decay behavior. |

## Anomaly + Novelty Metrics

| Metric ID | Units | Mathematical definition | Plain English |
|---|---:|---|---|
| `novelty_curve` | a.u. | `N(t) = sum_f max(M(f,t)-M(f,t-1), 0)` | Positive spectral-flux novelty over time. |
| `spectral_change_detection` | zscore | `Z(t) = (N(t)-mean(N))/std(N)` | Standardized novelty for threshold-based change detection. |
| `isolation_forest_score` | score | `-decision_function(X_t)` from Isolation Forest; fallback `||z_t||_2` | Frame-wise anomaly score from unsupervised isolation modeling. |
| `ocsvm_score` | score | `-decision_function(X_t)` from One-Class SVM; fallback `||z_t||_2` | Frame-wise anomaly score from one-class boundary modeling. |
| `autoencoder_recon_error` | mse | PCA/SVD low-rank proxy: `err_t = mean((X_t - Xhat_t)^2)` on normalized features | Reconstruction-error proxy compatible with autoencoder workflows. |
| `change_point_confidence` | ratio | Peak-prominence mapping on novelty z-score: `conf = clip(max_prominence/8,0,1)` | Confidence that salient acoustic change points exist. |

## Implementation Notes

- Many architectural/spatial metrics are explicitly marked as proxies when full geometry, microphone directivity, or standards-calibrated workflows are unavailable.
- Calibration-dependent metrics are still emitted without calibration, but values are interpreted as dBFS-derived proxies unless calibration metadata is provided.
- Metric IDs are stable API surface for JSON/CSV/Parquet/HDF5/ML exports.

## Visual Metric Topology

Cross-reference details:
- Core bibliography: [`docs/REFERENCES.md`](REFERENCES.md)
- Open-source attribution: [`docs/ATTRIBUTION.md`](ATTRIBUTION.md)

```mermaid
mindmap
  root((Metric Topology))
    Level + Loudness
      SPL A C Z
      Leq Percentiles SEL LAE
      LUFS LRA True Peak
    Spectral + Temporal
      STFT Features
      Octave Bands
      ZCR
    Ecoacoustics
      BAI ACI
      NDSI ADI AEI Entropy
    Spatial
      Coherence IACC
      ILD IPD ITD DOA
      FOA Diffuseness
    Architectural
      RT EDT T20 T30
      C50 C80 D50 Ts
      LF LFC Bass Ratio STI proxy
    Anomaly
      Novelty Change Z
      Isolation Forest
      OCSVM
      Reconstruction Error
```

```mermaid
flowchart LR
    A["Raw Signal"] --> B["Frame + STFT"]
    B --> C["Magnitude / Power"]
    C --> D["Spectral Features"]
    C --> E["Ecoacoustic Indices"]
    C --> F["Novelty"]
    C --> G["Similarity Matrix"]
    D --> H["Anomaly Feature Matrix"]
```

```mermaid
flowchart TD
    A["Calibration Profile"] --> B["dBFS Reference"]
    A --> C["SPL Reference"]
    B --> D["Offset"]
    C --> D
    D --> E["SPL / dBA / dBC Output"]
    A --> F["Calibration Tone"]
    F --> G["Drift Estimation"]
    G --> E
```

```mermaid
flowchart LR
    A["Impulse Response"] --> B["Schroeder Decay"]
    B --> C["Regression Windows"]
    C --> D["RT60 EDT T20 T30"]
    A --> E["Early-Late Split"]
    E --> F["C50 C80 D50"]
    A --> G["Energy Moment"]
    G --> H["Ts"]
```

```mermaid
flowchart LR
    A["Stereo/Multichannel Frames"] --> B["Cross-Correlation"]
    B --> C["IACC / ITD"]
    A --> D["RMS Ratio"]
    D --> E["ILD"]
    A --> F["Cross-Spectrum Angle"]
    F --> G["IPD"]
    C --> H["DOA Proxy"]
```

```mermaid
flowchart TD
    A["Feature Matrix"] --> B["Isolation Forest"]
    A --> C["One-Class SVM"]
    A --> D["Low-rank Reconstruction"]
    B --> E["IF Score"]
    C --> F["OCSVM Score"]
    D --> G["AE Proxy Error"]
    A --> H["Novelty Z-Score"]
    H --> I["Change Point Confidence"]
```

```mermaid
stateDiagram-v2
    [*] --> Measured
    Measured --> Proxy: "Missing calibration or geometry"
    Measured --> StandardAligned: "Calibration and assumptions available"
    Proxy --> Reported
    StandardAligned --> Reported
    Reported --> [*]
```

## Citation Coverage Matrix

- STFT, spectral descriptors, and novelty: see [D1], [N1], [N3], [N4] in [`docs/REFERENCES.md`](REFERENCES.md)
- Loudness / true-peak: see [S1], [S2]
- Room acoustics: see [S3], [S4], [A1]
- Spatial delay estimation: see [P1]
- Ecoacoustic index families: see [E1], [E2], [E3]
- Anomaly detection models: see [M1], [M2], [M3]

## Open-Source Attribution Pointers

- K-weighting implementation context and attribution notes: [`src/esl/metrics/extended.py`](../src/esl/metrics/extended.py)
- Novelty/similarity algorithm attribution notes: [`src/esl/viz/plotting.py`](../src/esl/viz/plotting.py)
- Full attribution log: [`docs/ATTRIBUTION.md`](ATTRIBUTION.md)

## Related Workflows

- Interesting moments extraction (clip export + timestamp CSV): [`docs/MOMENTS_EXTRACTION.md`](MOMENTS_EXTRACTION.md)
