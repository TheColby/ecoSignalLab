"""Extended metric plugins for standards, ecoacoustics, spatial, QC, anomaly, and campaign proxies.

Attribution and references:
- Loudness gating and K-weighted program loudness concepts:
  ITU-R BS.1770-4 and EBU Tech 3341 (LRA).
- K-weighting filter coefficient pattern is implementation-aligned with common
  open-source practice (e.g., pyloudnorm's BS.1770 realization), while this file's
  surrounding orchestration and metric contract plumbing is original to esl.
- RT/clarity/definition conventions:
  ISO 3382-1:2009 and ISO 3382-2:2008.
- NDSI / ACI / ADI / AEI ecoacoustic families:
  Kasten et al. (2012), Pieretti et al. (2011), Villanueva-Rivera et al. ecosystem docs.
- GCC-PHAT time-delay estimation:
  Knapp & Carter (1976), IEEE TASSP 24(4):320-327.
- Isolation Forest:
  Liu, Ting, Zhou (2008), ICDM.
- One-Class SVM:
  Schölkopf et al. (2001), Neural Computation 13(7):1443-1471.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, lfilter, resample_poly

from esl.core.audio import read_audio
from esl.core.calibration import db, dbfs_to_spl, weighted_rms
from esl.core.context import AnalysisContext
from esl.metrics.base import MetricResult, MetricSpec
from esl.metrics.helpers import (
    fit_decay_time,
    frame_rms,
    frame_signal,
    novelty_from_spectrum,
    schroeder_decay,
    spectral_features,
    summarize,
)


EPS = 1e-12
SPEED_OF_SOUND = 343.0


def _safe_summary(series: np.ndarray) -> dict[str, float]:
    if series.size == 0 or np.all(~np.isfinite(series)):
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "p50": nan, "p95": nan}
    valid = series[np.isfinite(series)]
    if valid.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "p50": nan, "p95": nan}
    return summarize(valid)


def _scalar_result(name: str, units: str, value: float, confidence: float = 1.0, extra: dict[str, Any] | None = None) -> MetricResult:
    return MetricResult(
        name=name,
        units=units,
        summary=_safe_summary(np.array([value], dtype=np.float64)),
        series=[float(value)] if np.isfinite(value) else [],
        timestamps_s=[0.0] if np.isfinite(value) else [],
        confidence=float(confidence),
        extra=extra or {},
    )


def _spectral_cache(ctx: AnalysisContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = "spectral_ext"
    if key not in ctx.cache:
        ctx.cache[key] = spectral_features(ctx.mono(), ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    return ctx.cache[key]


def _frame_weighted_level_db(ctx: AnalysisContext, weighting: str = "Z") -> tuple[np.ndarray, np.ndarray]:
    framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    vals = np.zeros(framed.frames.shape[0], dtype=np.float64)
    for i in range(framed.frames.shape[0]):
        wrms = np.mean(weighted_rms(framed.frames[i], ctx.sample_rate, weighting=weighting))
        vals[i] = float(db(wrms))
    vals = dbfs_to_spl(vals, ctx.calibration)
    return vals, framed.times_s


def _frame_dbfs(ctx: AnalysisContext) -> tuple[np.ndarray, np.ndarray]:
    framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    rms = frame_rms(framed.frames)
    return db(rms), framed.times_s


def _leq(levels_db: np.ndarray) -> float:
    if levels_db.size == 0:
        return float("nan")
    return float(10.0 * np.log10(np.mean(np.power(10.0, levels_db / 10.0)) + EPS))


def _percentile_exceedance(levels_db: np.ndarray, exceedance_percent: float) -> float:
    # L10 = level exceeded 10% of time = 90th percentile.
    p = 100.0 - exceedance_percent
    return float(np.percentile(levels_db, p)) if levels_db.size else float("nan")


def _ir_cache(ctx: AnalysisContext) -> dict[str, Any]:
    key = "ir_ext"
    if key in ctx.cache:
        return ctx.cache[key]

    mono = ctx.mono().astype(np.float64)
    peak_idx = int(np.argmax(np.abs(mono)))
    ir = mono[peak_idx:]
    if ir.size < max(16, int(0.1 * ctx.sample_rate)):
        ir = mono
    t = np.arange(len(ir), dtype=np.float64) / float(ctx.sample_rate)
    decay = schroeder_decay(ir)
    e = np.square(ir)
    total_e = float(np.sum(e)) + EPS

    cache = {
        "ir": ir,
        "time": t,
        "decay": decay,
        "energy": e,
        "total_energy": total_e,
        "t20": fit_decay_time(decay, t, lo_db=-5.0, hi_db=-25.0),
        "t30": fit_decay_time(decay, t, lo_db=-5.0, hi_db=-35.0),
    }
    ctx.cache[key] = cache
    return cache


def _bandpass_rt(ir: np.ndarray, sr: int, center_hz: float) -> float:
    lo = center_hz / np.sqrt(2.0)
    hi = center_hz * np.sqrt(2.0)
    nyq = sr / 2.0
    lo = max(20.0, lo)
    hi = min(nyq * 0.98, hi)
    if hi <= lo:
        return float("nan")
    try:
        # Butterworth + zero-phase filtering is a practical RT-band proxy workflow.
        # Reference for Butterworth filters: Butterworth (1930), Wireless Engineer.
        b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
        y = filtfilt(b, a, ir)
    except Exception:
        return float("nan")
    decay = schroeder_decay(y)
    t = np.arange(len(decay), dtype=np.float64) / float(sr)
    return fit_decay_time(decay, t, lo_db=-5.0, hi_db=-35.0)


def _k_weight_signal(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    target_sr = 48000
    if sr != target_sr:
        up = target_sr
        down = sr
        y = np.zeros((int(np.ceil(x.shape[0] * target_sr / sr)), x.shape[1]), dtype=np.float64)
        for c in range(x.shape[1]):
            y[:, c] = resample_poly(x[:, c], up, down)
    else:
        y = x.astype(np.float64)

    # BS.1770-style K-weighting biquads (commonly used digital coefficients).
    # Attribution: these coefficient values follow the widely used realization seen in
    # open implementations such as pyloudnorm, derived from ITU-R BS.1770 design intent.
    b1 = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64)
    a1 = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
    b2 = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    a2 = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64)

    for c in range(y.shape[1]):
        z = lfilter(b1, a1, y[:, c])
        z = lfilter(b2, a2, z)
        y[:, c] = z

    return y, target_sr


def _block_loudness(y: np.ndarray, sr: int, win_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = y.shape[0]
    win = max(1, int(round(win_s * sr)))
    hop = max(1, int(round(hop_s * sr)))

    if n < win:
        pad = np.zeros((win - n, y.shape[1]), dtype=np.float64)
        y = np.vstack([y, pad])
        n = y.shape[0]

    times = []
    energies = []
    loudness = []
    for start in range(0, n - win + 1, hop):
        block = y[start : start + win]
        # Channel weights default to 1 for general N-channel workflows.
        e = float(np.sum(np.mean(np.square(block), axis=0)))
        # ITU-R BS.1770 loudness block equation constant.
        l = float(-0.691 + 10.0 * np.log10(max(e, EPS)))
        energies.append(e)
        loudness.append(l)
        times.append((start + 0.5 * win) / sr)

    return np.array(times, dtype=np.float64), np.array(energies, dtype=np.float64), np.array(loudness, dtype=np.float64)


def _loudness_cache(ctx: AnalysisContext) -> dict[str, Any]:
    key = "loudness_ext"
    if key in ctx.cache:
        return ctx.cache[key]

    y, sr = _k_weight_signal(ctx.signal, ctx.sample_rate)
    t_m, e_m, l_m = _block_loudness(y, sr, win_s=0.4, hop_s=0.1)  # momentary
    t_s, e_s, l_s = _block_loudness(y, sr, win_s=3.0, hop_s=0.1)  # short-term

    # Absolute and relative loudness gating aligned with BS.1770-style workflows.
    mask_abs = l_m > -70.0
    if np.any(mask_abs):
        l_ungated = float(-0.691 + 10.0 * np.log10(max(float(np.mean(e_m[mask_abs])), EPS)))
        gate_rel = l_ungated - 10.0
        mask = mask_abs & (l_m >= gate_rel)
        if np.any(mask):
            integrated = float(-0.691 + 10.0 * np.log10(max(float(np.mean(e_m[mask])), EPS)))
        else:
            integrated = l_ungated
    else:
        integrated = -120.0

    mask_lra = l_s > -70.0
    if np.any(mask_lra):
        lra = float(np.percentile(l_s[mask_lra], 95.0) - np.percentile(l_s[mask_lra], 10.0))
    else:
        lra = 0.0

    cache = {
        "momentary_time": t_m,
        "momentary_lufs": l_m,
        "short_time": t_s,
        "short_lufs": l_s,
        "integrated_lufs": integrated,
        "lra_lu": lra,
    }
    ctx.cache[key] = cache
    return cache


def _true_peak_dbfs(ctx: AnalysisContext, oversample: int = 4) -> float:
    # True peak via oversampling is the practical approximation used in loudness standards.
    peaks = []
    for c in range(ctx.channels):
        osig = resample_poly(ctx.signal[:, c], oversample, 1)
        peaks.append(np.max(np.abs(osig)))
    return float(db(max(peaks) if peaks else 0.0))


def _fractional_octave_centers(sr: int, fraction: int) -> np.ndarray:
    base = 31.5 if fraction == 1 else 25.0
    ratio = 2.0 ** (1.0 / fraction)
    nyq = sr / 2.0 * 0.98
    centers = []
    f = base
    while f <= nyq:
        centers.append(f)
        f *= ratio
    return np.array(centers, dtype=np.float64)


def _fractional_octave_levels(ctx: AnalysisContext, fraction: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f, t, mag = _spectral_cache(ctx)
    p = np.square(mag)
    centers = _fractional_octave_centers(ctx.sample_rate, fraction=fraction)
    if centers.size == 0:
        return centers, t, np.zeros((0, len(t))), np.zeros((0,))

    factor = 2.0 ** (1.0 / (2.0 * fraction))
    band_series = np.zeros((len(centers), p.shape[1]), dtype=np.float64)

    for i, fc in enumerate(centers):
        lo = fc / factor
        hi = fc * factor
        mask = (f >= lo) & (f < hi)
        if np.any(mask):
            band_pow = np.mean(p[mask], axis=0)
            band_series[i] = 10.0 * np.log10(np.maximum(band_pow, EPS))
        else:
            band_series[i] = -120.0

    band_mean = np.zeros(len(centers), dtype=np.float64)
    for i in range(len(centers)):
        band_mean[i] = _leq(band_series[i])

    return centers, t, band_series, band_mean


def _ndsi(ctx: AnalysisContext) -> float:
    # NDSI popularized in soundscape ecology literature (e.g., Kasten et al., 2012).
    f, _, mag = _spectral_cache(ctx)
    p = np.square(mag)
    anthro = np.sum(p[(f >= 1000.0) & (f < 2000.0)])
    bio = np.sum(p[(f >= 2000.0) & (f < min(11000.0, ctx.sample_rate / 2.0))])
    return float((bio - anthro) / (bio + anthro + EPS))


def _adi_aei(ctx: AnalysisContext) -> tuple[float, float]:
    f, _, mag = _spectral_cache(ctx)
    p = np.mean(np.square(mag), axis=1)
    edges = np.arange(1000.0, min(10000.0, ctx.sample_rate / 2.0) + 1000.0, 1000.0)
    if len(edges) < 2:
        return 0.0, 0.0

    vals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        vals.append(float(np.sum(p[m])) if np.any(m) else 0.0)
    arr = np.array(vals, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    arr_sum = float(np.sum(arr))
    if arr_sum <= 0:
        return 0.0, 0.0
    q = arr / arr_sum

    # ADI proxy uses Shannon entropy over eco-frequency occupancy bins.
    adi = float(-np.sum(q * np.log(q + EPS)) / np.log(len(q)))
    # AEI proxy uses Simpson-style evenness.
    aei = float((1.0 / np.sum(np.square(q) + EPS)) / len(q))
    return adi, aei


def _acoustic_entropy(ctx: AnalysisContext) -> float:
    # Composite entropy: normalized spectral entropy times normalized temporal entropy.
    f, _, mag = _spectral_cache(ctx)
    pf = np.mean(np.square(mag), axis=1)
    pf = pf / (np.sum(pf) + EPS)
    hf = float(-np.sum(pf * np.log(pf + EPS)) / np.log(len(pf) + EPS))

    env = np.abs(ctx.mono())
    hist, _ = np.histogram(env, bins=64, range=(0.0, max(float(np.max(env)), EPS)))
    pt = hist.astype(np.float64)
    pt = pt / (np.sum(pt) + EPS)
    ht = float(-np.sum(pt * np.log(pt + EPS)) / np.log(len(pt) + EPS))

    return float(hf * ht)


def _stereo_frames(ctx: AnalysisContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    if ctx.channels < 2:
        x = framed.frames[:, :, 0]
        return x, x, framed.times_s
    return framed.frames[:, :, 0], framed.frames[:, :, 1], framed.times_s


def _max_norm_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0) + EPS)
    c = np.correlate(a0, b0, mode="full") / denom
    mid = len(c) // 2
    lo = max(0, mid - max_lag)
    hi = min(len(c), mid + max_lag + 1)
    return float(np.max(np.abs(c[lo:hi]))) if hi > lo else 0.0


def _gcc_phat_itd(a: np.ndarray, b: np.ndarray, sr: int, max_tau_s: float = 0.001) -> float:
    # GCC-PHAT time-delay estimation after Knapp & Carter (1976).
    n = 1
    target = len(a) + len(b)
    while n < target:
        n <<= 1
    A = np.fft.rfft(a, n=n)
    B = np.fft.rfft(b, n=n)
    R = A * np.conj(B)
    R /= np.maximum(np.abs(R), EPS)
    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-(n // 2) :], cc[: n // 2 + 1]))
    max_shift = int(max_tau_s * sr)
    mid = len(cc) // 2
    lo = max(0, mid - max_shift)
    hi = min(len(cc), mid + max_shift + 1)
    if hi <= lo:
        return 0.0
    idx = int(np.argmax(cc[lo:hi])) + lo
    lag = idx - mid
    return float(lag / sr)


def _doa_azimuth_from_itd(itd_s: float, mic_spacing_m: float = 0.2) -> float:
    arg = float(np.clip(itd_s * SPEED_OF_SOUND / max(mic_spacing_m, EPS), -1.0, 1.0))
    return float(np.degrees(np.arcsin(arg)))


def _qc_cache(ctx: AnalysisContext) -> dict[str, Any]:
    key = "qc_ext"
    if key in ctx.cache:
        return ctx.cache[key]

    x = ctx.signal
    clipped = np.abs(x) >= 0.999
    clipping_ratio = float(np.mean(clipped))

    # Count clip events by rising edges over any channel.
    clip_any = np.any(clipped, axis=1).astype(np.int8)
    clip_events = int(np.sum((clip_any[1:] == 1) & (clip_any[:-1] == 0)))

    dc_offset = float(np.mean(x))

    framed = frame_signal(x, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    rms = frame_rms(framed.frames)
    abs_mean = np.mean(np.abs(framed.frames), axis=(1, 2))

    dropout_mask = (rms < 1e-4) & (abs_mean < 1e-4)
    silence_mask = abs_mean < 1e-4

    dropout_ratio = float(np.mean(dropout_mask)) if dropout_mask.size else 0.0
    silence_ratio = float(np.mean(silence_mask)) if silence_mask.size else 0.0

    calibration_drift = float("nan")
    drift_conf = 0.2
    if ctx.calibration and ctx.calibration.calibration_tone_file:
        tone_path = Path(ctx.calibration.calibration_tone_file)
        if not tone_path.is_absolute():
            tone_path = Path(ctx.config.input_path).parent / tone_path
        if tone_path.exists():
            try:
                tone = read_audio(tone_path)
                tone_rms = np.sqrt(np.mean(np.square(tone.samples)))
                measured_dbfs = float(db(tone_rms))
                calibration_drift = measured_dbfs - float(ctx.calibration.dbfs_reference)
                drift_conf = 1.0
            except Exception:
                pass

    cache = {
        "clipping_ratio": clipping_ratio,
        "clipping_event_count": clip_events,
        "dc_offset": dc_offset,
        "dropout_ratio": dropout_ratio,
        "silence_ratio": silence_ratio,
        "calibration_drift_db": calibration_drift,
        "calibration_drift_confidence": drift_conf,
    }
    ctx.cache[key] = cache
    return cache


def _feature_matrix(ctx: AnalysisContext) -> tuple[np.ndarray, np.ndarray]:
    f, t, mag = _spectral_cache(ctx)
    p = np.square(mag)
    den = np.sum(mag, axis=0) + EPS
    centroid = np.sum(f[:, None] * mag, axis=0) / den
    spread = np.sqrt(np.sum(((f[:, None] - centroid[None, :]) ** 2) * mag, axis=0) / den)
    gm = np.exp(np.mean(np.log(np.maximum(mag, EPS)), axis=0))
    am = np.mean(mag, axis=0) + EPS
    flatness = gm / am
    csum = np.cumsum(p, axis=0)
    target = 0.85 * csum[-1]
    idx = np.argmax(csum >= target[None, :], axis=0)
    rolloff = f[idx]
    novelty = novelty_from_spectrum(mag)

    X = np.stack([centroid, spread, flatness, rolloff, novelty], axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return t, X


def _anomaly_fallback_score(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.zeros((0,), dtype=np.float64)
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True) + EPS
    z = (X - mu) / sd
    return np.sqrt(np.sum(np.square(z), axis=1))


def _parse_datetime_from_name(path: str | Path) -> datetime | None:
    stem = Path(path).stem

    m = re.search(r"(\d{8})[_-]?(\d{6})", stem)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except Exception:
            pass

    m2 = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})[T_ -]?(\d{2})[:_-]?(\d{2})[:_-]?(\d{2})", stem)
    if m2:
        try:
            y, mo, d, hh, mm, ss = [int(m2.group(i)) for i in range(1, 7)]
            return datetime(y, mo, d, hh, mm, ss)
        except Exception:
            pass

    return None


def _campaign_proxy_cache(ctx: AnalysisContext) -> dict[str, Any]:
    key = "campaign_ext"
    if key in ctx.cache:
        return ctx.cache[key]

    qc = _qc_cache(ctx)
    uptime_ratio = float(np.clip(1.0 - qc["dropout_ratio"], 0.0, 1.0))

    finite_ratio = float(np.mean(np.isfinite(ctx.signal))) if ctx.signal.size else 0.0
    completeness_ratio = float(np.clip(uptime_ratio * finite_ratio, 0.0, 1.0))

    start_dt = _parse_datetime_from_name(ctx.audio.source_path)
    if start_dt is not None:
        end_dt = start_dt + timedelta(seconds=ctx.duration_s)
        covered_hours = set()
        cur = start_dt
        while cur <= end_dt:
            covered_hours.add(cur.hour)
            cur += timedelta(minutes=30)
        diurnal_ratio = float(len(covered_hours) / 24.0)
        diurnal_conf = 1.0
    else:
        diurnal_ratio = float(np.clip(ctx.duration_s / 86400.0, 0.0, 1.0))
        diurnal_conf = 0.3

    has_cal = 1.0 if ctx.calibration is not None else 0.0
    sr_score = 1.0 if ctx.sample_rate in {16000, 22050, 32000, 44100, 48000} else 0.7
    layout_score = 1.0 if ctx.layout in {"mono", "stereo", "ambisonic_b_format"} else 0.8
    clip_score = float(np.clip(1.0 - qc["clipping_ratio"], 0.0, 1.0))
    site_comparability = float(np.clip(0.4 * has_cal + 0.2 * sr_score + 0.2 * layout_score + 0.2 * clip_score, 0.0, 1.0))

    cache = {
        "uptime_ratio": uptime_ratio,
        "completeness_ratio": completeness_ratio,
        "diurnal_coverage_ratio": diurnal_ratio,
        "diurnal_confidence": diurnal_conf,
        "site_comparability_score": site_comparability,
    }
    ctx.cache[key] = cache
    return cache


# -----------------------------------------------------------------------------
# Standards SPL and loudness
# -----------------------------------------------------------------------------
# References:
# - ITU-R BS.1770-4 (integrated/short-term/momentary loudness and true-peak context)
# - EBU Tech 3341 (LRA definition context)
# - IEC 61672-1 (A/C/Z weighting intent in level reporting)


class LeqMetric:
    spec = MetricSpec(
        name="leq_db",
        category="Level & Loudness",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Equivalent continuous sound level over the analyzed duration.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        levels, _ = _frame_weighted_level_db(ctx, "Z")
        value = _leq(levels)
        conf = 1.0 if ctx.calibration else 0.7
        return _scalar_result(self.spec.name, self.spec.units, value, conf)


class LmaxMetric:
    spec = MetricSpec(
        name="lmax_db",
        category="Level & Loudness",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Maximum frame-level Z-weighted sound level.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        levels, _ = _frame_weighted_level_db(ctx, "Z")
        v = float(np.max(levels)) if levels.size else float("nan")
        return _scalar_result(self.spec.name, self.spec.units, v, 1.0 if ctx.calibration else 0.7)


class LminMetric:
    spec = MetricSpec(
        name="lmin_db",
        category="Level & Loudness",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Minimum frame-level Z-weighted sound level.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        levels, _ = _frame_weighted_level_db(ctx, "Z")
        v = float(np.min(levels)) if levels.size else float("nan")
        return _scalar_result(self.spec.name, self.spec.units, v, 1.0 if ctx.calibration else 0.7)


class LpeakMetric:
    spec = MetricSpec(
        name="lpeak_dbfs",
        category="Level & Loudness",
        units="dBFS",
        window_size=0,
        hop_size=0,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Based on full-signal absolute peak",
        description="Absolute sample peak level in dBFS.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        peak = float(np.max(np.abs(ctx.signal)))
        return _scalar_result(self.spec.name, self.spec.units, float(db(peak)), 1.0)


class PercentileLevelMetric:
    def __init__(self, exceedance_percent: float) -> None:
        ptxt = str(int(exceedance_percent))
        self.exceedance = float(exceedance_percent)
        self.spec = MetricSpec(
            name=f"l{ptxt}_db",
            category="Level & Loudness",
            units="dB",
            window_size=2048,
            hop_size=512,
            streaming_capable=True,
            calibration_dependency=True,
            ml_compatible=True,
            confidence_logic="Percentile from frame level distribution",
            description=f"Level exceeded {ptxt}% of the time.",
        )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        levels, _ = _frame_weighted_level_db(ctx, "Z")
        v = _percentile_exceedance(levels, self.exceedance)
        return _scalar_result(self.spec.name, self.spec.units, v, 1.0 if ctx.calibration else 0.7)


class SELMetric:
    spec = MetricSpec(
        name="sel_db",
        category="Level & Loudness",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Sound exposure level from Leq and duration.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        levels, _ = _frame_weighted_level_db(ctx, "Z")
        leq = _leq(levels)
        v = float(leq + 10.0 * np.log10(max(ctx.duration_s, EPS)))
        return _scalar_result(self.spec.name, self.spec.units, v, 1.0 if ctx.calibration else 0.7)


class LAEMetric:
    spec = MetricSpec(
        name="lae_db",
        category="Level & Loudness",
        units="dBA",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="A-weighted sound exposure level.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        levels, _ = _frame_weighted_level_db(ctx, "A")
        leq = _leq(levels)
        v = float(leq + 10.0 * np.log10(max(ctx.duration_s, EPS)))
        return _scalar_result(self.spec.name, self.spec.units, v, 1.0 if ctx.calibration else 0.7)


class IntegratedLUFSMetric:
    spec = MetricSpec(
        name="integrated_lufs",
        category="Level & Loudness",
        units="LUFS",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="ITU-style gating over momentary loudness blocks",
        description="Integrated loudness with absolute and relative gating.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l = _loudness_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(l["integrated_lufs"]), 0.85)


class ShortTermLUFSMetric:
    spec = MetricSpec(
        name="short_term_lufs",
        category="Level & Loudness",
        units="LUFS",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="3 s sliding loudness blocks",
        description="Short-term loudness trajectory.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l = _loudness_cache(ctx)
        series = np.array(l["short_lufs"], dtype=np.float64)
        t = np.array(l["short_time"], dtype=np.float64)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(series),
            series=series.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.85,
        )


class MomentaryLUFSMetric:
    spec = MetricSpec(
        name="momentary_lufs",
        category="Level & Loudness",
        units="LUFS",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="400 ms sliding loudness blocks",
        description="Momentary loudness trajectory.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l = _loudness_cache(ctx)
        series = np.array(l["momentary_lufs"], dtype=np.float64)
        t = np.array(l["momentary_time"], dtype=np.float64)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(series),
            series=series.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.85,
        )


class LoudnessRangeMetric:
    spec = MetricSpec(
        name="loudness_range_lu",
        category="Level & Loudness",
        units="LU",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="95th-10th percentile of short-term loudness above gate",
        description="Loudness range (LRA) descriptor.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l = _loudness_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(l["lra_lu"]), 0.85)


class TruePeakMetric:
    spec = MetricSpec(
        name="true_peak_dbfs",
        category="Level & Loudness",
        units="dBFS",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="4x oversampled peak estimate",
        description="Estimated true peak level in dBFS.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        return _scalar_result(self.spec.name, self.spec.units, _true_peak_dbfs(ctx), 0.9)


# -----------------------------------------------------------------------------
# Intelligibility and architectural extensions
# -----------------------------------------------------------------------------
# References:
# - ISO 3382-1 / ISO 3382-2 (room-acoustic parameter conventions)
# - Schroeder (1965) for decay integration
# - IEC 60268-16 context for STI family (proxy here, not full standard STI)


class STIProxyMetric:
    spec = MetricSpec(
        name="sti_proxy",
        category="Architectural Acoustics",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Proxy from C50, SNR, and RT behavior",
        description="Speech intelligibility proxy in [0,1].",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        ir = _ir_cache(ctx)
        e = ir["energy"]
        sr = ctx.sample_rate
        n50 = min(len(e), max(1, int(round(0.05 * sr))))
        e50 = float(np.sum(e[:n50]))
        late = max(float(np.sum(e)) - e50, EPS)
        c50 = 10.0 * np.log10(e50 / late)

        levels, _ = _frame_dbfs(ctx)
        snr = float(np.percentile(levels, 90.0) - np.percentile(levels, 10.0)) if levels.size else 0.0
        rt = float(ir["t30"]) if np.isfinite(ir["t30"]) else 1.0

        s1 = 1.0 / (1.0 + np.exp(-(c50 - 2.0) / 2.0))
        s2 = 1.0 / (1.0 + np.exp(-(snr - 6.0) / 4.0))
        s3 = np.exp(-max(rt - 0.6, 0.0) / 1.8)
        sti = float(np.clip(s1 * s2 * s3, 0.0, 1.0))
        return _scalar_result(self.spec.name, self.spec.units, sti, 0.6, extra={"c50_db": c50, "snr_db": snr, "rt30_s": rt})


class T20Metric:
    spec = MetricSpec(
        name="t20_s",
        category="Architectural Acoustics",
        units="s",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Decay regression from -5 to -25 dB",
        description="T20 reverberation time estimate.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        ir = _ir_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(ir["t20"]), 0.8)


class T30Metric:
    spec = MetricSpec(
        name="t30_s",
        category="Architectural Acoustics",
        units="s",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Decay regression from -5 to -35 dB",
        description="T30 reverberation time estimate.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        ir = _ir_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(ir["t30"]), 0.85)


class TsMetric:
    spec = MetricSpec(
        name="ts_ms",
        category="Architectural Acoustics",
        units="ms",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Energy-weighted first moment",
        description="Center time (Ts) from impulse response energy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        ir = _ir_cache(ctx)
        t = ir["time"]
        e = ir["energy"]
        ts = float(np.sum(t * e) / (np.sum(e) + EPS) * 1000.0)
        return _scalar_result(self.spec.name, self.spec.units, ts, 0.9)


class GStrengthMetric:
    spec = MetricSpec(
        name="g_strength_db",
        category="Architectural Acoustics",
        units="dB",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Proxy from total IR energy relative to fixed reference",
        description="Strength factor proxy (relative room gain).",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        ir = _ir_cache(ctx)
        g = float(10.0 * np.log10(ir["total_energy"] / EPS))
        return _scalar_result(self.spec.name, self.spec.units, g, 0.5, extra={"note": "proxy"})


class LFMetric:
    spec = MetricSpec(
        name="lf_ratio",
        category="Architectural Acoustics",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Requires >=2 channels for lateral energy proxy",
        description="Lateral fraction proxy in the 5-80 ms early window.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        if ctx.channels < 2:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)

        n0 = max(0, int(round(0.005 * ctx.sample_rate)))
        n1 = min(ctx.signal.shape[0], int(round(0.080 * ctx.sample_rate)))
        early = ctx.signal[n0:n1]
        if early.size == 0:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)

        e_total = float(np.sum(np.square(early))) + EPS
        e_lat = float(np.sum(np.square(early[:, 1:])))
        return _scalar_result(self.spec.name, self.spec.units, float(np.clip(e_lat / e_total, 0.0, 1.0)), 0.6)


class LFCMetric:
    spec = MetricSpec(
        name="lfc_ratio",
        category="Architectural Acoustics",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Correlation-weighted lateral fraction proxy",
        description="Lateral fraction coefficient proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        if ctx.channels < 2:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)

        n0 = max(0, int(round(0.005 * ctx.sample_rate)))
        n1 = min(ctx.signal.shape[0], int(round(0.080 * ctx.sample_rate)))
        early = ctx.signal[n0:n1]
        if early.size == 0:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)

        ref = early[:, 0]
        corr_vals = []
        for c in range(1, ctx.channels):
            tgt = early[:, c]
            if np.std(ref) < EPS or np.std(tgt) < EPS:
                corr_vals.append(0.0)
            else:
                corr_vals.append(float(np.abs(np.corrcoef(ref, tgt)[0, 1])))
        corr = float(np.mean(corr_vals)) if corr_vals else 0.0

        e_total = float(np.sum(np.square(early))) + EPS
        e_lat = float(np.sum(np.square(early[:, 1:])))
        lf = float(np.clip(e_lat / e_total, 0.0, 1.0))
        return _scalar_result(self.spec.name, self.spec.units, float(np.clip(lf * corr, 0.0, 1.0)), 0.6)


class BassRatioMetric:
    spec = MetricSpec(
        name="bass_ratio",
        category="Architectural Acoustics",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Band-limited RT30 ratio",
        description="(RT125 + RT250) / (RT500 + RT1000) reverberation ratio.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        ir = _ir_cache(ctx)["ir"]
        rt125 = _bandpass_rt(ir, ctx.sample_rate, 125.0)
        rt250 = _bandpass_rt(ir, ctx.sample_rate, 250.0)
        rt500 = _bandpass_rt(ir, ctx.sample_rate, 500.0)
        rt1000 = _bandpass_rt(ir, ctx.sample_rate, 1000.0)

        num = np.nansum([rt125, rt250])
        den = np.nansum([rt500, rt1000])
        value = float(num / den) if den > EPS else float("nan")
        return _scalar_result(
            self.spec.name,
            self.spec.units,
            value,
            0.7,
            extra={"rt125": rt125, "rt250": rt250, "rt500": rt500, "rt1000": rt1000},
        )


# -----------------------------------------------------------------------------
# Ecoacoustics extension set
# -----------------------------------------------------------------------------
# References:
# - Pieretti et al. (2011) ACI
# - Kasten et al. (2012) NDSI family context
# - Sueur et al. (2008) acoustic diversity/entropy context


class NDSIMetric:
    spec = MetricSpec(
        name="ndsi",
        category="Ecoacoustics",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Band energy contrast between anthropophony and biophony",
        description="Normalized Difference Soundscape Index.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        return _scalar_result(self.spec.name, self.spec.units, _ndsi(ctx), 0.9)


class ADIMetric:
    spec = MetricSpec(
        name="adi",
        category="Ecoacoustics",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Shannon diversity over eco-frequency bins",
        description="Acoustic Diversity Index proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        adi, _ = _adi_aei(ctx)
        return _scalar_result(self.spec.name, self.spec.units, adi, 0.85)


class AEIMetric:
    spec = MetricSpec(
        name="aei",
        category="Ecoacoustics",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Evenness over eco-frequency bins",
        description="Acoustic Evenness Index proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        _, aei = _adi_aei(ctx)
        return _scalar_result(self.spec.name, self.spec.units, aei, 0.85)


class AcousticEntropyMetric:
    spec = MetricSpec(
        name="acoustic_entropy",
        category="Ecoacoustics",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Product of spectral and temporal normalized entropies",
        description="Composite acoustic entropy index.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        h = _acoustic_entropy(ctx)
        return _scalar_result(self.spec.name, self.spec.units, h, 0.85)


class EcoOctaveTrendMetric:
    spec = MetricSpec(
        name="eco_octave_trends",
        category="Ecoacoustics",
        units="dB/s",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Linear trend fit per octave band",
        description="Slope trends across octave-band eco trajectories.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        centers, t, band_series, band_mean = _fractional_octave_levels(ctx, fraction=1)
        if band_series.size == 0 or t.size < 2:
            return MetricResult(
                name=self.spec.name,
                units=self.spec.units,
                summary=_safe_summary(np.array([], dtype=np.float64)),
                confidence=0.2,
            )

        slopes = []
        for i in range(band_series.shape[0]):
            y = band_series[i]
            if np.all(np.isfinite(y)) and np.std(y) > EPS:
                a, _ = np.polyfit(t, y, deg=1)
                slopes.append(float(a))
            else:
                slopes.append(0.0)

        s = np.array(slopes, dtype=np.float64)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(s),
            series=s.tolist(),
            timestamps_s=list(range(len(s))),
            confidence=0.8,
            extra={
                "band_centers_hz": centers.tolist(),
                "band_mean_db": band_mean.tolist(),
                "band_slopes_db_per_s": s.tolist(),
            },
        )


# -----------------------------------------------------------------------------
# Spatial and ambisonic extensions
# -----------------------------------------------------------------------------
# References:
# - Knapp & Carter (1976) GCC-PHAT
# - Blauert, \"Spatial Hearing\" (interaural cue interpretation)
# - Ambisonic energy/intensity vector conventions (first-order proxy formulation)


class IACCMetric:
    spec = MetricSpec(
        name="iacc",
        category="Spatial",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Max normalized cross-correlation over +/-1 ms lag",
        description="Interaural cross-correlation proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l, r, ts = _stereo_frames(ctx)
        max_lag = max(1, int(0.001 * ctx.sample_rate))
        vals = np.array([_max_norm_xcorr(l[i], r[i], max_lag=max_lag) for i in range(l.shape[0])], dtype=np.float64)
        conf = 1.0 if ctx.channels >= 2 else 0.3
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(vals),
            series=vals.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
        )


class ILDMetric:
    spec = MetricSpec(
        name="ild_db",
        category="Spatial",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Frame RMS ratio of first two channels",
        description="Interaural level difference proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l, r, ts = _stereo_frames(ctx)
        lrms = np.sqrt(np.mean(np.square(l), axis=1))
        rrms = np.sqrt(np.mean(np.square(r), axis=1))
        ild = 20.0 * np.log10(np.maximum(lrms, EPS) / np.maximum(rrms, EPS))
        conf = 1.0 if ctx.channels >= 2 else 0.3
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(ild),
            series=ild.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
        )


class IPDMetric:
    spec = MetricSpec(
        name="ipd_rad",
        category="Spatial",
        units="rad",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Cross-spectrum phase angle of first two channels",
        description="Interaural phase difference proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l, r, ts = _stereo_frames(ctx)
        ipd = np.zeros(l.shape[0], dtype=np.float64)
        for i in range(l.shape[0]):
            L = np.fft.rfft(l[i])
            R = np.fft.rfft(r[i])
            cs = np.sum(L * np.conj(R))
            ipd[i] = float(np.angle(cs))
        conf = 1.0 if ctx.channels >= 2 else 0.3
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(ipd),
            series=ipd.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
        )


class ITDMetric:
    spec = MetricSpec(
        name="itd_s",
        category="Spatial",
        units="s",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="GCC-PHAT lag estimate over +/-1 ms",
        description="Interaural time difference proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l, r, ts = _stereo_frames(ctx)
        itd = np.array([_gcc_phat_itd(l[i], r[i], ctx.sample_rate) for i in range(l.shape[0])], dtype=np.float64)
        conf = 1.0 if ctx.channels >= 2 else 0.3
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(itd),
            series=itd.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
        )


class AmbisonicDiffusenessMetric:
    spec = MetricSpec(
        name="ambisonic_diffuseness",
        category="Spatial",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Energy-vector magnitude relative to total B-format energy",
        description="B-format diffuseness proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        if ctx.channels < 4:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)

        w = ctx.signal[:, 0]
        x = ctx.signal[:, 1]
        y = ctx.signal[:, 2]
        z = ctx.signal[:, 3]
        e = float(np.mean(np.square(w) + np.square(x) + np.square(y) + np.square(z))) + EPS
        iv = np.array([np.mean(w * x), np.mean(w * y), np.mean(w * z)], dtype=np.float64)
        ratio = float(np.linalg.norm(iv) / e)
        diff = float(np.clip(1.0 - ratio, 0.0, 1.0))
        return _scalar_result(self.spec.name, self.spec.units, diff, 0.8)


class AmbisonicEnergyAzimuthMetric:
    spec = MetricSpec(
        name="ambisonic_energy_vector_azimuth_deg",
        category="Spatial",
        units="deg",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Azimuth from B-format energy/intensity vector",
        description="B-format energy vector azimuth.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        if ctx.channels < 4:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)
        w = ctx.signal[:, 0]
        x = ctx.signal[:, 1]
        y = ctx.signal[:, 2]
        ix = float(np.mean(w * x))
        iy = float(np.mean(w * y))
        az = float(np.degrees(np.arctan2(iy, ix)))
        return _scalar_result(self.spec.name, self.spec.units, az, 0.8)


class AmbisonicEnergyElevationMetric:
    spec = MetricSpec(
        name="ambisonic_energy_vector_elevation_deg",
        category="Spatial",
        units="deg",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Elevation from B-format energy/intensity vector",
        description="B-format energy vector elevation.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        if ctx.channels < 4:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)
        w = ctx.signal[:, 0]
        x = ctx.signal[:, 1]
        y = ctx.signal[:, 2]
        z = ctx.signal[:, 3]
        ix = float(np.mean(w * x))
        iy = float(np.mean(w * y))
        iz = float(np.mean(w * z))
        el = float(np.degrees(np.arctan2(iz, np.sqrt(ix * ix + iy * iy) + EPS)))
        return _scalar_result(self.spec.name, self.spec.units, el, 0.8)


class DOAAzimuthProxyMetric:
    spec = MetricSpec(
        name="doa_azimuth_proxy_deg",
        category="Spatial",
        units="deg",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="DOA from mean ITD and assumed 0.2 m baseline",
        description="Direction-of-arrival azimuth proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        l, r, _ = _stereo_frames(ctx)
        if l.shape[0] == 0:
            return _scalar_result(self.spec.name, self.spec.units, float("nan"), 0.2)
        itd = np.array([_gcc_phat_itd(l[i], r[i], ctx.sample_rate) for i in range(l.shape[0])], dtype=np.float64)
        mean_itd = float(np.mean(itd))
        az = _doa_azimuth_from_itd(mean_itd, mic_spacing_m=0.2)
        conf = 0.7 if ctx.channels >= 2 else 0.2
        return _scalar_result(self.spec.name, self.spec.units, az, conf, extra={"mean_itd_s": mean_itd, "mic_spacing_m": 0.2})


# -----------------------------------------------------------------------------
# Quality-control metrics
# -----------------------------------------------------------------------------
# References:
# - Practical QC conventions from measurement workflows (clipping, dropouts, DC bias)
# - IEC meter and instrumentation practice context (non-normative here)


class ClippingRatioMetric:
    spec = MetricSpec(
        name="clipping_ratio",
        category="Basic",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Fraction of clipped samples |x|>=0.999",
        description="Share of samples at or near digital full scale.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        qc = _qc_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(qc["clipping_ratio"]), 1.0)


class ClippingEventCountMetric:
    spec = MetricSpec(
        name="clipping_event_count",
        category="Basic",
        units="count",
        window_size=0,
        hop_size=0,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Rising-edge count of clipping mask",
        description="Number of clipping event segments.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        qc = _qc_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(qc["clipping_event_count"]), 1.0)


class DCOffsetMetric:
    spec = MetricSpec(
        name="dc_offset",
        category="Basic",
        units="linear",
        window_size=0,
        hop_size=0,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Mean sample value",
        description="DC bias in waveform amplitude.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        qc = _qc_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(qc["dc_offset"]), 1.0)


class DropoutRatioMetric:
    spec = MetricSpec(
        name="dropout_ratio",
        category="Basic",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Low-RMS and low-amplitude frame fraction",
        description="Fraction of likely sensor or stream dropouts.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        qc = _qc_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(qc["dropout_ratio"]), 0.9)


class SilenceRatioMetric:
    spec = MetricSpec(
        name="silence_ratio",
        category="Basic",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Low mean-absolute-amplitude frame fraction",
        description="Fraction of near-silent frames.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        qc = _qc_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(qc["silence_ratio"]), 0.9)


class CalibrationDriftMetric:
    spec = MetricSpec(
        name="calibration_drift_db",
        category="Level & Loudness",
        units="dB",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Difference between measured and expected calibration tone level",
        description="Estimated calibration drift from calibration tone reference.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        qc = _qc_cache(ctx)
        return _scalar_result(
            self.spec.name,
            self.spec.units,
            float(qc["calibration_drift_db"]),
            float(qc["calibration_drift_confidence"]),
        )


# -----------------------------------------------------------------------------
# Model-backed anomaly metrics
# -----------------------------------------------------------------------------
# References:
# - Liu et al. (2008) Isolation Forest
# - Schölkopf et al. (2001) One-Class SVM
# - Pimentel et al. (2014) novelty/anomaly survey context


class IsolationForestScoreMetric:
    spec = MetricSpec(
        name="isolation_forest_score",
        category="Anomaly / Novelty",
        units="score",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Isolation Forest decision score over frame feature vectors",
        description="Mean anomaly score from Isolation Forest (or fallback z-score).",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        t, X = _feature_matrix(ctx)
        score: np.ndarray
        conf = 0.6
        try:
            from sklearn.ensemble import IsolationForest

            # Isolation Forest: Liu, Ting, Zhou (ICDM 2008).
            model = IsolationForest(random_state=ctx.config.seed, n_estimators=200, contamination="auto")
            model.fit(X)
            score = -model.decision_function(X)
            conf = 0.9
        except Exception:
            score = _anomaly_fallback_score(X)

        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(score),
            series=score.tolist(),
            timestamps_s=t.tolist(),
            confidence=conf,
        )


class OCSVMScoreMetric:
    spec = MetricSpec(
        name="ocsvm_score",
        category="Anomaly / Novelty",
        units="score",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="One-Class SVM decision score over frame feature vectors",
        description="Mean anomaly score from One-Class SVM (or fallback z-score).",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        t, X = _feature_matrix(ctx)
        score: np.ndarray
        conf = 0.55
        try:
            from sklearn.svm import OneClassSVM

            # One-Class SVM: Schölkopf et al. (Neural Computation, 2001).
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
            model.fit(X)
            score = -model.decision_function(X)
            conf = 0.85
        except Exception:
            score = _anomaly_fallback_score(X)

        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(score),
            series=score.tolist(),
            timestamps_s=t.tolist(),
            confidence=conf,
        )


class AutoencoderReconErrorMetric:
    spec = MetricSpec(
        name="autoencoder_recon_error",
        category="Anomaly / Novelty",
        units="mse",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Low-rank reconstruction proxy for autoencoder residual",
        description="Per-frame reconstruction error proxy for autoencoder pipelines.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        t, X = _feature_matrix(ctx)
        if X.size == 0:
            return MetricResult(name=self.spec.name, units=self.spec.units, summary=_safe_summary(np.array([])), confidence=0.2)

        mu = np.mean(X, axis=0, keepdims=True)
        sd = np.std(X, axis=0, keepdims=True) + EPS
        Xn = (X - mu) / sd
        U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
        k = min(3, max(1, Xn.shape[1] - 1), len(S))
        Xr = (U[:, :k] * S[:k]) @ Vt[:k, :]
        err = np.mean(np.square(Xn - Xr), axis=1)

        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(err),
            series=err.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.8,
        )


class ChangePointConfidenceMetric:
    spec = MetricSpec(
        name="change_point_confidence",
        category="Anomaly / Novelty",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Prominence and z-score strength of novelty peaks",
        description="Confidence that significant acoustic change points exist.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        _, t, mag = _spectral_cache(ctx)
        nov = novelty_from_spectrum(mag)
        z = (nov - np.mean(nov)) / (np.std(nov) + EPS)
        peaks, props = find_peaks(z, prominence=1.0, distance=max(1, ctx.hop_size // 256))
        if peaks.size == 0:
            conf = 0.0
        else:
            prom = props.get("prominences", np.array([0.0]))
            conf = float(np.clip(np.max(prom) / 8.0, 0.0, 1.0))

        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(np.array([conf])),
            series=z.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.9,
            extra={"peak_indices": peaks.tolist(), "peak_count": int(peaks.size), "confidence": conf},
        )


# -----------------------------------------------------------------------------
# Fractional octave outputs
# -----------------------------------------------------------------------------
# References:
# - ISO octave/third-octave center-frequency conventions (implementation-proxy bands)


class OctaveBandLevelMetric:
    spec = MetricSpec(
        name="octave_band_level_db",
        category="Spectral",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Band-averaged levels over ISO-like octave center frequencies",
        description="Octave-band mean levels.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        centers, _, _, band_mean = _fractional_octave_levels(ctx, fraction=1)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(band_mean),
            series=band_mean.tolist(),
            timestamps_s=list(range(len(band_mean))),
            confidence=0.9,
            extra={"band_centers_hz": centers.tolist()},
        )


class ThirdOctaveBandLevelMetric:
    spec = MetricSpec(
        name="third_octave_band_level_db",
        category="Spectral",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Band-averaged levels over 1/3-octave center frequencies",
        description="Third-octave-band mean levels.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        centers, _, _, band_mean = _fractional_octave_levels(ctx, fraction=3)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_safe_summary(band_mean),
            series=band_mean.tolist(),
            timestamps_s=list(range(len(band_mean))),
            confidence=0.9,
            extra={"band_centers_hz": centers.tolist()},
        )


# -----------------------------------------------------------------------------
# Campaign-level proxy metrics (single-file inference)
# -----------------------------------------------------------------------------
# References:
# - Field recording QA literature and passive acoustic monitoring workflow conventions.


class UptimeRatioMetric:
    spec = MetricSpec(
        name="uptime_ratio",
        category="Basic",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="1 - dropout ratio",
        description="Estimated sensor/audio uptime ratio.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _campaign_proxy_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(c["uptime_ratio"]), 0.8)


class CompletenessRatioMetric:
    spec = MetricSpec(
        name="completeness_ratio",
        category="Basic",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Finite-sample ratio weighted by uptime",
        description="Estimated recording completeness ratio.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _campaign_proxy_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(c["completeness_ratio"]), 0.75)


class DiurnalCoverageMetric:
    spec = MetricSpec(
        name="diurnal_coverage_ratio",
        category="Basic",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Derived from parsed timestamp and duration (or low-confidence fallback)",
        description="Estimated fraction of 24 h cycle represented by this recording.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _campaign_proxy_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(c["diurnal_coverage_ratio"]), float(c["diurnal_confidence"]))


class SiteComparabilityMetric:
    spec = MetricSpec(
        name="site_comparability_score",
        category="Basic",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Composite of calibration availability, sampling convention, layout and clipping health",
        description="Single-score proxy for cross-site comparability.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _campaign_proxy_cache(ctx)
        return _scalar_result(self.spec.name, self.spec.units, float(c["site_comparability_score"]), 0.7)


def extended_plugins() -> list[object]:
    """Return all extended metrics."""
    return [
        # Standards SPL/Loudness
        LeqMetric(),
        LmaxMetric(),
        LminMetric(),
        LpeakMetric(),
        PercentileLevelMetric(10),
        PercentileLevelMetric(50),
        PercentileLevelMetric(90),
        PercentileLevelMetric(95),
        SELMetric(),
        LAEMetric(),
        IntegratedLUFSMetric(),
        ShortTermLUFSMetric(),
        MomentaryLUFSMetric(),
        LoudnessRangeMetric(),
        TruePeakMetric(),
        # Intelligibility + architectural extensions
        STIProxyMetric(),
        T20Metric(),
        T30Metric(),
        TsMetric(),
        GStrengthMetric(),
        LFMetric(),
        LFCMetric(),
        BassRatioMetric(),
        # Ecoacoustics extension set
        NDSIMetric(),
        ADIMetric(),
        AEIMetric(),
        AcousticEntropyMetric(),
        EcoOctaveTrendMetric(),
        # Spatial and ambisonic
        IACCMetric(),
        ILDMetric(),
        IPDMetric(),
        ITDMetric(),
        AmbisonicDiffusenessMetric(),
        AmbisonicEnergyAzimuthMetric(),
        AmbisonicEnergyElevationMetric(),
        DOAAzimuthProxyMetric(),
        # QC metrics
        ClippingRatioMetric(),
        ClippingEventCountMetric(),
        DCOffsetMetric(),
        DropoutRatioMetric(),
        SilenceRatioMetric(),
        CalibrationDriftMetric(),
        # Anomaly model-backed metrics
        IsolationForestScoreMetric(),
        OCSVMScoreMetric(),
        AutoencoderReconErrorMetric(),
        ChangePointConfidenceMetric(),
        # Band outputs
        OctaveBandLevelMetric(),
        ThirdOctaveBandLevelMetric(),
        # Campaign-level proxy metrics
        UptimeRatioMetric(),
        CompletenessRatioMetric(),
        DiurnalCoverageMetric(),
        SiteComparabilityMetric(),
    ]
