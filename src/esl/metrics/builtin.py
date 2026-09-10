"""Built-in metric plugins for esl.

References:
- Spectral descriptors and STFT-based feature extraction:
  Klapuri & Davy (2007), \"Signal Processing Methods for Music Transcription\".
- Spectral flux novelty and change detection:
  Dixon (2006), DAFx.
- Room acoustics decay metrics:
  ISO 3382-1 / ISO 3382-2 and Schroeder (1965).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from esl.core.calibration import db, dbfs_to_spl, weighted_rms
from esl.core.context import AnalysisContext
from esl.metrics.base import MetricResult, MetricSpec
from esl.metrics.helpers import (
    crest_factor_db,
    estimate_snr_db,
    fit_decay_diagnostics,
    frame_peak,
    frame_rms,
    frame_signal,
    novelty_from_spectrum,
    schroeder_decay,
    spectral_features,
    summarize,
)


def _frame_level_dbfs(ctx: AnalysisContext) -> tuple[np.ndarray, np.ndarray]:
    framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    rms = frame_rms(framed.frames)
    dbfs = db(rms)
    return dbfs, framed.times_s


def _frame_weighted_level_db(ctx: AnalysisContext, weighting: str) -> tuple[np.ndarray, np.ndarray]:
    framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    vals = np.zeros(framed.frames.shape[0], dtype=np.float64)
    for i in range(framed.frames.shape[0]):
        wrms = np.mean(weighted_rms(framed.frames[i], ctx.sample_rate, weighting=weighting))
        vals[i] = float(db(wrms))
    vals = dbfs_to_spl(vals, ctx.calibration)
    return vals, framed.times_s


def _spectral_cache(ctx: AnalysisContext) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Shared STFT cache for spectral descriptors and novelty features.
    key = "spectral"
    if key not in ctx.cache:
        ctx.cache[key] = spectral_features(ctx.mono(), ctx.sample_rate, ctx.frame_size, ctx.hop_size)
    return ctx.cache[key]


def _architectural_cache(ctx: AnalysisContext) -> dict[str, np.ndarray | float]:
    key = "architectural"
    if key in ctx.cache:
        return ctx.cache[key]

    mono = ctx.mono().astype(np.float64)
    peak_idx = int(np.argmax(np.abs(mono)))
    ir = mono[peak_idx:]
    if ir.size < max(16, int(0.1 * ctx.sample_rate)):
        ir = mono
    # Energy decay via Schroeder reverse integration.
    decay = schroeder_decay(ir)
    t = np.arange(len(decay), dtype=np.float64) / float(ctx.sample_rate)

    # ISO-style decay-window fits.
    rt60_fit = fit_decay_diagnostics(decay, t, lo_db=-5.0, hi_db=-35.0)
    edt_fit = fit_decay_diagnostics(decay, t, lo_db=0.0, hi_db=-10.0)
    rt60 = float(rt60_fit["rt60_s"])
    edt = float(edt_fit["rt60_s"])

    e = np.square(ir)
    e_total = float(np.sum(e)) + 1e-20
    n50 = min(len(e), max(1, int(round(0.05 * ctx.sample_rate))))
    n80 = min(len(e), max(1, int(round(0.08 * ctx.sample_rate))))
    e50 = float(np.sum(e[:n50]))
    e80 = float(np.sum(e[:n80]))
    late50 = max(e_total - e50, 1e-20)
    late80 = max(e_total - e80, 1e-20)
    c50 = float(10.0 * np.log10(e50 / late50))
    c80 = float(10.0 * np.log10(e80 / late80))
    d50 = float(e50 / e_total)

    dynamic_range = float(np.nanmax(decay) - np.nanmin(decay))
    fit_r2 = float(rt60_fit["r2"]) if np.isfinite(float(rt60_fit["r2"])) else 0.0
    rt_conf = float(np.clip(0.5 * min(dynamic_range / 35.0, 1.0) + 0.5 * max(fit_r2, 0.0), 0.2, 1.0))

    cache = {
        "rt60": rt60,
        "edt": edt,
        "c50": c50,
        "c80": c80,
        "d50": d50,
        "decay": decay,
        "time": t,
        "confidence": float(np.clip(rt_conf, 0.0, 1.0)),
        "dynamic_range_db": dynamic_range,
        "rt60_fit": rt60_fit,
        "edt_fit": edt_fit,
    }
    cache.update(_decay_validity(cache))
    ctx.cache[key] = cache
    return cache


def _finite_summary(series: np.ndarray) -> dict[str, float]:
    """Summary over the finite entries of a series that may contain NaN.

    A frame marked unavailable should not poison the summary of the frames that
    were measurable; if nothing is finite the summary is NaN throughout, which
    is the correct report for "no measurement was possible".
    """
    arr = np.asarray(series, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "p50": nan, "p95": nan}
    return summarize(finite)


DECAY_MIN_DYNAMIC_RANGE_DB = 35.0
DECAY_MIN_FIT_R2 = 0.95


def _decay_validity(cache: dict[str, Any]) -> dict[str, Any]:
    """Decide whether a decay-derived time constant means anything.

    A reverse-integrated energy curve can be fitted to any signal, so a decay
    time is always *computable*.  It is only a reverberation time when the input
    actually decays: enough dynamic range for the ISO evaluation window, a
    negative slope, and a fit the data supports.  Without this test a steady tone
    or a DC offset yields a confident-looking RT.
    """
    fit = cache.get("rt60_fit") or {}
    dynamic_range = float(cache.get("dynamic_range_db", 0.0) or 0.0)
    r2 = float(fit.get("r2", float("nan")))
    slope = float(fit.get("slope_db_per_s", float("nan")))
    rt60 = float(cache.get("rt60", float("nan")))

    reasons: list[str] = []
    if not np.isfinite(rt60):
        reasons.append("decay fit did not converge to a finite decay time")
    if dynamic_range < DECAY_MIN_DYNAMIC_RANGE_DB:
        reasons.append(
            f"decay curve spans only {dynamic_range:.1f} dB; ISO 3382-1 T30 needs at least "
            f"{DECAY_MIN_DYNAMIC_RANGE_DB:.0f} dB below the starting level"
        )
    if not (np.isfinite(slope) and slope < 0.0):
        reasons.append("energy decay curve does not decrease over the evaluation range")
    if not (np.isfinite(r2) and r2 >= DECAY_MIN_FIT_R2):
        reasons.append(f"decay fit r2 = {r2:.3f} is below {DECAY_MIN_FIT_R2}")

    return {
        "decay_valid": not reasons,
        "decay_invalid_reason": "; ".join(reasons) if reasons else None,
    }


class RMSDbfsMetric:
    spec = MetricSpec(
        name="rms_dbfs",
        category="Basic",
        units="dBFS",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="1.0 unless NaN present",
        description="Frame-level RMS level in dBFS.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        series, ts = _frame_level_dbfs(ctx)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(series),
            series=series.tolist(),
            timestamps_s=ts.tolist(),
            confidence=1.0,
        )


class PeakDbfsMetric:
    spec = MetricSpec(
        name="peak_dbfs",
        category="Basic",
        units="dBFS",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="1.0 unless clipping invalidates peak",
        description="Frame-level peak amplitude in dBFS.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
        p = frame_peak(framed.frames)
        series = db(p)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(series),
            series=series.tolist(),
            timestamps_s=framed.times_s.tolist(),
            confidence=1.0,
        )


class CrestFactorMetric:
    spec = MetricSpec(
        name="crest_factor_db",
        category="Level & Loudness",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Depends on stable RMS floor",
        description="Frame-level crest factor in dB.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
        rms = frame_rms(framed.frames)
        peak = frame_peak(framed.frames)
        cf = crest_factor_db(rms, peak)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(cf),
            series=cf.tolist(),
            timestamps_s=framed.times_s.tolist(),
            confidence=1.0,
        )


class SPLZMetric:
    spec = MetricSpec(
        name="spl_z_db",
        category="Level & Loudness",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Frame-level Z-weighted SPL proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        series, ts = _frame_weighted_level_db(ctx, "Z")
        conf = 1.0 if ctx.calibration is not None else 0.6
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(series),
            series=series.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
            extra={"interpretation": "dBFS proxy" if ctx.calibration is None else "calibrated SPL"},
        )


class SPLAMetric:
    spec = MetricSpec(
        name="spl_a_db",
        category="Level & Loudness",
        units="dBA",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Frame-level A-weighted SPL proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        series, ts = _frame_weighted_level_db(ctx, "A")
        conf = 1.0 if ctx.calibration is not None else 0.6
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(series),
            series=series.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
            extra={"interpretation": "dBFS proxy" if ctx.calibration is None else "calibrated SPL"},
        )


class SPLCMetric:
    spec = MetricSpec(
        name="spl_c_db",
        category="Level & Loudness",
        units="dBC",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=True,
        ml_compatible=True,
        confidence_logic="Reduced if calibration missing",
        description="Frame-level C-weighted SPL proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        series, ts = _frame_weighted_level_db(ctx, "C")
        conf = 1.0 if ctx.calibration is not None else 0.6
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(series),
            series=series.tolist(),
            timestamps_s=ts.tolist(),
            confidence=conf,
            extra={"interpretation": "dBFS proxy" if ctx.calibration is None else "calibrated SPL"},
        )


class SNRMetric:
    spec = MetricSpec(
        name="snr_db",
        category="Noise & SNR",
        units="dB",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Based on percentile-separation estimator",
        description="Robust percentile SNR estimate.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        level_dbfs, ts = _frame_level_dbfs(ctx)
        snr_series = estimate_snr_db(level_dbfs)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(snr_series),
            series=snr_series.tolist(),
            timestamps_s=ts.tolist(),
            confidence=0.8,
            extra={
                "method": "percentile_90_minus_10",
                "confidence_notes": "Percentile estimator; assumes representative noise floor in signal distribution.",
            },
        )


class SpectralCentroidMetric:
    spec = MetricSpec(
        name="spectral_centroid_hz",
        category="Spectral",
        units="Hz",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="High for non-silent frames",
        description="Spectral centroid over time.",
    )

    # spectral_features adds 1e-12 to every magnitude bin, so an all-zero frame
    # still sums to bins * 1e-12.  Anything within an order of magnitude of that
    # floor carries no signal.
    SILENCE_FLOOR_FACTOR = 10.0

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        f, t, mag = _spectral_cache(ctx)
        num = np.sum(f[:, None] * mag, axis=0)
        den = np.sum(mag, axis=0)
        centroid = num / np.maximum(den, 1e-12)
        # With no signal, num/den collapses to the mean bin frequency -- roughly
        # half of Nyquist -- which looks like a measurement and is not one.
        floor = float(mag.shape[0]) * 1e-12 * self.SILENCE_FLOOR_FACTOR
        silent = den <= floor
        centroid = np.where(silent, np.nan, centroid)
        silent_share = float(np.mean(silent)) if silent.size else 0.0
        valid = silent_share < 1.0
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=_finite_summary(centroid),
            series=centroid.tolist(),
            timestamps_s=t.tolist(),
            confidence=1.0 if silent_share == 0.0 else max(0.2, 1.0 - silent_share),
            extra={
                "valid": valid,
                "invalid_reason": (
                    None if valid else "every frame is below the spectral energy floor; centroid is undefined"
                ),
                "silent_frame_share": silent_share,
                "silence_floor_magnitude_sum": floor,
            },
        )


class SpectralBandwidthMetric:
    spec = MetricSpec(
        name="spectral_bandwidth_hz",
        category="Spectral",
        units="Hz",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="High for non-silent frames",
        description="Spectral spread around centroid.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        f, t, mag = _spectral_cache(ctx)
        centroid = np.sum(f[:, None] * mag, axis=0) / np.maximum(np.sum(mag, axis=0), 1e-12)
        spread = np.sqrt(np.sum(((f[:, None] - centroid[None, :]) ** 2) * mag, axis=0) / np.maximum(np.sum(mag, axis=0), 1e-12))
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(spread),
            series=spread.tolist(),
            timestamps_s=t.tolist(),
            confidence=1.0,
        )


class SpectralFlatnessMetric:
    spec = MetricSpec(
        name="spectral_flatness",
        category="Spectral",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Stable with sufficient bins",
        description="Geometric-to-arithmetic spectral magnitude ratio.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        _, t, mag = _spectral_cache(ctx)
        gm = np.exp(np.mean(np.log(np.maximum(mag, 1e-12)), axis=0))
        am = np.mean(mag, axis=0)
        flat = gm / np.maximum(am, 1e-12)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(flat),
            series=flat.tolist(),
            timestamps_s=t.tolist(),
            confidence=1.0,
        )


class SpectralRolloffMetric:
    spec = MetricSpec(
        name="spectral_rolloff_hz",
        category="Spectral",
        units="Hz",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="High when spectra are non-silent",
        description="Frequency below which 85% spectral energy lies.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        f, t, mag = _spectral_cache(ctx)
        power = np.square(mag)
        csum = np.cumsum(power, axis=0)
        target = 0.85 * csum[-1]
        idx = np.argmax(csum >= target[None, :], axis=0)
        rolloff = f[idx]
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(rolloff),
            series=rolloff.tolist(),
            timestamps_s=t.tolist(),
            confidence=1.0,
        )


class ZeroCrossingRateMetric:
    spec = MetricSpec(
        name="zero_crossing_rate",
        category="Temporal",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Reliable for band-limited low-noise inputs",
        description="Frame-level zero crossing ratio.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
        mono = np.mean(framed.frames, axis=2)
        sgn = np.sign(mono)
        zc = np.sum(np.abs(np.diff(sgn, axis=1)) > 0, axis=1) / float(ctx.frame_size)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(zc),
            series=zc.tolist(),
            timestamps_s=framed.times_s.tolist(),
            confidence=0.9,
        )


class BioacousticIndexMetric:
    spec = MetricSpec(
        name="bioacoustic_index",
        category="Ecoacoustics",
        units="a.u.",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Depends on frequency coverage in 2-8 kHz band",
        description="Band-limited high-frequency energy proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        f, t, mag = _spectral_cache(ctx)
        band = (f >= 2000.0) & (f <= 8000.0)
        if np.count_nonzero(band) == 0:
            series = np.zeros_like(t)
            conf = 0.3
        else:
            series = np.sum(mag[band], axis=0)
            conf = 1.0
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(series),
            series=series.tolist(),
            timestamps_s=t.tolist(),
            confidence=conf,
        )


class AcousticComplexityIndexMetric:
    spec = MetricSpec(
        name="acoustic_complexity_index",
        category="Ecoacoustics",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Flux normalization by energy floor",
        description="Temporal spectral fluctuation ratio proxy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        _, t, mag = _spectral_cache(ctx)
        d = np.abs(np.diff(mag, axis=1, prepend=mag[:, :1]))
        aci = np.sum(d, axis=0) / np.maximum(np.sum(mag, axis=0), 1e-12)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(aci),
            series=aci.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.9,
        )


class InterchannelCorrelationZeroLagMetric:
    """Zero-lag normalized cross-correlation between adjacent channels.

    This is a Pearson correlation coefficient computed per frame, so it is
    bounded in [-1, 1] and *can be negative*.  It is deliberately not called
    coherence: magnitude-squared coherence is a frequency-domain quantity
    bounded in [0, 1] and is provided separately as
    ``interchannel_coherence_msc``.  The two answer different questions -- a
    pair of broadband signals differing only by a short delay has near-zero
    zero-lag correlation but near-unity coherence.
    """

    spec = MetricSpec(
        name="interchannel_correlation_zerolag",
        category="Spatial",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Reduced for low-energy frames",
        description="Mean adjacent-channel zero-lag correlation per frame, in [-1, 1].",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        framed = frame_signal(ctx.signal, ctx.sample_rate, ctx.frame_size, ctx.hop_size)
        if ctx.channels < 2:
            coh = np.ones(framed.frames.shape[0], dtype=np.float64)
            conf = 1.0
        else:
            coh = np.zeros(framed.frames.shape[0], dtype=np.float64)
            for i in range(framed.frames.shape[0]):
                vals = []
                for c in range(ctx.channels - 1):
                    a = framed.frames[i, :, c]
                    b = framed.frames[i, :, c + 1]
                    da = np.std(a)
                    dbb = np.std(b)
                    if da < 1e-12 or dbb < 1e-12:
                        vals.append(0.0)
                    else:
                        vals.append(float(np.corrcoef(a, b)[0, 1]))
                coh[i] = float(np.mean(vals))
            conf = 0.9
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(coh),
            series=coh.tolist(),
            timestamps_s=framed.times_s.tolist(),
            confidence=conf,
        )


class InterchannelCoherenceMSCMetric:
    """Magnitude-squared coherence between the first two channels.

    Welch estimate with ``nperseg = frame_size`` and ``noverlap = frame_size -
    hop_size``.  Bounded in [0, 1] by construction; a single segment would give
    unity everywhere, so the value is only meaningful because several segments
    are averaged.  The reported scalar is the power-weighted mean of
    Gamma^2(f) over ``band_hz``; the full function is summarized in ``extra``.
    """

    BAND_HZ = (20.0, None)

    spec = MetricSpec(
        name="interchannel_coherence_msc",
        category="Spatial",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Requires several Welch segments; reduced for short inputs",
        description="Welch magnitude-squared coherence of channels 1 and 2, in [0, 1].",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        from scipy.signal import coherence as _coherence

        if ctx.channels < 2:
            return MetricResult(
                name=self.spec.name,
                units=self.spec.units,
                summary=summarize(np.array([float("nan")], dtype=np.float64)),
                series=[float("nan")],
                timestamps_s=[0.0],
                confidence=0.0,
                extra={
                    "valid": False,
                    "invalid_reason": "magnitude-squared coherence requires at least two channels",
                    "channel_pair": None,
                },
            )

        a = np.asarray(ctx.signal[:, 0], dtype=np.float64)
        b = np.asarray(ctx.signal[:, 1], dtype=np.float64)
        nperseg = int(min(ctx.frame_size, a.size))
        noverlap = int(max(0, min(nperseg - 1, ctx.frame_size - ctx.hop_size)))
        segments = 0 if nperseg <= 0 else 1 + max(0, (a.size - nperseg) // max(1, nperseg - noverlap))
        if segments < 8:
            return MetricResult(
                name=self.spec.name,
                units=self.spec.units,
                summary=summarize(np.array([float("nan")], dtype=np.float64)),
                series=[float("nan")],
                timestamps_s=[0.0],
                confidence=0.0,
                extra={
                    "valid": False,
                    "invalid_reason": f"only {segments} Welch segments available; at least 8 required",
                    "welch_segments": int(segments),
                    "channel_pair": [0, 1],
                },
            )

        freqs, gamma2 = _coherence(a, b, fs=ctx.sample_rate, nperseg=nperseg, noverlap=noverlap)
        lo, hi = self.BAND_HZ
        mask = freqs >= float(lo)
        if hi is not None:
            mask &= freqs <= float(hi)
        if not np.any(mask):
            mask = np.ones_like(freqs, dtype=bool)
        weights = np.abs(np.fft.rfft(a[:nperseg]))
        weights = weights[: mask.size] if weights.size >= mask.size else np.ones_like(freqs)
        w = np.maximum(weights[mask], 0.0)
        band = np.clip(gamma2[mask], 0.0, 1.0)
        value = float(np.sum(band * w) / np.sum(w)) if float(np.sum(w)) > 0.0 else float(np.mean(band))
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(np.array([value], dtype=np.float64)),
            series=[value],
            timestamps_s=[0.0],
            confidence=0.9,
            extra={
                "valid": True,
                "channel_pair": [0, 1],
                "welch_nperseg": int(nperseg),
                "welch_noverlap": int(noverlap),
                "welch_segments": int(segments),
                "band_hz": [float(lo), None if hi is None else float(hi)],
                "gamma2_unweighted_mean": float(np.mean(band)),
                "gamma2_max": float(np.max(band)),
            },
        )


class NoveltyCurveMetric:
    spec = MetricSpec(
        name="novelty_curve",
        category="Anomaly / Novelty",
        units="a.u.",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Spectral flux reliability on stationary noise",
        description="Positive spectral flux novelty curve.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        _, t, mag = _spectral_cache(ctx)
        # Positive spectral-flux novelty (Dixon 2006).
        nov = novelty_from_spectrum(mag)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(nov),
            series=nov.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.95,
        )


class SpectralChangeMetric:
    spec = MetricSpec(
        name="spectral_change_detection",
        category="Anomaly / Novelty",
        units="zscore",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Thresholdability of standardized novelty",
        description="Z-scored novelty for change event detection.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        _, t, mag = _spectral_cache(ctx)
        # Z-normalized novelty; thresholding approach inspired by MIR onset/change practice.
        nov = novelty_from_spectrum(mag)
        z = (nov - np.mean(nov)) / (np.std(nov) + 1e-12)
        events = np.where(z > 2.5)[0].tolist()
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(z),
            series=z.tolist(),
            timestamps_s=t.tolist(),
            confidence=0.9,
            extra={"event_frame_indices": events},
        )


class RT60Metric:
    spec = MetricSpec(
        name="rt60_s",
        category="Architectural Acoustics",
        units="s",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Decay fit quality and dynamic range",
        description="Reverberation time extrapolated from T30 decay fit.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _architectural_cache(ctx)
        v = float(c["rt60"])
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(np.array([v])),
            confidence=float(c["confidence"]),
            extra={
                "value": v,
                "time": c["time"].tolist(),
                "decay_db": c["decay"].tolist(),
                "dynamic_range_db": float(c["dynamic_range_db"]),
                "fit": c["rt60_fit"],
                "valid": bool(c["decay_valid"]),
                "invalid_reason": c["decay_invalid_reason"],
            },
        )


class EDTMetric:
    spec = MetricSpec(
        name="edt_s",
        category="Architectural Acoustics",
        units="s",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Decay fit quality and dynamic range",
        description="Early decay time from 0..-10 dB fit.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _architectural_cache(ctx)
        v = float(c["edt"])
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(np.array([v])),
            confidence=float(c["confidence"]),
            extra={
                "value": v,
                "fit": c["edt_fit"],
                "valid": bool(c["decay_valid"]),
                "invalid_reason": c["decay_invalid_reason"],
            },
        )


class C50Metric:
    spec = MetricSpec(
        name="c50_db",
        category="Architectural Acoustics",
        units="dB",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Energy partition validity",
        description="Speech clarity ratio with 50 ms split.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _architectural_cache(ctx)
        v = float(c["c50"])
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(np.array([v])),
            confidence=float(c["confidence"]),
            extra={"value": v},
        )


class C80Metric:
    spec = MetricSpec(
        name="c80_db",
        category="Architectural Acoustics",
        units="dB",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Energy partition validity",
        description="Music clarity ratio with 80 ms split.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _architectural_cache(ctx)
        v = float(c["c80"])
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(np.array([v])),
            confidence=float(c["confidence"]),
            extra={"value": v},
        )


class D50Metric:
    spec = MetricSpec(
        name="d50",
        category="Architectural Acoustics",
        units="ratio",
        window_size=0,
        hop_size=0,
        streaming_capable=False,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Energy partition validity",
        description="Definition ratio using 50 ms early energy.",
    )

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        c = _architectural_cache(ctx)
        v = float(c["d50"])
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(np.array([v])),
            confidence=float(c["confidence"]),
            extra={"value": v},
        )


def builtin_plugins() -> list[object]:
    """Return instances of all built-in metrics."""
    return [
        RMSDbfsMetric(),
        PeakDbfsMetric(),
        CrestFactorMetric(),
        SPLZMetric(),
        SPLAMetric(),
        SPLCMetric(),
        SNRMetric(),
        SpectralCentroidMetric(),
        SpectralBandwidthMetric(),
        SpectralFlatnessMetric(),
        SpectralRolloffMetric(),
        ZeroCrossingRateMetric(),
        BioacousticIndexMetric(),
        AcousticComplexityIndexMetric(),
        InterchannelCorrelationZeroLagMetric(),
        InterchannelCoherenceMSCMetric(),
        NoveltyCurveMetric(),
        SpectralChangeMetric(),
        RT60Metric(),
        EDTMetric(),
        C50Metric(),
        C80Metric(),
        D50Metric(),
    ]
