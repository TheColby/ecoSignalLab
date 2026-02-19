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
    ctx.cache[key] = cache
    return cache


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

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        f, t, mag = _spectral_cache(ctx)
        num = np.sum(f[:, None] * mag, axis=0)
        den = np.sum(mag, axis=0)
        centroid = num / np.maximum(den, 1e-12)
        return MetricResult(
            name=self.spec.name,
            units=self.spec.units,
            summary=summarize(centroid),
            series=centroid.tolist(),
            timestamps_s=t.tolist(),
            confidence=1.0,
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


class InterchannelCoherenceMetric:
    spec = MetricSpec(
        name="interchannel_coherence",
        category="Spatial",
        units="ratio",
        window_size=2048,
        hop_size=512,
        streaming_capable=True,
        calibration_dependency=False,
        ml_compatible=True,
        confidence_logic="Reduced for low-energy frames",
        description="Mean adjacent-channel correlation per frame.",
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
            extra={"value": v, "fit": c["edt_fit"]},
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
        InterchannelCoherenceMetric(),
        NoveltyCurveMetric(),
        SpectralChangeMetric(),
        RT60Metric(),
        EDTMetric(),
        C50Metric(),
        C80Metric(),
        D50Metric(),
    ]
