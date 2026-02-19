"""Top-level analysis orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import math
import platform
import socket
from typing import Any

import numpy as np

from esl import __version__
from esl.core.audio import AudioBuffer, detect_signal_layout, read_audio, stream_audio
from esl.core.calibration import calibration_to_dict
from esl.core.config import AnalysisConfig
from esl.core.context import AnalysisContext
from esl.core.utils import canonicalize, config_hash, library_versions, pipeline_hash, set_seed
from esl.metrics.base import MetricResult
from esl.metrics.registry import METRIC_CATALOG_VERSION, MetricRegistry, create_registry
from esl.schema import SCHEMA_VERSION


def _serialize_metric(result: MetricResult, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "units": result.units,
        "summary": result.summary,
        "series": result.series,
        "timestamps_s": result.timestamps_s,
        "confidence": result.confidence,
        "extra": result.extra,
        "spec": spec,
    }


def _channel_summary(audio: AudioBuffer) -> dict[str, Any]:
    x = np.asarray(audio.samples, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {"channels": [], "aggregate": {}, "aggregation_rules": {}}

    rms_ch = np.sqrt(np.mean(np.square(x), axis=0))
    peak_ch = np.max(np.abs(x), axis=0)
    dc_ch = np.mean(x, axis=0)
    clip_ch = np.mean(np.abs(x) >= 0.999, axis=0)

    channels = []
    for i in range(x.shape[1]):
        channels.append(
            {
                "id": f"ch{i + 1}",
                "rms_dbfs": float(20.0 * np.log10(max(rms_ch[i], 1e-12))),
                "peak_dbfs": float(20.0 * np.log10(max(peak_ch[i], 1e-12))),
                "dc_offset": float(dc_ch[i]),
                "clipping_ratio": float(clip_ch[i]),
            }
        )

    agg_rms = float(np.sqrt(np.mean(np.square(rms_ch))))
    aggregate = {
        "rms_dbfs": float(20.0 * np.log10(max(agg_rms, 1e-12))),
        "peak_dbfs": float(20.0 * np.log10(max(float(np.max(peak_ch)), 1e-12))),
        "dc_offset": float(np.mean(dc_ch)),
        "clipping_ratio": float(np.mean(clip_ch)),
    }
    aggregation_rules = {
        "rms_dbfs": "20*log10(sqrt(mean(channel_rms_linear^2)))",
        "peak_dbfs": "max(channel_peak_dbfs)",
        "dc_offset": "mean(channel_dc_offset)",
        "clipping_ratio": "mean(channel_clipping_ratio)",
    }
    return {"channels": channels, "aggregate": aggregate, "aggregation_rules": aggregation_rules}


def _ir_detected(audio: AudioBuffer) -> bool:
    if audio.samples.size == 0:
        return False
    mono = np.mean(audio.samples, axis=1)
    if mono.size < 8:
        return False
    peak_idx = int(np.argmax(np.abs(mono)))
    if peak_idx > max(1, mono.size // 8):
        return False
    tail = mono[peak_idx:]
    if tail.size < 8:
        return False
    env = np.abs(tail)
    first = float(np.mean(env[: max(4, min(64, env.size // 8))]))
    last = float(np.mean(env[-max(4, min(64, env.size // 8)) :]))
    return bool(first > 0.0 and last < first)


def _validity_flags(
    audio: AudioBuffer,
    channel_summary: dict[str, Any],
    calibration_applied: bool,
    metrics: dict[str, MetricResult],
) -> dict[str, Any]:
    agg = channel_summary.get("aggregate", {})
    clipping_ratio = float(agg.get("clipping_ratio", 0.0))
    dc_offset = float(agg.get("dc_offset", 0.0))
    snr_conf = float(metrics.get("snr_db").confidence) if "snr_db" in metrics else None
    ir_detected = _ir_detected(audio)
    ir_fit_r2: float | None = None
    ir_dynamic_range_db: float | None = None
    ir_tail_low_snr = False
    if ir_detected:
        rt_extra = metrics.get("rt60_s").extra if "rt60_s" in metrics else {}
        fit = rt_extra.get("fit", {}) if isinstance(rt_extra, dict) else {}
        ir_fit_r2 = float(fit["r2"]) if isinstance(fit, dict) and fit.get("r2") is not None else None
        ir_dynamic_range_db = (
            float(rt_extra["dynamic_range_db"])
            if isinstance(rt_extra, dict) and rt_extra.get("dynamic_range_db") is not None
            else None
        )
        if ir_dynamic_range_db is not None and ir_dynamic_range_db < 35.0:
            ir_tail_low_snr = True
        if ir_fit_r2 is not None and ir_fit_r2 < 0.85:
            ir_tail_low_snr = True
    return {
        "clipping": clipping_ratio > 0.0,
        "clipping_ratio": clipping_ratio,
        "dc_offset_excessive": abs(dc_offset) > 1e-3,
        "dc_offset": dc_offset,
        "calibration_applied": bool(calibration_applied),
        "ir_detected": ir_detected,
        "ir_fit_r2": ir_fit_r2,
        "ir_dynamic_range_db": ir_dynamic_range_db,
        "ir_tail_low_snr": ir_tail_low_snr,
        "snr_confidence": snr_conf,
        "snr_confidence_low": bool(snr_conf is not None and snr_conf < 0.7),
    }


def _assemble_result(
    config: AnalysisConfig,
    audio: AudioBuffer,
    metrics: dict[str, MetricResult],
    registry: MetricRegistry,
    mode: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    selected = config.metrics or registry.names()
    lib_versions = library_versions()
    p_hash = pipeline_hash(
        config=config,
        metric_names=list(selected),
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        library_version_map=lib_versions,
    )
    metric_payload: dict[str, Any] = {}
    for name in selected:
        spec = asdict(registry.get(name).spec)
        metric_payload[name] = _serialize_metric(metrics[name], spec)

    assumptions: list[str] = []
    if config.calibration is None:
        assumptions.append("No calibration provided; SPL fields are dBFS-derived proxies.")
    assumptions.append("All timestamps are in seconds from start of input stream.")
    channel_summary = _channel_summary(audio)
    validity = _validity_flags(
        audio=audio,
        channel_summary=channel_summary,
        calibration_applied=config.calibration is not None,
        metrics=metrics,
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "esl_version": __version__,
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_time_local": datetime.now().astimezone().isoformat(),
        "config_hash": config_hash(config),
        "pipeline_hash": p_hash,
        "analysis_mode": mode,
        "metric_catalog": {
            "version": METRIC_CATALOG_VERSION,
            "selected_metrics": list(selected),
            "count": len(selected),
        },
        "library_versions": lib_versions,
        "metadata": {
            "input_path": str(Path(audio.source_path).resolve()),
            "sample_rate": audio.sample_rate,
            "num_samples": audio.num_samples,
            "channels": audio.channels,
            "duration_s": audio.duration_s,
            "format_name": audio.format_name,
            "subtype": audio.subtype,
            "backend": audio.source_backend,
            "channel_layout_hint": detect_signal_layout(audio.channels, audio.source_path),
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "seed": config.seed,
            "project": config.project,
            "variant": config.variant,
            "decoder": {
                "decoder_used": audio.decoder_provenance.get("decoder_used", audio.source_backend),
                "ffmpeg_version": audio.decoder_provenance.get("ffmpeg_version"),
                "ffprobe": audio.decoder_provenance.get("ffprobe"),
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            },
            "config_snapshot": canonicalize(asdict(config)),
            "resolved_metric_list": list(selected),
            "metric_catalog_version": METRIC_CATALOG_VERSION,
            "channel_metrics": channel_summary,
            "validity_flags": validity,
            "calibration": calibration_to_dict(config.calibration),
            "assumptions": assumptions,
            "warnings": warnings or [],
        },
        "metrics": metric_payload,
    }
    return result


def _analyze_full(config: AnalysisConfig, registry: MetricRegistry, metric_names: list[str]) -> dict[str, Any]:
    audio = read_audio(config.input_path, target_sr=config.sample_rate)
    ctx = AnalysisContext(audio=audio, config=config, calibration=config.calibration)
    metric_results = registry.compute(ctx, metric_names)
    return _assemble_result(config, audio, metric_results, registry=registry, mode="full")


def _analyze_streaming(config: AnalysisConfig, registry: MetricRegistry, metric_names: list[str]) -> dict[str, Any]:
    chunks = list(stream_audio(config.input_path, chunk_size=int(config.chunk_size or 131072), target_sr=config.sample_rate))
    if not chunks:
        raise RuntimeError("No audio chunks produced for streaming analysis.")

    base_audio = read_audio(config.input_path, target_sr=config.sample_rate)
    merged: dict[str, dict[str, Any]] = {
        name: {"series": [], "timestamps": [], "units": None, "conf": []} for name in metric_names
    }

    for chunk in chunks:
        chunk_buffer = AudioBuffer(
            samples=chunk.samples,
            sample_rate=chunk.sample_rate,
            source_path=str(config.input_path),
            format_name=base_audio.format_name,
            subtype=base_audio.subtype,
            source_backend=base_audio.source_backend,
        )
        chunk_cfg = AnalysisConfig(
            input_path=config.input_path,
            output_dir=config.output_dir,
            frame_size=config.frame_size,
            hop_size=config.hop_size,
            sample_rate=config.sample_rate,
            chunk_size=config.chunk_size,
            metrics=list(metric_names),
            calibration=config.calibration,
            project=config.project,
            variant=config.variant,
            verbosity=config.verbosity,
            debug=config.debug,
            seed=config.seed,
            make_plots=False,
            ml_export=False,
        )
        ctx = AnalysisContext(audio=chunk_buffer, config=chunk_cfg, calibration=config.calibration)
        chunk_metrics = registry.compute(ctx, metric_names)
        chunk_offset = chunk.start_sample / float(chunk.sample_rate)

        for name, r in chunk_metrics.items():
            merged[name]["series"].extend(r.series)
            merged[name]["timestamps"].extend([chunk_offset + t for t in r.timestamps_s])
            merged[name]["units"] = r.units
            merged[name]["conf"].append(r.confidence)

    metric_results: dict[str, MetricResult] = {}
    for name in metric_names:
        series = np.array(merged[name]["series"], dtype=np.float64)
        if series.size == 0:
            summary = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "p50": float("nan"), "p95": float("nan")}
        else:
            summary = {
                "mean": float(np.mean(series)),
                "std": float(np.std(series)),
                "min": float(np.min(series)),
                "max": float(np.max(series)),
                "p50": float(np.percentile(series, 50.0)),
                "p95": float(np.percentile(series, 95.0)),
            }
        metric_results[name] = MetricResult(
            name=name,
            units=str(merged[name]["units"]),
            summary=summary,
            series=merged[name]["series"],
            timestamps_s=merged[name]["timestamps"],
            confidence=float(np.mean(merged[name]["conf"]) if merged[name]["conf"] else 0.0),
        )

    return _assemble_result(
        config,
        base_audio,
        metric_results,
        registry=registry,
        mode="streaming",
        warnings=["Streaming mode computes per-chunk metrics and merges timeseries."],
    )


def analyze(config: AnalysisConfig, registry: MetricRegistry | None = None) -> dict[str, Any]:
    """Run analysis and return serializable result document."""
    set_seed(config.seed)
    reg = registry or create_registry(with_external=True)
    metric_names = config.metrics or reg.names()

    if config.chunk_size:
        non_streaming = [name for name in metric_names if not reg.get(name).spec.streaming_capable]
        if non_streaming:
            return _analyze_full(config, reg, metric_names)
        return _analyze_streaming(config, reg, metric_names)

    return _analyze_full(config, reg, metric_names)
