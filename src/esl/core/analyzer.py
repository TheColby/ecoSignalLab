"""Top-level analysis orchestration."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from esl import __version__
from esl.core.audio import AudioBuffer, read_audio, stream_audio
from esl.core.calibration import calibration_to_dict
from esl.core.config import AnalysisConfig
from esl.core.context import AnalysisContext
from esl.core.utils import config_hash, set_seed
from esl.metrics.base import MetricResult
from esl.metrics.registry import MetricRegistry, create_registry


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


def _assemble_result(
    config: AnalysisConfig,
    audio: AudioBuffer,
    metrics: dict[str, MetricResult],
    registry: MetricRegistry,
    mode: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    selected = config.metrics or registry.names()
    metric_payload: dict[str, Any] = {}
    for name in selected:
        spec = asdict(registry.get(name).spec)
        metric_payload[name] = _serialize_metric(metrics[name], spec)

    assumptions: list[str] = []
    if config.calibration is None:
        assumptions.append("No calibration provided; SPL fields are dBFS-derived proxies.")
    assumptions.append("All timestamps are in seconds from start of input stream.")

    result = {
        "esl_version": __version__,
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "analysis_mode": mode,
        "metadata": {
            "input_path": str(Path(audio.source_path).resolve()),
            "sample_rate": audio.sample_rate,
            "num_samples": audio.num_samples,
            "channels": audio.channels,
            "duration_s": audio.duration_s,
            "format_name": audio.format_name,
            "subtype": audio.subtype,
            "backend": audio.source_backend,
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "seed": config.seed,
            "project": config.project,
            "variant": config.variant,
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
