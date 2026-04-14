"""Top-level analysis orchestration.

The chunked path is intentionally out-of-core so long recordings do not need to
fit in RAM. Full-file analysis remains available for metrics that require global
context such as impulse-response decay fitting.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
from typing import Any

import numpy as np

from esl import __version__
from esl.core.audio import AudioBuffer, detect_signal_layout, probe_audio_metadata, read_audio, stream_audio
from esl.core.calibration import calibration_to_dict
from esl.core.config import AnalysisConfig
from esl.core.context import AnalysisContext
from esl.core.out_of_core import (
    AudioAccumulator,
    FrameTableCsvWriter,
    FrameTableHdf5Writer,
    FrameTableParquetDatasetWriter,
    MetricAccumulator,
    RUNNING_SUMMARY_METHOD,
    load_checkpoint,
    save_checkpoint,
)
from esl.core.spatial_metadata import infer_spatial_metadata
from esl.core.utils import canonicalize, config_hash, library_versions, pipeline_hash, set_seed
from esl.metrics.base import MetricResult
from esl.metrics.registry import METRIC_CATALOG_VERSION, MetricRegistry, create_registry
from esl.schema import SCHEMA_VERSION


CHECKPOINT_FILENAME = "analysis_state.json"


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


def _validity_flags_from_audio(
    audio: AudioBuffer,
    channel_summary: dict[str, Any],
    calibration_applied: bool,
    metrics: dict[str, MetricResult],
) -> dict[str, Any]:
    agg = channel_summary.get("aggregate", {})
    clipping_ratio = float(agg.get("clipping_ratio", 0.0))
    dc_offset = float(agg.get("dc_offset", 0.0))
    snr_metric = metrics.get("snr_db")
    snr_conf = float(snr_metric.confidence) if isinstance(snr_metric, MetricResult) else None
    ir_detected = _ir_detected(audio)
    ir_fit_r2: float | None = None
    ir_dynamic_range_db: float | None = None
    ir_tail_low_snr = False
    if ir_detected:
        rt_metric = metrics.get("rt60_s")
        rt_extra = rt_metric.extra if isinstance(rt_metric, MetricResult) else {}
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


def _validity_flags_from_summary(
    channel_summary: dict[str, Any],
    calibration_applied: bool,
    metrics: dict[str, MetricResult],
) -> dict[str, Any]:
    agg = channel_summary.get("aggregate", {})
    clipping_ratio = float(agg.get("clipping_ratio", 0.0))
    dc_offset = float(agg.get("dc_offset", 0.0))
    snr_metric = metrics.get("snr_db")
    snr_conf = float(snr_metric.confidence) if isinstance(snr_metric, MetricResult) else None
    return {
        "clipping": clipping_ratio > 0.0,
        "clipping_ratio": clipping_ratio,
        "dc_offset_excessive": abs(dc_offset) > 1e-3,
        "dc_offset": dc_offset,
        "calibration_applied": bool(calibration_applied),
        "ir_detected": False,
        "ir_fit_r2": None,
        "ir_dynamic_range_db": None,
        "ir_tail_low_snr": False,
        "snr_confidence": snr_conf,
        "snr_confidence_low": bool(snr_conf is not None and snr_conf < 0.7),
    }


def _resolve_metric_names(config: AnalysisConfig, registry: MetricRegistry) -> tuple[list[str], list[str]]:
    requested = config.metrics or registry.names()
    if not config.streamable_only:
        return list(requested), []
    selected = [name for name in requested if registry.get(name).spec.streaming_capable]
    dropped = [name for name in requested if name not in selected]
    return selected, dropped


def _checkpoint_path(config: AnalysisConfig) -> Path | None:
    if config.checkpoint_dir is None:
        return None
    return Path(config.checkpoint_dir) / CHECKPOINT_FILENAME


def _assemble_result(
    config: AnalysisConfig,
    *,
    audio: AudioBuffer | None,
    audio_metadata: dict[str, Any] | None,
    metrics: dict[str, MetricResult],
    registry: MetricRegistry,
    mode: str,
    selected_metrics: list[str],
    warnings: list[str] | None = None,
    channel_summary: dict[str, Any] | None = None,
    validity_flags: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    analysis_strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lib_versions = library_versions()
    p_hash = pipeline_hash(
        config=config,
        metric_names=list(selected_metrics),
        frame_size=config.frame_size,
        hop_size=config.hop_size,
        library_version_map=lib_versions,
    )
    metric_payload: dict[str, Any] = {}
    for name in selected_metrics:
        spec = asdict(registry.get(name).spec)
        metric_payload[name] = _serialize_metric(metrics[name], spec)

    assumptions: list[str] = []
    if config.calibration is None:
        assumptions.append("No calibration provided; SPL fields are dBFS-derived proxies.")
    assumptions.append("All timestamps are in seconds from start of input stream.")
    from esl.ml import device_resolution_dict, resolve_compute_device

    device_info = resolve_compute_device(config.compute_device, strict=False)
    if device_info.reason:
        assumptions.append(f"Compute device resolution: {device_info.reason}")

    if audio is not None:
        channel_summary = channel_summary or _channel_summary(audio)
        validity_flags = validity_flags or _validity_flags_from_audio(
            audio=audio,
            channel_summary=channel_summary,
            calibration_applied=config.calibration is not None,
            metrics=metrics,
        )
        resolved_meta = {
            "input_path": str(Path(audio.source_path).resolve()),
            "sample_rate": audio.sample_rate,
            "num_samples": audio.num_samples,
            "channels": audio.channels,
            "duration_s": audio.duration_s,
            "format_name": audio.format_name,
            "subtype": audio.subtype,
            "backend": audio.source_backend,
            "decoder": {
                "decoder_used": audio.decoder_provenance.get("decoder_used", audio.source_backend),
                "ffmpeg_version": audio.decoder_provenance.get("ffmpeg_version"),
                "ffprobe": audio.decoder_provenance.get("ffprobe"),
            },
            "channel_layout_hint": detect_signal_layout(audio.channels, audio.source_path),
            "spatial_metadata": infer_spatial_metadata(
                audio.channels,
                audio.source_path,
                source_channel_layout=(
                    str(audio.decoder_provenance.get("ffprobe", {}).get("channel_layout"))
                    if isinstance(audio.decoder_provenance.get("ffprobe"), dict)
                    else None
                ),
            ).to_dict(),
        }
    else:
        resolved_meta = dict(audio_metadata or {})
        resolved_meta.setdefault("channel_layout_hint", detect_signal_layout(int(resolved_meta.get("channels", 1)), config.input_path))
        resolved_meta.setdefault(
            "spatial_metadata",
            infer_spatial_metadata(
                int(resolved_meta.get("channels", 1)),
                config.input_path,
                source_channel_layout=(
                    str(resolved_meta.get("decoder", {}).get("ffprobe", {}).get("channel_layout"))
                    if isinstance(resolved_meta.get("decoder"), dict)
                    and isinstance(resolved_meta.get("decoder", {}).get("ffprobe"), dict)
                    and resolved_meta.get("decoder", {}).get("ffprobe", {}).get("channel_layout") is not None
                    else None
                ),
            ).to_dict(),
        )
        resolved_meta.setdefault("decoder", resolved_meta.pop("decoder_provenance", None))
        channel_summary = channel_summary or {"channels": [], "aggregate": {}, "aggregation_rules": {}}
        validity_flags = validity_flags or _validity_flags_from_summary(
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
            "selected_metrics": list(selected_metrics),
            "count": len(selected_metrics),
        },
        "library_versions": lib_versions,
        "artifacts": canonicalize(artifacts or {}),
        "metadata": {
            **canonicalize(resolved_meta),
            "frame_size": config.frame_size,
            "hop_size": config.hop_size,
            "seed": config.seed,
            "compute_device": device_resolution_dict(device_info),
            "project": config.project,
            "variant": config.variant,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            },
            "config_snapshot": canonicalize(asdict(config)),
            "resolved_metric_list": list(selected_metrics),
            "metric_catalog_version": METRIC_CATALOG_VERSION,
            "channel_metrics": channel_summary,
            "validity_flags": validity_flags,
            "calibration": calibration_to_dict(config.calibration),
            "assumptions": assumptions,
            "warnings": warnings or [],
            "analysis_strategy": {
                "out_of_core": bool(mode == "streaming"),
                "summary_statistics": RUNNING_SUMMARY_METHOD if mode == "streaming" else "exact_from_materialized_signal",
                "store_series_in_json": bool(not config.summary_only),
                "max_series_points": config.max_series_points,
                "frame_table_csv": str(config.frame_table_csv) if config.frame_table_csv else None,
                "checkpoint_dir": str(config.checkpoint_dir) if config.checkpoint_dir else None,
                "resume": bool(config.resume),
                **canonicalize(analysis_strategy or {}),
            },
        },
        "metrics": metric_payload,
    }
    return result


def _analyze_full(config: AnalysisConfig, registry: MetricRegistry, metric_names: list[str]) -> dict[str, Any]:
    audio = read_audio(config.input_path, target_sr=config.sample_rate)
    ctx = AnalysisContext(audio=audio, config=config, calibration=config.calibration)
    metric_results = registry.compute(ctx, metric_names)
    return _assemble_result(
        config,
        audio=audio,
        audio_metadata=None,
        metrics=metric_results,
        registry=registry,
        mode="full",
        selected_metrics=metric_names,
    )


def _load_stream_state(
    config: AnalysisConfig,
    metric_names: list[str],
) -> tuple[int, AudioAccumulator, dict[str, MetricAccumulator]]:
    checkpoint_path = _checkpoint_path(config)
    if checkpoint_path is None or not config.resume or not checkpoint_path.exists():
        metric_acc = {
            name: MetricAccumulator(
                name=name,
                store_series=not config.summary_only,
                max_series_points=config.max_series_points,
            )
            for name in metric_names
        }
        return 0, AudioAccumulator(), metric_acc

    payload = load_checkpoint(checkpoint_path)
    resume_from = int(payload.get("next_chunk_index", 0))
    audio_acc = AudioAccumulator.from_dict(payload.get("audio_accumulator", {}))
    metric_acc = {
        name: MetricAccumulator.from_dict(payload["metric_accumulators"][name])
        for name in metric_names
        if isinstance(payload.get("metric_accumulators", {}).get(name), dict)
    }
    for name in metric_names:
        metric_acc.setdefault(
            name,
            MetricAccumulator(
                name=name,
                store_series=not config.summary_only,
                max_series_points=config.max_series_points,
            ),
        )
    return resume_from, audio_acc, metric_acc


def _save_stream_state(
    config: AnalysisConfig,
    *,
    next_chunk_index: int,
    metric_names: list[str],
    audio_acc: AudioAccumulator,
    metric_acc: dict[str, MetricAccumulator],
    frame_table_csv: Path | None,
) -> Path | None:
    checkpoint_path = _checkpoint_path(config)
    if checkpoint_path is None:
        return None
    payload = {
        "input_path": str(Path(config.input_path).resolve()),
        "config_hash": config_hash(config),
        "metric_names": list(metric_names),
        "next_chunk_index": int(next_chunk_index),
        "audio_accumulator": audio_acc.to_dict(),
        "metric_accumulators": {name: acc.to_dict() for name, acc in metric_acc.items()},
        "frame_table_csv": str(frame_table_csv) if frame_table_csv else None,
    }
    return save_checkpoint(checkpoint_path, payload)


def _analyze_streaming(config: AnalysisConfig, registry: MetricRegistry, metric_names: list[str]) -> dict[str, Any]:
    probe = probe_audio_metadata(config.input_path)
    format_name = str(probe.get("format_name") or Path(config.input_path).suffix.lstrip(".").upper() or "unknown")
    subtype = probe.get("subtype")
    source_backend = str(probe.get("backend") or "unknown")
    decoder_provenance = probe.get("decoder_provenance") or {
        "decoder_used": source_backend,
        "ffmpeg_version": None,
        "ffprobe": None,
    }

    frame_writer: FrameTableCsvWriter | None = None
    frame_parquet_writer: FrameTableParquetDatasetWriter | None = None
    frame_hdf5_writer: FrameTableHdf5Writer | None = None
    if config.frame_table_csv is not None:
        frame_writer = FrameTableCsvWriter(config.frame_table_csv, append=bool(config.resume))
    if config.frame_table_parquet_dir is not None:
        frame_parquet_writer = FrameTableParquetDatasetWriter(config.frame_table_parquet_dir, append=bool(config.resume))
    if config.frame_table_hdf5 is not None:
        frame_hdf5_writer = FrameTableHdf5Writer(config.frame_table_hdf5, append=bool(config.resume))

    resume_from, audio_acc, metric_acc = _load_stream_state(config=config, metric_names=metric_names)
    processed_chunks = 0
    for chunk in stream_audio(
        config.input_path,
        chunk_size=int(config.chunk_size or 131072),
        target_sr=config.sample_rate,
    ):
        if chunk.index < resume_from:
            continue
        chunk_buffer = AudioBuffer(
            samples=chunk.samples,
            sample_rate=chunk.sample_rate,
            source_path=str(config.input_path),
            format_name=format_name,
            subtype=str(subtype) if subtype is not None else None,
            source_backend=source_backend,
            decoder_provenance=dict(decoder_provenance),
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
            compute_device=config.compute_device,
            make_plots=False,
            ml_export=False,
            summary_only=config.summary_only,
            streamable_only=config.streamable_only,
            allow_full_read=config.allow_full_read,
            max_series_points=config.max_series_points,
            frame_table_csv=config.frame_table_csv,
            checkpoint_dir=config.checkpoint_dir,
            resume=config.resume,
        )
        ctx = AnalysisContext(audio=chunk_buffer, config=chunk_cfg, calibration=config.calibration)
        chunk_metrics = registry.compute(ctx, metric_names)
        chunk_offset = chunk.start_sample / float(chunk.sample_rate)
        audio_acc.update(chunk.samples, chunk.sample_rate)
        for name, result in chunk_metrics.items():
            metric_acc[name].update(result, chunk_offset_s=chunk_offset)
        if frame_writer is not None:
            frame_writer.append_chunk(chunk_metrics, chunk_offset_s=chunk_offset)
        if frame_parquet_writer is not None:
            frame_parquet_writer.append_chunk(chunk_metrics, chunk_offset_s=chunk_offset)
        if frame_hdf5_writer is not None:
            frame_hdf5_writer.append_chunk(chunk_metrics, chunk_offset_s=chunk_offset)
        processed_chunks += 1
        _save_stream_state(
            config,
            next_chunk_index=chunk.index + 1,
            metric_names=metric_names,
            audio_acc=audio_acc,
            metric_acc=metric_acc,
            frame_table_csv=config.frame_table_csv,
        )

    if frame_writer is not None:
        frame_writer.close()

    if processed_chunks == 0 and audio_acc.total_samples == 0:
        raise RuntimeError("No audio chunks produced for streaming analysis.")

    metric_results = {name: acc.finalize() for name, acc in metric_acc.items()}
    channel_summary = audio_acc.channel_summary()
    resolved_sample_rate = int(audio_acc.sample_rate or config.sample_rate or probe.get("sample_rate") or 0)
    resolved_num_samples = int(audio_acc.total_samples)
    resolved_duration = float(resolved_num_samples / resolved_sample_rate) if resolved_sample_rate > 0 else 0.0

    artifacts: dict[str, Any] = {}
    warnings = ["Streaming mode merged chunk-local frame metrics without full-file materialization."]
    if config.summary_only:
        warnings.append("Series payloads were omitted from JSON; use frame table artifacts for long-form rows.")
    if config.frame_table_csv is not None:
        artifacts["frame_table_csv"] = str(Path(config.frame_table_csv).resolve())
        artifacts["frame_table_metadata_json"] = str(
            Path(config.frame_table_csv).with_suffix(Path(config.frame_table_csv).suffix + ".meta.json").resolve()
        )
        meta_payload = frame_writer.metadata_payload(
            frame_size=config.frame_size,
            hop_size=config.hop_size,
            source_channels=int(probe.get("channels") or audio_acc.channels or 1),
            esl_version=__version__,
        ) if frame_writer is not None else {
            "frame_table_version": "1.0.0",
            "frame_size": int(config.frame_size),
            "hop_size": int(config.hop_size),
            "source_channels": int(probe.get("channels") or audio_acc.channels or 1),
            "tensor_layout": "[channels, frames, features]",
            "channel_feature_mode": "aggregate_mixdown",
            "esl_version": __version__,
            "columns": ["timestamp_s"],
        }
        Path(artifacts["frame_table_metadata_json"]).write_text(
            json.dumps(canonicalize(meta_payload), indent=2),
            encoding="utf-8",
        )
    if config.frame_table_parquet_dir is not None:
        artifacts["frame_table_parquet_dir"] = str(Path(config.frame_table_parquet_dir).resolve())
        artifacts["frame_table_parquet_metadata_json"] = str(
            (Path(config.frame_table_parquet_dir) / "metadata.json").resolve()
        )
        if frame_parquet_writer is not None:
            frame_parquet_writer.metadata_payload(
                frame_size=config.frame_size,
                hop_size=config.hop_size,
                source_channels=int(probe.get("channels") or audio_acc.channels or 1),
                esl_version=__version__,
            )
    if config.frame_table_hdf5 is not None:
        artifacts["frame_table_hdf5"] = str(Path(config.frame_table_hdf5).resolve())
        if frame_hdf5_writer is not None:
            frame_hdf5_writer.metadata_payload(
                frame_size=config.frame_size,
                hop_size=config.hop_size,
                source_channels=int(probe.get("channels") or audio_acc.channels or 1),
                esl_version=__version__,
            )
    if frame_parquet_writer is not None:
        frame_parquet_writer.close()
    if frame_hdf5_writer is not None:
        frame_hdf5_writer.close()
    checkpoint_path = _checkpoint_path(config)
    if checkpoint_path is not None:
        artifacts["checkpoint_state_json"] = str(checkpoint_path.resolve())

    audio_metadata = {
        "input_path": str(Path(config.input_path).resolve()),
        "sample_rate": resolved_sample_rate,
        "num_samples": resolved_num_samples,
        "channels": int(audio_acc.channels or probe.get("channels") or 1),
        "duration_s": resolved_duration,
        "source_sample_rate": int(probe.get("sample_rate") or resolved_sample_rate),
        "source_num_samples": probe.get("num_samples"),
        "source_duration_s": probe.get("duration_s"),
        "format_name": probe.get("format_name"),
        "subtype": probe.get("subtype"),
        "backend": source_backend,
        "decoder_provenance": decoder_provenance,
        "channel_layout_hint": detect_signal_layout(int(audio_acc.channels or probe.get("channels") or 1), config.input_path),
    }
    validity = _validity_flags_from_summary(
        channel_summary=channel_summary,
        calibration_applied=config.calibration is not None,
        metrics=metric_results,
    )
    if checkpoint_path is not None and checkpoint_path.exists():
        warnings.append("Checkpoint state was written during analysis and can be reused with --resume.")
    return _assemble_result(
        config,
        audio=None,
        audio_metadata=audio_metadata,
        metrics=metric_results,
        registry=registry,
        mode="streaming",
        selected_metrics=metric_names,
        warnings=warnings,
        channel_summary=channel_summary,
        validity_flags=validity,
        artifacts=artifacts,
        analysis_strategy={
            "processed_chunks": int(processed_chunks),
            "resume_from_chunk": int(resume_from),
            "streaming_capable_metrics_only": True,
        },
    )


def analyze(config: AnalysisConfig, registry: MetricRegistry | None = None) -> dict[str, Any]:
    """Run analysis and return serializable result document."""
    set_seed(config.seed)
    reg = registry or create_registry(with_external=True)
    metric_names, dropped = _resolve_metric_names(config, reg)
    if not metric_names:
        raise RuntimeError("No metrics selected for analysis after applying streamability filters.")

    if config.chunk_size:
        non_streaming = [name for name in metric_names if not reg.get(name).spec.streaming_capable]
        if non_streaming:
            if config.allow_full_read:
                return _analyze_full(config, reg, metric_names)
            joined = ", ".join(non_streaming)
            raise RuntimeError(
                "Chunked analysis requested, but these metrics require full-file context: "
                f"{joined}. Use --streamable-only, choose streamable metrics explicitly, or pass --allow-full-read."
            )
        result = _analyze_streaming(config, reg, metric_names)
        warnings = list(result.get("metadata", {}).get("warnings", []))
        if dropped:
            warnings.append("Dropped non-streaming metrics because --streamable-only was enabled: " + ", ".join(dropped))
            result["metadata"]["warnings"] = warnings
        return result

    result = _analyze_full(config, reg, metric_names)
    if dropped:
        warnings = list(result.get("metadata", {}).get("warnings", []))
        warnings.append("Dropped non-streaming metrics because --streamable-only was enabled: " + ", ".join(dropped))
        result["metadata"]["warnings"] = warnings
    return result
