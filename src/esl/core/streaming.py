"""Streaming-friendly analysis loop with threshold-based alerting.

This module keeps chunk details on disk so long-duration scans do not need to
retain every chunk summary in memory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from esl.core.audio import AudioBuffer, probe_audio_metadata, stream_audio
from esl.core.config import AnalysisConfig, CalibrationProfile
from esl.core.context import AnalysisContext
from esl.core.out_of_core import RunningStats, load_checkpoint, save_checkpoint
from esl.metrics.registry import MetricRegistry, create_registry


CHUNK_REPORT_FILENAME = "stream_chunks.jsonl"
CHECKPOINT_FILENAME = "stream_state.json"


@dataclass(slots=True)
class StreamRunConfig:
    input_path: Path
    output_dir: Path
    metrics: list[str]
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int = 131072
    calibration: CalibrationProfile | None = None
    seed: int = 42
    rules_path: str | None = None
    max_chunks: int | None = None
    checkpoint_dir: Path | None = None
    resume: bool = False
    chunks_jsonl: Path | None = None
    max_chunks_in_report: int = 256


def _load_rules(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stream alert rules file not found: {p}")
    raw_text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML stream rules require pyyaml") from exc
        payload = yaml.safe_load(raw_text) or {}
    else:
        payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stream alert rules must be an object: {p}")
    return payload


def _metric_mean(metric_payload: dict[str, Any]) -> float | None:
    summary = metric_payload.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _checkpoint_path(cfg: StreamRunConfig) -> Path | None:
    if cfg.checkpoint_dir is None:
        return None
    return Path(cfg.checkpoint_dir) / CHECKPOINT_FILENAME


def _load_state(cfg: StreamRunConfig, metric_names: list[str]) -> tuple[int, int, int, dict[str, RunningStats]]:
    checkpoint_path = _checkpoint_path(cfg)
    if checkpoint_path is None or not cfg.resume or not checkpoint_path.exists():
        return 0, 0, 0, {name: RunningStats() for name in metric_names}
    payload = load_checkpoint(checkpoint_path)
    metric_stats = {
        name: RunningStats.from_dict(stats)
        for name, stats in payload.get("metric_stats", {}).items()
        if name in metric_names and isinstance(stats, dict)
    }
    for name in metric_names:
        metric_stats.setdefault(name, RunningStats())
    return (
        int(payload.get("next_chunk_index", 0)),
        int(payload.get("chunks_processed", 0)),
        int(payload.get("alert_count", 0)),
        metric_stats,
    )


def _save_state(
    cfg: StreamRunConfig,
    *,
    next_chunk_index: int,
    chunks_processed: int,
    alert_count: int,
    metric_stats: dict[str, RunningStats],
    chunks_jsonl: Path,
) -> Path | None:
    checkpoint_path = _checkpoint_path(cfg)
    if checkpoint_path is None:
        return None
    payload = {
        "input_path": str(cfg.input_path.resolve()),
        "next_chunk_index": int(next_chunk_index),
        "chunks_processed": int(chunks_processed),
        "alert_count": int(alert_count),
        "metric_stats": {name: stats.to_dict() for name, stats in metric_stats.items()},
        "chunks_jsonl": str(chunks_jsonl.resolve()),
    }
    return save_checkpoint(checkpoint_path, payload)


def run_stream_analysis(
    cfg: StreamRunConfig,
    registry: MetricRegistry | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run chunk-based streaming analysis and emit alert report artifacts."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rules = _load_rules(cfg.rules_path)
    metric_rules = rules.get("metric_thresholds", {})
    if metric_rules and not isinstance(metric_rules, dict):
        raise RuntimeError("rules.metric_thresholds must be an object")

    selected_metrics = list(cfg.metrics)
    if not selected_metrics and isinstance(metric_rules, dict):
        selected_metrics = [str(x) for x in metric_rules.keys()]
    if not selected_metrics:
        selected_metrics = ["rms_dbfs", "ndsi", "novelty_curve"]

    reg = registry or create_registry(with_external=True)
    for m in selected_metrics:
        reg.get(m)

    probe = probe_audio_metadata(cfg.input_path)
    format_name = str(probe.get("format_name") or cfg.input_path.suffix.lstrip(".").upper() or "unknown")
    subtype = probe.get("subtype")
    source_backend = str(probe.get("backend") or "unknown")
    decoder_provenance = probe.get("decoder_provenance") or {
        "decoder_used": source_backend,
        "ffmpeg_version": None,
        "ffprobe": None,
    }

    chunks_jsonl = Path(cfg.chunks_jsonl) if cfg.chunks_jsonl else (cfg.output_dir / CHUNK_REPORT_FILENAME)
    chunks_jsonl.parent.mkdir(parents=True, exist_ok=True)
    chunks_mode = "a" if cfg.resume and chunks_jsonl.exists() else "w"
    resume_from, chunks_processed, alert_count, metric_stats = _load_state(cfg, selected_metrics)
    alerts_csv = cfg.output_dir / "stream_alerts.csv"
    alerts_mode = "a" if cfg.resume and alerts_csv.exists() else "w"
    in_memory_preview: list[dict[str, Any]] = []
    chunk_preview_complete = True

    with chunks_jsonl.open(chunks_mode, encoding="utf-8") as chunk_f, alerts_csv.open(
        alerts_mode,
        encoding="utf-8",
        newline="",
    ) as alerts_f:
        alerts_writer = csv.DictWriter(
            alerts_f,
            fieldnames=["chunk_index", "metric", "value", "condition", "threshold"],
        )
        if alerts_mode == "w" or alerts_csv.stat().st_size == 0:
            alerts_writer.writeheader()

        for idx, chunk in enumerate(
            stream_audio(cfg.input_path, chunk_size=cfg.chunk_size, target_sr=cfg.sample_rate)
        ):
            if idx < resume_from:
                continue
            if cfg.max_chunks is not None and chunks_processed >= cfg.max_chunks:
                break

            chunk_buffer = AudioBuffer(
                samples=chunk.samples,
                sample_rate=chunk.sample_rate,
                source_path=str(cfg.input_path),
                format_name=format_name,
                subtype=str(subtype) if subtype is not None else None,
                source_backend=source_backend,
                decoder_provenance=dict(decoder_provenance),
            )
            chunk_cfg = AnalysisConfig(
                input_path=cfg.input_path,
                output_dir=cfg.output_dir,
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate,
                chunk_size=cfg.chunk_size,
                metrics=selected_metrics,
                calibration=cfg.calibration,
                verbosity=0,
                debug=0,
                seed=cfg.seed,
                compute_device="auto",
                summary_only=True,
            )
            ctx = AnalysisContext(audio=chunk_buffer, config=chunk_cfg, calibration=cfg.calibration)
            metric_results = reg.compute(ctx, selected_metrics)

            metric_map: dict[str, dict[str, Any]] = {}
            metric_means: dict[str, float | None] = {}
            for name in selected_metrics:
                res = metric_results[name]
                payload = {
                    "summary": res.summary,
                    "confidence": res.confidence,
                    "units": res.units,
                }
                metric_map[name] = payload
                metric_means[name] = _metric_mean(payload)
                value = metric_means[name]
                if value is not None:
                    metric_stats[name].update([value])

            chunk_alerts: list[dict[str, Any]] = []
            if isinstance(metric_rules, dict):
                for metric_name, threshold in metric_rules.items():
                    m_name = str(metric_name)
                    thr = threshold if isinstance(threshold, dict) else {}
                    min_val = thr.get("min")
                    max_val = thr.get("max")
                    value = metric_means.get(m_name)
                    if value is None:
                        continue
                    if isinstance(min_val, (int, float)) and value < float(min_val):
                        alert = {
                            "chunk_index": idx,
                            "metric": m_name,
                            "value": value,
                            "condition": "min",
                            "threshold": float(min_val),
                        }
                        chunk_alerts.append(alert)
                        alerts_writer.writerow(alert)
                        alert_count += 1
                    if isinstance(max_val, (int, float)) and value > float(max_val):
                        alert = {
                            "chunk_index": idx,
                            "metric": m_name,
                            "value": value,
                            "condition": "max",
                            "threshold": float(max_val),
                        }
                        chunk_alerts.append(alert)
                        alerts_writer.writerow(alert)
                        alert_count += 1

            chunk_start_s = float(chunk.start_sample / chunk.sample_rate)
            chunk_end_s = float((chunk.start_sample + chunk.samples.shape[0]) / chunk.sample_rate)
            record = {
                "index": idx,
                "start_s": chunk_start_s,
                "end_s": chunk_end_s,
                "num_samples": int(chunk.samples.shape[0]),
                "metric_means": metric_means,
                "metrics": metric_map,
                "alerts": chunk_alerts,
            }
            chunk_f.write(json.dumps(record) + "\n")
            chunk_f.flush()
            if len(in_memory_preview) < max(int(cfg.max_chunks_in_report), 0):
                in_memory_preview.append(record)
            else:
                chunk_preview_complete = False

            chunks_processed += 1
            _save_state(
                cfg,
                next_chunk_index=idx + 1,
                chunks_processed=chunks_processed,
                alert_count=alert_count,
                metric_stats=metric_stats,
                chunks_jsonl=chunks_jsonl,
            )

    report = {
        "mode": "file_stream",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(cfg.input_path.resolve()),
        "sample_rate": int(cfg.sample_rate or probe.get("sample_rate") or 0),
        "channels": int(probe.get("channels") or 1),
        "chunk_size": int(cfg.chunk_size),
        "metrics": selected_metrics,
        "rules": rules,
        "chunks_processed": int(chunks_processed),
        "alert_count": int(alert_count),
        "alerts": [],
        "chunks": in_memory_preview,
        "chunk_details_complete": bool(chunk_preview_complete),
        "decoder_provenance": decoder_provenance,
        "source_duration_s": probe.get("duration_s"),
        "aggregate_metric_means": {
            name: (stats.summary().get("mean") if stats.count > 0 else None)
            for name, stats in metric_stats.items()
        },
        "artifacts": {
            "report_json": str((cfg.output_dir / "stream_report.json").resolve()),
            "alerts_csv": str(alerts_csv.resolve()),
            "chunks_jsonl": str(chunks_jsonl.resolve()),
            "checkpoint_state_json": str(_checkpoint_path(cfg).resolve()) if _checkpoint_path(cfg) else None,
        },
    }

    report_path = cfg.output_dir / "stream_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report
