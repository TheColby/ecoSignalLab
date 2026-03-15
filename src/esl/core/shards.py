"""Shard-manifest workflows for long-duration archives.

This module treats an audio archive as an ordered manifest of shard files
instead of one monolithic recording. That keeps long deployments operationally
manageable while preserving a cumulative timeline.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from esl.core import AnalysisConfig, analyze, load_calibration
from esl.core.audio import iter_supported_files, probe_audio_metadata
from esl.io import save_json
from esl.metrics.registry import create_registry


SHARD_MANIFEST_VERSION = "1.0.0"
SUPPORTED_PATTERNS = (
    "*.wav",
    "*.flac",
    "*.aiff",
    "*.aif",
    "*.rf64",
    "*.caf",
    "*.mp3",
    "*.aac",
    "*.ogg",
    "*.opus",
    "*.wma",
    "*.alac",
    "*.m4a",
    "*.sofa",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_") or "shard"


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


@dataclass(slots=True)
class ShardManifestConfig:
    input_dir: Path
    output_path: Path
    recursive: bool = True
    patterns: tuple[str, ...] = SUPPORTED_PATTERNS
    order_by: str = "path"  # path|mtime


@dataclass(slots=True)
class ShardAnalyzeConfig:
    manifest_path: Path
    output_dir: Path
    calibration_path: str | None = None
    metrics: list[str] = field(default_factory=list)
    report_metrics: list[str] = field(default_factory=list)
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int | None = None
    seed: int = 42
    compute_device: str = "auto"
    summary_only: bool = False
    streamable_only: bool = False
    allow_full_read: bool = False
    max_series_points: int | None = None
    frame_table_dir: Path | None = None
    checkpoint_root: Path | None = None
    resume: bool = False
    force: bool = False


def build_shard_manifest(cfg: ShardManifestConfig) -> tuple[Path, dict[str, Any]]:
    """Create an ordered shard manifest from a directory of audio files."""
    files = iter_supported_files(cfg.input_dir, patterns=cfg.patterns, recursive=cfg.recursive)
    if cfg.order_by == "mtime":
        files = sorted(files, key=lambda p: (p.stat().st_mtime, str(p.resolve())))
    else:
        files = sorted(files, key=lambda p: str(p.relative_to(cfg.input_dir.resolve())))

    items: list[dict[str, Any]] = []
    cumulative_start_s = 0.0
    total_size_bytes = 0
    root = cfg.input_dir.resolve()

    for idx, fp in enumerate(files):
        meta = probe_audio_metadata(fp)
        duration_s = float(meta.get("duration_s") or 0.0)
        size_bytes = int(meta.get("size_bytes") or 0)
        start_s = cumulative_start_s
        end_s = cumulative_start_s + duration_s
        cumulative_start_s = end_s
        total_size_bytes += size_bytes
        rel = fp.resolve().relative_to(root)
        items.append(
            {
                "shard_index": idx,
                "path": str(fp.resolve()),
                "relative_path": str(rel),
                "name": fp.name,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration_s,
                "size_bytes": size_bytes,
                "size_gb": float(size_bytes / 1_000_000_000.0),
                "sample_rate": meta.get("sample_rate"),
                "channels": meta.get("channels"),
                "num_samples": meta.get("num_samples"),
                "format_name": meta.get("format_name"),
                "subtype": meta.get("subtype"),
                "backend": meta.get("backend"),
                "codec_name": meta.get("codec_name"),
                "channel_layout": meta.get("channel_layout"),
                "mtime_utc": datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )

    manifest = {
        "schema_version": SHARD_MANIFEST_VERSION,
        "created_utc": _now_utc(),
        "root_dir": str(root),
        "order_by": cfg.order_by,
        "recursive": bool(cfg.recursive),
        "patterns": list(cfg.patterns),
        "num_shards": len(items),
        "total_duration_s": float(cumulative_start_s),
        "total_size_bytes": int(total_size_bytes),
        "total_size_gb": float(total_size_bytes / 1_000_000_000.0),
        "items": items,
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return cfg.output_path, manifest


def load_shard_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RuntimeError(f"Invalid shard manifest: {p}")
    return payload


def _analysis_json_path(base_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    return (base_dir / rel.parent / f"{rel.stem}.json").resolve()


def _frame_table_path(base_dir: Path | None, relative_path: str) -> Path | None:
    if base_dir is None:
        return None
    rel = Path(relative_path)
    return (base_dir / rel.parent / f"{rel.stem}_frame_table.csv").resolve()


def _checkpoint_dir(root: Path | None, relative_path: str) -> Path | None:
    if root is None:
        return None
    rel = Path(relative_path)
    return (root / rel.parent / _safe_name(rel.stem)).resolve()


def run_shard_analysis(cfg: ShardAnalyzeConfig) -> tuple[Path, dict[str, Any]]:
    """Analyze a shard manifest as one ordered archive."""
    manifest = load_shard_manifest(cfg.manifest_path)
    items = [item for item in manifest.get("items", []) if isinstance(item, dict)]
    if not items:
        raise RuntimeError(f"No shard items found in manifest: {cfg.manifest_path}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    reg = create_registry(with_external=True)
    calibration = load_calibration(cfg.calibration_path) if cfg.calibration_path else None

    report_metrics = list(dict.fromkeys(cfg.report_metrics or cfg.metrics or ["rms_dbfs", "snr_db", "ndsi"]))
    rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    errors = 0
    weighted_metric_sums: dict[str, float] = {name: 0.0 for name in report_metrics}
    weighted_metric_durations: dict[str, float] = {name: 0.0 for name in report_metrics}

    for item in items:
        path = Path(str(item["path"]))
        relative_path = str(item.get("relative_path") or path.name)
        json_path = _analysis_json_path(cfg.output_dir / "shards", relative_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        status = "processed"
        result: dict[str, Any] | None = None
        if json_path.exists() and not cfg.force:
            try:
                result = json.loads(json_path.read_text(encoding="utf-8"))
                status = "skipped"
                skipped += 1
            except Exception:
                result = None

        if result is None:
            try:
                result = analyze(
                    AnalysisConfig(
                        input_path=path,
                        output_dir=json_path.parent,
                        frame_size=cfg.frame_size,
                        hop_size=cfg.hop_size,
                        sample_rate=cfg.sample_rate,
                        chunk_size=cfg.chunk_size,
                        metrics=list(cfg.metrics),
                        calibration=calibration,
                        verbosity=0,
                        debug=0,
                        seed=cfg.seed,
                        compute_device=cfg.compute_device,
                        summary_only=cfg.summary_only,
                        streamable_only=cfg.streamable_only,
                        allow_full_read=cfg.allow_full_read,
                        max_series_points=cfg.max_series_points,
                        frame_table_csv=_frame_table_path(cfg.frame_table_dir, relative_path),
                        checkpoint_dir=_checkpoint_dir(cfg.checkpoint_root, relative_path),
                        resume=cfg.resume,
                    ),
                    registry=reg,
                )
                save_json(result, json_path)
                processed += 1
            except Exception as exc:
                errors += 1
                rows.append(
                    {
                        "shard_index": int(item.get("shard_index", -1)),
                        "relative_path": relative_path,
                        "input": str(path),
                        "json": str(json_path),
                        "status": "error",
                        "error": str(exc),
                        "timeline_start_s": float(item.get("start_s") or 0.0),
                        "timeline_end_s": float(item.get("end_s") or 0.0),
                    }
                )
                continue

        duration_s = float(result.get("metadata", {}).get("duration_s") or item.get("duration_s") or 0.0)
        metrics_payload = result.get("metrics", {})
        row: dict[str, Any] = {
            "shard_index": int(item.get("shard_index", -1)),
            "relative_path": relative_path,
            "input": str(path),
            "json": str(json_path),
            "status": status,
            "timeline_start_s": float(item.get("start_s") or 0.0),
            "timeline_end_s": float(item.get("end_s") or 0.0),
            "duration_s": duration_s,
            "channels": int(result.get("metadata", {}).get("channels") or item.get("channels") or 0),
            "sample_rate": int(result.get("metadata", {}).get("sample_rate") or item.get("sample_rate") or 0),
            "analysis_mode": result.get("analysis_mode"),
        }
        for metric_name in report_metrics:
            mean_v = None
            if isinstance(metrics_payload, dict):
                payload = metrics_payload.get(metric_name, {})
                if isinstance(payload, dict):
                    summary = payload.get("summary", {})
                    if isinstance(summary, dict):
                        mean_v = _as_float(summary.get("mean"))
            row[f"{metric_name}_mean"] = mean_v
            if mean_v is not None and duration_s > 0.0:
                weighted_metric_sums[metric_name] += float(mean_v) * duration_s
                weighted_metric_durations[metric_name] += duration_s
        rows.append(row)

    index_csv = (cfg.output_dir / "shard_analysis_index.csv").resolve()
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shard_index",
        "relative_path",
        "input",
        "json",
        "status",
        "timeline_start_s",
        "timeline_end_s",
        "duration_s",
        "channels",
        "sample_rate",
        "analysis_mode",
        *[f"{name}_mean" for name in report_metrics],
        "error",
    ]
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    weighted_metric_means = {
        name: (
            float(weighted_metric_sums[name] / weighted_metric_durations[name])
            if weighted_metric_durations[name] > 0.0
            else None
        )
        for name in report_metrics
    }
    report = {
        "created_utc": _now_utc(),
        "manifest_path": str(Path(cfg.manifest_path).resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "num_shards": int(len(items)),
        "processed": int(processed),
        "skipped": int(skipped),
        "errors": int(errors),
        "archive_duration_s": float(manifest.get("total_duration_s") or 0.0),
        "archive_size_gb": float(manifest.get("total_size_gb") or 0.0),
        "report_metrics": report_metrics,
        "weighted_metric_means": weighted_metric_means,
        "rows": rows,
        "artifacts": {
            "index_csv": str(index_csv),
        },
    }
    report_path = (cfg.output_dir / "shard_analysis_report.json").resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["artifacts"]["report_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report
