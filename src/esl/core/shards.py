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

import numpy as np

from esl.core import AnalysisConfig, analyze, load_calibration
from esl.core.similarity import (
    SimilaritySearchConfig,
    _aggregate_feature_vector,
    _distance,
    _metric_vector,
    _mode as _similarity_mode,
)
from esl.core.audio import iter_supported_files, probe_audio_metadata
from esl.core.moments import (
    _clip_with_ffmpeg,
    _clip_with_soundfile,
    _codec_from_subtype,
    _collect_windows,
    _ffmpeg_available,
    _iter_stream_chunks,
    _load_stream_report,
    _read_segment,
    _rerank_windows,
    _sec_to_hms,
    _select_windows,
)
from esl.core.streaming import StreamRunConfig, run_stream_analysis
from esl.io import save_json
from esl.metrics.registry import create_registry
from esl.viz.feature_vectors import extract_feature_vectors, extract_feature_vectors_from_array


SHARD_MANIFEST_VERSION = "1.0.0"
DEFAULT_SPATIAL_SIMILARITY_METRICS = (
    "interchannel_coherence",
    "iacc",
    "ild_db",
    "itd_s",
    "doa_azimuth_proxy_deg",
    "ambisonic_diffuseness",
    "ambisonic_energy_vector_azimuth_deg",
    "ambisonic_energy_vector_elevation_deg",
)
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
    frame_table_parquet_root: Path | None = None
    frame_table_hdf5_root: Path | None = None
    checkpoint_root: Path | None = None
    resume: bool = False
    force: bool = False


@dataclass(slots=True)
class ShardMomentsConfig:
    manifest_path: Path
    output_dir: Path
    stream_root: Path | None = None
    rules_path: str | None = None
    metrics: list[str] = field(default_factory=list)
    calibration_path: str | None = None
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int = 131072
    seed: int = 42
    max_chunks: int | None = None
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0
    merge_gap_s: float = 2.0
    min_alerts_per_chunk: int = 1
    selection_mode: str = "all"
    top_k: int | None = None
    rank_metric: str = "novelty_curve"
    rank_scope: str = "downmix"
    event_window_s: float | None = None
    window_before_s: float | None = None
    window_after_s: float | None = None
    resume: bool = False
    force_stream: bool = False
    report_path: Path | None = None


@dataclass(slots=True)
class ShardSimilarConfig:
    manifest_path: Path
    query_path: Path
    output_dir: Path
    top_k: int = 5
    mode: str = "auto"
    metric: str = "novelty_curve"
    metrics: list[str] = field(default_factory=list)
    distance: str = "cosine"
    feature_set: str = "auto"
    frame_size: int = 1024
    hop_size: int = 256
    sample_rate: int | None = None
    normalize: bool = True
    calibration_path: str | None = None
    seed: int = 42
    include_query_if_present: bool = False
    max_shards: int | None = None
    spatial_mode: str = "off"  # off|append|only
    spatial_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_SPATIAL_SIMILARITY_METRICS))
    spatial_weight: float = 0.5


@dataclass(slots=True)
class ShardRetrieveConfig:
    manifest_path: Path
    query_path: Path
    output_dir: Path
    top_k: int = 10
    window_s: float = 10.0
    hop_s: float = 5.0
    feature_set: str = "core"
    distance: str = "cosine"
    frame_size: int = 1024
    hop_size: int = 256
    sample_rate: int | None = None
    max_shards: int | None = None
    write_clips: bool = True


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


def _frame_table_parquet_dir(base_dir: Path | None, relative_path: str) -> Path | None:
    if base_dir is None:
        return None
    rel = Path(relative_path)
    return (base_dir / rel.parent / f"{rel.stem}_frame_table.parquet").resolve()


def _frame_table_hdf5_path(base_dir: Path | None, relative_path: str) -> Path | None:
    if base_dir is None:
        return None
    rel = Path(relative_path)
    return (base_dir / rel.parent / f"{rel.stem}_frame_table.h5").resolve()


def _checkpoint_dir(root: Path | None, relative_path: str) -> Path | None:
    if root is None:
        return None
    rel = Path(relative_path)
    return (root / rel.parent / _safe_name(rel.stem)).resolve()


def _resolve_spatial_metrics(cfg: ShardSimilarConfig) -> list[str]:
    names = [name for name in cfg.spatial_metrics if name]
    return list(dict.fromkeys(names or list(DEFAULT_SPATIAL_SIMILARITY_METRICS)))


def _shared_finite_vectors(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return None
    return a[mask], b[mask]


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
                        frame_table_parquet_dir=_frame_table_parquet_dir(cfg.frame_table_parquet_root, relative_path),
                        frame_table_hdf5=_frame_table_hdf5_path(cfg.frame_table_hdf5_root, relative_path),
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


def _stream_report_path(stream_root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    return (stream_root / rel.parent / rel.stem / "stream_report.json").resolve()


def run_shard_similarity(cfg: ShardSimilarConfig) -> tuple[Path, dict[str, Any]]:
    """Rank manifest shards by similarity to a query file."""
    manifest = load_shard_manifest(cfg.manifest_path)
    items = [item for item in manifest.get("items", []) if isinstance(item, dict)]
    if not items:
        raise RuntimeError(f"No shard items found in manifest: {cfg.manifest_path}")
    if not cfg.query_path.exists():
        raise FileNotFoundError(f"Query file not found: {cfg.query_path}")
    if int(cfg.top_k) < 1:
        raise ValueError("top_k must be >= 1")

    if cfg.max_shards is not None and int(cfg.max_shards) >= 0:
        items = items[: int(cfg.max_shards)]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(cfg.calibration_path) if cfg.calibration_path else None
    spatial_mode = str(cfg.spatial_mode).lower().strip()
    if spatial_mode not in {"off", "append", "only"}:
        raise ValueError("spatial_mode must be one of off|append|only")
    spatial_metrics = _resolve_spatial_metrics(cfg)
    selected_mode = _similarity_mode(
        SimilaritySearchConfig(
            input_path=cfg.query_path,
            corpus_dir=cfg.output_dir,
            output_dir=cfg.output_dir,
            mode=cfg.mode,
        )
    )

    skipped: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    query_payload: dict[str, Any]

    if selected_mode == "feature":
        query_fv = extract_feature_vectors(
            cfg.query_path,
            feature_set=cfg.feature_set,
            frame_size=cfg.frame_size,
            hop_size=cfg.hop_size,
            sample_rate=cfg.sample_rate,
        )
        query_vec = _aggregate_feature_vector(query_fv.matrix)
        query_payload = {
            "path": str(cfg.query_path.resolve()),
            "feature_backend": query_fv.backend,
            "num_frames": int(query_fv.matrix.shape[0]),
            "num_features": int(query_fv.matrix.shape[1]),
        }
        spatial_query_vec = None
        registry = None
        if spatial_mode in {"append", "only"}:
            registry = create_registry(with_external=True)
            qres = analyze(
                AnalysisConfig(
                    input_path=cfg.query_path,
                    output_dir=cfg.output_dir,
                    metrics=list(spatial_metrics),
                    frame_size=cfg.frame_size,
                    hop_size=cfg.hop_size,
                    sample_rate=cfg.sample_rate,
                    calibration=calibration,
                    verbosity=0,
                    debug=0,
                    seed=cfg.seed,
                ),
                registry=registry,
            )
            spatial_query_vec = _metric_vector(qres, spatial_metrics)

        for item in items:
            shard_path = Path(str(item["path"]))
            if not cfg.include_query_if_present and shard_path.resolve() == cfg.query_path.resolve():
                continue
            try:
                fv = extract_feature_vectors(
                    shard_path,
                    feature_set=cfg.feature_set,
                    frame_size=cfg.frame_size,
                    hop_size=cfg.hop_size,
                    sample_rate=cfg.sample_rate,
                )
                shard_vec = _aggregate_feature_vector(fv.matrix)
                feature_dist, feature_sim = _distance(query_vec, shard_vec, cfg.distance)
                dist = feature_dist
                sim = feature_sim
                distance_components: dict[str, float] | None = None
                if spatial_mode in {"append", "only"} and spatial_query_vec is not None:
                    assert registry is not None
                    cres = analyze(
                        AnalysisConfig(
                            input_path=shard_path,
                            output_dir=cfg.output_dir,
                            metrics=list(spatial_metrics),
                            frame_size=cfg.frame_size,
                            hop_size=cfg.hop_size,
                            sample_rate=cfg.sample_rate,
                            calibration=calibration,
                            verbosity=0,
                            debug=0,
                            seed=cfg.seed,
                        ),
                        registry=registry,
                    )
                    spatial_vec = _metric_vector(cres, spatial_metrics)
                    shared = _shared_finite_vectors(spatial_query_vec, spatial_vec)
                    if shared is not None:
                        spatial_dist, spatial_sim = _distance(shared[0], shared[1], cfg.distance)
                        weight = float(np.clip(float(cfg.spatial_weight), 0.0, 1.0))
                        if spatial_mode == "only":
                            dist = spatial_dist
                            sim = spatial_sim
                        else:
                            dist = ((1.0 - weight) * feature_dist) + (weight * spatial_dist)
                            sim = ((1.0 - weight) * feature_sim) + (weight * spatial_sim)
                        distance_components = {
                            "feature_distance": float(feature_dist),
                            "spatial_distance": float(spatial_dist),
                            "feature_similarity": float(feature_sim),
                            "spatial_similarity": float(spatial_sim),
                        }
                rows.append(
                    {
                        "path": str(shard_path.resolve()),
                        "relative_path": str(item.get("relative_path") or shard_path.name),
                        "shard_index": int(item.get("shard_index", -1)),
                        "archive_start_s": float(item.get("start_s") or 0.0),
                        "archive_end_s": float(item.get("end_s") or 0.0),
                        "duration_s": float(item.get("duration_s") or 0.0),
                        "channels": int(item.get("channels") or 0),
                        "sample_rate": int(item.get("sample_rate") or 0),
                        "distance": float(dist),
                        "similarity": float(sim),
                        "distance_kind": cfg.distance,
                        "feature_backend": fv.backend,
                        "num_frames": int(fv.matrix.shape[0]),
                        "num_features": int(fv.matrix.shape[1]),
                        "distance_components": distance_components,
                    }
                )
            except Exception as exc:
                skipped.append({"path": str(shard_path.resolve()), "reason": str(exc)})

        rows.sort(key=lambda r: (float(r["distance"]), str(r["path"])))
        rows = rows[: max(1, int(cfg.top_k))]
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
        body = {
            "mode": "feature",
            "feature_set": cfg.feature_set,
            "distance": cfg.distance,
            "query": query_payload,
            "results": rows,
            "skipped": skipped,
        }
    else:
        metric_names = list(cfg.metrics or [])
        if not metric_names:
            metric_names = [cfg.metric]
        if spatial_mode == "only":
            metric_names = list(spatial_metrics)
        elif spatial_mode == "append":
            metric_names = list(dict.fromkeys([*metric_names, *spatial_metrics]))
        if selected_mode == "metric":
            metric_names = [metric_names[0]]

        registry = create_registry(with_external=True)
        query_result = analyze(
            AnalysisConfig(
                input_path=cfg.query_path,
                output_dir=cfg.output_dir,
                metrics=list(metric_names),
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate,
                calibration=calibration,
                verbosity=0,
                debug=0,
                seed=cfg.seed,
            ),
            registry=registry,
        )
        query_vec = _metric_vector(query_result, metric_names)
        if not np.isfinite(query_vec).any():
            raise RuntimeError(f"Query file has non-finite metric means for: {metric_names}")

        cand_rows: list[dict[str, Any]] = []
        cand_vecs: list[Any] = []
        for item in items:
            shard_path = Path(str(item["path"]))
            if not cfg.include_query_if_present and shard_path.resolve() == cfg.query_path.resolve():
                continue
            try:
                result = analyze(
                    AnalysisConfig(
                        input_path=shard_path,
                        output_dir=cfg.output_dir,
                        metrics=list(metric_names),
                        frame_size=cfg.frame_size,
                        hop_size=cfg.hop_size,
                        sample_rate=cfg.sample_rate,
                        calibration=calibration,
                        verbosity=0,
                        debug=0,
                        seed=cfg.seed,
                    ),
                    registry=registry,
                )
                cand_vec = _metric_vector(result, metric_names)
                if not np.isfinite(cand_vec).any():
                    skipped.append({"path": str(shard_path.resolve()), "reason": "non-finite metric mean(s)"})
                    continue
                cand_rows.append(
                    {
                        "path": str(shard_path.resolve()),
                        "relative_path": str(item.get("relative_path") or shard_path.name),
                        "shard_index": int(item.get("shard_index", -1)),
                        "archive_start_s": float(item.get("start_s") or 0.0),
                        "archive_end_s": float(item.get("end_s") or 0.0),
                        "duration_s": float(result.get("metadata", {}).get("duration_s", item.get("duration_s") or 0.0)),
                        "channels": int(result.get("metadata", {}).get("channels", item.get("channels") or 0)),
                        "sample_rate": int(result.get("metadata", {}).get("sample_rate", item.get("sample_rate") or 0)),
                        "metric_means": {name: float(v) for name, v in zip(metric_names, cand_vec.tolist())},
                    }
                )
                cand_vecs.append(cand_vec)
            except Exception as exc:
                skipped.append({"path": str(shard_path.resolve()), "reason": str(exc)})

        if cand_rows:
            cand_mat = np.vstack(cand_vecs)
            q_work = query_vec.copy()
            c_work = cand_mat.copy()
            if selected_mode == "metrics" and cfg.normalize:
                all_mat = np.vstack([q_work[None, :], c_work])
                mu = np.nanmean(all_mat, axis=0)
                sigma = np.nanstd(all_mat, axis=0)
                sigma = np.where(sigma < 1e-12, 1.0, sigma)
                q_work = (q_work - mu) / sigma
                c_work = (c_work - mu[None, :]) / sigma[None, :]

            rows = []
            for row, raw_vec, work_vec in zip(cand_rows, cand_mat, c_work):
                shared_raw = _shared_finite_vectors(query_vec, raw_vec)
                if shared_raw is None:
                    skipped.append({"path": row["path"], "reason": "no shared finite metric dimensions"})
                    continue
                if selected_mode == "metrics":
                    shared_work = _shared_finite_vectors(q_work, work_vec)
                    if shared_work is None:
                        skipped.append({"path": row["path"], "reason": "no shared finite normalized metric dimensions"})
                        continue
                    dist, sim = _distance(shared_work[0], shared_work[1], cfg.distance)
                    dist_kind = cfg.distance
                else:
                    dist = float(abs(shared_raw[1][0] - shared_raw[0][0]))
                    sim = float(1.0 / (1.0 + dist))
                    dist_kind = "abs_diff"
                rows.append(
                    {
                        **row,
                        "distance": float(dist),
                        "similarity": float(sim),
                        "distance_kind": dist_kind,
                    }
                )
            rows.sort(key=lambda r: (float(r["distance"]), str(r["path"])))
            rows = rows[: max(1, int(cfg.top_k))]
            for idx, row in enumerate(rows, start=1):
                row["rank"] = idx

        body = {
            "mode": selected_mode,
            "metrics": metric_names,
            "distance": cfg.distance,
            "normalize": bool(cfg.normalize),
            "query": {
                "path": str(cfg.query_path.resolve()),
                "metric_means": {name: float(v) for name, v in zip(metric_names, query_vec.tolist())},
            },
            "results": rows,
            "skipped": skipped,
        }

    report = {
        "schema_version": SHARD_MANIFEST_VERSION,
        "manifest_path": str(Path(cfg.manifest_path).resolve()),
        "query_path": str(cfg.query_path.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "top_k": int(cfg.top_k),
        "mode_requested": str(cfg.mode),
        "mode_used": body.get("mode"),
        "distance": cfg.distance,
        "candidates_scanned": len(items),
        "max_shards": cfg.max_shards,
        "include_query_if_present": bool(cfg.include_query_if_present),
        "archive_duration_s": float(manifest.get("total_duration_s") or 0.0),
        "archive_size_gb": float(manifest.get("total_size_gb") or 0.0),
        "config": {
            "feature_set": cfg.feature_set,
            "metric": cfg.metric,
            "metrics": cfg.metrics or [],
            "frame_size": int(cfg.frame_size),
            "hop_size": int(cfg.hop_size),
            "sample_rate": cfg.sample_rate,
            "normalize": bool(cfg.normalize),
            "seed": int(cfg.seed),
            "spatial_mode": spatial_mode,
            "spatial_metrics": spatial_metrics,
            "spatial_weight": float(cfg.spatial_weight),
        },
        **body,
    }
    report_path = (cfg.output_dir / f"{cfg.query_path.stem}_shard_similarity.json").resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report


def _retrieval_windows(
    duration_s: float,
    window_s: float,
    hop_s: float,
) -> list[tuple[float, float]]:
    duration = max(0.0, float(duration_s))
    if duration <= 0.0:
        return []
    window = min(duration, max(1e-6, float(window_s)))
    hop = max(1e-6, float(hop_s))
    starts: list[float] = []
    start = 0.0
    while start + window < duration - 1e-9:
        starts.append(start)
        start += hop
    final = max(0.0, duration - window)
    if not starts or abs(starts[-1] - final) > 1e-6:
        starts.append(final)
    return [(float(s), float(min(duration, s + window))) for s in starts]


def _write_retrieval_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "clip_id",
        "shard_index",
        "relative_path",
        "source_path",
        "local_start_s",
        "local_end_s",
        "archive_start_s",
        "archive_end_s",
        "archive_start_hms",
        "archive_end_hms",
        "duration_s",
        "distance",
        "similarity",
        "distance_kind",
        "feature_backend",
        "num_frames",
        "num_features",
        "wav_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def run_shard_event_retrieval(cfg: ShardRetrieveConfig) -> tuple[Path, dict[str, Any]]:
    """Find query-like time windows inside an ordered shard archive."""
    manifest = load_shard_manifest(cfg.manifest_path)
    items = [item for item in manifest.get("items", []) if isinstance(item, dict)]
    if not items:
        raise RuntimeError(f"No shard items found in manifest: {cfg.manifest_path}")
    if not cfg.query_path.exists():
        raise FileNotFoundError(f"Query file not found: {cfg.query_path}")
    if int(cfg.top_k) < 1:
        raise ValueError("top_k must be >= 1")
    if float(cfg.window_s) <= 0.0:
        raise ValueError("window_s must be > 0")
    if float(cfg.hop_s) <= 0.0:
        raise ValueError("hop_s must be > 0")

    if cfg.max_shards is not None and int(cfg.max_shards) >= 0:
        items = items[: int(cfg.max_shards)]

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    query_fv = extract_feature_vectors(
        cfg.query_path,
        feature_set=cfg.feature_set,
        frame_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        sample_rate=cfg.sample_rate,
    )
    query_vec = _aggregate_feature_vector(query_fv.matrix)
    if not np.isfinite(query_vec).any():
        raise RuntimeError("Query feature vector has no finite dimensions.")

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    windows_scanned = 0

    for item in items:
        shard_path = Path(str(item["path"]))
        relative_path = str(item.get("relative_path") or shard_path.name)
        duration_s = float(item.get("duration_s") or 0.0)
        archive_offset_s = float(item.get("start_s") or 0.0)
        for local_start_s, local_end_s in _retrieval_windows(duration_s, cfg.window_s, cfg.hop_s):
            windows_scanned += 1
            try:
                segment, sr = _read_segment(shard_path, local_start_s, local_end_s, cfg.sample_rate)
                if segment.size == 0:
                    skipped.append(
                        {
                            "path": str(shard_path.resolve()),
                            "local_start_s": local_start_s,
                            "local_end_s": local_end_s,
                            "reason": "empty segment",
                        }
                    )
                    continue
                fv = extract_feature_vectors_from_array(
                    segment,
                    sample_rate=sr,
                    feature_set=cfg.feature_set,
                    frame_size=cfg.frame_size,
                    hop_size=cfg.hop_size,
                )
                cand_vec = _aggregate_feature_vector(fv.matrix)
                shared = _shared_finite_vectors(query_vec, cand_vec)
                if shared is None:
                    skipped.append(
                        {
                            "path": str(shard_path.resolve()),
                            "local_start_s": local_start_s,
                            "local_end_s": local_end_s,
                            "reason": "no shared finite feature dimensions",
                        }
                    )
                    continue
                dist, sim = _distance(shared[0], shared[1], cfg.distance)
                archive_start_s = archive_offset_s + local_start_s
                archive_end_s = archive_offset_s + local_end_s
                candidates.append(
                    {
                        "shard_index": int(item.get("shard_index", -1)),
                        "relative_path": relative_path,
                        "source_path": str(shard_path.resolve()),
                        "local_start_s": float(local_start_s),
                        "local_end_s": float(local_end_s),
                        "archive_start_s": float(archive_start_s),
                        "archive_end_s": float(archive_end_s),
                        "archive_start_hms": _sec_to_hms(archive_start_s),
                        "archive_end_hms": _sec_to_hms(archive_end_s),
                        "duration_s": float(max(0.0, local_end_s - local_start_s)),
                        "distance": float(dist),
                        "similarity": float(sim),
                        "distance_kind": str(cfg.distance),
                        "feature_backend": fv.backend,
                        "num_frames": int(fv.matrix.shape[0]),
                        "num_features": int(fv.matrix.shape[1]),
                        "wav_path": "",
                    }
                )
            except Exception as exc:
                skipped.append(
                    {
                        "path": str(shard_path.resolve()),
                        "local_start_s": local_start_s,
                        "local_end_s": local_end_s,
                        "reason": str(exc),
                    }
                )

    candidates.sort(
        key=lambda r: (
            float(r["distance"]),
            str(r["source_path"]),
            float(r["local_start_s"]),
        )
    )
    results = candidates[: max(1, int(cfg.top_k))]
    clips_dir = cfg.output_dir / "retrieved_clips"
    ffmpeg_ok = _ffmpeg_available()
    if cfg.write_clips:
        clips_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(results, start=1):
        row["rank"] = idx
        row["clip_id"] = f"retrieved_{idx:04d}"
        if not cfg.write_clips:
            continue
        source_path = Path(str(row["source_path"]))
        clip_path = clips_dir / f"retrieved_{idx:04d}.wav"
        info = probe_audio_metadata(source_path)
        codec = _codec_from_subtype(
            str(info.get("subtype")) if info.get("subtype") is not None else None
        )
        wrote = False
        if ffmpeg_ok:
            wrote = _clip_with_ffmpeg(
                input_path=source_path,
                output_path=clip_path,
                start_s=float(row["local_start_s"]),
                end_s=float(row["local_end_s"]),
                codec=codec,
                sample_rate=(
                    int(cfg.sample_rate)
                    if cfg.sample_rate is not None
                    else int(info.get("sample_rate") or 0) or None
                ),
                channels=int(info.get("channels") or 1),
            )
        if not wrote:
            _clip_with_soundfile(
                source_path,
                clip_path,
                float(row["local_start_s"]),
                float(row["local_end_s"]),
                cfg.sample_rate,
            )
        row["wav_path"] = str(clip_path.resolve())

    csv_path = (cfg.output_dir / "event_retrieval.csv").resolve()
    _write_retrieval_csv(csv_path, results)
    report = {
        "schema_version": SHARD_MANIFEST_VERSION,
        "retrieval_version": "1.0.0",
        "created_utc": _now_utc(),
        "manifest_path": str(Path(cfg.manifest_path).resolve()),
        "query_path": str(cfg.query_path.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "archive_duration_s": float(manifest.get("total_duration_s") or 0.0),
        "archive_size_gb": float(manifest.get("total_size_gb") or 0.0),
        "top_k": int(cfg.top_k),
        "candidates_scanned": int(len(items)),
        "windows_scanned": int(windows_scanned),
        "candidate_windows": int(len(candidates)),
        "selected_windows": int(len(results)),
        "max_shards": cfg.max_shards,
        "config": {
            "window_s": float(cfg.window_s),
            "hop_s": float(cfg.hop_s),
            "feature_set": str(cfg.feature_set),
            "distance": str(cfg.distance),
            "frame_size": int(cfg.frame_size),
            "hop_size": int(cfg.hop_size),
            "sample_rate": cfg.sample_rate,
            "write_clips": bool(cfg.write_clips),
        },
        "query": {
            "path": str(cfg.query_path.resolve()),
            "feature_backend": query_fv.backend,
            "num_frames": int(query_fv.matrix.shape[0]),
            "num_features": int(query_fv.matrix.shape[1]),
        },
        "results": results,
        "skipped": skipped,
        "artifacts": {
            "event_retrieval_csv": str(csv_path),
            "clips_dir": str(clips_dir.resolve()) if cfg.write_clips else None,
        },
    }
    report_path = (cfg.output_dir / "event_retrieval.json").resolve()
    report["artifacts"]["event_retrieval_json"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report




def run_shard_moments(cfg: ShardMomentsConfig) -> tuple[Path, dict[str, Any]]:
    """Find top-ranked moments across a shard manifest and export clips + CSV."""
    manifest = load_shard_manifest(cfg.manifest_path)
    items = [item for item in manifest.get("items", []) if isinstance(item, dict)]
    if not items:
        raise RuntimeError(f"No shard items found in manifest: {cfg.manifest_path}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = cfg.output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.output_dir / "moments.csv"
    report_path = cfg.report_path if cfg.report_path is not None else (cfg.output_dir / "archive_moments_report.json")
    calibration = load_calibration(cfg.calibration_path) if cfg.calibration_path else None
    stream_root = (cfg.output_dir / "stream") if cfg.stream_root is None else Path(cfg.stream_root)

    windows_all: list[dict[str, Any]] = []
    shards_processed = 0
    ffmpeg_ok = _ffmpeg_available()

    for item in items:
        shard_path = Path(str(item["path"]))
        relative_path = str(item.get("relative_path") or shard_path.name)
        stream_report_path = _stream_report_path(stream_root, relative_path)
        if not stream_report_path.exists() or cfg.force_stream:
            stream_report_path.parent.mkdir(parents=True, exist_ok=True)
            run_stream_analysis(
                StreamRunConfig(
                    input_path=shard_path,
                    output_dir=stream_report_path.parent,
                    metrics=list(cfg.metrics or [cfg.rank_metric]),
                    frame_size=cfg.frame_size,
                    hop_size=cfg.hop_size,
                    sample_rate=cfg.sample_rate,
                    chunk_size=cfg.chunk_size,
                    calibration=calibration,
                    seed=cfg.seed,
                    rules_path=cfg.rules_path,
                    max_chunks=cfg.max_chunks,
                    checkpoint_dir=(stream_report_path.parent / "checkpoints"),
                    resume=cfg.resume,
                )
            )
        stream_report = _load_stream_report(stream_report_path)
        duration_s = float(stream_report.get("source_duration_s") or item.get("duration_s") or 0.0)
        if duration_s <= 0.0:
            duration_s = float(item.get("duration_s") or 0.0)
        rules_payload = stream_report.get("rules", {})
        has_threshold_rules = isinstance(rules_payload, dict) and bool(rules_payload.get("metric_thresholds"))
        shard_windows = _collect_windows(
            chunks=_iter_stream_chunks(stream_report, report_path=stream_report_path),
            pre_roll_s=cfg.pre_roll_s,
            post_roll_s=cfg.post_roll_s,
            merge_gap_s=cfg.merge_gap_s,
            min_alerts_per_chunk=cfg.min_alerts_per_chunk,
            duration_s=duration_s if duration_s > 0.0 else float("inf"),
            rank_metric=cfg.rank_metric,
            event_window_s=cfg.event_window_s,
            window_before_s=cfg.window_before_s,
            window_after_s=cfg.window_after_s,
            allow_metric_only_candidates=not has_threshold_rules,
        )
        shard_windows = _rerank_windows(
            shard_windows,
            input_path=shard_path,
            rank_metric=cfg.rank_metric,
            rank_scope=cfg.rank_scope,
            frame_size=cfg.frame_size,
            hop_size=cfg.hop_size,
            sample_rate=cfg.sample_rate,
            calibration=calibration,
        )
        archive_offset = float(item.get("start_s") or 0.0)
        for win in shard_windows:
            win_copy = dict(win)
            win_copy["source_path"] = str(shard_path.resolve())
            win_copy["relative_path"] = relative_path
            win_copy["archive_start_s"] = float(win["start_s"]) + archive_offset
            win_copy["archive_end_s"] = float(win["end_s"]) + archive_offset
            win_copy["archive_event_center_s"] = float(win["event_center_s"]) + archive_offset
            windows_all.append(win_copy)
        shards_processed += 1

    effective_mode = cfg.selection_mode
    selected = _select_windows(windows_all, selection_mode=effective_mode, top_k=cfg.top_k)

    rows: list[dict[str, Any]] = []
    for idx, win in enumerate(selected, start=1):
        source_path = Path(str(win["source_path"]))
        info = probe_audio_metadata(source_path)
        clip_path = clips_dir / f"moment_{idx:04d}.wav"
        start_s = float(win["start_s"])
        end_s = float(win["end_s"])
        codec = _codec_from_subtype(str(info.get("subtype")) if info.get("subtype") is not None else None)
        wrote = False
        if ffmpeg_ok:
            wrote = _clip_with_ffmpeg(
                input_path=source_path,
                output_path=clip_path,
                start_s=start_s,
                end_s=end_s,
                codec=codec,
                sample_rate=(int(cfg.sample_rate) if cfg.sample_rate is not None else int(info.get("sample_rate") or 0) or None),
                channels=int(info.get("channels") or 1),
            )
        if not wrote:
            _clip_with_soundfile(source_path, clip_path, start_s, end_s, cfg.sample_rate)
        rows.append(
            {
                "clip_id": f"moment_{idx:04d}",
                "relative_path": str(win.get("relative_path", "")),
                "source_path": str(source_path),
                "start_s": f"{start_s:.3f}",
                "end_s": f"{end_s:.3f}",
                "archive_start_s": f"{float(win['archive_start_s']):.3f}",
                "archive_end_s": f"{float(win['archive_end_s']):.3f}",
                "archive_start_hms": _sec_to_hms(float(win["archive_start_s"])),
                "archive_end_hms": _sec_to_hms(float(win["archive_end_s"])),
                "duration_s": f"{max(0.0, end_s - start_s):.3f}",
                "rank_metric": str(win["rank_metric"]),
                "rank_scope": str(win.get("rank_scope", cfg.rank_scope)),
                "rank_channel": str(win.get("rank_channel", "mix")),
                "rank_score": f"{float(win['rank_score']):.6f}",
                "wav_path": str(clip_path),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "relative_path",
                "source_path",
                "start_s",
                "end_s",
                "archive_start_s",
                "archive_end_s",
                "archive_start_hms",
                "archive_end_hms",
                "duration_s",
                "rank_metric",
                "rank_scope",
                "rank_channel",
                "rank_score",
                "wav_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = {
        "created_utc": _now_utc(),
        "manifest_path": str(Path(cfg.manifest_path).resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "stream_root": str(stream_root.resolve()),
        "rank_metric": cfg.rank_metric,
        "rank_scope": cfg.rank_scope,
        "selection_mode": effective_mode,
        "top_k": cfg.top_k,
        "shards_processed": int(shards_processed),
        "candidate_windows": int(len(windows_all)),
        "selected_windows": int(len(selected)),
        "artifacts": {
            "moments_csv": str(csv_path.resolve()),
            "clips_dir": str(clips_dir.resolve()),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report
