"""Archive-level insights for shard manifests and shard analysis reports.

The functions here intentionally avoid decoding shard audio. They operate on
manifest metadata and `shard_analysis_report.json` rows, so they scale to
multi-day and multi-year archives.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esl import __version__
from esl.core.insights import run_archive_drift
from esl.core.shards import load_shard_manifest
from esl.schema import SCHEMA_VERSION

EPS = 1e-12


@dataclass(slots=True)
class ShardInsightPaths:
    primary: Path
    report: dict[str, Any]


def _base_report(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "esl_version": __version__,
        "insight_kind": kind,
        "config": config,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid JSON object: {path}")
    return payload


def _numeric_row_value(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if np.isfinite(parsed) else None
    return None


def _metric_column(metric: str) -> str:
    return metric if metric.endswith("_mean") else f"{metric}_mean"


def _available_metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    cols: set[str] = set()
    for row in rows:
        for key in row:
            if key.endswith("_mean") and _numeric_row_value(row, key) is not None:
                cols.add(str(key))
    return sorted(cols)


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    mean = np.nanmean(matrix, axis=0, keepdims=True)
    std = np.nanstd(matrix, axis=0, keepdims=True)
    z = (matrix - mean) / np.maximum(std, EPS)
    return np.where(np.isfinite(z), z, 0.0)


def run_shard_manifest_summary(manifest_path: Path, output_dir: Path) -> ShardInsightPaths:
    """Summarize a shard manifest without decoding audio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_shard_manifest(manifest_path)
    items = [item for item in manifest.get("items", []) if isinstance(item, dict)]

    timeline_csv = output_dir / "shard_timeline.csv"
    fields = [
        "shard_index",
        "relative_path",
        "start_s",
        "end_s",
        "start_time_utc",
        "end_time_utc",
        "start_time_local",
        "end_time_local",
        "duration_s",
        "size_gb",
        "sample_rate",
        "channels",
        "format_name",
        "subtype",
    ]
    with timeline_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow({field: item.get(field) for field in fields})

    gaps: list[dict[str, Any]] = []
    sorted_items = sorted(items, key=lambda x: float(x.get("start_s") or 0.0))
    for prev, cur in zip(sorted_items, sorted_items[1:], strict=False):
        prev_end = float(prev.get("end_s") or 0.0)
        cur_start = float(cur.get("start_s") or 0.0)
        delta = cur_start - prev_end
        if abs(delta) > 1e-6:
            gaps.append(
                {
                    "previous_shard_index": int(prev.get("shard_index", -1)),
                    "current_shard_index": int(cur.get("shard_index", -1)),
                    "delta_s": float(delta),
                    "kind": "gap" if delta > 0.0 else "overlap",
                }
            )

    durations = [float(item.get("duration_s") or 0.0) for item in items]
    sizes = [float(item.get("size_gb") or 0.0) for item in items]
    report = _base_report(
        "shard_manifest_summary",
        {"manifest_path": str(manifest_path), "output_dir": str(output_dir)},
    )
    report.update(
        {
            "manifest_path": str(manifest_path.resolve()),
            "num_shards": int(len(items)),
            "archive_duration_s": float(manifest.get("total_duration_s") or sum(durations)),
            "archive_size_gb": float(manifest.get("total_size_gb") or sum(sizes)),
            "calendar": manifest.get("calendar", {"timeline_mode": "archive_relative"}),
            "duration_s": {
                "min": float(np.min(durations)) if durations else 0.0,
                "max": float(np.max(durations)) if durations else 0.0,
                "mean": float(np.mean(durations)) if durations else 0.0,
                "median": float(np.median(durations)) if durations else 0.0,
            },
            "sample_rates": dict(Counter(str(item.get("sample_rate")) for item in items)),
            "channels": dict(Counter(str(item.get("channels")) for item in items)),
            "formats": dict(Counter(str(item.get("format_name")) for item in items)),
            "timeline_integrity": {
                "non_contiguous_count": int(len(gaps)),
                "gaps_or_overlaps": gaps,
            },
            "artifacts": {"timeline_csv": str(timeline_csv)},
        }
    )
    summary_json = output_dir / "shard_insights_summary.json"
    return ShardInsightPaths(_write_json(summary_json, report), report)


def run_shard_report_scene_changes(
    report_path: Path,
    output_dir: Path,
    *,
    metrics: list[str] | None = None,
    threshold_z: float = 1.5,
    max_changes: int | None = None,
) -> ShardInsightPaths:
    """Detect shard-to-shard scene changes from archive report metric means."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_report(report_path)
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    metric_cols = [_metric_column(m) for m in metrics or []]
    if not metric_cols:
        metric_cols = _available_metric_columns(rows)
    if not metric_cols:
        raise RuntimeError("No numeric *_mean metric columns found in shard analysis report.")

    usable: list[dict[str, Any]] = []
    values: list[list[float]] = []
    for row in sorted(rows, key=lambda r: float(r.get("timeline_start_s") or 0.0)):
        vec = [_numeric_row_value(row, col) for col in metric_cols]
        if any(v is None for v in vec):
            continue
        usable.append(row)
        values.append([float(v) for v in vec if v is not None])

    matrix = np.array(values, dtype=np.float64)
    if matrix.shape[0] <= 1:
        scores = np.zeros((matrix.shape[0],), dtype=np.float64)
    else:
        z = _zscore_columns(matrix)
        scores = np.concatenate([[0.0], np.linalg.norm(np.diff(z, axis=0), axis=1)])
    cutoff = float(np.mean(scores) + float(threshold_z) * np.std(scores)) if scores.size else 0.0
    candidates = [
        (idx, score)
        for idx, score in enumerate(scores)
        if idx > 0 and float(score) >= cutoff
    ]
    candidates = sorted(candidates, key=lambda x: float(x[1]), reverse=True)
    if max_changes is not None:
        candidates = candidates[: max(0, int(max_changes))]
    candidates = sorted(
        candidates,
        key=lambda x: float(usable[x[0]].get("timeline_start_s") or 0.0),
    )

    changes: list[dict[str, Any]] = []
    for rank, (idx, score) in enumerate(candidates, start=1):
        row = usable[idx]
        prev = usable[idx - 1]
        changes.append(
            {
                "rank": rank,
                "from_shard_index": int(prev.get("shard_index", -1)),
                "to_shard_index": int(row.get("shard_index", -1)),
                "from_relative_path": str(prev.get("relative_path", "")),
                "to_relative_path": str(row.get("relative_path", "")),
                "archive_time_s": float(row.get("timeline_start_s") or 0.0),
                "change_score": float(score),
                "threshold": float(cutoff),
            }
        )

    csv_path = output_dir / "shard_scene_changes.csv"
    fields = [
        "rank",
        "from_shard_index",
        "to_shard_index",
        "from_relative_path",
        "to_relative_path",
        "archive_time_s",
        "change_score",
        "threshold",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(changes)

    out = _base_report(
        "shard_scene_changes",
        {
            "report_path": str(report_path),
            "metrics": [col.removesuffix("_mean") for col in metric_cols],
            "threshold_z": float(threshold_z),
            "max_changes": max_changes,
        },
    )
    out.update(
        {
            "method": "adjacent_shard_metric_mean_distance",
            "num_shards_considered": int(len(usable)),
            "metric_columns": metric_cols,
            "threshold": float(cutoff),
            "changes": changes,
            "artifacts": {"csv": str(csv_path)},
        }
    )
    return ShardInsightPaths(_write_json(output_dir / "shard_scene_changes.json", out), out)


def run_shard_report_calmness(
    report_path: Path,
    output_path: Path,
    *,
    level_metric: str = "rms_dbfs",
    activity_metrics: list[str] | None = None,
) -> ShardInsightPaths:
    """Estimate archive calmness/chaos/diversity from shard-level metric means."""
    payload = _load_report(report_path)
    rows = [
        row
        for row in payload.get("rows", [])
        if isinstance(row, dict) and str(row.get("status", "")) != "error"
    ]
    level_col = _metric_column(level_metric)
    level = np.array(
        [v for row in rows if (v := _numeric_row_value(row, level_col)) is not None],
        dtype=np.float64,
    )
    if level.size == 0 and level_metric == "rms_dbfs":
        level_col = "spl_a_db_mean"
        level = np.array(
            [v for row in rows if (v := _numeric_row_value(row, level_col)) is not None],
            dtype=np.float64,
        )

    activity_cols = [_metric_column(m) for m in activity_metrics or []]
    if not activity_cols:
        activity_cols = [
            col
            for col in ("novelty_curve_mean", "spectral_change_detection_mean", "ndsi_mean")
            if any(_numeric_row_value(row, col) is not None for row in rows)
        ]
    activity_values: list[float] = []
    for col in activity_cols:
        series = np.array(
            [v for row in rows if (v := _numeric_row_value(row, col)) is not None],
            dtype=np.float64,
        )
        if series.size:
            span = max(float(np.nanmax(series) - np.nanmin(series)), EPS)
            activity_values.append(float(np.mean(np.abs(series - np.nanmean(series))) / span))

    level_std = float(np.nanstd(level) / 60.0) if level.size else 0.0
    level_step = float(np.nanmean(np.abs(np.diff(level))) / 30.0) if level.size > 1 else 0.0
    activity = float(np.mean(activity_values)) if activity_values else 0.0
    chaos = float(max(0.0, level_std + level_step + activity))
    calmness = float(1.0 / (1.0 + chaos))
    if level.size:
        hist, _ = np.histogram(level, bins=min(10, max(2, level.size)))
        probs = hist.astype(np.float64) / max(float(np.sum(hist)), EPS)
        diversity = float(
            -np.sum(probs * np.log2(np.maximum(probs, EPS)))
            / max(np.log2(max(probs.size, 2)), EPS)
        )
    else:
        diversity = 0.0

    out = _base_report(
        "shard_calmness_chaos_diversity",
        {
            "report_path": str(report_path),
            "level_metric": level_col.removesuffix("_mean"),
            "activity_metrics": [col.removesuffix("_mean") for col in activity_cols],
        },
    )
    out.update(
        {
            "method": "shard_metric_level_stability_activity_entropy",
            "calmness_score": calmness,
            "chaos_score": chaos,
            "diversity_score": diversity,
            "level_std_component": level_std,
            "level_step_component": level_step,
            "activity_component": activity,
            "num_shards_considered": int(len(rows)),
        }
    )
    return ShardInsightPaths(_write_json(output_path, out), out)


def run_shard_soundscape_report(report_path: Path, output_dir: Path) -> ShardInsightPaths:
    """Create a compact HTML report from shard_analysis_report.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_report(report_path)
    weighted = payload.get("weighted_metric_means", {})
    rows = payload.get("rows", [])
    metric_rows = ""
    if isinstance(weighted, dict):
        for name, value in sorted(weighted.items()):
            metric_rows += f"<tr><td>{name}</td><td>{value}</td></tr>\n"

    mermaid = """flowchart LR
  A["Shard manifest"] --> B["esl shard analyze"]
  B --> C["shard_analysis_report.json"]
  C --> D["esl shard insights report"]
  D --> E["Archive decisions"]
"""
    html_path = output_dir / "shard_soundscape_report.html"
    script = (
        "<script type='module'>"
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
        "mermaid.initialize({startOnLoad:true});"
        "</script>"
    )
    style = (
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ddd;padding:.45rem}"
        "</style>"
    )
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>ecoSignalLab Shard Soundscape Report</title>",
        script,
        style,
        "</head><body><h1>ecoSignalLab Shard Soundscape Report</h1>",
        "<p>Archive-level summary from shard analysis outputs. No decade-sized gulp required.</p>",
        "<pre class='mermaid'>",
        mermaid,
        "</pre>",
        f"<p>Shards: {payload.get('num_shards', len(rows))}</p>",
        f"<p>Archive duration seconds: {payload.get('archive_duration_s')}</p>",
        "<h2>Weighted Metric Means</h2>",
        "<table><tr><th>Metric</th><th>Weighted Mean</th></tr>",
        metric_rows,
        "</table></body></html>",
    ]
    html_path.write_text("\n".join(html_parts), encoding="utf-8")

    out = _base_report("shard_soundscape_report", {"report_path": str(report_path)})
    out.update(
        {
            "html_path": str(html_path),
            "num_shards": int(payload.get("num_shards", len(rows)) or 0),
            "archive_duration_s": float(payload.get("archive_duration_s") or 0.0),
            "weighted_metric_means": weighted if isinstance(weighted, dict) else {},
        }
    )
    return ShardInsightPaths(_write_json(output_dir / "shard_soundscape_report.json", out), out)


def run_shard_report_drift(
    baseline_report: Path,
    candidate_report: Path,
    output_path: Path,
) -> ShardInsightPaths:
    """Compare two shard analysis reports with the shared archive-drift implementation."""
    result = run_archive_drift(baseline_report, candidate_report, output_path)
    return ShardInsightPaths(result.primary, result.report)
