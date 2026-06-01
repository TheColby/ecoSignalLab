"""Archive-scale plots for shard-manifest workflows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROLLUP_SECONDS: dict[str, float] = {
    "day": 24.0 * 3600.0,
    "month": 30.0 * 24.0 * 3600.0,
    "year": 365.0 * 24.0 * 3600.0,
}


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not np.isfinite(out):
        return None
    return out


def _report_metric_names(report: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    names = [str(name) for name in report.get("report_metrics", [])]
    if names:
        return names
    found: set[str] = set()
    for row in rows:
        for key in row:
            if key.endswith("_mean"):
                found.add(key[: -len("_mean")])
    return sorted(found)


def _write_rollup_csv(path: Path, rollup_rows: list[dict[str, Any]], metric_names: list[str]) -> None:
    fieldnames = [
        "period_index",
        "period_start_s",
        "period_end_s",
        "shard_count",
        "total_duration_s",
        *[f"{name}_mean" for name in metric_names],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rollup_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _rollup_rows(rows: list[dict[str, Any]], metric_names: list[str], bucket_s: float) -> list[dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    for row in rows:
        start_s = _finite_float(row.get("timeline_start_s")) or 0.0
        duration_s = max(0.0, _finite_float(row.get("duration_s")) or 0.0)
        period_index = int(np.floor(start_s / bucket_s))
        group = groups.setdefault(
            period_index,
            {
                "period_index": period_index,
                "period_start_s": period_index * bucket_s,
                "period_end_s": (period_index + 1) * bucket_s,
                "shard_count": 0,
                "total_duration_s": 0.0,
                "_weighted_sums": {name: 0.0 for name in metric_names},
                "_weights": {name: 0.0 for name in metric_names},
            },
        )
        group["shard_count"] = int(group["shard_count"]) + 1
        group["total_duration_s"] = float(group["total_duration_s"]) + duration_s
        for name in metric_names:
            value = _finite_float(row.get(f"{name}_mean"))
            if value is None:
                continue
            weight = duration_s if duration_s > 0.0 else 1.0
            group["_weighted_sums"][name] += value * weight
            group["_weights"][name] += weight

    output: list[dict[str, Any]] = []
    for period_index in sorted(groups):
        group = groups[period_index]
        clean = {
            "period_index": int(group["period_index"]),
            "period_start_s": float(group["period_start_s"]),
            "period_end_s": float(group["period_end_s"]),
            "shard_count": int(group["shard_count"]),
            "total_duration_s": float(group["total_duration_s"]),
        }
        for name in metric_names:
            weight = float(group["_weights"][name])
            clean[f"{name}_mean"] = float(group["_weighted_sums"][name] / weight) if weight > 0.0 else ""
        output.append(clean)
    return output


def _plot_rollup(path: Path, rollup_rows: list[dict[str, Any]], metric_names: list[str], label: str) -> None:
    x = np.array([int(row["period_index"]) for row in rollup_rows], dtype=np.int64)
    durations_h = np.array([float(row["total_duration_s"]) / 3600.0 for row in rollup_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, durations_h, width=0.8, alpha=0.65, label="Analyzed duration")
    ax.set_title(f"Archive Rollup by {label.capitalize()}")
    ax.set_xlabel(f"Archive-relative {label} index")
    ax.set_ylabel("Analyzed duration (hours)")
    ax.grid(True, axis="y", alpha=0.3)

    for metric_name in metric_names[:1]:
        y = np.array(
            [
                float(row.get(f"{metric_name}_mean"))
                if isinstance(row.get(f"{metric_name}_mean"), (int, float))
                else np.nan
                for row in rollup_rows
            ],
            dtype=np.float64,
        )
        if np.isfinite(y).any():
            ax2 = ax.twinx()
            ax2.plot(x, y, color="black", marker="o", linewidth=1.2, label=f"{metric_name} mean")
            ax2.set_ylabel(f"{metric_name} mean")
            break

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_shard_report(report_path: str | Path, output_dir: str | Path, *, rollup: str = "none") -> list[Path]:
    """Render archive-level overview plots from a shard analysis report."""
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    rows = [row for row in report.get("rows", []) if isinstance(row, dict) and row.get("status") != "error"]
    if not rows:
        return []

    starts = np.array([float(row.get("timeline_start_s", 0.0)) for row in rows], dtype=np.float64)
    durations = np.array([float(row.get("duration_s", 0.0)) for row in rows], dtype=np.float64)
    paths: list[Path] = []

    duration_path = out_dir / "archive_duration_timeline.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(starts / 3600.0, durations / 3600.0, marker="o", linewidth=1.0)
    ax.set_title("Shard Duration Timeline")
    ax.set_xlabel("Archive time (hours)")
    ax.set_ylabel("Shard duration (hours)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(duration_path, dpi=150)
    plt.close(fig)
    paths.append(duration_path)

    report_metrics = _report_metric_names(report, rows)
    for metric_name in report_metrics:
        y = np.array(
            [
                float(row.get(f"{metric_name}_mean"))
                if isinstance(row.get(f"{metric_name}_mean"), (int, float))
                else np.nan
                for row in rows
            ],
            dtype=np.float64,
        )
        if not np.isfinite(y).any():
            continue
        metric_path = out_dir / f"archive_metric_{metric_name}.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(starts / 3600.0, y, marker="o", linewidth=1.0)
        ax.set_title(f"Archive Metric Timeline: {metric_name}")
        ax.set_xlabel("Archive time (hours)")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(metric_path, dpi=150)
        plt.close(fig)
        paths.append(metric_path)

    rollup_names = list(_ROLLUP_SECONDS) if rollup == "all" else [rollup]
    for rollup_name in rollup_names:
        if rollup_name not in _ROLLUP_SECONDS:
            continue
        rollup_rows = _rollup_rows(rows, report_metrics, _ROLLUP_SECONDS[rollup_name])
        if not rollup_rows:
            continue
        csv_path = out_dir / f"archive_rollup_{rollup_name}.csv"
        png_path = out_dir / f"archive_rollup_{rollup_name}.png"
        _write_rollup_csv(csv_path, rollup_rows, report_metrics)
        _plot_rollup(png_path, rollup_rows, report_metrics, rollup_name)
        paths.extend([csv_path, png_path])

    return paths
