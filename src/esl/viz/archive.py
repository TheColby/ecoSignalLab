"""Archive-scale plots for shard-manifest workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_shard_report(report_path: str | Path, output_dir: str | Path) -> list[Path]:
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

    report_metrics = [str(name) for name in report.get("report_metrics", [])]
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

    return paths
