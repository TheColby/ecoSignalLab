"""Project/variant tracking for architectural design comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _project_dir(root: str | Path, project: str) -> Path:
    return Path(root) / "projects" / project


def record_project_variant(
    result: dict[str, Any],
    project: str,
    variant: str,
    root: str | Path,
) -> Path:
    """Persist run summary into project index for variant-level comparison."""
    pdir = _project_dir(root, project)
    pdir.mkdir(parents=True, exist_ok=True)
    index_path = pdir / "index.json"

    rows: list[dict[str, Any]] = []
    if index_path.exists():
        rows = json.loads(index_path.read_text(encoding="utf-8"))

    summary = {
        metric: payload.get("summary", {}).get("mean")
        for metric, payload in result.get("metrics", {}).items()
    }
    rows.append(
        {
            "variant": variant,
            "analysis_time_utc": result.get("analysis_time_utc"),
            "input_path": result.get("metadata", {}).get("input_path"),
            "duration_s": result.get("metadata", {}).get("duration_s"),
            "sample_rate": result.get("metadata", {}).get("sample_rate"),
            "channels": result.get("metadata", {}).get("channels"),
            "summary": summary,
        }
    )

    index_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Emit a tabular comparison snapshot for quick review.
    compare_path = pdir / "comparison.csv"
    flat_rows = []
    for row in rows:
        for metric, value in row.get("summary", {}).items():
            flat_rows.append(
                {
                    "variant": row.get("variant"),
                    "metric": metric,
                    "value_mean": value,
                    "analysis_time_utc": row.get("analysis_time_utc"),
                }
            )

    with compare_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "metric", "value_mean", "analysis_time_utc"])
        writer.writeheader()
        for item in flat_rows:
            writer.writerow(item)

    return index_path
