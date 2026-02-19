"""Project/variant tracking for architectural design comparisons."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _project_dir(root: str | Path, project: str) -> Path:
    return Path(root) / "projects" / project


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    confidence = {
        metric: payload.get("confidence")
        for metric, payload in result.get("metrics", {}).items()
    }
    metadata = result.get("metadata", {})
    rows.append(
        {
            "variant": variant,
            "analysis_time_utc": result.get("analysis_time_utc"),
            "analysis_time_local": result.get("analysis_time_local"),
            "schema_version": result.get("schema_version"),
            "esl_version": result.get("esl_version"),
            "analysis_mode": result.get("analysis_mode"),
            "config_hash": result.get("config_hash"),
            "pipeline_hash": result.get("pipeline_hash"),
            "input_path": result.get("metadata", {}).get("input_path"),
            "duration_s": result.get("metadata", {}).get("duration_s"),
            "sample_rate": result.get("metadata", {}).get("sample_rate"),
            "channels": result.get("metadata", {}).get("channels"),
            "summary": summary,
            "confidence": confidence,
            "validity_flags": metadata.get("validity_flags", {}),
            "warnings": metadata.get("warnings", []),
            "assumptions": metadata.get("assumptions", []),
            "metric_catalog_version": metadata.get("metric_catalog_version"),
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


def compare_project_variants(
    project: str,
    root: str | Path,
    baseline_variant: str | None = None,
    metrics: list[str] | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Compare project variants and emit delta reports."""
    pdir = _project_dir(root, project)
    index_path = pdir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Project index not found: {index_path}")

    loaded = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list) or not loaded:
        raise RuntimeError(f"Project index is empty or invalid: {index_path}")

    latest_by_variant: dict[str, dict[str, Any]] = {}
    variant_order: list[str] = []
    for row in loaded:
        if not isinstance(row, dict):
            continue
        variant = str(row.get("variant", "")).strip()
        if not variant:
            continue
        if variant not in latest_by_variant:
            variant_order.append(variant)
        latest_by_variant[variant] = row

    variants = [v for v in variant_order if v in latest_by_variant]
    if not variants:
        raise RuntimeError(f"No valid variants in project index: {index_path}")

    base = baseline_variant or variants[0]
    if base not in latest_by_variant:
        raise ValueError(f"Baseline variant '{base}' not found in project '{project}'.")

    if metrics:
        metric_list = list(dict.fromkeys([m for m in metrics if m]))
    else:
        metric_set: set[str] = set()
        for row in latest_by_variant.values():
            summary = row.get("summary", {})
            if isinstance(summary, dict):
                metric_set.update(str(k) for k in summary.keys())
        metric_list = sorted(metric_set)

    baseline_summary = latest_by_variant[base].get("summary", {})
    if not isinstance(baseline_summary, dict):
        baseline_summary = {}

    comparison_rows: list[dict[str, Any]] = []
    variant_payloads: list[dict[str, Any]] = []
    for variant in variants:
        row = latest_by_variant[variant]
        summary = row.get("summary", {})
        confidence = row.get("confidence", {})
        if not isinstance(summary, dict):
            summary = {}
        if not isinstance(confidence, dict):
            confidence = {}

        values: dict[str, float | None] = {}
        deltas: dict[str, float | None] = {}
        for metric in metric_list:
            value = _coerce_float(summary.get(metric))
            baseline_value = _coerce_float(baseline_summary.get(metric))
            delta = None if value is None or baseline_value is None else float(value - baseline_value)
            delta_pct = None
            if delta is not None and baseline_value is not None and abs(baseline_value) > 1e-12:
                delta_pct = float(100.0 * delta / abs(baseline_value))

            values[metric] = value
            deltas[metric] = delta
            comparison_rows.append(
                {
                    "project": project,
                    "baseline_variant": base,
                    "variant": variant,
                    "metric": metric,
                    "baseline_value": baseline_value,
                    "value": value,
                    "delta": delta,
                    "delta_pct": delta_pct,
                    "confidence": _coerce_float(confidence.get(metric)),
                    "analysis_time_utc": row.get("analysis_time_utc"),
                }
            )

        variant_payloads.append(
            {
                "variant": variant,
                "input_path": row.get("input_path"),
                "analysis_time_utc": row.get("analysis_time_utc"),
                "validity_flags": row.get("validity_flags", {}),
                "warnings": row.get("warnings", []),
                "values": values,
                "deltas_vs_baseline": deltas,
            }
        )

    json_path = Path(output_json) if output_json else pdir / "comparison_report.json"
    csv_path = Path(output_csv) if output_csv else pdir / "comparison_deltas.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "project": project,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_variant": base,
        "variants": variants,
        "metrics": metric_list,
        "comparison_rows": comparison_rows,
        "variant_reports": variant_payloads,
        "artifacts": {"json": str(json_path), "csv": str(csv_path)},
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "project",
                "baseline_variant",
                "variant",
                "metric",
                "baseline_value",
                "value",
                "delta",
                "delta_pct",
                "confidence",
                "analysis_time_utc",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)

    return report
