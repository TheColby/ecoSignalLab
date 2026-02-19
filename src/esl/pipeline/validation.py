"""Dataset-level validation harness for regression and quality checks."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esl.core import AnalysisConfig, analyze, load_calibration
from esl.core.audio import iter_supported_files


SUPPORTED_PATTERNS = [
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
]


@dataclass(slots=True)
class ValidationRunConfig:
    input_dir: Path
    output_dir: Path
    rules_path: str | None = None
    calibration_path: str | None = None
    metrics: list[str] | None = None
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int | None = None
    recursive: bool = True
    seed: int = 42


def _load_rules(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Validation rules file not found: {p}")
    raw_text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML rules require pyyaml") from exc
        payload = yaml.safe_load(raw_text) or {}
    else:
        payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Validation rules must be an object: {p}")
    return payload


def _metric_mean(result: dict[str, Any], metric_name: str) -> float | None:
    payload = result.get("metrics", {}).get(metric_name)
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get("mean")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def run_validation(cfg: ValidationRunConfig) -> tuple[Path, dict[str, Any]]:
    """Run dataset-level validation and write JSON/CSV reports."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rules = _load_rules(cfg.rules_path)
    metric_rules = rules.get("metric_thresholds", {})
    flag_rules = rules.get("validity_flags", {})
    if not isinstance(metric_rules, dict):
        raise RuntimeError("rules.metric_thresholds must be an object")
    if not isinstance(flag_rules, dict):
        raise RuntimeError("rules.validity_flags must be an object")

    requested_metrics = list(cfg.metrics or [])
    for metric_name in metric_rules.keys():
        metric_str = str(metric_name)
        if metric_str not in requested_metrics:
            requested_metrics.append(metric_str)

    files = iter_supported_files(
        cfg.input_dir,
        patterns=SUPPORTED_PATTERNS,
        recursive=cfg.recursive,
    )
    calibration = load_calibration(cfg.calibration_path) if cfg.calibration_path else None

    summary_rows: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    files_failed = 0
    for fp in files:
        result = analyze(
            AnalysisConfig(
                input_path=fp,
                output_dir=cfg.output_dir,
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate,
                chunk_size=cfg.chunk_size,
                metrics=requested_metrics,
                calibration=calibration,
                verbosity=0,
                debug=0,
                seed=cfg.seed,
            )
        )
        validity = result.get("metadata", {}).get("validity_flags", {})
        if not isinstance(validity, dict):
            validity = {}

        failures: list[dict[str, Any]] = []
        for metric_name, threshold in metric_rules.items():
            name = str(metric_name)
            thr = threshold if isinstance(threshold, dict) else {}
            min_val = thr.get("min")
            max_val = thr.get("max")
            value = _metric_mean(result, name)
            status = "pass"
            message = ""
            if value is None:
                status = "fail"
                message = "metric missing"
            elif isinstance(min_val, (int, float)) and value < float(min_val):
                status = "fail"
                message = f"{value} < min {float(min_val)}"
            elif isinstance(max_val, (int, float)) and value > float(max_val):
                status = "fail"
                message = f"{value} > max {float(max_val)}"

            row = {
                "file": str(fp),
                "check_type": "metric",
                "name": name,
                "value": value,
                "expected_min": float(min_val) if isinstance(min_val, (int, float)) else None,
                "expected_max": float(max_val) if isinstance(max_val, (int, float)) else None,
                "expected_flag": None,
                "status": status,
                "message": message,
            }
            summary_rows.append(row)
            if status == "fail":
                failures.append(row)

        for flag_name, expected in flag_rules.items():
            name = str(flag_name)
            value = validity.get(name)
            status = "pass" if value == expected else "fail"
            message = "" if status == "pass" else f"{value!r} != expected {expected!r}"
            row = {
                "file": str(fp),
                "check_type": "validity_flag",
                "name": name,
                "value": value,
                "expected_min": None,
                "expected_max": None,
                "expected_flag": expected,
                "status": status,
                "message": message,
            }
            summary_rows.append(row)
            if status == "fail":
                failures.append(row)

        if failures:
            files_failed += 1

        file_reports.append(
            {
                "file": str(fp),
                "passed": not failures,
                "failures": failures,
                "validity_flags": validity,
                "selected_metric_means": {name: _metric_mean(result, name) for name in requested_metrics},
            }
        )

    summary_csv_path = cfg.output_dir / "validation_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "check_type",
                "name",
                "value",
                "expected_min",
                "expected_max",
                "expected_flag",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    report = {
        "input_dir": str(cfg.input_dir.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "rules": rules,
        "files_checked": len(files),
        "files_failed": files_failed,
        "files_passed": len(files) - files_failed,
        "summary_csv": str(summary_csv_path),
        "file_reports": file_reports,
    }
    report_path = cfg.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report
