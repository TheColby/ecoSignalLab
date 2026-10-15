"""Explicit, calibration-aware baseline comparisons; no inferred alarm thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from esl.version import __version__


def _assignments(items: list[str]) -> dict[str, float]:
    result = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"Expected unique METRIC=VALUE, got {item!r}")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Thresholds, standard deviations and steps must be finite and >= 0")
        result[name] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
        raise ValueError("Expected a JSON object containing a metrics object")
    return report


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Metric values must be finite numbers")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Metric values must be finite numbers")
    return number


def compare_reports(
    baseline: dict[str, Any],
    current: dict[str, Any],
    thresholds: dict[str, float],
    *,
    relative_thresholds: dict[str, float] | None = None,
    statistic: str = "mean",
    allow_uncalibrated: bool = False,
) -> dict[str, Any]:
    """Compare only explicitly selected metrics; equality at the limit passes."""
    relative = relative_thresholds or {}
    names = sorted(thresholds.keys() | relative.keys())
    if not names:
        raise ValueError("Supply at least one explicit threshold")
    for limit in [*thresholds.values(), *relative.values()]:
        if _number(limit) < 0:
            raise ValueError("Thresholds must be nonnegative")
    metadata = [report.get("metadata", {}) for report in (baseline, current)]
    if any(not isinstance(item, dict) for item in metadata):
        raise ValueError("metadata must be an object")
    calibrations = [item.get("calibration") for item in metadata]
    if any(item is not None and (not isinstance(item, dict) or not item) for item in calibrations):
        raise ValueError("Calibration metadata must be a nonempty object or null")
    if calibrations[0] != calibrations[1]:
        raise ValueError("Incompatible calibration metadata")
    if calibrations[0] is None and not allow_uncalibrated:
        raise ValueError("Calibration missing; explicitly use --allow-uncalibrated")
    for field in ("sample_rate", "channels", "frame_size", "hop_size"):
        if metadata[0].get(field) != metadata[1].get(field):
            raise ValueError(f"Incompatible analysis metadata: {field}")
    for field in ("schema_version",):
        if baseline.get(field) != current.get(field):
            raise ValueError(f"Incompatible report metadata: {field}")
    rows = []
    for name in names:
        values = []
        payloads = []
        for report in (baseline, current):
            metrics = report.get("metrics")
            if not isinstance(metrics, dict) or name not in metrics:
                raise ValueError(f"Missing selected metric: {name}")
            payload = metrics[name]
            payloads.append(payload)
            if isinstance(payload, dict):
                summary = payload.get("summary")
                if not isinstance(summary, dict):
                    raise ValueError(f"Missing summary object: {name}")
                values.append(_number(summary.get(statistic)))
            else:
                values.append(_number(payload))
        if isinstance(payloads[0], dict) != isinstance(payloads[1], dict):
            raise ValueError(f"Incompatible metric representations: {name}")
        units = None
        if isinstance(payloads[0], dict):
            units = payloads[0].get("units")
            if units != payloads[1].get("units"):
                raise ValueError(f"Incompatible units: {name}")
            if payloads[0].get("spec") != payloads[1].get("spec"):
                raise ValueError(f"Incompatible metric specifications: {name}")
            if any((p.get("spec") or {}).get("calibration_dependency") for p in payloads):
                if calibrations[0] is None:
                    raise ValueError(f"Calibration required for metric: {name}")
        delta = values[1] - values[0]
        if not math.isfinite(delta):
            raise ValueError(f"Numeric overflow for metric: {name}")
        fraction = abs(delta) / abs(values[0]) if values[0] != 0 else None
        if fraction is not None and not math.isfinite(fraction):
            raise ValueError(f"Numeric overflow computing relative change: {name}")
        absolute_ok = name not in thresholds or abs(delta) <= thresholds[name]
        relative_ok = name not in relative or (
            (fraction is not None and fraction <= relative[name]) or delta == 0
        )
        rows.append(
            {
                "metric": name,
                "baseline": values[0],
                "current": values[1],
                "delta": delta,
                "relative_change": fraction,
                "units": units,
                "absolute_threshold": thresholds.get(name),
                "relative_threshold": relative.get(name),
                "status": "pass" if absolute_ok and relative_ok else "alert",
            }
        )
    return {
        "workflow": "monitor",
        "artifact_kind": "esl.monitor.comparison",
        "schema_version": "1.0.0",
        "esl_version": __version__,
        "config": {
            "thresholds": thresholds,
            "relative_thresholds": relative,
            "allow_uncalibrated": allow_uncalibrated,
        },
        "statistic": statistic,
        "calibration_status": "matched" if calibrations[0] is not None else "unverified",
        "status": "alert" if any(r["status"] == "alert" for r in rows) else "pass",
        "metrics": rows,
        "limitations": [
            "Metadata equality does not prove instrument calibration or station equivalence.",
            "Missing metadata in both inputs is unverified, not evidence of compatibility.",
            "Only selected metrics and explicit thresholds are tested; no statistical alarm model.",
            "Pipeline hashes include file paths and are not used for compatibility.",
        ],
    }


def _run(args: argparse.Namespace) -> int:
    if args.output and args.output.resolve() in (args.baseline.resolve(), args.current.resolve()):
        raise ValueError("Output must not overwrite an input report")
    hashes = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.baseline, args.current)
    ]
    report = compare_reports(
        _load(args.baseline),
        _load(args.current),
        _assignments(args.threshold),
        relative_thresholds=_assignments(args.relative_threshold),
        statistic=args.statistic,
        allow_uncalibrated=args.allow_uncalibrated,
    )
    report["provenance"] = {
        "baseline": str(args.baseline.resolve()),
        "current": str(args.current.resolve()),
        "baseline_sha256": hashes[0],
        "current_sha256": hashes[1],
    }
    if hashes != [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.baseline, args.current)
    ]:
        raise ValueError("Input changed during monitoring comparison")
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 2 if report["status"] == "alert" else 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``esl monitor compare``."""
    parser = subparsers.add_parser("monitor", help="Compare a baseline with current metrics")
    commands = parser.add_subparsers(dest="monitor_command", required=True)
    compare = commands.add_parser("compare", help="Apply explicit absolute/relative thresholds")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("current", type=Path)
    compare.add_argument("--threshold", action="append", default=[], metavar="METRIC=VALUE")
    compare.add_argument(
        "--relative-threshold", action="append", default=[], metavar="METRIC=FRACTION"
    )
    compare.add_argument("--statistic", default="mean")
    compare.add_argument("--allow-uncalibrated", action="store_true")
    compare.add_argument("--output", type=Path)
    compare.set_defaults(func=_run)
