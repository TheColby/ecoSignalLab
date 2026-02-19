"""Streaming-friendly analysis loop with threshold-based alerting."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from esl.core.audio import AudioBuffer, read_audio, stream_audio
from esl.core.config import AnalysisConfig, CalibrationProfile
from esl.core.context import AnalysisContext
from esl.metrics.registry import MetricRegistry, create_registry


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
        selected_metrics = ["spl_a_db", "ndsi", "novelty_curve"]

    reg = registry or create_registry(with_external=True)
    for m in selected_metrics:
        reg.get(m)

    base_audio = read_audio(cfg.input_path, target_sr=cfg.sample_rate)
    chunk_iter = stream_audio(cfg.input_path, chunk_size=cfg.chunk_size, target_sr=cfg.sample_rate)

    chunks_out: list[dict[str, Any]] = []
    alert_rows: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunk_iter):
        if cfg.max_chunks is not None and idx >= cfg.max_chunks:
            break

        chunk_buffer = AudioBuffer(
            samples=chunk.samples,
            sample_rate=chunk.sample_rate,
            source_path=str(cfg.input_path),
            format_name=base_audio.format_name,
            subtype=base_audio.subtype,
            source_backend=base_audio.source_backend,
            decoder_provenance=base_audio.decoder_provenance,
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
                    alert_rows.append(alert)
                if isinstance(max_val, (int, float)) and value > float(max_val):
                    alert = {
                        "chunk_index": idx,
                        "metric": m_name,
                        "value": value,
                        "condition": "max",
                        "threshold": float(max_val),
                    }
                    chunk_alerts.append(alert)
                    alert_rows.append(alert)

        chunk_start_s = float(chunk.start_sample / chunk.sample_rate)
        chunk_end_s = float((chunk.start_sample + chunk.samples.shape[0]) / chunk.sample_rate)
        chunks_out.append(
            {
                "index": idx,
                "start_s": chunk_start_s,
                "end_s": chunk_end_s,
                "num_samples": int(chunk.samples.shape[0]),
                "metric_means": metric_means,
                "metrics": metric_map,
                "alerts": chunk_alerts,
            }
        )

    report = {
        "mode": "file_stream",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(cfg.input_path.resolve()),
        "sample_rate": int(base_audio.sample_rate),
        "channels": int(base_audio.channels),
        "chunk_size": int(cfg.chunk_size),
        "metrics": selected_metrics,
        "rules": rules,
        "chunks_processed": len(chunks_out),
        "alert_count": len(alert_rows),
        "alerts": alert_rows,
        "chunks": chunks_out,
        "decoder_provenance": base_audio.decoder_provenance,
        "aggregate_metric_means": {
            m: (
                float(np.mean([c["metric_means"][m] for c in chunks_out if c["metric_means"].get(m) is not None]))
                if any(c["metric_means"].get(m) is not None for c in chunks_out)
                else None
            )
            for m in selected_metrics
        },
    }

    report_path = cfg.output_dir / "stream_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    alerts_csv = cfg.output_dir / "stream_alerts.csv"
    with alerts_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chunk_index", "metric", "value", "condition", "threshold"],
        )
        writer.writeheader()
        for row in alert_rows:
            writer.writerow(row)

    report["artifacts"] = {
        "report_json": str(report_path),
        "alerts_csv": str(alerts_csv),
    }
    return report_path, report
