"""CLI-first staged pipeline runner with manifest/status tracking."""

from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from esl.core import AnalysisConfig, analyze, load_calibration
from esl.core.audio import iter_supported_files
from esl.core.utils import config_hash
from esl.io import save_json
from esl.metrics.registry import create_registry


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
class PipelineRunConfig:
    """Configuration for staged pipeline execution."""

    input_dir: Path
    output_dir: Path
    calibration_path: str | None = None
    metrics: list[str] | None = None
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int | None = None
    seed: int = 42
    plot: bool = False
    interactive: bool = False
    plot_metrics: list[str] | None = None
    include_spectral: bool = True
    ml_export: bool = False
    project: str | None = None
    force: bool = False
    stages: list[str] | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / "pipeline_manifest.json"


def _analysis_index_path(output_dir: Path) -> Path:
    return output_dir / "pipeline_analysis_index.json"


def _digest_csv_path(output_dir: Path) -> Path:
    return output_dir / "pipeline_digest.csv"


def _digest_json_path(output_dir: Path) -> Path:
    return output_dir / "pipeline_digest.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_stages(cfg: PipelineRunConfig) -> list[str]:
    out = ["analyze"]
    if cfg.plot:
        out.append("plot")
    if cfg.ml_export:
        out.append("ml_export")
    out.append("digest")
    return out


def _init_manifest(cfg: PipelineRunConfig) -> dict[str, Any]:
    stages = cfg.stages or _default_stages(cfg)
    pipeline_cfg = {
        "input_dir": str(cfg.input_dir.resolve()),
        "output_dir": str(cfg.output_dir.resolve()),
        "calibration_path": cfg.calibration_path,
        "metrics": cfg.metrics or [],
        "frame_size": cfg.frame_size,
        "hop_size": cfg.hop_size,
        "sample_rate": cfg.sample_rate,
        "chunk_size": cfg.chunk_size,
        "seed": cfg.seed,
        "plot": cfg.plot,
        "interactive": cfg.interactive,
        "plot_metrics": cfg.plot_metrics or [],
        "include_spectral": cfg.include_spectral,
        "ml_export": cfg.ml_export,
        "project": cfg.project,
        "force": cfg.force,
        "stages": stages,
    }
    return {
        "pipeline_id": str(uuid.uuid4()),
        "created_utc": _now_utc_iso(),
        "updated_utc": _now_utc_iso(),
        "status": "running",
        "config_hash": config_hash(pipeline_cfg),
        "config": pipeline_cfg,
        "stages": {},
        "artifacts": {},
    }


def _stage_start(manifest: dict[str, Any], stage: str) -> float:
    t0 = time.perf_counter()
    manifest["stages"][stage] = {
        "status": "running",
        "started_utc": _now_utc_iso(),
        "finished_utc": None,
        "duration_s": None,
        "counts": {},
        "errors": [],
    }
    manifest["updated_utc"] = _now_utc_iso()
    return t0


def _stage_finish(
    manifest: dict[str, Any],
    stage: str,
    t0: float,
    status: str,
    counts: dict[str, Any] | None = None,
    errors: list[str] | None = None,
) -> None:
    row = manifest["stages"][stage]
    row["status"] = status
    row["finished_utc"] = _now_utc_iso()
    row["duration_s"] = round(time.perf_counter() - t0, 4)
    row["counts"] = counts or {}
    row["errors"] = errors or []
    manifest["updated_utc"] = _now_utc_iso()


def _scan_results_from_index_or_tree(output_dir: Path) -> list[dict[str, Any]]:
    index_path = _analysis_index_path(output_dir)
    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return payload.get("results", [])

    results: list[dict[str, Any]] = []
    for p in sorted(output_dir.rglob("*.json")):
        if p.name in {"pipeline_manifest.json", "pipeline_analysis_index.json", "pipeline_digest.json"}:
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        if "metadata" in doc and "metrics" in doc and "config_hash" in doc:
            results.append({
                "input": doc.get("metadata", {}).get("input_path"),
                "json": str(p),
                "config_hash": doc.get("config_hash"),
            })
    return results


def _run_stage_analyze(cfg: PipelineRunConfig, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = iter_supported_files(cfg.input_dir, patterns=SUPPORTED_PATTERNS, recursive=True)
    if not files:
        return [], {"input_files": 0, "processed": 0, "skipped": 0, "errors": 0}

    calibration = load_calibration(cfg.calibration_path) if cfg.calibration_path else None
    reg = create_registry(with_external=True)

    processed = 0
    skipped = 0
    errors = 0
    results: list[dict[str, Any]] = []
    in_root = cfg.input_dir.resolve()

    for fp in files:
        rel = fp.relative_to(in_root)
        run_out = cfg.output_dir / rel.parent
        run_out.mkdir(parents=True, exist_ok=True)
        json_path = run_out / f"{fp.stem}.json"

        if json_path.exists() and not cfg.force:
            skipped += 1
            results.append({"input": str(fp), "json": str(json_path), "status": "skipped"})
            continue

        try:
            acfg = AnalysisConfig(
                input_path=fp,
                output_dir=run_out,
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate,
                chunk_size=cfg.chunk_size,
                metrics=list(cfg.metrics or []),
                calibration=calibration,
                project=cfg.project,
                variant=str(rel).replace("/", "__"),
                verbosity=0,
                debug=0,
                seed=cfg.seed,
                make_plots=False,
                ml_export=False,
            )
            result = analyze(acfg, registry=reg)
            save_json(result, json_path)
            processed += 1
            results.append(
                {
                    "input": str(fp),
                    "json": str(json_path),
                    "status": "processed",
                    "config_hash": result.get("config_hash"),
                }
            )
        except Exception as exc:
            errors += 1
            results.append({"input": str(fp), "json": str(json_path), "status": "error", "error": str(exc)})

    idx_payload = {"results": results, "generated_utc": _now_utc_iso()}
    _write_json(_analysis_index_path(cfg.output_dir), idx_payload)
    return results, {
        "input_files": len(files),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
    }


def _run_stage_plot(cfg: PipelineRunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    from esl.viz import plot_from_json

    results = _scan_results_from_index_or_tree(cfg.output_dir)
    made = 0
    skipped = 0
    errors = 0
    for item in results:
        json_path = Path(item["json"])
        if not json_path.exists():
            continue
        plot_dir = json_path.parent / f"{json_path.stem}_plots"
        if plot_dir.exists() and any(plot_dir.iterdir()) and not cfg.force:
            skipped += 1
            continue
        try:
            plot_from_json(
                json_path=json_path,
                output_dir=plot_dir,
                interactive=cfg.interactive,
                audio_path=item.get("input"),
                include_metrics=cfg.plot_metrics,
                include_spectral=cfg.include_spectral,
            )
            made += 1
        except Exception:
            errors += 1
    return {"analysis_items": len(results), "generated": made, "skipped": skipped, "errors": errors}


def _run_stage_ml_export(cfg: PipelineRunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    from esl.ml import export_ml_features

    results = _scan_results_from_index_or_tree(cfg.output_dir)
    made = 0
    skipped = 0
    errors = 0
    for item in results:
        json_path = Path(item["json"])
        if not json_path.exists():
            continue
        ml_dir = json_path.parent / f"{json_path.stem}_ml"
        if ml_dir.exists() and any(ml_dir.iterdir()) and not cfg.force:
            skipped += 1
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            export_ml_features(payload, output_dir=ml_dir, prefix=json_path.stem, seed=cfg.seed)
            made += 1
        except Exception:
            errors += 1
    return {"analysis_items": len(results), "generated": made, "skipped": skipped, "errors": errors}


def _run_stage_digest(cfg: PipelineRunConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    results = _scan_results_from_index_or_tree(cfg.output_dir)
    rows: list[dict[str, Any]] = []
    for item in results:
        json_path = Path(item["json"])
        if not json_path.exists():
            continue
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = doc.get("metrics", {})
        rows.append(
            {
                "input": doc.get("metadata", {}).get("input_path"),
                "json": str(json_path),
                "duration_s": doc.get("metadata", {}).get("duration_s"),
                "channels": doc.get("metadata", {}).get("channels"),
                "sample_rate": doc.get("metadata", {}).get("sample_rate"),
                "spl_a_mean": metrics.get("spl_a_db", {}).get("summary", {}).get("mean"),
                "spl_c_mean": metrics.get("spl_c_db", {}).get("summary", {}).get("mean"),
                "snr_mean": metrics.get("snr_db", {}).get("summary", {}).get("mean"),
                "rt60_mean": metrics.get("rt60_s", {}).get("summary", {}).get("mean"),
                "config_hash": doc.get("config_hash"),
            }
        )

    digest_csv = _digest_csv_path(cfg.output_dir)
    digest_csv.parent.mkdir(parents=True, exist_ok=True)
    with digest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input",
                "json",
                "duration_s",
                "channels",
                "sample_rate",
                "spl_a_mean",
                "spl_c_mean",
                "snr_mean",
                "rt60_mean",
                "config_hash",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    digest_json = {
        "generated_utc": _now_utc_iso(),
        "num_items": len(rows),
        "rows": rows,
    }
    _write_json(_digest_json_path(cfg.output_dir), digest_json)

    manifest["artifacts"]["digest_csv"] = str(digest_csv)
    manifest["artifacts"]["digest_json"] = str(_digest_json_path(cfg.output_dir))
    return {"rows": len(rows)}


def run_pipeline(cfg: PipelineRunConfig) -> tuple[Path, dict[str, Any]]:
    """Run selected pipeline stages and persist a manifest."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _init_manifest(cfg)
    manifest_path = _manifest_path(cfg.output_dir)
    _write_json(manifest_path, manifest)

    stages = cfg.stages or _default_stages(cfg)

    for stage in stages:
        t0 = _stage_start(manifest, stage)
        _write_json(manifest_path, manifest)

        try:
            if stage == "analyze":
                results, counts = _run_stage_analyze(cfg, manifest)
                manifest["artifacts"]["analysis_index"] = str(_analysis_index_path(cfg.output_dir))
                manifest["artifacts"]["analysis_json_count"] = len(results)
                _stage_finish(manifest, stage, t0, status="completed", counts=counts)
            elif stage == "plot":
                counts = _run_stage_plot(cfg, manifest)
                _stage_finish(manifest, stage, t0, status="completed", counts=counts)
            elif stage == "ml_export":
                counts = _run_stage_ml_export(cfg, manifest)
                _stage_finish(manifest, stage, t0, status="completed", counts=counts)
            elif stage == "digest":
                counts = _run_stage_digest(cfg, manifest)
                _stage_finish(manifest, stage, t0, status="completed", counts=counts)
            else:
                _stage_finish(
                    manifest,
                    stage,
                    t0,
                    status="failed",
                    counts={},
                    errors=[f"Unknown stage: {stage}"],
                )
        except Exception as exc:
            _stage_finish(
                manifest,
                stage,
                t0,
                status="failed",
                counts={},
                errors=[str(exc)],
            )
            break

        _write_json(manifest_path, manifest)

    any_failed = any(v.get("status") != "completed" for v in manifest.get("stages", {}).values())
    manifest["status"] = "failed" if any_failed else "completed"
    manifest["updated_utc"] = _now_utc_iso()
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def read_pipeline_status(path: str | Path) -> dict[str, Any]:
    """Load and return pipeline manifest."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
