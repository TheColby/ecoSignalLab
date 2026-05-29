"""Command line interface for ecoSignalLab (esl)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from esl.core import AnalysisConfig, IngestConfig, analyze, load_calibration
from esl.core.audio import iter_supported_files, probe_sample_rate
from esl.docsgen import build_docs
from esl.io import (
    save_apx_csv,
    save_csv,
    save_hdf5,
    save_head_csv,
    save_json,
    save_mat,
    save_parquet,
    save_series_csv,
    save_soundcheck_csv,
)
from esl.metrics.registry import create_registry
from esl.project import compare_project_variants, record_project_variant
from esl.schema import SCHEMA_VERSION, analysis_output_schema


def _metric_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _csv_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _stage_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_") or "profile"


def _to_samples(seconds: float, sample_rate: int, flag_name: str) -> int:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError(f"{flag_name} must be a positive finite number.")
    samples = int(round(seconds * float(sample_rate)))
    if samples < 1:
        raise ValueError(
            f"{flag_name}={seconds} is too small for sample_rate={sample_rate}; resolved sample count is < 1."
        )
    return samples


def _resolve_chunk_duration_seconds(args: argparse.Namespace) -> float | None:
    choices = [
        ("chunk_seconds", 1.0),
        ("chunk_minutes", 60.0),
        ("chunk_hours", 3600.0),
        ("chunk_days", 86400.0),
    ]
    specified: list[tuple[str, float]] = []
    for name, scale in choices:
        raw = getattr(args, name, None)
        if raw is not None:
            if not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) <= 0.0:
                raise ValueError(f"--{name.replace('_', '-')} must be a positive finite number.")
            specified.append((name, float(raw) * scale))
    if not specified:
        return None
    if len(specified) > 1:
        labels = ", ".join(f"--{name.replace('_', '-')}" for name, _ in specified)
        raise ValueError(f"Specify only one chunk-duration flag at a time: {labels}")
    return specified[0][1]


def _resolve_window_samples(
    args: argparse.Namespace,
    *,
    input_path: Path | None,
    default_frame_size: int | None = None,
    default_hop_size: int | None = None,
    default_chunk_size: int | None = None,
) -> tuple[int | None, int | None, int | None, int | None]:
    frame_size = int(getattr(args, "frame_size", default_frame_size) or 0) if default_frame_size is not None or hasattr(args, "frame_size") else None
    hop_size = int(getattr(args, "hop_size", default_hop_size) or 0) if default_hop_size is not None or hasattr(args, "hop_size") else None
    chunk_size_raw = getattr(args, "chunk_size", default_chunk_size)
    chunk_size = int(chunk_size_raw) if chunk_size_raw is not None else None
    sample_rate_raw = getattr(args, "sample_rate", None)
    sample_rate = int(sample_rate_raw) if sample_rate_raw is not None else None

    frame_seconds = getattr(args, "frame_seconds", None)
    hop_seconds = getattr(args, "hop_seconds", None)
    chunk_duration_s = _resolve_chunk_duration_seconds(args)
    needs_sr = frame_seconds is not None or hop_seconds is not None or chunk_duration_s is not None
    resolved_sr = sample_rate
    if needs_sr and resolved_sr is None:
        if input_path is None:
            raise ValueError(
                "Duration-based window flags require --sample-rate when input path metadata is unavailable."
            )
        try:
            resolved_sr = probe_sample_rate(input_path)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to infer sample rate from {input_path}. Set --sample-rate explicitly."
            ) from exc

    if resolved_sr is not None:
        if frame_seconds is not None:
            frame_size = _to_samples(float(frame_seconds), resolved_sr, "--frame-seconds")
        if hop_seconds is not None:
            hop_size = _to_samples(float(hop_seconds), resolved_sr, "--hop-seconds")
        if chunk_duration_s is not None:
            chunk_size = _to_samples(float(chunk_duration_s), resolved_sr, "--chunk-*")

    if frame_size is not None and frame_size < 1:
        raise ValueError("--frame-size must be >= 1")
    if hop_size is not None and hop_size < 1:
        raise ValueError("--hop-size must be >= 1")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    return frame_size, hop_size, chunk_size, resolved_sr


def _build_analysis_config(args: argparse.Namespace, input_path: Path, out_dir: Path) -> AnalysisConfig:
    calibration = load_calibration(args.calibration) if args.calibration else None
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=input_path,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=None,
    )
    return AnalysisConfig(
        input_path=input_path,
        output_dir=out_dir,
        frame_size=int(frame_size or 2048),
        hop_size=int(hop_size or 512),
        sample_rate=resolved_sr,
        chunk_size=chunk_size,
        metrics=_metric_list(args.metrics),
        calibration=calibration,
        project=args.project,
        variant=args.variant,
        verbosity=args.verbosity,
        debug=args.debug,
        seed=args.seed,
        compute_device=args.device,
        make_plots=args.plot,
        ml_export=args.ml_export,
        summary_only=bool(getattr(args, "summary_only", False)),
        streamable_only=bool(getattr(args, "streamable_only", False)),
        allow_full_read=bool(getattr(args, "allow_full_read", False)),
        max_series_points=getattr(args, "max_series_points", None),
        frame_table_csv=(Path(args.frame_table_csv) if getattr(args, "frame_table_csv", None) else None),
        frame_table_parquet_dir=(
            Path(args.frame_table_parquet_dir) if getattr(args, "frame_table_parquet_dir", None) else None
        ),
        frame_table_hdf5=(Path(args.frame_table_hdf5) if getattr(args, "frame_table_hdf5", None) else None),
        checkpoint_dir=(Path(args.checkpoint_dir) if getattr(args, "checkpoint_dir", None) else None),
        resume=bool(getattr(args, "resume", False)),
    )


def _run_profile_analyze(args: argparse.Namespace, base_cfg: AnalysisConfig, out_dir: Path) -> int:
    from esl import __version__
    from esl.core.profiles import load_resolution_profiles, with_resolution_profile

    profiles = load_resolution_profiles(args.profile)
    input_stem = base_cfg.input_path.stem
    runs: list[dict[str, Any]] = []

    for prof in profiles:
        run_cfg = with_resolution_profile(base_cfg, prof)
        result = analyze(run_cfg)
        run_name = _safe_name(prof.name)
        run_json = out_dir / f"{input_stem}__{run_name}.json"
        save_json(result, run_json)

        if args.plot:
            from esl.viz import plot_analysis

            plot_analysis(
                result,
                output_dir=out_dir / f"{input_stem}__{run_name}_plots",
                audio_path=base_cfg.input_path,
                interactive=args.interactive,
                include_metrics=_metric_list(args.plot_metrics),
                include_spectral=not args.no_spectral,
                include_similarity_matrix=args.similarity_matrix,
                include_novelty_matrix=args.novelty_matrix,
                similarity_feature_set=args.sim_feature_set,
                feature_vectors_path=args.feature_vectors,
            )
        if args.ml_export:
            from esl.ml import export_ml_features

            export_ml_features(
                result,
                output_dir=out_dir / f"{input_stem}__{run_name}_ml",
                prefix=f"{input_stem}__{run_name}",
                seed=run_cfg.seed,
                device=run_cfg.compute_device,
                strict_device=False,
            )

        if run_cfg.project and run_cfg.variant:
            record_project_variant(result, project=run_cfg.project, variant=run_cfg.variant, root=out_dir)

        def _mean(name: str) -> float | None:
            payload = result.get("metrics", {}).get(name)
            if not isinstance(payload, dict):
                return None
            summary = payload.get("summary")
            if not isinstance(summary, dict):
                return None
            value = summary.get("mean")
            return float(value) if isinstance(value, (int, float)) else None

        runs.append(
            {
                "name": prof.name,
                "frame_size": run_cfg.frame_size,
                "hop_size": run_cfg.hop_size,
                "sample_rate": run_cfg.sample_rate,
                "chunk_size": run_cfg.chunk_size,
                "metrics": list(run_cfg.metrics),
                "json": str(run_json),
                "summary": {
                    "duration_s": round(float(result["metadata"]["duration_s"]), 6),
                    "channels": int(result["metadata"]["channels"]),
                    "sample_rate": int(result["metadata"]["sample_rate"]),
                    "compute_device": result.get("metadata", {}).get("compute_device", {}).get("resolved"),
                    "spl_a_mean": _mean("spl_a_db"),
                    "snr_mean": _mean("snr_db"),
                    "rt60": _mean("rt60_s"),
                },
            }
        )

    profile_index: dict[str, Any] = {
        "profile_version": "esl-profile-1.0.0",
        "esl_version": __version__,
        "profile_source": str(Path(args.profile).resolve()),
        "input": str(base_cfg.input_path.resolve()),
        "created_runs": len(runs),
        "runs": runs,
    }
    index_path = Path(args.json) if args.json else out_dir / f"{input_stem}_profile.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(profile_index, indent=2), encoding="utf-8")

    if args.csv:
        summary_csv = Path(args.csv)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "frame_size",
                    "hop_size",
                    "sample_rate",
                    "chunk_size",
                    "json",
                    "duration_s",
                    "channels",
                    "result_sample_rate",
                    "compute_device",
                    "spl_a_mean",
                    "snr_mean",
                    "rt60",
                ],
            )
            writer.writeheader()
            for run in runs:
                summary = run["summary"]
                writer.writerow(
                    {
                        "name": run["name"],
                        "frame_size": run["frame_size"],
                        "hop_size": run["hop_size"],
                        "sample_rate": run["sample_rate"],
                        "chunk_size": run["chunk_size"],
                        "json": run["json"],
                        "duration_s": summary["duration_s"],
                        "channels": summary["channels"],
                        "result_sample_rate": summary["sample_rate"],
                        "compute_device": summary["compute_device"],
                        "spl_a_mean": summary["spl_a_mean"],
                        "snr_mean": summary["snr_mean"],
                        "rt60": summary["rt60"],
                    }
                )

    if base_cfg.verbosity >= 1:
        print(f"profile source: {args.profile}")
        print(f"profile runs: {len(runs)}")
        print(f"profile index: {index_path}")
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir)
    _mkdir(out_dir)

    cfg = _build_analysis_config(args, input_path=input_path, out_dir=out_dir)
    if args.profile:
        return _run_profile_analyze(args, base_cfg=cfg, out_dir=out_dir)

    result = analyze(cfg)

    stem = input_path.stem
    json_path = Path(args.json) if args.json else out_dir / f"{stem}.json"
    save_json(result, json_path)

    if args.csv:
        save_csv(result, Path(args.csv))
    if args.series_csv:
        save_series_csv(result, Path(args.series_csv))
    if args.parquet:
        save_parquet(result, Path(args.parquet))
    if args.hdf5:
        save_hdf5(result, Path(args.hdf5))
    if args.mat:
        save_mat(result, Path(args.mat))
    if args.head_csv:
        save_head_csv(result, Path(args.head_csv))
    if args.apx_csv:
        save_apx_csv(result, Path(args.apx_csv))
    if args.soundcheck_csv:
        save_soundcheck_csv(result, Path(args.soundcheck_csv))

    if args.plot:
        from esl.viz import plot_analysis, spawn_plot_paths

        plot_dir = out_dir / f"{stem}_plots"
        plots = plot_analysis(
            result,
            output_dir=plot_dir,
            audio_path=input_path,
            interactive=args.interactive,
            include_metrics=_metric_list(args.plot_metrics),
            include_spectral=not args.no_spectral,
            include_similarity_matrix=args.similarity_matrix,
            include_novelty_matrix=args.novelty_matrix,
            similarity_feature_set=args.sim_feature_set,
            feature_vectors_path=args.feature_vectors,
        )
        if cfg.verbosity >= 1:
            print(f"plots: {len(plots)} files -> {plot_dir}")
        if args.show:
            spawn_summary = spawn_plot_paths(plots, limit=args.show_limit)
            print(
                f"plot spawn: opened={spawn_summary['opened']} "
                f"failed={spawn_summary['failed']} "
                f"skipped={spawn_summary['skipped_by_limit']}"
            )

    if args.ml_export:
        from esl.ml import export_ml_features

        ml_dir = out_dir / f"{stem}_ml"
        artifacts = export_ml_features(
            result,
            output_dir=ml_dir,
            prefix=stem,
            seed=cfg.seed,
            device=cfg.compute_device,
            strict_device=False,
        )
        if cfg.verbosity >= 1:
            print(f"ml artifacts: {len(artifacts)} -> {ml_dir}")

    if args.project and args.variant:
        record_project_variant(result, project=args.project, variant=args.variant, root=out_dir)

    if cfg.verbosity >= 1:
        metrics = result.get("metrics", {})

        def _mean(name: str) -> float | None:
            payload = metrics.get(name)
            if not isinstance(payload, dict):
                return None
            summary = payload.get("summary")
            if not isinstance(summary, dict):
                return None
            value = summary.get("mean")
            return float(value) if isinstance(value, (int, float)) else None

        print(f"json: {json_path}")
        print(
            "summary:",
            {
                "duration_s": round(float(result["metadata"]["duration_s"]), 3),
                "channels": int(result["metadata"]["channels"]),
                "sample_rate": int(result["metadata"]["sample_rate"]),
                "compute_device": result.get("metadata", {}).get("compute_device", {}).get("resolved"),
                "spl_a_mean": _mean("spl_a_db"),
                "snr_mean": _mean("snr_db"),
                "rt60": _mean("rt60_s"),
            },
        )

    if cfg.debug >= 1:
        print(f"config_hash: {result.get('config_hash')}")
    if cfg.debug >= 2:
        print(json.dumps(result.get("metadata", {}), indent=2))

    return 0


def _run_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    _mkdir(out_dir)
    in_root = in_dir.resolve()

    files = iter_supported_files(in_dir, patterns=["*.wav", "*.flac", "*.aiff", "*.aif", "*.rf64", "*.caf", "*.mp3", "*.aac", "*.ogg", "*.opus", "*.wma", "*.alac", "*.m4a", "*.sofa"], recursive=not args.no_recursive)
    if not files:
        print("No supported files found.")
        return 0

    report_metrics = _metric_list(args.report_metrics) or ["snr_db", "spl_a_db", "rt60_s"]
    report_cols: list[str] = []
    col_counts: dict[str, int] = {}
    report_metric_col: dict[str, str] = {}
    for metric_name in report_metrics:
        base = f"{_safe_name(metric_name)}_mean"
        idx = col_counts.get(base, 0)
        col_counts[base] = idx + 1
        col = base if idx == 0 else f"{base}_{idx + 1}"
        report_cols.append(col)
        report_metric_col[metric_name] = col

    rows: list[dict[str, Any]] = []
    plot_artifacts: list[str] = []
    for fp in files:
        rel = fp.relative_to(in_root)
        run_out = out_dir / rel.parent
        _mkdir(run_out)

        cfg = _build_analysis_config(args, input_path=fp, out_dir=run_out)
        result = analyze(cfg)

        base = run_out / f"{fp.stem}.json"
        save_json(result, base)

        if args.csv:
            save_csv(result, run_out / f"{fp.stem}.csv")
        if args.parquet:
            save_parquet(result, run_out / f"{fp.stem}.parquet")
        if args.hdf5:
            save_hdf5(result, run_out / f"{fp.stem}.h5")
        if args.mat:
            save_mat(result, run_out / f"{fp.stem}.mat")

        row: dict[str, Any] = {
            "input": str(fp),
            "json": str(base),
            "duration_s": result["metadata"]["duration_s"],
            "channels": result["metadata"]["channels"],
            "sample_rate": result["metadata"]["sample_rate"],
            "compute_device": result.get("metadata", {}).get("compute_device", {}).get("resolved"),
        }
        metrics_payload = result.get("metrics", {})
        for metric_name, col_name in report_metric_col.items():
            value = None
            if isinstance(metrics_payload, dict):
                payload = metrics_payload.get(metric_name, {})
                if isinstance(payload, dict):
                    summary = payload.get("summary", {})
                    if isinstance(summary, dict):
                        m = summary.get("mean")
                        if isinstance(m, (int, float)):
                            value = float(m)
            row[col_name] = value
        rows.append(row)

        if args.plot:
            from esl.viz import plot_analysis

            plots = plot_analysis(
                result,
                output_dir=run_out / f"{fp.stem}_plots",
                audio_path=fp,
                interactive=args.interactive,
                include_metrics=_metric_list(args.plot_metrics),
                include_spectral=not args.no_spectral,
                include_similarity_matrix=args.similarity_matrix,
                include_novelty_matrix=args.novelty_matrix,
                similarity_feature_set=args.sim_feature_set,
                feature_vectors_path=args.feature_vectors,
            )
            plot_artifacts.extend(plots)

        if args.ml_export:
            from esl.ml import export_ml_features

            export_ml_features(
                result,
                output_dir=run_out / f"{fp.stem}_ml",
                prefix=fp.stem,
                seed=cfg.seed,
                device=cfg.compute_device,
                strict_device=False,
            )

    idx = out_dir / "batch_index.csv"
    with idx.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input",
                "json",
                "duration_s",
                "channels",
                "sample_rate",
                "compute_device",
                *report_cols,
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"batch complete: {len(rows)} files -> {out_dir}")
    print(f"index: {idx}")
    if args.plot and args.show:
        from esl.viz import spawn_plot_paths

        spawn_summary = spawn_plot_paths(plot_artifacts, limit=args.show_limit)
        print(
            f"plot spawn: opened={spawn_summary['opened']} "
            f"failed={spawn_summary['failed']} "
            f"skipped={spawn_summary['skipped_by_limit']}"
        )
    return 0


def _run_plot(args: argparse.Namespace) -> int:
    from esl.viz import plot_from_json, spawn_plot_paths

    plots = plot_from_json(
        json_path=args.results_json,
        output_dir=args.out,
        interactive=args.interactive,
        audio_path=args.audio,
        include_metrics=_metric_list(args.metrics),
        include_spectral=not args.no_spectral,
        include_similarity_matrix=args.similarity_matrix,
        include_novelty_matrix=args.novelty_matrix,
        similarity_feature_set=args.sim_feature_set,
        feature_vectors_path=args.feature_vectors,
    )
    print(f"generated {len(plots)} plot files in {args.out}")
    if args.show:
        spawn_summary = spawn_plot_paths(plots, limit=args.show_limit)
        print(
            f"plot spawn: opened={spawn_summary['opened']} "
            f"failed={spawn_summary['failed']} "
            f"skipped={spawn_summary['skipped_by_limit']}"
        )
    return 0


def _run_similar(args: argparse.Namespace) -> int:
    from esl.core.similarity import SimilaritySearchConfig, run_similarity_search

    input_path = Path(args.input)
    corpus_dir = Path(args.corpus_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    if int(args.top_k) < 1:
        raise ValueError("--top-k must be >= 1")

    out_dir = Path(args.out_dir)
    _mkdir(out_dir)
    calibration = load_calibration(args.calibration) if args.calibration else None
    cfg = SimilaritySearchConfig(
        input_path=input_path,
        corpus_dir=corpus_dir,
        output_dir=out_dir,
        top_k=int(args.top_k),
        mode=str(args.mode),
        metric=str(args.metric),
        metrics=_metric_list(args.metrics) if args.metrics else None,
        distance=str(args.distance),
        feature_set=str(args.feature_set),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
        normalize=bool(args.normalize),
        include_self=bool(args.include_self),
        recursive=not bool(args.no_recursive),
        max_files=args.max_files,
        calibration=calibration,
        seed=int(args.seed),
    )
    report = run_similarity_search(cfg)
    json_path = Path(args.json) if args.json else out_dir / f"{input_path.stem}_similarity.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "path", "distance", "similarity", "distance_kind", "duration_s", "channels", "sample_rate", "metric_means_json"],
            )
            writer.writeheader()
            for row in report.get("results", []):
                metric_map = row.get("metric_means") if isinstance(row, dict) else None
                writer.writerow(
                    {
                        "rank": row.get("rank"),
                        "path": row.get("path"),
                        "distance": row.get("distance"),
                        "similarity": row.get("similarity"),
                        "distance_kind": row.get("distance_kind"),
                        "duration_s": row.get("duration_s"),
                        "channels": row.get("channels"),
                        "sample_rate": row.get("sample_rate"),
                        "metric_means_json": json.dumps(metric_map, sort_keys=True) if isinstance(metric_map, dict) else "",
                    }
                )

    if int(args.verbosity) >= 1:
        print(f"json: {json_path}")
        if args.csv:
            print(f"csv: {args.csv}")
        print(
            "summary:",
            {
                "mode": report.get("mode_used"),
                "candidates_scanned": report.get("candidates_scanned"),
                "results": len(report.get("results", [])),
            },
        )
        for row in report.get("results", [])[: int(args.top_k)]:
            print(
                f"rank={row.get('rank')} dist={row.get('distance'):.6f} "
                f"sim={row.get('similarity'):.6f} path={row.get('path')}"
            )
    if int(args.debug) >= 1:
        print(f"mode_requested={report.get('mode_requested')} mode_used={report.get('mode_used')}")
    if int(args.debug) >= 2:
        print(json.dumps(report.get("config", {}), indent=2))
    return 0


def _run_schema(args: argparse.Namespace) -> int:
    schema = analysis_output_schema()
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"schema_version: {SCHEMA_VERSION}")
        print(str(p))
    else:
        print(json.dumps(schema, indent=2))
        print(f"schema_version: {SCHEMA_VERSION}", file=sys.stderr)
    return 0


def _run_project_compare(args: argparse.Namespace) -> int:
    report = compare_project_variants(
        project=args.project,
        root=Path(args.root),
        baseline_variant=args.baseline,
        metrics=_metric_list(args.metrics) or None,
        output_json=args.json_out,
        output_csv=args.csv_out,
    )
    print(f"project: {report.get('project')}")
    print(f"baseline_variant: {report.get('baseline_variant')}")
    print(f"variants: {len(report.get('variants', []))}")
    print(f"metrics: {len(report.get('metrics', []))}")
    artifacts = report.get("artifacts", {})
    print(f"json: {artifacts.get('json')}")
    print(f"csv: {artifacts.get('csv')}")
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    from esl.pipeline import ValidationRunConfig, run_validation

    cfg = ValidationRunConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.out),
        rules_path=args.rules,
        calibration_path=args.calibration,
        metrics=_metric_list(args.metrics) or None,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        recursive=not args.no_recursive,
        seed=args.seed,
    )
    report_path, report = run_validation(cfg)
    print(f"validation_report: {report_path}")
    print(
        "summary:",
        {
            "files_checked": report.get("files_checked"),
            "files_passed": report.get("files_passed"),
            "files_failed": report.get("files_failed"),
            "summary_csv": report.get("summary_csv"),
        },
    )
    return 0 if int(report.get("files_failed", 0)) == 0 else 2


def _run_stream(args: argparse.Namespace) -> int:
    from esl.core.streaming import StreamRunConfig, run_stream_analysis

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    out_dir = Path(args.out)
    calibration = load_calibration(args.calibration) if args.calibration else None
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=input_path,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=131072,
    )
    cfg = StreamRunConfig(
        input_path=input_path,
        output_dir=out_dir,
        metrics=_metric_list(args.metrics),
        frame_size=int(frame_size or 2048),
        hop_size=int(hop_size or 512),
        sample_rate=resolved_sr,
        chunk_size=int(chunk_size or 131072),
        calibration=calibration,
        seed=args.seed,
        rules_path=args.rules,
        max_chunks=args.max_chunks,
        checkpoint_dir=(Path(args.checkpoint_dir) if args.checkpoint_dir else None),
        resume=bool(args.resume),
        chunks_jsonl=(Path(args.chunks_jsonl) if args.chunks_jsonl else None),
    )
    report_path, report = run_stream_analysis(cfg)
    if args.verbosity >= 1:
        print(f"stream report: {report_path}")
        print(
            "summary:",
            {
                "chunks_processed": report.get("chunks_processed"),
                "alert_count": report.get("alert_count"),
                "metrics": report.get("metrics"),
                "alerts_csv": report.get("artifacts", {}).get("alerts_csv"),
            },
        )
    if args.debug >= 1:
        print(f"chunk_size: {args.chunk_size} sample_rate: {report.get('sample_rate')}")
    if args.debug >= 2:
        print(json.dumps(report.get("rules", {}), indent=2))
    return 0


def _run_shard_index(args: argparse.Namespace) -> int:
    from esl.core.shards import ShardManifestConfig, build_shard_manifest

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_path = Path(args.out)
    manifest_path, manifest = build_shard_manifest(
        ShardManifestConfig(
            input_dir=input_dir,
            output_path=output_path,
            recursive=not args.no_recursive,
            order_by=args.order_by,
        )
    )
    print(f"manifest: {manifest_path}")
    print(
        "summary:",
        {
            "num_shards": manifest.get("num_shards"),
            "total_duration_s": manifest.get("total_duration_s"),
            "total_size_gb": manifest.get("total_size_gb"),
            "order_by": manifest.get("order_by"),
        },
    )
    return 0


def _run_shard_analyze(args: argparse.Namespace) -> int:
    from esl.core.shards import ShardAnalyzeConfig, load_shard_manifest, run_shard_analysis

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Shard manifest not found: {manifest_path}")
    out_dir = Path(args.out)
    manifest = load_shard_manifest(manifest_path)
    duration_flags_used = any(
        getattr(args, name, None) is not None
        for name in ("frame_seconds", "hop_seconds", "chunk_seconds", "chunk_minutes", "chunk_hours", "chunk_days")
    )
    if args.sample_rate is None and duration_flags_used:
        rates = {
            int(item["sample_rate"])
            for item in manifest.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("sample_rate"), int)
        }
        if len(rates) == 1:
            args.sample_rate = rates.pop()
        elif len(rates) > 1:
            raise ValueError(
                "Duration-based window flags on a shard manifest with mixed sample rates require --sample-rate explicitly."
            )
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=None,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=None,
    )
    report_path, report = run_shard_analysis(
        ShardAnalyzeConfig(
            manifest_path=manifest_path,
            output_dir=out_dir,
            calibration_path=args.calibration,
            metrics=_metric_list(args.metrics),
            report_metrics=_metric_list(args.report_metrics),
            frame_size=int(frame_size or 2048),
            hop_size=int(hop_size or 512),
            sample_rate=resolved_sr,
            chunk_size=chunk_size,
            seed=args.seed,
            compute_device=args.device,
            summary_only=bool(args.summary_only),
            streamable_only=bool(args.streamable_only),
            allow_full_read=bool(args.allow_full_read),
            max_series_points=args.max_series_points,
            frame_table_dir=(Path(args.frame_table_dir) if args.frame_table_dir else None),
            frame_table_parquet_root=(Path(args.frame_table_parquet_dir) if args.frame_table_parquet_dir else None),
            frame_table_hdf5_root=(Path(args.frame_table_hdf5_dir) if args.frame_table_hdf5_dir else None),
            checkpoint_root=(Path(args.checkpoint_dir) if args.checkpoint_dir else None),
            resume=bool(args.resume),
            force=bool(args.force),
        )
    )
    print(f"shard_report: {report_path}")
    print(
        "summary:",
        {
            "num_shards": report.get("num_shards"),
            "processed": report.get("processed"),
            "skipped": report.get("skipped"),
            "errors": report.get("errors"),
            "archive_duration_s": report.get("archive_duration_s"),
            "index_csv": report.get("artifacts", {}).get("index_csv"),
        },
    )
    if args.debug >= 1:
        print(json.dumps(report.get("weighted_metric_means", {}), indent=2))
    return 0


def _run_shard_moments(args: argparse.Namespace) -> int:
    from esl.core.shards import ShardMomentsConfig, load_shard_manifest, run_shard_moments

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Shard manifest not found: {manifest_path}")
    manifest = load_shard_manifest(manifest_path)
    duration_flags_used = any(
        getattr(args, name, None) is not None
        for name in ("frame_seconds", "hop_seconds", "chunk_seconds", "chunk_minutes", "chunk_hours", "chunk_days")
    )
    if args.sample_rate is None and duration_flags_used:
        rates = {
            int(item["sample_rate"])
            for item in manifest.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("sample_rate"), int)
        }
        if len(rates) == 1:
            args.sample_rate = rates.pop()
        elif len(rates) > 1:
            raise ValueError(
                "Duration-based window flags on a shard manifest with mixed sample rates require --sample-rate explicitly."
            )
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=None,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=131072,
    )
    selection_mode = "all"
    top_k: int | None = None
    if bool(args.single):
        selection_mode = "single"
        top_k = 1
    elif args.top_k is not None:
        if int(args.top_k) < 1:
            raise ValueError("--top-k must be >= 1")
        selection_mode = "top_k"
        top_k = int(args.top_k)

    report_path, report = run_shard_moments(
        ShardMomentsConfig(
            manifest_path=manifest_path,
            output_dir=Path(args.out),
            stream_root=(Path(args.stream_root) if args.stream_root else None),
            rules_path=args.rules,
            metrics=_metric_list(args.metrics),
            calibration_path=args.calibration,
            frame_size=int(frame_size or 2048),
            hop_size=int(hop_size or 512),
            sample_rate=resolved_sr,
            chunk_size=int(chunk_size or 131072),
            seed=int(args.seed),
            max_chunks=args.max_chunks,
            pre_roll_s=float(args.pre_roll),
            post_roll_s=float(args.post_roll),
            merge_gap_s=float(args.merge_gap),
            min_alerts_per_chunk=int(args.min_alerts_per_chunk),
            selection_mode=selection_mode,
            top_k=top_k,
            rank_metric=str(args.rank_metric),
            rank_scope=str(args.rank_scope),
            event_window_s=(float(args.event_window) if args.event_window is not None else None),
            window_before_s=(float(args.window_before) if args.window_before is not None else None),
            window_after_s=(float(args.window_after) if args.window_after is not None else None),
            resume=bool(args.resume),
            force_stream=bool(args.force_stream),
            report_path=(Path(args.report) if args.report else None),
        )
    )
    print(f"shard_moments_report: {report_path}")
    print(
        "summary:",
        {
            "shards_processed": report.get("shards_processed"),
            "candidate_windows": report.get("candidate_windows"),
            "selected_windows": report.get("selected_windows"),
            "rank_metric": report.get("rank_metric"),
            "rank_scope": report.get("rank_scope"),
        },
    )
    return 0


def _run_shard_similar(args: argparse.Namespace) -> int:
    from esl.core.shards import ShardSimilarConfig, load_shard_manifest, run_shard_similarity

    manifest_path = Path(args.manifest)
    query_path = Path(args.query)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Shard manifest not found: {manifest_path}")
    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")

    manifest = load_shard_manifest(manifest_path)
    duration_flags_used = any(
        getattr(args, name, None) is not None for name in ("frame_seconds", "hop_seconds")
    )
    if args.sample_rate is None and duration_flags_used:
        rates = {
            int(item["sample_rate"])
            for item in manifest.get("items", [])
            if isinstance(item, dict) and isinstance(item.get("sample_rate"), int)
        }
        if len(rates) == 1:
            args.sample_rate = rates.pop()
        elif len(rates) > 1:
            raise ValueError(
                "Duration-based window flags on a shard manifest with mixed sample rates require --sample-rate explicitly."
            )

    frame_size, hop_size, _, resolved_sr = _resolve_window_samples(
        args,
        input_path=query_path,
        default_frame_size=1024,
        default_hop_size=256,
        default_chunk_size=None,
    )
    report_path, report = run_shard_similarity(
        ShardSimilarConfig(
            manifest_path=manifest_path,
            query_path=query_path,
            output_dir=Path(args.out),
            top_k=int(args.top_k),
            mode=str(args.mode),
            metric=str(args.metric),
            metrics=_metric_list(args.metrics),
            distance=str(args.distance),
            feature_set=str(args.feature_set),
            frame_size=int(frame_size or 1024),
            hop_size=int(hop_size or 256),
            sample_rate=resolved_sr,
            normalize=bool(args.normalize),
            calibration_path=args.calibration,
            seed=int(args.seed),
            include_query_if_present=bool(args.include_query),
            max_shards=args.max_shards,
            spatial_mode=str(getattr(args, "spatial_mode", "off")),
            spatial_metrics=_metric_list(getattr(args, "spatial_metrics", None)),
            spatial_weight=float(getattr(args, "spatial_weight", 0.5)),
        )
    )

    json_path = Path(args.json) if args.json else report_path
    if json_path != report_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "shard_index",
                    "relative_path",
                    "path",
                    "archive_start_s",
                    "archive_end_s",
                    "distance",
                    "similarity",
                    "distance_kind",
                    "duration_s",
                    "channels",
                    "sample_rate",
                    "metric_means_json",
                ],
            )
            writer.writeheader()
            for row in report.get("results", []):
                writer.writerow(
                    {
                        "rank": row.get("rank"),
                        "shard_index": row.get("shard_index"),
                        "relative_path": row.get("relative_path"),
                        "path": row.get("path"),
                        "archive_start_s": row.get("archive_start_s"),
                        "archive_end_s": row.get("archive_end_s"),
                        "distance": row.get("distance"),
                        "similarity": row.get("similarity"),
                        "distance_kind": row.get("distance_kind"),
                        "duration_s": row.get("duration_s"),
                        "channels": row.get("channels"),
                        "sample_rate": row.get("sample_rate"),
                        "metric_means_json": json.dumps(row.get("metric_means", {}), sort_keys=True),
                    }
                )
    if int(args.verbosity) >= 1:
        print(f"shard_similarity_json: {json_path}")
        if args.csv:
            print(f"shard_similarity_csv: {args.csv}")
        for row in report.get("results", []):
            print(
                f"#{row.get('rank')} shard={row.get('relative_path')} "
                f"dist={float(row.get('distance', 0.0)):.6f} sim={float(row.get('similarity', 0.0)):.6f}"
            )
    return 0


def _run_shard_plot(args: argparse.Namespace) -> int:
    from esl.viz import plot_shard_report

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Shard analysis report not found: {report_path}")
    paths = plot_shard_report(report_path, Path(args.out))
    print(f"archive_plot_dir: {Path(args.out).resolve()}")
    print("plots:", [str(path) for path in paths])
    return 0


def _run_spatial_analyze(args: argparse.Namespace) -> int:
    from esl.core.spatial import (
        SPATIAL_DEFAULT_METRICS,
        load_array_config,
        run_spatial_analysis,
        stereo_beam_map,
        write_beam_map_csv,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    out_dir = Path(args.out_dir)
    _mkdir(out_dir)
    calibration = load_calibration(args.calibration) if args.calibration else None
    array_config = load_array_config(args.array_config)
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=input_path,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=None,
    )

    metric_list = _metric_list(args.metrics) or list(SPATIAL_DEFAULT_METRICS)
    if args.doa and "doa_azimuth_proxy_deg" not in metric_list:
        metric_list.append("doa_azimuth_proxy_deg")
    if args.doa and "itd_s" not in metric_list:
        metric_list.append("itd_s")

    cfg = AnalysisConfig(
        input_path=input_path,
        output_dir=out_dir,
        frame_size=int(frame_size or 2048),
        hop_size=int(hop_size or 512),
        sample_rate=resolved_sr,
        chunk_size=chunk_size,
        metrics=metric_list,
        calibration=calibration,
        verbosity=args.verbosity,
        debug=args.debug,
        seed=args.seed,
        compute_device=args.device,
        project=args.project,
        variant=args.variant,
    )
    result = run_spatial_analysis(cfg, array_config=array_config)
    json_path = Path(args.json) if args.json else out_dir / f"{input_path.stem}_spatial.json"
    save_json(result, json_path)

    beam_map_csv: Path | None = None
    if args.beam_map:
        spacing = 0.2
        if isinstance(array_config, dict) and isinstance(array_config.get("mic_spacing_m"), (int, float)):
            spacing = float(array_config["mic_spacing_m"])
        rows = stereo_beam_map(
            input_path,
            mic_spacing_m=spacing,
            azimuth_step_deg=args.azimuth_step_deg,
            target_sr=args.sample_rate,
        )
        beam_map_csv = (
            Path(args.beam_map_csv)
            if args.beam_map_csv
            else out_dir / f"{input_path.stem}_beam_map.csv"
        )
        write_beam_map_csv(rows, beam_map_csv)

    if args.verbosity >= 1:
        print(f"json: {json_path}")
        if beam_map_csv:
            print(f"beam_map_csv: {beam_map_csv}")
        print(
            "summary:",
            {
                "channels": result.get("metadata", {}).get("channels"),
                "layout": result.get("metadata", {}).get("channel_layout_hint"),
                "metrics": len(result.get("metrics", {})),
            },
        )
    return 0


def _run_calibrate_check(args: argparse.Namespace) -> int:
    from esl.core.calibration_check import CalibrationCheckConfig, run_calibration_check

    tone_path = Path(args.tone)
    if not tone_path.exists():
        raise FileNotFoundError(f"Tone file not found: {tone_path}")

    profile = load_calibration(args.calibration) if args.calibration else None
    dbfs_reference = (
        float(args.dbfs_reference)
        if args.dbfs_reference is not None
        else float(profile.dbfs_reference if profile is not None else 0.0)
    )
    spl_reference_db = (
        float(args.spl_reference_db)
        if args.spl_reference_db is not None
        else float(profile.spl_reference_db if profile is not None else 94.0)
    )
    weighting = (
        str(args.weighting).upper()
        if args.weighting is not None
        else str(profile.weighting if profile is not None else "Z").upper()
    )
    mic_sensitivity_mv_pa = (
        float(args.mic_sensitivity_mv_pa)
        if args.mic_sensitivity_mv_pa is not None
        else profile.mic_sensitivity_mv_pa
        if profile is not None
        else None
    )
    preamp_gain_db = (
        float(args.preamp_gain_db)
        if args.preamp_gain_db is not None
        else profile.preamp_gain_db
        if profile is not None
        else None
    )
    adc_full_scale_vrms = (
        float(args.adc_full_scale_vrms)
        if args.adc_full_scale_vrms is not None
        else profile.adc_full_scale_vrms
        if profile is not None
        else None
    )

    out_path = Path(args.out)
    cfg = CalibrationCheckConfig(
        tone_path=tone_path,
        output_path=out_path,
        dbfs_reference=dbfs_reference,
        spl_reference_db=spl_reference_db,
        weighting=weighting,
        mic_sensitivity_mv_pa=mic_sensitivity_mv_pa,
        preamp_gain_db=preamp_gain_db,
        adc_full_scale_vrms=adc_full_scale_vrms,
        calibration_profile=profile,
        device_id=args.device_id,
        history_csv=Path(args.history) if args.history else None,
        max_drift_db=float(args.max_drift_db),
        sample_rate=args.sample_rate,
    )
    report_path, report, within_tolerance = run_calibration_check(cfg)
    print(f"calibration_report: {report_path}")
    print(
        "summary:",
        {
            "device_id": report.get("device_id"),
            "measured_dbfs": report.get("measured_dbfs"),
            "dbfs_reference": report.get("dbfs_reference"),
            "drift_db": report.get("drift_db"),
            "max_drift_db": report.get("max_drift_db"),
            "within_tolerance": report.get("within_tolerance"),
            "pressure_chain_supported": report.get("pressure_chain_supported"),
            "measured_pa_rms": report.get("measured_pa_rms"),
            "measured_db_spl_from_pa": report.get("measured_db_spl_from_pa"),
        },
    )
    return 0 if within_tolerance else 2


def _run_calibrate_verify(args: argparse.Namespace) -> int:
    from esl.core.calibration_check import CalibrationVerifyConfig, run_calibration_verify

    profile = load_calibration(args.calibration) if args.calibration else None
    report_path, report, ok = run_calibration_verify(
        CalibrationVerifyConfig(
            fixture=str(args.fixture),
            output_path=Path(args.out),
            calibration_profile=profile,
            max_abs_error_db=float(args.max_abs_error_db),
            write_tone_path=(Path(args.write_tone) if args.write_tone else None),
        )
    )
    print(f"calibration_verify_report: {report_path}")
    print(
        "summary:",
        {
            "fixture": report.get("fixture"),
            "expected_dbfs_rms": report.get("expected_dbfs_rms"),
            "measured_dbfs_rms": report.get("measured_dbfs_rms"),
            "abs_error_db": report.get("abs_error_db"),
            "within_tolerance": report.get("within_tolerance"),
            "pressure_chain_error_db": report.get("pressure_chain_error_db"),
        },
    )
    return 0 if ok else 2


def _run_features_extract(args: argparse.Namespace) -> int:
    from esl.viz.feature_vectors import extract_feature_vectors, save_feature_vectors

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    vectors = extract_feature_vectors(
        audio_path=input_path,
        feature_set=args.feature_set,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
    )
    out_path = save_feature_vectors(vectors, args.out)
    meta = {
        "input_path": str(input_path.resolve()),
        "output_path": str(out_path.resolve()),
        "feature_set": args.feature_set,
        "backend": vectors.backend,
        "frames": int(vectors.matrix.shape[0]),
        "features": int(vectors.matrix.shape[1]),
        "sample_rate": int(vectors.sample_rate),
        "frame_size": int(vectors.frame_size),
        "hop_size": int(vectors.hop_size),
        "feature_names": vectors.feature_names,
    }
    if args.meta_json:
        meta_path = Path(args.meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"meta_json: {meta_path}")
    print(f"feature_vectors: {out_path}")
    print(
        "summary:",
        {
            "backend": vectors.backend,
            "frames": vectors.matrix.shape[0],
            "features": vectors.matrix.shape[1],
        },
    )
    return 0


def _run_features_manifest(args: argparse.Namespace) -> int:
    from esl.ml import build_dataset_manifest_from_ml_metadata

    split_values = tuple(float(x) for x in _csv_list(args.split_ratios)) if args.split_ratios else (0.8, 0.1, 0.1)
    if len(split_values) != 3:
        raise ValueError("--split-ratios must contain exactly three comma-separated values: train,val,test")
    out_path, manifest = build_dataset_manifest_from_ml_metadata(
        input_dir=Path(args.input_dir),
        output_path=Path(args.out),
        pattern=str(args.pattern),
        split_ratios=(split_values[0], split_values[1], split_values[2]),
    )
    print(f"dataset_manifest: {out_path}")
    print("summary:", {"num_samples": manifest.get("num_samples"), "split_counts": manifest.get("split_counts")})
    return 0


def _run_moments_extract(args: argparse.Namespace) -> int:
    from esl.core.moments import MomentsExtractConfig, run_moments_extract

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    calibration = load_calibration(args.calibration) if args.calibration else None
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=input_path,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=131072,
    )

    selection_mode = "all"
    top_k: int | None = None
    if bool(args.single):
        selection_mode = "single"
        top_k = 1
    elif args.top_k is not None:
        if int(args.top_k) < 1:
            raise ValueError("--top-k must be >= 1")
        selection_mode = "top_k"
        top_k = int(args.top_k)
    elif args.max_clips is not None:
        if int(args.max_clips) < 1:
            raise ValueError("--max-clips must be >= 1")
        selection_mode = "top_k"
        top_k = int(args.max_clips)

    cfg = MomentsExtractConfig(
        input_path=input_path,
        output_dir=Path(args.out),
        rules_path=args.rules,
        metrics=_metric_list(args.metrics) or None,
        calibration=calibration,
        frame_size=int(frame_size or 2048),
        hop_size=int(hop_size or 512),
        sample_rate=resolved_sr,
        chunk_size=int(chunk_size or 131072),
        seed=args.seed,
        max_chunks=args.max_chunks,
        stream_report_path=args.stream_report,
        pre_roll_s=float(args.pre_roll),
        post_roll_s=float(args.post_roll),
        merge_gap_s=float(args.merge_gap),
        min_alerts_per_chunk=int(args.min_alerts_per_chunk),
        selection_mode=selection_mode,
        top_k=top_k,
        rank_metric=str(args.rank_metric),
        rank_scope=str(getattr(args, "rank_scope", "downmix")),
        event_window_s=(float(args.event_window) if args.event_window is not None else None),
        window_before_s=(float(args.window_before) if args.window_before is not None else None),
        window_after_s=(float(args.window_after) if args.window_after is not None else None),
        max_clips=args.max_clips,
        csv_out=args.csv,
        clips_dir=args.clips_dir,
        report_out=args.report,
    )
    report_path, report = run_moments_extract(cfg)
    print(f"moments_report: {report_path}")
    print(
        "summary:",
        {
            "clips_written": report.get("clips_written"),
            "windows_selected": report.get("windows_selected"),
            "selection_mode": report.get("selection_mode"),
            "rank_metric": report.get("rank_metric"),
            "csv_path": report.get("csv_path"),
            "clips_dir": report.get("clips_dir"),
            "stream_report_path": report.get("stream_report_path"),
        },
    )
    return 0


def _run_insights_scene(args: argparse.Namespace) -> int:
    from esl.core.insights import run_scene_changes

    result = run_scene_changes(
        Path(args.input),
        Path(args.out),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
        threshold_z=float(args.threshold_z),
        max_changes=args.max_changes,
        feature_set=str(args.feature_set),
    )
    print(f"scene_changes_json: {result.primary}")
    print(f"scene_changes_csv: {result.report.get('csv_path')}")
    return 0


def _run_insights_calmness(args: argparse.Namespace) -> int:
    from esl.core.insights import run_calmness

    result = run_calmness(
        Path(args.input),
        Path(args.out),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
    )
    print(f"calmness_json: {result.primary}")
    print(
        "summary:",
        {
            "calmness": result.report.get("calmness_score"),
            "chaos": result.report.get("chaos_score"),
            "diversity": result.report.get("diversity_score"),
        },
    )
    return 0


def _run_insights_spatial(args: argparse.Namespace) -> int:
    from esl.core.insights import run_spatial_timeline

    result = run_spatial_timeline(
        Path(args.input),
        Path(args.out),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
    )
    print(f"spatial_timeline_json: {result.primary}")
    print(f"spatial_timeline_csv: {result.report.get('csv_path')}")
    return 0


def _run_insights_occupancy(args: argparse.Namespace) -> int:
    from esl.core.insights import run_bio_occupancy

    result = run_bio_occupancy(
        Path(args.input),
        Path(args.out),
        bands=str(args.bands),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
        threshold_ratio=float(args.threshold_ratio),
    )
    print(f"bio_occupancy_json: {result.primary}")
    print(f"bio_occupancy_csv: {result.report.get('csv_path')}")
    return 0


def _run_insights_drift(args: argparse.Namespace) -> int:
    from esl.core.insights import run_archive_drift

    result = run_archive_drift(Path(args.baseline_report), Path(args.candidate_report), Path(args.out))
    print(f"archive_drift_json: {result.primary}")
    print("summary:", {"drift_score": result.report.get("drift_score")})
    return 0


def _run_insights_retrieve(args: argparse.Namespace) -> int:
    from esl.core.insights import run_event_retrieval

    result = run_event_retrieval(
        Path(args.query),
        Path(args.corpus_dir),
        Path(args.out),
        top_k=int(args.top_k),
        mode=str(args.mode),
        metric=str(args.metric),
        metrics=_metric_list(args.metrics),
        distance=str(args.distance),
        feature_set=str(args.feature_set),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
        max_files=args.max_files,
    )
    print(f"event_retrieval_json: {result.primary}")
    print(f"event_retrieval_csv: {result.report.get('csv_path')}")
    return 0


def _run_insights_embeddings(args: argparse.Namespace) -> int:
    from esl.core.insights import run_embeddings

    result = run_embeddings(
        Path(args.input_dir),
        Path(args.out),
        feature_set=str(args.feature_set),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
        max_files=args.max_files,
        device=str(args.device),
    )
    print(f"embeddings_manifest: {result.primary}")
    print(f"embeddings_npz: {result.report.get('npz_path')}")
    print(f"embeddings_csv: {result.report.get('csv_path')}")
    return 0


def _run_insights_report(args: argparse.Namespace) -> int:
    from esl.core.insights import run_soundscape_report

    result = run_soundscape_report(Path(args.analysis_json), Path(args.out))
    print(f"soundscape_report_json: {result.primary}")
    print(f"soundscape_report_html: {result.report.get('html_path')}")
    return 0


def _run_insights_simulation_compare(args: argparse.Namespace) -> int:
    from esl.core.insights import run_simulation_compare

    result = run_simulation_compare(Path(args.simulated_json), Path(args.measured_json), Path(args.out))
    print(f"simulation_compare_json: {result.primary}")
    return 0


def _run_insights_storyboard(args: argparse.Namespace) -> int:
    from esl.core.insights import run_storyboard

    result = run_storyboard(
        Path(args.input),
        Path(args.out),
        clips=int(args.clips),
        window_s=float(args.window),
        frame_size=int(args.frame_size),
        hop_size=int(args.hop_size),
        sample_rate=args.sample_rate,
        feature_set=str(args.feature_set),
        write_clips=not bool(args.no_clips),
    )
    print(f"storyboard_json: {result.primary}")
    print(f"storyboard_csv: {result.report.get('csv_path')}")
    return 0


def _run_ingest(args: argparse.Namespace) -> int:
    from esl.ingest import ingest

    cfg = IngestConfig(
        source=args.source,
        query=args.query,
        limit=args.limit,
        output_dir=Path(args.out),
        auto_analyze=args.auto_analyze,
    )
    manifest = ingest(cfg)
    print(f"ingested {manifest['num_items']} items")
    print(f"manifest: {manifest['manifest_path']}")

    if args.auto_analyze and manifest["items"]:
        out_dir = Path(args.out) / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        reg = create_registry()
        analyzed = 0
        for item in manifest["items"]:
            local_path = item.get("local_path")
            if not local_path:
                continue
            p = Path(local_path)
            if not p.exists():
                continue
            try:
                acfg = AnalysisConfig(
                    input_path=p,
                    output_dir=out_dir,
                    verbosity=0,
                    debug=0,
                    seed=42,
                )
                result = analyze(acfg, registry=reg)
                save_json(result, out_dir / f"{p.stem}.json")
                analyzed += 1
            except Exception:
                continue
        print(f"auto-analyzed: {analyzed}")

    return 0


def _run_pipeline_run(args: argparse.Namespace) -> int:
    from esl.pipeline import PipelineRunConfig, run_pipeline

    cfg = PipelineRunConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.out),
        calibration_path=args.calibration,
        metrics=_metric_list(args.metrics),
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        seed=args.seed,
        compute_device=args.device,
        plot=args.plot,
        interactive=args.interactive,
        plot_metrics=_metric_list(args.plot_metrics),
        include_spectral=not args.no_spectral,
        include_similarity_matrix=args.similarity_matrix,
        include_novelty_matrix=args.novelty_matrix,
        show_plots=args.show,
        show_limit=args.show_limit,
        ml_export=args.ml_export,
        project=args.project,
        force=args.force,
        stages=_stage_list(args.stages) or None,
    )
    manifest_path, manifest = run_pipeline(cfg)
    print(f"pipeline status: {manifest.get('status')}")
    print(f"manifest: {manifest_path}")
    for stage, payload in manifest.get("stages", {}).items():
        print(f"- {stage}: {payload.get('status')} counts={payload.get('counts', {})}")
    return 0


def _run_pipeline_status(args: argparse.Namespace) -> int:
    from esl.pipeline import read_pipeline_status

    payload = read_pipeline_status(args.manifest)
    print(f"pipeline_id: {payload.get('pipeline_id')}")
    print(f"status: {payload.get('status')}")
    print(f"created_utc: {payload.get('created_utc')}")
    print(f"updated_utc: {payload.get('updated_utc')}")
    print("stages:")
    for stage, info in payload.get("stages", {}).items():
        print(f"- {stage}: {info.get('status')} ({info.get('duration_s')}s)")
    return 0


def _run_docs(args: argparse.Namespace) -> int:
    formats = {x.lower() for x in _csv_list(args.formats)}
    report = build_docs(
        root=Path(args.root),
        output_root=Path(args.out),
        formats=formats,
        title=args.title,
    )
    print(f"docs root: {report.root}")
    print(f"html artifacts: {len(report.html_pages)} -> {report.output_root / 'html'}")
    print(f"pdf artifacts: {len(report.pdf_pages)} -> {report.output_root / 'pdf'}")
    return 0


def _run_doctor(args: argparse.Namespace) -> int:
    from esl.core.doctor import DoctorConfig, run_doctor

    input_path = Path(args.input) if args.input else None
    if input_path is not None and not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    report = run_doctor(
        DoctorConfig(
            input_path=input_path,
            requested_device=args.device,
            strict=bool(args.strict),
        )
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"doctor_json: {out}")

    print(f"status: {report.get('status')}")
    print(f"esl_version: {report.get('esl_version')}")
    platform_info = report.get("platform", {})
    print(
        "platform:",
        {
            "python": platform_info.get("python"),
            "platform": platform_info.get("platform"),
        },
    )
    core = report.get("core_dependencies", {})
    print(
        "core_dependencies:",
        {
            "soundfile": core.get("soundfile", {}).get("installed"),
            "ffmpeg": core.get("ffmpeg", {}).get("available"),
            "ffprobe": core.get("ffprobe", {}).get("available"),
        },
    )
    device = report.get("device", {})
    print(
        "device:",
        {
            "requested": device.get("requested"),
            "resolved": device.get("resolved"),
            "torch_available": device.get("torch_available"),
            "device_name": device.get("device_name"),
        },
    )
    if report.get("input"):
        inp = report["input"]
        print(
            "input:",
            {
                "path": inp.get("path"),
                "format": inp.get("format_name"),
                "duration_s": inp.get("duration_s"),
                "channels": inp.get("channels"),
                "sample_rate": inp.get("sample_rate"),
                "layout_hint": inp.get("layout_hint"),
                "size_gb": inp.get("size_gb"),
            },
        )
    if report.get("blockers"):
        print("blockers:")
        for item in report["blockers"]:
            print(f"- {item}")
    if report.get("warnings"):
        print("warnings:")
        for item in report["warnings"]:
            print(f"- {item}")
    if report.get("recommendations"):
        print("recommendations:")
        for item in report["recommendations"]:
            print(f"- {item}")
    return 2 if bool(args.strict) and (report.get("blockers") or report.get("warnings")) else 0


def _run_simple(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    calibration = load_calibration(args.calibration) if args.calibration else None
    frame_size, hop_size, chunk_size, resolved_sr = _resolve_window_samples(
        args,
        input_path=input_path,
        default_frame_size=2048,
        default_hop_size=512,
        default_chunk_size=None,
    )
    cfg = AnalysisConfig(
        input_path=input_path,
        output_dir=Path("."),
        frame_size=int(frame_size or 2048),
        hop_size=int(hop_size or 512),
        sample_rate=resolved_sr,
        chunk_size=chunk_size,
        metrics=["rms_dbfs", "peak_dbfs", "spl_a_db", "snr_db"],
        calibration=calibration,
        verbosity=0,
        debug=0,
        seed=42,
        compute_device=args.device,
    )
    result = analyze(cfg)
    meta = result.get("metadata", {})
    metrics = result.get("metrics", {})
    validity = meta.get("validity_flags", {})

    def _fmt(metric_name: str) -> str:
        payload = metrics.get(metric_name, {})
        if not isinstance(payload, dict):
            return "N/A"
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            return "N/A"
        value = summary.get("mean")
        units = payload.get("units")
        if not isinstance(value, (int, float)):
            return "N/A"
        return f"{float(value):.2f} {units or ''}".strip()

    print(f"file: {meta.get('input_path')}")
    print(
        "signal:",
        {
            "duration_s": round(float(meta.get("duration_s", 0.0)), 3),
            "channels": meta.get("channels"),
            "sample_rate": meta.get("sample_rate"),
            "layout_hint": meta.get("channel_layout_hint"),
        },
    )
    print(
        "summary:",
        {
            "rms_dbfs": _fmt("rms_dbfs"),
            "peak_dbfs": _fmt("peak_dbfs"),
            "spl_a_db": _fmt("spl_a_db"),
            "snr_db": _fmt("snr_db"),
            "clipping": validity.get("clipping"),
            "calibration_applied": validity.get("calibration_applied"),
        },
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"json: {out}")
    return 0


def _run_benchmark_device(args: argparse.Namespace) -> int:
    from esl.ml import benchmark_tensor_backend

    report = benchmark_tensor_backend(
        device=args.device,
        channels=args.channels,
        frames=args.frames,
        features=args.features,
        iters=args.iters,
        seed=args.seed,
        strict=bool(args.strict),
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"benchmark_json: {out}")
    print(
        "summary:",
        {
            "backend": report.get("backend"),
            "device": report.get("device", {}).get("resolved"),
            "seconds_per_iter": report.get("seconds_per_iter"),
            "throughput_mel_per_s": report.get("throughput_mel_per_s"),
        },
    )
    if args.debug >= 1:
        print(json.dumps(report, indent=2))
    return 0


def _run_quickstart(args: argparse.Namespace) -> int:
    in_file = str(args.input or "input.wav")
    long_file = str(args.long_input or in_file)
    in_arg = f'"{in_file}"' if any(ch.isspace() for ch in in_file) else in_file
    long_arg = f'"{long_file}"' if any(ch.isspace() for ch in long_file) else long_file
    goals: dict[str, list[str]] = {
        "doctor": [
            "1) Check your environment first:",
            f"   esl doctor {in_arg}",
        ],
        "analyze": [
            "1) Analyze one file (JSON + plots):",
            f"   esl analyze {in_arg} --out-dir out --plot --json out/{Path(in_file).stem}.json",
        ],
        "moments": [
            "1) Extract interesting moments as clips + CSV:",
            f"   esl moments extract {in_arg} --out out/moments --single --rank-metric novelty_curve --event-window 8",
        ],
        "features": [
            "1) Extract ML-ready feature vectors:",
            f"   esl features extract {in_arg} --out out/vectors.npz --feature-set all --meta-json out/vectors_meta.json",
        ],
        "similar": [
            "1) Find the most similar files in a folder:",
            f"   esl similar {in_arg} corpus_dir --top-k 5 --json out/similarity.json --csv out/similarity.csv",
        ],
        "batch": [
            "1) Batch analyze a folder:",
            "   esl batch input_dir --out out_batch --csv --parquet --hdf5 --mat --plot",
        ],
        "long": [
            "1) Inspect the long file first:",
            f"   esl doctor {long_arg}",
            "",
            "2) Run out-of-core chunked analysis:",
            f"   esl analyze {long_arg} --out-dir out --chunk-minutes 10 --streamable-only --summary-only --frame-table-csv out/frame_table.csv --checkpoint-dir out/checkpoints --resume",
            "",
            "3) Extract the most novel event safely:",
            f"   esl moments extract {long_arg} --out out/moments --single --rank-metric novelty_curve --chunk-minutes 10 --event-window 8",
        ],
    }
    lines = ["ecoSignalLab Quickstart", ""]
    if args.goal == "all":
        lines.extend(
            [
                "You have one audio file and want results fast. Start here:",
                "",
                "1) Check your environment:",
                f"   esl doctor {in_arg}",
                "",
                "2) Analyze one file (JSON + plots):",
                f"   esl analyze {in_arg} --out-dir out --plot --json out/{Path(in_file).stem}.json",
                "",
                "3) Extract interesting moments as clips + CSV:",
                f"   esl moments extract {in_arg} --out out/moments --single --rank-metric novelty_curve --event-window 8",
                "",
                "4) Extract ML-ready feature vectors:",
                f"   esl features extract {in_arg} --out out/vectors.npz --feature-set all --meta-json out/vectors_meta.json",
                "",
                "5) If the file is huge, use long-file mode:",
                f"   esl quickstart --goal long --input {long_arg}",
            ]
        )
    else:
        lines.extend(goals[str(args.goal)])
    lines.extend(
        [
            "",
            "Need help with a command? Use:",
            "   esl <command> --help",
            "",
            "Beginner docs:",
            "   docs/GETTING_STARTED.md",
            "   docs/TASK_RECIPES.md",
            "   docs/TROUBLESHOOTING.md",
            "   docs/ANNOUNCEMENT_FAQ.md",
            "",
            "If decode fails on compressed audio, install ffmpeg and ensure ffprobe is on PATH.",
        ]
    )
    for line in lines:
        print(line)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esl",
        description="ecoSignalLab CLI for acoustic analysis, ML export, and reproducible reporting.",
        epilog=(
            "First time here? Run: esl doctor or esl quickstart\n"
            "Decode behavior: native formats use soundfile first; compressed formats fall back to ffmpeg/ffprobe.\n"
            "Compute device: use --device auto|cpu|cuda|mps on analyze/batch/spatial/pipeline run.\n"
            "Calibration file keys: dbfs_reference, spl_reference_db, weighting (A|C|Z), "
            "mic_sensitivity_mv_pa, preamp_gain_db, adc_full_scale_vrms, calibration_tone_file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # doctor
    pdoc = sub.add_parser("doctor", help="Check environment, dependencies, and optional input readiness")
    pdoc.add_argument("input", nargs="?", default=None, help="Optional input audio file to inspect")
    pdoc.add_argument("--json-out", default=None, help="Optional doctor report JSON path")
    pdoc.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference to validate for tensor workflows: auto|cpu|cuda|mps",
    )
    pdoc.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero if warnings or blockers are found",
    )
    pdoc.set_defaults(func=_run_doctor)

    # simple
    psimple = sub.add_parser("simple", help="Print a compact human-readable summary for one file")
    psimple.add_argument("input", help="Input audio file path")
    psimple.add_argument("--json-out", default=None, help="Optional full analysis JSON output path")
    psimple.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    psimple.add_argument("--sample-rate", type=int, default=None, help="Optional target sample rate")
    psimple.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    psimple.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    psimple.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    psimple.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    psimple.add_argument("--chunk-size", type=int, default=None, help="Chunk size in samples")
    psimple.add_argument("--chunk-seconds", type=float, default=None, help="Chunk size in seconds (overrides --chunk-size)")
    psimple.add_argument("--chunk-minutes", type=float, default=None, help="Chunk size in minutes (overrides --chunk-size)")
    psimple.add_argument("--chunk-hours", type=float, default=None, help="Chunk size in hours (overrides --chunk-size)")
    psimple.add_argument("--chunk-days", type=float, default=None, help="Chunk size in days (overrides --chunk-size)")
    psimple.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute backend preference for ML/tensor workflows: auto|cpu|cuda|mps",
    )
    psimple.set_defaults(func=_run_simple)

    # analyze
    pa = sub.add_parser("analyze", help="Analyze an audio file")
    pa.add_argument(
        "input",
        help=(
            "Input audio file path.\n"
            "Decoding: soundfile for native formats; ffmpeg fallback for compressed formats."
        ),
    )
    pa.add_argument("--json", dest="json", default=None, help="JSON output path (default: <out-dir>/<stem>.json)")
    pa.add_argument("--csv", dest="csv", default=None, help="Summary CSV output path")
    pa.add_argument("--series-csv", dest="series_csv", default=None, help="Frame/series CSV output path")
    pa.add_argument("--parquet", dest="parquet", default=None, help="Summary Parquet output path")
    pa.add_argument("--hdf5", dest="hdf5", default=None, help="HDF5 output path")
    pa.add_argument("--mat", dest="mat", default=None, help="MATLAB .mat output path")
    pa.add_argument("--head-csv", dest="head_csv", default=None, help="HEAD-compatible CSV path")
    pa.add_argument("--apx-csv", dest="apx_csv", default=None, help="APx-compatible CSV path")
    pa.add_argument("--soundcheck-csv", dest="soundcheck_csv", default=None, help="SoundCheck-compatible CSV path")
    pa.add_argument(
        "--calibration",
        dest="calibration",
        default=None,
        help=(
            "Calibration YAML/JSON path.\n"
            "Supports 0 dBFS to SPL mapping, weighting (A/C/Z), mic sensitivity, preamp gain, "
            "ADC full-scale voltage, and calibration tone."
        ),
    )
    pa.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=silent, 1=summary, 2=detailed, 3=full diagnostic",
    )
    pa.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal metric traces",
    )
    pa.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute backend preference for ML/tensor workflows: auto|cpu|cuda|mps",
    )
    pa.add_argument("--plot", action="store_true", help="Generate plots")
    pa.add_argument("--interactive", action="store_true", help="Generate Plotly interactive plots")
    pa.add_argument("--plot-metrics", default=None, help="Comma-separated metrics to include in plots")
    pa.add_argument("--no-spectral", action="store_true", help="Skip spectrogram/mel/log/waterfall/LTSA plots")
    pa.add_argument("--similarity-matrix", action="store_true", help="Generate self-similarity matrix plot")
    pa.add_argument("--novelty-matrix", action="store_true", help="Generate novelty matrix plot")
    pa.add_argument(
        "--sim-feature-set",
        default="auto",
        choices=["auto", "core", "librosa", "all"],
        help="Feature set for similarity/novelty matrices: auto|core|librosa|all",
    )
    pa.add_argument(
        "--feature-vectors",
        default=None,
        help="Optional feature vectors (.npz/.npy/.csv) for similarity/novelty matrix plots",
    )
    pa.add_argument("--show", action="store_true", help="Open generated plots with the system default viewer")
    pa.add_argument("--show-limit", type=int, default=12, help="Maximum number of plots to open with --show")
    pa.add_argument("--ml-export", action="store_true", help="Export ML-ready features")
    pa.add_argument("--project", default=None, help="Project name")
    pa.add_argument("--variant", default=None, help="Variant name")
    pa.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    pa.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    pa.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    pa.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    pa.add_argument("--sample-rate", type=int, default=None)
    pa.add_argument("--chunk-size", type=int, default=None, help="Chunk size in samples (enables chunked mode)")
    pa.add_argument("--chunk-seconds", type=float, default=None, help="Chunk size in seconds (overrides --chunk-size)")
    pa.add_argument("--chunk-minutes", type=float, default=None, help="Chunk size in minutes (overrides --chunk-size)")
    pa.add_argument("--chunk-hours", type=float, default=None, help="Chunk size in hours (overrides --chunk-size)")
    pa.add_argument("--chunk-days", type=float, default=None, help="Chunk size in days (overrides --chunk-size)")
    pa.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit frame series from JSON and keep only summaries; recommended for very long files",
    )
    pa.add_argument(
        "--streamable-only",
        action="store_true",
        help="Keep only streaming-capable metrics; useful with --chunk-* for long-duration analysis",
    )
    pa.add_argument(
        "--allow-full-read",
        action="store_true",
        help="Allow fallback to full-file loading when selected metrics are not streaming-capable",
    )
    pa.add_argument(
        "--max-series-points",
        type=int,
        default=None,
        help="Maximum frame-series points kept in JSON before truncation",
    )
    pa.add_argument(
        "--frame-table-csv",
        default=None,
        help="Write canonical FrameTable CSV incrementally during chunked analysis",
    )
    pa.add_argument(
        "--frame-table-parquet-dir",
        default=None,
        help="Write canonical FrameTable as an appendable Parquet dataset directory during chunked analysis",
    )
    pa.add_argument(
        "--frame-table-hdf5",
        default=None,
        help="Write canonical FrameTable as an appendable HDF5 file during chunked analysis",
    )
    pa.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory for resumable chunk-analysis checkpoints",
    )
    pa.add_argument(
        "--resume",
        action="store_true",
        help="Resume chunked analysis from --checkpoint-dir if a checkpoint exists",
    )
    pa.add_argument("--metrics", default=None, help="Comma-separated metric list")
    pa.add_argument(
        "--profile",
        default=None,
        help=(
            "Multi-resolution profile YAML/JSON path. "
            "Runs multiple analysis resolutions and writes a profile index JSON."
        ),
    )
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--out-dir", default=".")
    pa.set_defaults(func=_run_analyze)

    # batch
    pb = sub.add_parser("batch", help="Batch analyze an input directory")
    pb.add_argument("input_dir", help="Input directory")
    pb.add_argument("--out", required=True, help="Output directory")
    pb.add_argument("--calibration", dest="calibration", default=None)
    pb.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=silent, 1=summary, 2=detailed, 3=full diagnostic",
    )
    pb.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal metric traces",
    )
    pb.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute backend preference for ML/tensor workflows: auto|cpu|cuda|mps",
    )
    pb.add_argument("--plot", action="store_true")
    pb.add_argument("--interactive", action="store_true")
    pb.add_argument("--plot-metrics", default=None, help="Comma-separated metrics to include in plots")
    pb.add_argument("--no-spectral", action="store_true", help="Skip spectral plots in batch mode")
    pb.add_argument("--similarity-matrix", action="store_true", help="Generate self-similarity matrix plots")
    pb.add_argument("--novelty-matrix", action="store_true", help="Generate novelty matrix plots")
    pb.add_argument(
        "--sim-feature-set",
        default="auto",
        choices=["auto", "core", "librosa", "all"],
        help="Feature set for similarity/novelty matrices: auto|core|librosa|all",
    )
    pb.add_argument(
        "--feature-vectors",
        default=None,
        help="Optional feature vectors (.npz/.npy/.csv) for similarity/novelty matrix plots",
    )
    pb.add_argument("--show", action="store_true", help="Open generated plots with the system default viewer")
    pb.add_argument("--show-limit", type=int, default=12, help="Maximum number of plots to open with --show")
    pb.add_argument("--ml-export", action="store_true")
    pb.add_argument("--project", default=None)
    pb.add_argument("--variant", default=None)
    pb.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    pb.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    pb.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    pb.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    pb.add_argument("--sample-rate", type=int, default=None)
    pb.add_argument("--chunk-size", type=int, default=None, help="Chunk size in samples (enables chunked mode)")
    pb.add_argument("--chunk-seconds", type=float, default=None, help="Chunk size in seconds (overrides --chunk-size)")
    pb.add_argument("--chunk-minutes", type=float, default=None, help="Chunk size in minutes (overrides --chunk-size)")
    pb.add_argument("--chunk-hours", type=float, default=None, help="Chunk size in hours (overrides --chunk-size)")
    pb.add_argument("--chunk-days", type=float, default=None, help="Chunk size in days (overrides --chunk-size)")
    pb.add_argument("--metrics", default=None)
    pb.add_argument(
        "--report-metrics",
        default="snr_db,spl_a_db,rt60_s",
        help=(
            "Comma-separated metric IDs to include as <metric>_mean columns in batch_index.csv "
            "(for example: rms_dbfs,novelty_curve,spl_a_db)."
        ),
    )
    pb.add_argument("--seed", type=int, default=42)
    pb.add_argument("--csv", action="store_true", help="Write CSV per file")
    pb.add_argument("--parquet", action="store_true", help="Write Parquet per file")
    pb.add_argument("--hdf5", action="store_true", help="Write HDF5 per file")
    pb.add_argument("--mat", action="store_true", help="Write MATLAB .mat per file")
    pb.add_argument("--no-recursive", action="store_true")
    pb.add_argument("--out-dir", default=".", help=argparse.SUPPRESS)
    pb.set_defaults(func=_run_batch)

    # plot
    pp = sub.add_parser("plot", help="Generate plots from analysis JSON")
    pp.add_argument("results_json", help="Analysis JSON file")
    pp.add_argument("--out", required=True, help="Output plot directory")
    pp.add_argument("--interactive", action="store_true")
    pp.add_argument("--audio", default=None, help="Optional source audio path")
    pp.add_argument("--metrics", default=None, help="Comma-separated metric filter for plotting")
    pp.add_argument("--no-spectral", action="store_true", help="Skip spectral plot suite")
    pp.add_argument("--similarity-matrix", action="store_true", help="Generate self-similarity matrix plot")
    pp.add_argument("--novelty-matrix", action="store_true", help="Generate novelty matrix plot")
    pp.add_argument(
        "--sim-feature-set",
        default="auto",
        choices=["auto", "core", "librosa", "all"],
        help="Feature set for similarity/novelty matrices: auto|core|librosa|all",
    )
    pp.add_argument(
        "--feature-vectors",
        default=None,
        help="Optional feature vectors (.npz/.npy/.csv) for similarity/novelty matrix plots",
    )
    pp.add_argument("--show", action="store_true", help="Open generated plots with the system default viewer")
    pp.add_argument("--show-limit", type=int, default=12, help="Maximum number of plots to open with --show")
    pp.set_defaults(func=_run_plot)

    # similar
    psim = sub.add_parser("similar", help="Find the N most similar files in a folder to an input file")
    psim.add_argument("input", help="Query input audio file")
    psim.add_argument("corpus_dir", help="Folder of candidate files to compare against")
    psim.add_argument("--top-k", type=int, default=5, help="Number of most-similar files to report")
    psim.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "feature", "metric", "metrics"],
        help="Similarity mode: auto(feature), feature, metric(single), or metrics(multi)",
    )
    psim.add_argument("--metric", default="novelty_curve", help="Single metric ID for --mode metric")
    psim.add_argument("--metrics", default=None, help="Comma-separated metric IDs for --mode metrics")
    psim.add_argument(
        "--distance",
        default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance function for vector comparison",
    )
    psim.add_argument(
        "--feature-set",
        default="auto",
        choices=["auto", "core", "librosa", "all"],
        help="Feature extraction set for feature mode",
    )
    psim.add_argument("--frame-size", type=int, default=1024)
    psim.add_argument("--hop-size", type=int, default=256)
    psim.add_argument("--sample-rate", type=int, default=None)
    psim.add_argument("--normalize", dest="normalize", action="store_true", default=True, help="Normalize multi-metric vectors before distance")
    psim.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable multi-metric normalization")
    psim.add_argument("--include-self", action="store_true", help="Allow query file to appear in results if inside corpus")
    psim.add_argument("--no-recursive", action="store_true", help="Scan corpus directory non-recursively")
    psim.add_argument("--max-files", type=int, default=None, help="Optional cap on scanned candidate files")
    psim.add_argument("--calibration", default=None, help="Calibration YAML/JSON path (used for metric-based modes)")
    psim.add_argument("--json", default=None, help="Output JSON path (default: <out-dir>/<input_stem>_similarity.json)")
    psim.add_argument("--csv", default=None, help="Optional CSV output path")
    psim.add_argument("--seed", type=int, default=42)
    psim.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=silent, 1=summary, 2=detailed, 3=full diagnostic",
    )
    psim.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal traces",
    )
    psim.add_argument("--out-dir", default=".")
    psim.set_defaults(func=_run_similar)

    # ingest
    pi = sub.add_parser("ingest", help="Ingest online audio datasets")
    pi.add_argument("--source", required=True, choices=["freesound", "huggingface", "http"])
    pi.add_argument("--query", required=True, help="Query text, dataset id, or URL list")
    pi.add_argument("--limit", type=int, default=20)
    pi.add_argument("--out", default="ingest")
    pi.add_argument("--auto-analyze", action="store_true")
    pi.set_defaults(func=_run_ingest)

    # validate
    pv = sub.add_parser("validate", help="Run dataset regression/quality validation checks")
    pv.add_argument("input_dir", help="Input directory to validate")
    pv.add_argument("--out", required=True, help="Output directory for validation reports")
    pv.add_argument("--rules", default=None, help="Validation rules JSON/YAML path")
    pv.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    pv.add_argument("--metrics", default=None, help="Comma-separated metric subset")
    pv.add_argument("--frame-size", type=int, default=2048)
    pv.add_argument("--hop-size", type=int, default=512)
    pv.add_argument("--sample-rate", type=int, default=None)
    pv.add_argument("--chunk-size", type=int, default=None)
    pv.add_argument("--seed", type=int, default=42)
    pv.add_argument("--no-recursive", action="store_true")
    pv.set_defaults(func=_run_validate)

    # stream
    pst = sub.add_parser("stream", help="Run streaming-style chunk analysis with alert rules")
    pst.add_argument("input", help="Input audio file for chunked streaming analysis")
    pst.add_argument("--out", default="stream_out", help="Output directory")
    pst.add_argument("--rules", default=None, help="Alert rules JSON/YAML path")
    pst.add_argument("--metrics", default="spl_a_db,ndsi,novelty_curve", help="Comma-separated metric list")
    pst.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    pst.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    pst.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    pst.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    pst.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    pst.add_argument("--sample-rate", type=int, default=None)
    pst.add_argument("--chunk-size", type=int, default=131072, help="Chunk size in samples")
    pst.add_argument("--chunk-seconds", type=float, default=None, help="Chunk size in seconds (overrides --chunk-size)")
    pst.add_argument("--chunk-minutes", type=float, default=None, help="Chunk size in minutes (overrides --chunk-size)")
    pst.add_argument("--chunk-hours", type=float, default=None, help="Chunk size in hours (overrides --chunk-size)")
    pst.add_argument("--chunk-days", type=float, default=None, help="Chunk size in days (overrides --chunk-size)")
    pst.add_argument("--checkpoint-dir", default=None, help="Directory for resumable stream checkpoints")
    pst.add_argument("--resume", action="store_true", help="Resume stream analysis from --checkpoint-dir")
    pst.add_argument(
        "--chunks-jsonl",
        default=None,
        help="Path for disk-backed chunk summaries (default: <out>/stream_chunks.jsonl)",
    )
    pst.add_argument("--seed", type=int, default=42)
    pst.add_argument("--max-chunks", type=int, default=None, help="Optional cap on processed chunks")
    pst.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=silent, 1=summary, 2=detailed, 3=full diagnostic",
    )
    pst.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal metric traces",
    )
    pst.set_defaults(func=_run_stream)

    # spatial
    psp = sub.add_parser("spatial", help="Spatial and ambisonic analysis commands")
    psp_sub = psp.add_subparsers(dest="spatial_cmd", required=True)

    psp_an = psp_sub.add_parser("analyze", help="Analyze spatial metrics and optional beam map")
    psp_an.add_argument("input", help="Input audio file path")
    psp_an.add_argument("--json", default=None, help="JSON output path (default: <out-dir>/<stem>_spatial.json)")
    psp_an.add_argument("--array-config", default=None, help="Array config JSON/YAML path")
    psp_an.add_argument("--metrics", default=None, help="Comma-separated spatial metric list")
    psp_an.add_argument("--doa", action="store_true", help="Force inclusion of DOA/ITD metrics")
    psp_an.add_argument("--beam-map", action="store_true", help="Generate stereo delay-and-sum beam map CSV")
    psp_an.add_argument("--beam-map-csv", default=None, help="Beam map CSV output path")
    psp_an.add_argument("--azimuth-step-deg", type=int, default=5, help="Beam map azimuth step in degrees")
    psp_an.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    psp_an.add_argument("--project", default=None, help="Project name")
    psp_an.add_argument("--variant", default=None, help="Variant name")
    psp_an.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    psp_an.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    psp_an.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    psp_an.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    psp_an.add_argument("--sample-rate", type=int, default=None)
    psp_an.add_argument("--chunk-size", type=int, default=None, help="Chunk size in samples")
    psp_an.add_argument("--chunk-seconds", type=float, default=None, help="Chunk size in seconds (overrides --chunk-size)")
    psp_an.add_argument("--chunk-minutes", type=float, default=None, help="Chunk size in minutes (overrides --chunk-size)")
    psp_an.add_argument("--chunk-hours", type=float, default=None, help="Chunk size in hours (overrides --chunk-size)")
    psp_an.add_argument("--chunk-days", type=float, default=None, help="Chunk size in days (overrides --chunk-size)")
    psp_an.add_argument("--seed", type=int, default=42)
    psp_an.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=silent, 1=summary, 2=detailed, 3=full diagnostic",
    )
    psp_an.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal metric traces",
    )
    psp_an.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute backend preference for ML/tensor workflows: auto|cpu|cuda|mps",
    )
    psp_an.add_argument("--out-dir", default=".")
    psp_an.set_defaults(func=_run_spatial_analyze)

    # calibrate
    pcal = sub.add_parser("calibrate", help="Calibration tooling")
    pcal_sub = pcal.add_subparsers(dest="calibrate_cmd", required=True)

    pcal_check = pcal_sub.add_parser("check", help="Check calibration drift from a tone recording")
    pcal_check.add_argument("--tone", required=True, help="Calibration tone audio file")
    pcal_check.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    pcal_check.add_argument("--dbfs-reference", type=float, default=None, help="Expected tone level in dBFS")
    pcal_check.add_argument("--spl-reference-db", type=float, default=None, help="Reference SPL for dbfs mapping")
    pcal_check.add_argument("--weighting", default=None, help="Weighting hint (A/C/Z)")
    pcal_check.add_argument("--mic-sensitivity-mv-pa", type=float, default=None, help="Mic sensitivity metadata")
    pcal_check.add_argument(
        "--preamp-gain-db",
        type=float,
        default=None,
        help="Analog preamp gain in dB (required with mic sensitivity + ADC FS for Pa<->dBFS conversion)",
    )
    pcal_check.add_argument(
        "--adc-full-scale-vrms",
        type=float,
        default=None,
        help="ADC full-scale RMS voltage in volts (required with mic sensitivity + gain for Pa<->dBFS conversion)",
    )
    pcal_check.add_argument("--sample-rate", type=int, default=None, help="Optional resample rate for tone read")
    pcal_check.add_argument("--max-drift-db", type=float, default=1.0, help="Pass/fail absolute drift threshold")
    pcal_check.add_argument("--device-id", default=None, help="Device identifier for history tracking")
    pcal_check.add_argument("--history", default=None, help="History CSV path to append checks")
    pcal_check.add_argument("--out", default="calibration_check.json", help="Calibration report JSON path")
    pcal_check.set_defaults(func=_run_calibrate_check)

    pcal_verify = pcal_sub.add_parser("verify", help="Verify calibration math/check path with deterministic reference fixtures")
    pcal_verify.add_argument(
        "--fixture",
        default="sine_1khz_minus20dbfs",
        choices=[
            "sine_1khz_minus20dbfs",
            "sine_1khz_minus26dbfs",
            "sine_250hz_minus20dbfs",
            "sine_4khz_minus20dbfs",
            "sine_1khz_minus12dbfs",
            "sine_1khz_minus20dbfs_precision_chain",
        ],
        help="Built-in deterministic reference fixture",
    )
    pcal_verify.add_argument("--calibration", default=None, help="Optional calibration YAML/JSON path")
    pcal_verify.add_argument("--max-abs-error-db", type=float, default=0.25, help="Maximum allowed absolute dBFS error")
    pcal_verify.add_argument("--write-tone", default=None, help="Optional path to keep the synthesized reference tone WAV")
    pcal_verify.add_argument("--out", default="calibration_verify.json", help="Calibration verification report JSON path")
    pcal_verify.set_defaults(func=_run_calibrate_verify)

    # features
    pfeat = sub.add_parser("features", help="Feature vector extraction commands")
    pfeat_sub = pfeat.add_subparsers(dest="features_cmd", required=True)

    pfeat_ex = pfeat_sub.add_parser("extract", help="Extract frame-level feature vectors")
    pfeat_ex.add_argument("input", help="Input audio file path")
    pfeat_ex.add_argument("--out", required=True, help="Output feature vectors (.npz/.npy/.csv)")
    pfeat_ex.add_argument(
        "--feature-set",
        default="auto",
        choices=["auto", "core", "librosa", "all"],
        help="Feature extraction set: auto|core|librosa|all",
    )
    pfeat_ex.add_argument("--frame-size", type=int, default=1024)
    pfeat_ex.add_argument("--hop-size", type=int, default=256)
    pfeat_ex.add_argument("--n-mels", type=int, default=64)
    pfeat_ex.add_argument("--sample-rate", type=int, default=None)
    pfeat_ex.add_argument("--meta-json", default=None, help="Optional metadata JSON sidecar path")
    pfeat_ex.set_defaults(func=_run_features_extract)

    pfeat_mf = pfeat_sub.add_parser("manifest", help="Build an ML dataset manifest from exported *_ml_metadata.json files")
    pfeat_mf.add_argument("input_dir", help="Directory containing exported ML metadata sidecars")
    pfeat_mf.add_argument("--out", required=True, help="Output dataset manifest JSON path")
    pfeat_mf.add_argument("--pattern", default="*_ml_metadata.json", help="Glob pattern for metadata sidecars")
    pfeat_mf.add_argument(
        "--split-ratios",
        default="0.8,0.1,0.1",
        help="Comma-separated train,val,test ratios used deterministically over sorted samples",
    )
    pfeat_mf.set_defaults(func=_run_features_manifest)

    # moments
    pmom = sub.add_parser("moments", help="Find and export interesting timestamped moments as clips")
    pmom_sub = pmom.add_subparsers(dest="moments_cmd", required=True)

    pmom_ex = pmom_sub.add_parser("extract", help="Extract moments to WAV clips + CSV from alert criteria")
    pmom_ex.add_argument("input", help="Input audio file path")
    pmom_ex.add_argument("--out", default="moments_out", help="Output directory")
    pmom_ex.add_argument("--stream-report", default=None, help="Optional precomputed stream_report.json")
    pmom_ex.add_argument("--rules", default=None, help="Alert rules JSON/YAML path (used when stream report not provided)")
    pmom_ex.add_argument(
        "--metrics",
        default="novelty_curve,spectral_change_detection,isolation_forest_score,spl_a_db",
        help="Comma-separated metrics for detection pass",
    )
    pmom_ex.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    pmom_ex.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    pmom_ex.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    pmom_ex.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    pmom_ex.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    pmom_ex.add_argument("--sample-rate", type=int, default=None)
    pmom_ex.add_argument("--chunk-size", type=int, default=131072, help="Detection chunk size in samples")
    pmom_ex.add_argument("--chunk-seconds", type=float, default=None, help="Detection chunk size in seconds (overrides --chunk-size)")
    pmom_ex.add_argument("--chunk-minutes", type=float, default=None, help="Detection chunk size in minutes (overrides --chunk-size)")
    pmom_ex.add_argument("--chunk-hours", type=float, default=None, help="Detection chunk size in hours (overrides --chunk-size)")
    pmom_ex.add_argument("--chunk-days", type=float, default=None, help="Detection chunk size in days (overrides --chunk-size)")
    pmom_ex.add_argument("--seed", type=int, default=42)
    pmom_ex.add_argument("--max-chunks", type=int, default=None, help="Optional cap for detection chunks")
    pmom_ex.add_argument("--pre-roll", type=float, default=3.0, help="Seconds before each detected chunk")
    pmom_ex.add_argument("--post-roll", type=float, default=3.0, help="Seconds after each detected chunk")
    pmom_ex.add_argument("--merge-gap", type=float, default=2.0, help="Merge windows separated by <= this many seconds")
    pmom_ex.add_argument("--min-alerts-per-chunk", type=int, default=1, help="Minimum alerts needed for chunk selection")
    select_group = pmom_ex.add_mutually_exclusive_group()
    select_group.add_argument("--single", action="store_true", help="Extract only the single highest-ranked moment")
    select_group.add_argument("--top-k", type=int, default=None, help="Extract top K highest-ranked moments")
    select_group.add_argument("--all", action="store_true", help="Extract all detected moments (default)")
    pmom_ex.add_argument("--rank-metric", default="novelty_curve", help="Metric used to rank moments (default: novelty_curve)")
    pmom_ex.add_argument(
        "--rank-scope",
        default="downmix",
        choices=["downmix", "per_channel_max", "per_channel_mean"],
        help="Rank on downmix, max per-channel metric, or mean per-channel metric",
    )
    pmom_ex.add_argument(
        "--event-window",
        type=float,
        default=None,
        help="Symmetric window duration in seconds around event center (overrides chunk-edge rolls when set)",
    )
    pmom_ex.add_argument("--window-before", type=float, default=None, help="Seconds before event center for each clip")
    pmom_ex.add_argument("--window-after", type=float, default=None, help="Seconds after event center for each clip")
    pmom_ex.add_argument("--max-clips", type=int, default=None, help="Legacy alias for --top-k")
    pmom_ex.add_argument("--csv", default=None, help="Output CSV path (default: <out>/moments.csv)")
    pmom_ex.add_argument("--clips-dir", default=None, help="Output clips directory (default: <out>/clips)")
    pmom_ex.add_argument("--report", default=None, help="Output moments report JSON path")
    pmom_ex.set_defaults(func=_run_moments_extract)

    # insights
    pins = sub.add_parser("insights", help="Higher-level soundscape insight workflows")
    pins_sub = pins.add_subparsers(dest="insights_cmd", required=True)

    pins_scene = pins_sub.add_parser("scene", help="Detect acoustic scene-change candidates")
    pins_scene.add_argument("input", help="Input audio file path")
    pins_scene.add_argument("--out", required=True, help="Output directory for scene_changes.json/csv")
    pins_scene.add_argument("--feature-set", default="auto", choices=["auto", "core", "librosa", "all"])
    pins_scene.add_argument("--frame-size", type=int, default=2048)
    pins_scene.add_argument("--hop-size", type=int, default=512)
    pins_scene.add_argument("--sample-rate", type=int, default=None)
    pins_scene.add_argument("--threshold-z", type=float, default=1.5, help="Peak threshold in z-score units")
    pins_scene.add_argument("--max-changes", type=int, default=None, help="Optional maximum number of changes")
    pins_scene.set_defaults(func=_run_insights_scene)

    pins_calm = pins_sub.add_parser("calmness", help="Estimate calmness, chaos, and acoustic diversity")
    pins_calm.add_argument("input", help="Input audio file path")
    pins_calm.add_argument("--out", required=True, help="Output calmness JSON path")
    pins_calm.add_argument("--frame-size", type=int, default=2048)
    pins_calm.add_argument("--hop-size", type=int, default=512)
    pins_calm.add_argument("--sample-rate", type=int, default=None)
    pins_calm.set_defaults(func=_run_insights_calmness)

    pins_sp = pins_sub.add_parser("spatial", help="Write a frame-wise multichannel spatial activity timeline")
    pins_sp.add_argument("input", help="Input audio file path")
    pins_sp.add_argument("--out", required=True, help="Output directory for spatial_timeline.json/csv")
    pins_sp.add_argument("--frame-size", type=int, default=2048)
    pins_sp.add_argument("--hop-size", type=int, default=512)
    pins_sp.add_argument("--sample-rate", type=int, default=None)
    pins_sp.set_defaults(func=_run_insights_spatial)

    pins_occ = pins_sub.add_parser("occupancy", help="Estimate acoustic occupancy by named frequency band")
    pins_occ.add_argument("input", help="Input audio file path")
    pins_occ.add_argument("--out", required=True, help="Output directory for bio_occupancy.json/csv")
    pins_occ.add_argument("--bands", default="anthro:20-1000,bio:2000-8000", help="Comma bands like anthro:20-1000,bio:2000-8000")
    pins_occ.add_argument("--threshold-ratio", type=float, default=0.2, help="Band/total energy ratio counted as occupied")
    pins_occ.add_argument("--frame-size", type=int, default=4096)
    pins_occ.add_argument("--hop-size", type=int, default=2048)
    pins_occ.add_argument("--sample-rate", type=int, default=None)
    pins_occ.set_defaults(func=_run_insights_occupancy)

    pins_drift = pins_sub.add_parser("drift", help="Compare two analysis/shard reports for archive drift")
    pins_drift.add_argument("baseline_report", help="Baseline analysis JSON or shard_analysis_report.json")
    pins_drift.add_argument("candidate_report", help="Candidate analysis JSON or shard_analysis_report.json")
    pins_drift.add_argument("--out", required=True, help="Output drift JSON path")
    pins_drift.set_defaults(func=_run_insights_drift)

    pins_ret = pins_sub.add_parser("retrieve", help="Query-by-example event/file retrieval from a corpus folder")
    pins_ret.add_argument("query", help="Query audio file")
    pins_ret.add_argument("corpus_dir", help="Folder of candidate audio files")
    pins_ret.add_argument("--out", required=True, help="Output directory for event_retrieval.json/csv")
    pins_ret.add_argument("--top-k", type=int, default=5)
    pins_ret.add_argument("--mode", default="auto", choices=["auto", "feature", "metric", "metrics"])
    pins_ret.add_argument("--metric", default="novelty_curve", help="Single metric for --mode metric")
    pins_ret.add_argument("--metrics", default=None, help="Comma-separated metrics for --mode metrics")
    pins_ret.add_argument("--distance", default="cosine", choices=["cosine", "euclidean", "manhattan"])
    pins_ret.add_argument("--feature-set", default="auto", choices=["auto", "core", "librosa", "all"])
    pins_ret.add_argument("--frame-size", type=int, default=1024)
    pins_ret.add_argument("--hop-size", type=int, default=256)
    pins_ret.add_argument("--sample-rate", type=int, default=None)
    pins_ret.add_argument("--max-files", type=int, default=None)
    pins_ret.set_defaults(func=_run_insights_retrieve)

    pins_emb = pins_sub.add_parser("embeddings", help="Build deterministic clip-level feature embeddings")
    pins_emb.add_argument("input_dir", help="Input directory of audio files")
    pins_emb.add_argument("--out", required=True, help="Output directory for embeddings.npz/csv/manifest")
    pins_emb.add_argument("--feature-set", default="auto", choices=["auto", "core", "librosa", "all"])
    pins_emb.add_argument("--frame-size", type=int, default=1024)
    pins_emb.add_argument("--hop-size", type=int, default=256)
    pins_emb.add_argument("--sample-rate", type=int, default=None)
    pins_emb.add_argument("--max-files", type=int, default=None)
    pins_emb.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    pins_emb.set_defaults(func=_run_insights_embeddings)

    pins_rep = pins_sub.add_parser("report", help="Generate a compact HTML soundscape report from analysis JSON")
    pins_rep.add_argument("analysis_json", help="Input esl analysis JSON")
    pins_rep.add_argument("--out", required=True, help="Output directory for soundscape_report.html/json")
    pins_rep.set_defaults(func=_run_insights_report)

    pins_cmp = pins_sub.add_parser("simulation-compare", help="Compare simulated and measured analysis JSON files")
    pins_cmp.add_argument("simulated_json", help="Simulation analysis JSON")
    pins_cmp.add_argument("measured_json", help="Measured/field analysis JSON")
    pins_cmp.add_argument("--out", required=True, help="Output comparison JSON path")
    pins_cmp.set_defaults(func=_run_insights_simulation_compare)

    pins_story = pins_sub.add_parser("storyboard", help="Create timestamped storyboard clips from high-change moments")
    pins_story.add_argument("input", help="Input audio file path")
    pins_story.add_argument("--out", required=True, help="Output directory for storyboard.json/csv/clips")
    pins_story.add_argument("--clips", type=int, default=12, help="Number of storyboard moments")
    pins_story.add_argument("--window", type=float, default=5.0, help="Seconds around each selected moment")
    pins_story.add_argument("--feature-set", default="auto", choices=["auto", "core", "librosa", "all"])
    pins_story.add_argument("--frame-size", type=int, default=2048)
    pins_story.add_argument("--hop-size", type=int, default=512)
    pins_story.add_argument("--sample-rate", type=int, default=None)
    pins_story.add_argument("--no-clips", action="store_true", help="Write CSV/JSON only; do not export WAV clips")
    pins_story.set_defaults(func=_run_insights_storyboard)

    # schema
    ps = sub.add_parser("schema", help="Print/write output JSON schema")
    ps.add_argument("--out", default=None, help="Output schema path (prints schema_version and path)")
    ps.set_defaults(func=_run_schema)

    # shard
    psh = sub.add_parser("shard", help="Shard-manifest workflows for long-duration archives")
    psh_sub = psh.add_subparsers(dest="shard_cmd", required=True)

    psh_index = psh_sub.add_parser("index", help="Build an ordered shard manifest from a directory of audio files")
    psh_index.add_argument("input_dir", help="Directory containing shard audio files")
    psh_index.add_argument("--out", required=True, help="Output shard manifest JSON path")
    psh_index.add_argument(
        "--order-by",
        default="path",
        choices=["path", "mtime"],
        help="Shard ordering: path or file modification time",
    )
    psh_index.add_argument("--no-recursive", action="store_true", help="Scan only the top-level directory")
    psh_index.set_defaults(func=_run_shard_index)

    psh_an = psh_sub.add_parser("analyze", help="Analyze an ordered shard manifest as one archive")
    psh_an.add_argument("manifest", help="Path to shard manifest JSON")
    psh_an.add_argument("--out", required=True, help="Output directory for shard analysis products")
    psh_an.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    psh_an.add_argument("--metrics", default=None, help="Comma-separated metric list")
    psh_an.add_argument(
        "--report-metrics",
        default="rms_dbfs,snr_db,ndsi",
        help="Comma-separated metric IDs to summarize in the archive index/report",
    )
    psh_an.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    psh_an.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    psh_an.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    psh_an.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    psh_an.add_argument("--sample-rate", type=int, default=None)
    psh_an.add_argument("--chunk-size", type=int, default=None, help="Chunk size in samples")
    psh_an.add_argument("--chunk-seconds", type=float, default=None, help="Chunk size in seconds (overrides --chunk-size)")
    psh_an.add_argument("--chunk-minutes", type=float, default=None, help="Chunk size in minutes (overrides --chunk-size)")
    psh_an.add_argument("--chunk-hours", type=float, default=None, help="Chunk size in hours (overrides --chunk-size)")
    psh_an.add_argument("--chunk-days", type=float, default=None, help="Chunk size in days (overrides --chunk-size)")
    psh_an.add_argument("--summary-only", action="store_true", help="Omit frame series from per-shard JSON outputs")
    psh_an.add_argument("--streamable-only", action="store_true", help="Use only streaming-capable metrics")
    psh_an.add_argument("--allow-full-read", action="store_true", help="Allow full-read fallback for non-streaming metrics")
    psh_an.add_argument("--max-series-points", type=int, default=None, help="Maximum frame-series points kept in each shard JSON")
    psh_an.add_argument("--frame-table-dir", default=None, help="Directory for per-shard FrameTable CSV sidecars")
    psh_an.add_argument("--frame-table-parquet-dir", default=None, help="Directory for per-shard Parquet FrameTable datasets")
    psh_an.add_argument("--frame-table-hdf5-dir", default=None, help="Directory for per-shard HDF5 FrameTable files")
    psh_an.add_argument("--checkpoint-dir", default=None, help="Root directory for per-shard checkpoint state")
    psh_an.add_argument("--resume", action="store_true", help="Resume shard analysis using per-shard checkpoints")
    psh_an.add_argument("--force", action="store_true", help="Recompute shard outputs even if present")
    psh_an.add_argument("--seed", type=int, default=42)
    psh_an.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute backend preference for ML/tensor workflows: auto|cpu|cuda|mps",
    )
    psh_an.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal traces",
    )
    psh_an.set_defaults(func=_run_shard_analyze)

    psh_mom = psh_sub.add_parser("moments", help="Find top-ranked moments across an ordered shard manifest")
    psh_mom.add_argument("manifest", help="Path to shard manifest JSON")
    psh_mom.add_argument("--out", required=True, help="Output directory for archive-level moments products")
    psh_mom.add_argument("--stream-root", default=None, help="Optional root directory for per-shard stream reports")
    psh_mom.add_argument("--rules", default=None, help="Optional alert rules JSON/YAML path for shard stream passes")
    psh_mom.add_argument("--metrics", default="novelty_curve,spectral_change_detection,spl_a_db", help="Comma-separated metrics for shard stream pass")
    psh_mom.add_argument("--calibration", default=None, help="Calibration YAML/JSON path")
    psh_mom.add_argument("--frame-size", type=int, default=2048, help="Frame size in samples")
    psh_mom.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    psh_mom.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    psh_mom.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    psh_mom.add_argument("--sample-rate", type=int, default=None)
    psh_mom.add_argument("--chunk-size", type=int, default=131072, help="Detection chunk size in samples")
    psh_mom.add_argument("--chunk-seconds", type=float, default=None, help="Detection chunk size in seconds (overrides --chunk-size)")
    psh_mom.add_argument("--chunk-minutes", type=float, default=None, help="Detection chunk size in minutes (overrides --chunk-size)")
    psh_mom.add_argument("--chunk-hours", type=float, default=None, help="Detection chunk size in hours (overrides --chunk-size)")
    psh_mom.add_argument("--chunk-days", type=float, default=None, help="Detection chunk size in days (overrides --chunk-size)")
    psh_mom.add_argument("--seed", type=int, default=42)
    psh_mom.add_argument("--max-chunks", type=int, default=None, help="Optional cap for detection chunks per shard")
    psh_mom.add_argument("--pre-roll", type=float, default=3.0, help="Seconds before each detected chunk")
    psh_mom.add_argument("--post-roll", type=float, default=3.0, help="Seconds after each detected chunk")
    psh_mom.add_argument("--merge-gap", type=float, default=2.0, help="Merge windows separated by <= this many seconds")
    psh_mom.add_argument("--min-alerts-per-chunk", type=int, default=1, help="Minimum alerts needed for candidate chunk selection")
    psh_mom.add_argument("--single", action="store_true", help="Extract only the single highest-ranked archive moment")
    psh_mom.add_argument("--top-k", type=int, default=None, help="Extract top K highest-ranked archive moments")
    psh_mom.add_argument("--rank-metric", default="novelty_curve", help="Metric used to rank archive moments")
    psh_mom.add_argument(
        "--rank-scope",
        default="downmix",
        choices=["downmix", "per_channel_max", "per_channel_mean"],
        help="Rank on downmix, max per-channel metric, or mean per-channel metric",
    )
    psh_mom.add_argument("--event-window", type=float, default=None, help="Symmetric window duration in seconds around event center")
    psh_mom.add_argument("--window-before", type=float, default=None, help="Seconds before event center for each clip")
    psh_mom.add_argument("--window-after", type=float, default=None, help="Seconds after event center for each clip")
    psh_mom.add_argument("--force-stream", action="store_true", help="Recompute shard stream reports even if present under --stream-root")
    psh_mom.add_argument("--resume", action="store_true", help="Resume shard stream passes if checkpoints are present")
    psh_mom.add_argument("--report", default=None, help="Output archive moments report JSON path")
    psh_mom.set_defaults(func=_run_shard_moments)

    psh_sim = psh_sub.add_parser("similar", help="Find the most similar shards in an ordered archive manifest")
    psh_sim.add_argument("manifest", help="Path to shard manifest JSON")
    psh_sim.add_argument("query", help="Query input audio file")
    psh_sim.add_argument("--out", required=True, help="Output directory for shard similarity products")
    psh_sim.add_argument("--top-k", type=int, default=5, help="Number of most-similar shards to report")
    psh_sim.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "feature", "metric", "metrics"],
        help="Similarity mode: auto(feature), feature, metric(single), or metrics(multi)",
    )
    psh_sim.add_argument("--metric", default="novelty_curve", help="Single metric ID for --mode metric")
    psh_sim.add_argument("--metrics", default=None, help="Comma-separated metric IDs for --mode metrics")
    psh_sim.add_argument(
        "--distance",
        default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="Distance function for feature/multi-metric similarity",
    )
    psh_sim.add_argument(
        "--feature-set",
        default="auto",
        choices=["auto", "core", "librosa", "all"],
        help="Feature extraction set for feature mode: auto|core|librosa|all",
    )
    psh_sim.add_argument("--frame-size", type=int, default=1024, help="Frame size in samples")
    psh_sim.add_argument("--hop-size", type=int, default=256, help="Hop size in samples")
    psh_sim.add_argument("--frame-seconds", type=float, default=None, help="Frame size in seconds (overrides --frame-size)")
    psh_sim.add_argument("--hop-seconds", type=float, default=None, help="Hop size in seconds (overrides --hop-size)")
    psh_sim.add_argument("--sample-rate", type=int, default=None)
    psh_sim.add_argument("--normalize", dest="normalize", action="store_true", default=True, help="Normalize multi-metric vectors before distance")
    psh_sim.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable multi-metric normalization")
    psh_sim.add_argument("--calibration", default=None, help="Calibration YAML/JSON path (used for metric-based modes)")
    psh_sim.add_argument("--max-shards", type=int, default=None, help="Optional cap on scanned shard candidates")
    psh_sim.add_argument("--include-query", action="store_true", help="Allow the query file to appear in results if it is also a shard")
    psh_sim.add_argument(
        "--spatial-mode",
        default="off",
        choices=["off", "append", "only"],
        help="Use spatial metrics for archive retrieval: off, append to base similarity, or spatial-only",
    )
    psh_sim.add_argument(
        "--spatial-metrics",
        default="interchannel_coherence,iacc,ild_db,itd_s,doa_azimuth_proxy_deg,ambisonic_diffuseness,ambisonic_energy_vector_azimuth_deg,ambisonic_energy_vector_elevation_deg",
        help="Comma-separated spatial metrics used when --spatial-mode is append or only",
    )
    psh_sim.add_argument("--spatial-weight", type=float, default=0.5, help="Blend weight for spatial distance when --spatial-mode append")
    psh_sim.add_argument("--json", default=None, help="Output JSON path (default: <out>/<query_stem>_shard_similarity.json)")
    psh_sim.add_argument("--csv", default=None, help="Optional CSV output path")
    psh_sim.add_argument("--seed", type=int, default=42)
    psh_sim.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help="Verbosity level: 0=silent, 1=summary, 2=detailed, 3=full diagnostic",
    )
    psh_sim.add_argument(
        "--debug",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Debug level: 0=none, 1=processing details, 2=internal traces",
    )
    psh_sim.set_defaults(func=_run_shard_similar)

    psh_plot = psh_sub.add_parser("plot", help="Render archive-scale plots from shard_analysis_report.json")
    psh_plot.add_argument("report", help="Path to shard_analysis_report.json")
    psh_plot.add_argument("--out", required=True, help="Output directory for archive PNG plots")
    psh_plot.set_defaults(func=_run_shard_plot)

    # project
    pproj = sub.add_parser("project", help="Project mode reports and comparisons")
    pproj_sub = pproj.add_subparsers(dest="project_cmd", required=True)

    pproj_cmp = pproj_sub.add_parser("compare", help="Compare project variants from project index")
    pproj_cmp.add_argument("--project", required=True, help="Project name")
    pproj_cmp.add_argument("--root", default=".", help="Root containing projects/<name>/index.json")
    pproj_cmp.add_argument("--baseline", default=None, help="Baseline variant (default: first recorded)")
    pproj_cmp.add_argument("--metrics", default=None, help="Comma-separated metric subset")
    pproj_cmp.add_argument("--json", dest="json_out", default=None, help="Comparison JSON output path")
    pproj_cmp.add_argument("--csv", dest="csv_out", default=None, help="Comparison CSV output path")
    pproj_cmp.set_defaults(func=_run_project_compare)

    # pipeline
    ppl = sub.add_parser("pipeline", help="Run/status staged CLI pipeline")
    ppl_sub = ppl.add_subparsers(dest="pipeline_cmd", required=True)

    ppl_run = ppl_sub.add_parser("run", help="Run staged pipeline on an input directory")
    ppl_run.add_argument("input_dir", help="Input directory")
    ppl_run.add_argument("--out", required=True, help="Output directory")
    ppl_run.add_argument("--calibration", dest="calibration", default=None, help="Calibration YAML/JSON path")
    ppl_run.add_argument("--metrics", default=None, help="Comma-separated metric list")
    ppl_run.add_argument("--frame-size", type=int, default=2048)
    ppl_run.add_argument("--hop-size", type=int, default=512)
    ppl_run.add_argument("--sample-rate", type=int, default=None)
    ppl_run.add_argument("--chunk-size", type=int, default=None)
    ppl_run.add_argument("--seed", type=int, default=42)
    ppl_run.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute backend preference for ML/tensor workflows: auto|cpu|cuda|mps",
    )
    ppl_run.add_argument("--plot", action="store_true", help="Run plot stage")
    ppl_run.add_argument("--interactive", action="store_true", help="Interactive plot HTML in plot stage")
    ppl_run.add_argument("--plot-metrics", default=None, help="Comma-separated metrics to include in pipeline plots")
    ppl_run.add_argument("--no-spectral", action="store_true", help="Skip spectral plot suite in pipeline plot stage")
    ppl_run.add_argument("--similarity-matrix", action="store_true", help="Generate similarity matrix in pipeline plot stage")
    ppl_run.add_argument("--novelty-matrix", action="store_true", help="Generate novelty matrix in pipeline plot stage")
    ppl_run.add_argument("--show", action="store_true", help="Open generated plots with the system default viewer")
    ppl_run.add_argument("--show-limit", type=int, default=12, help="Maximum number of plots to open with --show")
    ppl_run.add_argument("--ml-export", action="store_true", help="Run ML export stage")
    ppl_run.add_argument("--project", default=None, help="Project name for provenance tagging")
    ppl_run.add_argument("--stages", default=None, help="Explicit stage list: analyze,plot,ml_export,digest")
    ppl_run.add_argument("--force", action="store_true", help="Recompute outputs even if present")
    ppl_run.set_defaults(func=_run_pipeline_run)

    ppl_status = ppl_sub.add_parser("status", help="Show pipeline manifest status")
    ppl_status.add_argument("--manifest", required=True, help="Path to pipeline_manifest.json")
    ppl_status.set_defaults(func=_run_pipeline_status)

    # docs
    pd = sub.add_parser("docs", help="Build documentation into hyperlink-rich HTML/PDF")
    pd.add_argument("--root", default=".", help="Repository root to scan for markdown docs")
    pd.add_argument("--out", default="docs/build", help="Output directory for generated docs")
    pd.add_argument("--formats", default="html,pdf", help="Comma-separated formats: html,pdf")
    pd.add_argument("--title", default="ecoSignalLab Documentation", help="Site/report title")
    pd.set_defaults(func=_run_docs)

    # quickstart
    pqs = sub.add_parser("quickstart", help="Print copy-paste commands for first-time users")
    pqs.add_argument(
        "--goal",
        default="all",
        choices=["all", "doctor", "analyze", "moments", "features", "similar", "batch", "long"],
        help="Print commands for one goal instead of the full starter set",
    )
    pqs.add_argument("--input", default="input.wav", help="Placeholder input filename to use in printed commands")
    pqs.add_argument(
        "--long-input",
        default=None,
        help="Placeholder long-file input filename to use in long-file commands",
    )
    pqs.set_defaults(func=_run_quickstart)

    # benchmark
    pbench = sub.add_parser("benchmark", help="Backend and tensor workload benchmarking")
    pbench_sub = pbench.add_subparsers(dest="benchmark_cmd", required=True)

    pbench_dev = pbench_sub.add_parser("device", help="Benchmark tensor workload on cpu/cuda/mps")
    pbench_dev.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    pbench_dev.add_argument("--channels", type=int, default=1)
    pbench_dev.add_argument("--frames", type=int, default=16384)
    pbench_dev.add_argument("--features", type=int, default=256)
    pbench_dev.add_argument("--iters", type=int, default=20)
    pbench_dev.add_argument("--seed", type=int, default=42)
    pbench_dev.add_argument("--strict", action="store_true", help="Fail if requested accelerator is unavailable")
    pbench_dev.add_argument("--json-out", default=None, help="Optional benchmark report JSON path")
    pbench_dev.add_argument("--debug", type=int, default=0, choices=[0, 1, 2])
    pbench_dev.set_defaults(func=_run_benchmark_device)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: verify the file/path exists and try again.", file=sys.stderr)
        print("hint: for command usage, run: esl <command> --help", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: check your option values and ranges in --help.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: run with --debug 1 (or --debug 2) for more details where available.", file=sys.stderr)
        print("hint: start with: esl doctor or esl quickstart", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
