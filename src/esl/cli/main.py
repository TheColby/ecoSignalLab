"""Command line interface for ecoSignalLab (esl)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from esl.core import AnalysisConfig, IngestConfig, analyze, load_calibration
from esl.core.audio import iter_supported_files
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


def _build_analysis_config(args: argparse.Namespace, input_path: Path, out_dir: Path) -> AnalysisConfig:
    calibration = load_calibration(args.calibration) if args.calibration else None
    return AnalysisConfig(
        input_path=input_path,
        output_dir=out_dir,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        metrics=_metric_list(args.metrics),
        calibration=calibration,
        project=args.project,
        variant=args.variant,
        verbosity=args.verbosity,
        debug=args.debug,
        seed=args.seed,
        make_plots=args.plot,
        ml_export=args.ml_export,
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
        artifacts = export_ml_features(result, output_dir=ml_dir, prefix=stem, seed=cfg.seed)
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

        rows.append(
            {
                "input": str(fp),
                "json": str(base),
                "duration_s": result["metadata"]["duration_s"],
                "channels": result["metadata"]["channels"],
                "sample_rate": result["metadata"]["sample_rate"],
                "snr_mean": result["metrics"].get("snr_db", {}).get("summary", {}).get("mean"),
                "spl_a_mean": result["metrics"].get("spl_a_db", {}).get("summary", {}).get("mean"),
                "rt60": result["metrics"].get("rt60_s", {}).get("summary", {}).get("mean"),
            }
        )

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

            export_ml_features(result, output_dir=run_out / f"{fp.stem}_ml", prefix=fp.stem, seed=cfg.seed)

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
                "snr_mean",
                "spl_a_mean",
                "rt60",
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
    cfg = StreamRunConfig(
        input_path=input_path,
        output_dir=out_dir,
        metrics=_metric_list(args.metrics),
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        calibration=calibration,
        seed=args.seed,
        rules_path=args.rules,
        max_chunks=args.max_chunks,
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

    metric_list = _metric_list(args.metrics) or list(SPATIAL_DEFAULT_METRICS)
    if args.doa and "doa_azimuth_proxy_deg" not in metric_list:
        metric_list.append("doa_azimuth_proxy_deg")
    if args.doa and "itd_s" not in metric_list:
        metric_list.append("itd_s")

    cfg = AnalysisConfig(
        input_path=input_path,
        output_dir=out_dir,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        metrics=metric_list,
        calibration=calibration,
        verbosity=args.verbosity,
        debug=args.debug,
        seed=args.seed,
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

    out_path = Path(args.out)
    cfg = CalibrationCheckConfig(
        tone_path=tone_path,
        output_path=out_path,
        dbfs_reference=dbfs_reference,
        spl_reference_db=spl_reference_db,
        weighting=weighting,
        mic_sensitivity_mv_pa=mic_sensitivity_mv_pa,
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
        },
    )
    return 0 if within_tolerance else 2


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


def _run_moments_extract(args: argparse.Namespace) -> int:
    from esl.core.moments import MomentsExtractConfig, run_moments_extract

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    calibration = load_calibration(args.calibration) if args.calibration else None

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
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="esl",
        description="ecoSignalLab CLI for acoustic analysis, ML export, and reproducible reporting.",
        epilog=(
            "Decode behavior: native formats use soundfile first; compressed formats fall back to ffmpeg/ffprobe.\n"
            "Calibration file keys: dbfs_reference, spl_reference_db, weighting (A|C|Z), "
            "mic_sensitivity_mv_pa, calibration_tone_file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

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
            "Supports 0 dBFS to SPL mapping, weighting (A/C/Z), mic sensitivity, and calibration tone."
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
    pa.add_argument("--frame-size", type=int, default=2048)
    pa.add_argument("--hop-size", type=int, default=512)
    pa.add_argument("--sample-rate", type=int, default=None)
    pa.add_argument("--chunk-size", type=int, default=None)
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
    pb.add_argument("--frame-size", type=int, default=2048)
    pb.add_argument("--hop-size", type=int, default=512)
    pb.add_argument("--sample-rate", type=int, default=None)
    pb.add_argument("--chunk-size", type=int, default=None)
    pb.add_argument("--metrics", default=None)
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
    pst.add_argument("--frame-size", type=int, default=2048)
    pst.add_argument("--hop-size", type=int, default=512)
    pst.add_argument("--sample-rate", type=int, default=None)
    pst.add_argument("--chunk-size", type=int, default=131072)
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
    psp_an.add_argument("--frame-size", type=int, default=2048)
    psp_an.add_argument("--hop-size", type=int, default=512)
    psp_an.add_argument("--sample-rate", type=int, default=None)
    psp_an.add_argument("--chunk-size", type=int, default=None)
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
    pcal_check.add_argument("--sample-rate", type=int, default=None, help="Optional resample rate for tone read")
    pcal_check.add_argument("--max-drift-db", type=float, default=1.0, help="Pass/fail absolute drift threshold")
    pcal_check.add_argument("--device-id", default=None, help="Device identifier for history tracking")
    pcal_check.add_argument("--history", default=None, help="History CSV path to append checks")
    pcal_check.add_argument("--out", default="calibration_check.json", help="Calibration report JSON path")
    pcal_check.set_defaults(func=_run_calibrate_check)

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
    pmom_ex.add_argument("--frame-size", type=int, default=2048)
    pmom_ex.add_argument("--hop-size", type=int, default=512)
    pmom_ex.add_argument("--sample-rate", type=int, default=None)
    pmom_ex.add_argument("--chunk-size", type=int, default=131072, help="Detection chunk size in samples")
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

    # schema
    ps = sub.add_parser("schema", help="Print/write output JSON schema")
    ps.add_argument("--out", default=None, help="Output schema path (prints schema_version and path)")
    ps.set_defaults(func=_run_schema)

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
