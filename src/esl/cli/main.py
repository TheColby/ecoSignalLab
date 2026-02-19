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
from esl.project import record_project_variant
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


def _run_analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = Path(args.out_dir)
    _mkdir(out_dir)

    cfg = _build_analysis_config(args, input_path=input_path, out_dir=out_dir)
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

    # schema
    ps = sub.add_parser("schema", help="Print/write output JSON schema")
    ps.add_argument("--out", default=None, help="Output schema path (prints schema_version and path)")
    ps.set_defaults(func=_run_schema)

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
