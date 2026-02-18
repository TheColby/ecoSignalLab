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
from esl.schema import analysis_output_schema


def _metric_list(raw: str | None) -> list[str]:
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
        from esl.viz import plot_analysis

        plot_dir = out_dir / f"{stem}_plots"
        plots = plot_analysis(result, output_dir=plot_dir, audio_path=input_path, interactive=args.interactive)
        if cfg.verbosity >= 1:
            print(f"plots: {len(plots)} files -> {plot_dir}")

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
            m = metrics.get(name, {})
            return m.get("summary", {}).get("mean")

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

    files = iter_supported_files(in_dir, patterns=["*.wav", "*.flac", "*.aiff", "*.aif", "*.rf64", "*.caf", "*.mp3", "*.aac", "*.ogg", "*.opus", "*.wma", "*.alac", "*.m4a", "*.sofa"], recursive=not args.no_recursive)
    if not files:
        print("No supported files found.")
        return 0

    rows: list[dict[str, Any]] = []
    for fp in files:
        rel = fp.relative_to(in_dir)
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

            plot_analysis(result, output_dir=run_out / f"{fp.stem}_plots", audio_path=fp, interactive=args.interactive)

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
    return 0


def _run_plot(args: argparse.Namespace) -> int:
    from esl.viz import plot_from_json

    plots = plot_from_json(
        json_path=args.results_json,
        output_dir=args.out,
        interactive=args.interactive,
        audio_path=args.audio,
    )
    print(f"generated {len(plots)} plot files in {args.out}")
    return 0


def _run_schema(args: argparse.Namespace) -> int:
    schema = analysis_output_schema()
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(str(p))
    else:
        print(json.dumps(schema, indent=2))
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esl", description="ecoSignalLab CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    pa = sub.add_parser("analyze", help="Analyze an audio file")
    pa.add_argument("input", help="Input audio file path")
    pa.add_argument("--json", dest="json", default=None, help="JSON output path")
    pa.add_argument("--csv", dest="csv", default=None, help="Summary CSV output path")
    pa.add_argument("--series-csv", dest="series_csv", default=None, help="Series CSV output path")
    pa.add_argument("--parquet", dest="parquet", default=None, help="Parquet output path")
    pa.add_argument("--hdf5", dest="hdf5", default=None, help="HDF5 output path")
    pa.add_argument("--mat", dest="mat", default=None, help="MATLAB MAT output path")
    pa.add_argument("--head-csv", dest="head_csv", default=None, help="HEAD-compatible CSV path")
    pa.add_argument("--apx-csv", dest="apx_csv", default=None, help="APx-compatible CSV path")
    pa.add_argument("--soundcheck-csv", dest="soundcheck_csv", default=None, help="SoundCheck-compatible CSV path")
    pa.add_argument("--calibration", dest="calibration", default=None, help="Calibration YAML/JSON path")
    pa.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2, 3])
    pa.add_argument("--debug", type=int, default=0, choices=[0, 1, 2])
    pa.add_argument("--plot", action="store_true", help="Generate plots")
    pa.add_argument("--interactive", action="store_true", help="Generate Plotly interactive plots")
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
    pb.add_argument("--verbosity", type=int, default=1, choices=[0, 1, 2, 3])
    pb.add_argument("--debug", type=int, default=0, choices=[0, 1, 2])
    pb.add_argument("--plot", action="store_true")
    pb.add_argument("--interactive", action="store_true")
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
    pb.add_argument("--no-recursive", action="store_true")
    pb.add_argument("--out-dir", default=".")
    pb.set_defaults(func=_run_batch)

    # plot
    pp = sub.add_parser("plot", help="Generate plots from analysis JSON")
    pp.add_argument("results_json", help="Analysis JSON file")
    pp.add_argument("--out", required=True, help="Output plot directory")
    pp.add_argument("--interactive", action="store_true")
    pp.add_argument("--audio", default=None, help="Optional source audio path")
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
    ps.add_argument("--out", default=None, help="Output schema path")
    ps.set_defaults(func=_run_schema)

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
