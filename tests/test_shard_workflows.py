from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main
from esl.core.shards import ShardManifestConfig, build_shard_manifest


def _write_wav(path: Path, seconds: float, sr: int = 8000, freq: float = 220.0) -> Path:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, x, sr)
    return path


def test_build_shard_manifest_assigns_cumulative_offsets(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    _write_wav(archive / "0001.wav", seconds=1.0)
    _write_wav(archive / "0002.wav", seconds=2.0)

    manifest_path, manifest = build_shard_manifest(
        ShardManifestConfig(
            input_dir=archive,
            output_path=tmp_path / "manifest.json",
        )
    )

    assert manifest_path.exists()
    assert int(manifest["num_shards"]) == 2
    items = manifest["items"]
    assert items[0]["start_s"] == 0.0
    assert items[0]["end_s"] == 1.0
    assert items[1]["start_s"] == 1.0
    assert items[1]["end_s"] == 3.0


def test_cli_shard_analyze_writes_archive_report(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    _write_wav(archive / "a.wav", seconds=1.0, freq=220.0)
    _write_wav(archive / "b.wav", seconds=1.0, freq=440.0)

    manifest_path = tmp_path / "manifest.json"
    code_index = main(["shard", "index", str(archive), "--out", str(manifest_path)])
    assert code_index == 0

    out_dir = tmp_path / "shard_out"
    code_analyze = main(
        [
            "shard",
            "analyze",
            str(manifest_path),
            "--out",
            str(out_dir),
            "--metrics",
            "rms_dbfs,novelty_curve",
            "--report-metrics",
            "rms_dbfs,novelty_curve",
            "--chunk-seconds",
            "0.5",
            "--streamable-only",
            "--summary-only",
            "--frame-table-dir",
            str(out_dir / "frame_tables"),
            "--frame-table-parquet-dir",
            str(out_dir / "frame_tables_parquet"),
            "--frame-table-hdf5-dir",
            str(out_dir / "frame_tables_hdf5"),
            "--checkpoint-dir",
            str(out_dir / "checkpoints"),
        ]
    )
    assert code_analyze == 0

    report_path = out_dir / "shard_analysis_report.json"
    index_csv = out_dir / "shard_analysis_index.csv"
    assert report_path.exists()
    assert index_csv.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(report["num_shards"]) == 2
    assert int(report["processed"]) == 2
    assert "rms_dbfs" in report["weighted_metric_means"]
    shard_json = out_dir / "shards" / "a.json"
    assert shard_json.exists()
    frame_table_csv = out_dir / "frame_tables" / "a_frame_table.csv"
    assert frame_table_csv.exists()
    assert (out_dir / "frame_tables_hdf5" / "a_frame_table.h5").exists()
    if importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None:
        assert any((out_dir / "frame_tables_parquet").rglob("part-*.parquet"))


def test_cli_shard_moments_writes_archive_level_clips(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    sr = 8000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    quiet = (0.01 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    loud = (0.8 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(archive / "a.wav", quiet, sr)
    sf.write(archive / "b.wav", loud, sr)

    manifest_path = tmp_path / "manifest.json"
    assert main(["shard", "index", str(archive), "--out", str(manifest_path)]) == 0
    out_dir = tmp_path / "moments_out"
    code = main(
        [
            "shard",
            "moments",
            str(manifest_path),
            "--out",
            str(out_dir),
            "--top-k",
            "1",
            "--rank-metric",
            "novelty_curve",
            "--window-before",
            "0.1",
            "--window-after",
            "0.1",
            "--merge-gap",
            "0",
        ]
    )
    assert code == 0
    assert (out_dir / "moments.csv").exists()
    assert (out_dir / "archive_moments_report.json").exists()
    assert any((out_dir / "clips").glob("moment_*.wav"))


def test_cli_shard_similar_feature_mode_ranks_closest_shard(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr

    query = (0.2 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    close = (0.2 * np.sin(2.0 * np.pi * 440.0 * t + 0.05)).astype(np.float32)
    far = (0.2 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)

    query_path = tmp_path / "query.wav"
    sf.write(query_path, query, sr)
    sf.write(archive / "0001_close.wav", close, sr)
    sf.write(archive / "0002_far.wav", far, sr)

    manifest_path = tmp_path / "manifest.json"
    assert main(["shard", "index", str(archive), "--out", str(manifest_path)]) == 0

    out_dir = tmp_path / "similar_out"
    out_json = out_dir / "query_shard_similarity.json"
    out_csv = out_dir / "query_shard_similarity.csv"
    code = main(
        [
            "shard",
            "similar",
            str(manifest_path),
            str(query_path),
            "--out",
            str(out_dir),
            "--top-k",
            "2",
            "--json",
            str(out_json),
            "--csv",
            str(out_csv),
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    assert out_json.exists()
    assert out_csv.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["mode_used"] == "feature"
    assert payload["results"]
    assert payload["results"][0]["relative_path"] == "0001_close.wav"


def test_cli_shard_similar_metric_mode_single_metric(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr

    query = (0.10 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    close = (0.11 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
    far = (0.35 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)

    query_path = tmp_path / "query.wav"
    sf.write(query_path, query, sr)
    sf.write(archive / "close.wav", close, sr)
    sf.write(archive / "far.wav", far, sr)

    manifest_path = tmp_path / "manifest.json"
    assert main(["shard", "index", str(archive), "--out", str(manifest_path)]) == 0

    out_dir = tmp_path / "similar_metric_out"
    out_json = out_dir / "metric_similarity.json"
    code = main(
        [
            "shard",
            "similar",
            str(manifest_path),
            str(query_path),
            "--out",
            str(out_dir),
            "--mode",
            "metric",
            "--metric",
            "rms_dbfs",
            "--top-k",
            "1",
            "--json",
            str(out_json),
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["mode_used"] == "metric"
    assert payload["results"]
    assert payload["results"][0]["relative_path"] == "close.wav"


def test_cli_shard_similar_spatial_append_and_plot(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr

    query_left = 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
    query_right = np.roll(query_left, 2)
    query = np.stack([query_left, query_right], axis=1).astype(np.float32)

    close_left = 0.15 * np.sin(2.0 * np.pi * 430.0 * t)
    close_right = np.roll(close_left, 2)
    close = np.stack([close_left, close_right], axis=1).astype(np.float32)

    far_left = 0.15 * np.sin(2.0 * np.pi * 430.0 * t)
    far_right = np.roll(far_left, 40)
    far = np.stack([far_left, far_right], axis=1).astype(np.float32)

    query_path = tmp_path / "query.wav"
    sf.write(query_path, query, sr)
    sf.write(archive / "close.wav", close, sr)
    sf.write(archive / "far.wav", far, sr)

    manifest_path = tmp_path / "manifest.json"
    assert main(["shard", "index", str(archive), "--out", str(manifest_path)]) == 0

    similar_out = tmp_path / "similar_spatial"
    code = main(
        [
            "shard",
            "similar",
            str(manifest_path),
            str(query_path),
            "--out",
            str(similar_out),
            "--top-k",
            "2",
            "--spatial-mode",
            "append",
            "--spatial-weight",
            "0.7",
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    payload = json.loads((similar_out / "query_shard_similarity.json").read_text(encoding="utf-8"))
    assert payload["config"]["spatial_mode"] == "append"
    assert payload["results"]
    assert "distance_components" in payload["results"][0]

    analysis_out = tmp_path / "analysis_out"
    assert (
        main(
            [
                "shard",
                "analyze",
                str(manifest_path),
                "--out",
                str(analysis_out),
                "--metrics",
                "rms_dbfs,interchannel_coherence",
                "--report-metrics",
                "rms_dbfs,interchannel_coherence",
                "--summary-only",
                "--streamable-only",
            ]
        )
        == 0
    )
    plot_out = tmp_path / "archive_plots"
    assert (
        main(
            [
                "shard",
                "plot",
                str(analysis_out / "shard_analysis_report.json"),
                "--out",
                str(plot_out),
            ]
        )
        == 0
    )
    assert any(plot_out.glob("archive_metric_*.png"))
