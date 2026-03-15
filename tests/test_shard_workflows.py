from __future__ import annotations

import json
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
