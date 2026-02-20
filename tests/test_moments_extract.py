from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data.astype(np.float32), sr, subtype="PCM_24")


def test_moments_extract_from_rules_writes_csv_and_clips(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(2 * sr, dtype=np.float64) / sr
    low = 0.05 * np.sin(2.0 * np.pi * 220.0 * t[:sr])
    high = 0.8 * np.sin(2.0 * np.pi * 220.0 * t[sr:])
    x = np.concatenate([low, high])
    wav = tmp_path / "long.wav"
    _write_wav(wav, x, sr)

    rules = {"metric_thresholds": {"rms_dbfs": {"min": -14.0}}}
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")

    out = tmp_path / "moments_out"
    code = main(
        [
            "moments",
            "extract",
            str(wav),
            "--out",
            str(out),
            "--rules",
            str(rules_path),
            "--metrics",
            "rms_dbfs",
            "--chunk-size",
            "8000",
            "--sample-rate",
            "16000",
            "--pre-roll",
            "0",
            "--post-roll",
            "0",
            "--merge-gap",
            "0",
        ]
    )
    assert code == 0

    report_path = out / "moments_report.json"
    csv_path = out / "moments.csv"
    clips_dir = out / "clips"
    assert report_path.exists()
    assert csv_path.exists()
    assert clips_dir.exists()
    clips = sorted(clips_dir.glob("moment_*.wav"))
    assert len(clips) >= 1

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) >= 1
    assert "start_hms" in rows[0]
    assert "chunk_indices" in rows[0]
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(payload["clips_written"]) >= 1
    assert int(payload["windows_selected"]) >= 1


def test_moments_extract_from_existing_stream_report(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.7 * np.sin(2.0 * np.pi * 330.0 * t)
    wav = tmp_path / "in.wav"
    _write_wav(wav, x, sr)

    rules = {"metric_thresholds": {"rms_dbfs": {"min": -8.0}}}
    rules_path = tmp_path / "stream_rules.json"
    rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")

    stream_out = tmp_path / "stream"
    code_stream = main(
        [
            "stream",
            str(wav),
            "--out",
            str(stream_out),
            "--rules",
            str(rules_path),
            "--metrics",
            "rms_dbfs",
            "--chunk-size",
            "4000",
            "--sample-rate",
            "16000",
            "--verbosity",
            "0",
        ]
    )
    assert code_stream == 0
    stream_report = stream_out / "stream_report.json"
    assert stream_report.exists()

    moments_out = tmp_path / "moments_from_stream"
    code_moments = main(
        [
            "moments",
            "extract",
            str(wav),
            "--out",
            str(moments_out),
            "--stream-report",
            str(stream_report),
            "--pre-roll",
            "0",
            "--post-roll",
            "0",
            "--merge-gap",
            "0",
        ]
    )
    assert code_moments == 0
    assert (moments_out / "moments.csv").exists()
    assert (moments_out / "moments_report.json").exists()
