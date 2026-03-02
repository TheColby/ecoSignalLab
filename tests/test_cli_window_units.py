from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from esl.cli.main import main


def _write_wav(path: Path, sr: int = 8000, seconds: float = 1.0) -> Path:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(path, x, sr)
    return path


def test_analyze_duration_windows_convert_to_samples(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "in.wav", sr=8000, seconds=1.0)

    code = main(
        [
            "analyze",
            str(wav),
            "--out-dir",
            str(tmp_path),
            "--metrics",
            "rms_dbfs",
            "--frame-seconds",
            "0.1",
            "--hop-seconds",
            "0.05",
            "--chunk-seconds",
            "0.2",
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "in.json").read_text(encoding="utf-8"))
    meta = payload.get("metadata", {})
    assert int(meta["sample_rate"]) == 8000
    assert int(meta["frame_size"]) == 800
    assert int(meta["hop_size"]) == 400
    snapshot = meta.get("config_snapshot", {})
    assert int(snapshot["chunk_size"]) == 1600


def test_stream_chunk_hours_overrides_chunk_size_samples(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "in.wav", sr=8000, seconds=2.0)
    out = tmp_path / "stream_out"
    code = main(
        [
            "stream",
            str(wav),
            "--out",
            str(out),
            "--metrics",
            "rms_dbfs",
            "--chunk-size",
            "1",
            "--chunk-hours",
            "0.001",
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    report = json.loads((out / "stream_report.json").read_text(encoding="utf-8"))
    assert int(report["sample_rate"]) == 8000
    assert int(report["chunk_size"]) == 28800


def test_multiple_chunk_duration_flags_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wav = _write_wav(tmp_path / "in.wav", sr=8000, seconds=0.5)
    code = main(
        [
            "analyze",
            str(wav),
            "--chunk-minutes",
            "1",
            "--chunk-hours",
            "1",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "Specify only one chunk-duration flag" in err
