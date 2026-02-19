from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data.astype(np.float32), sr)


def test_validate_cli_writes_reports(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    tone = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    _write_wav(in_dir / "tone.wav", tone, sr)

    code = main(
        [
            "validate",
            str(in_dir),
            "--out",
            str(out_dir),
            "--metrics",
            "rms_dbfs,snr_db",
        ]
    )
    assert code == 0
    report_path = out_dir / "validation_report.json"
    summary_path = out_dir / "validation_summary.csv"
    assert report_path.exists()
    assert summary_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(payload["files_checked"]) == 1
    assert int(payload["files_failed"]) == 0


def test_validate_cli_rule_failure_returns_nonzero(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    clean = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    clipped = np.clip(1.3 * np.sin(2.0 * np.pi * 440.0 * t), -1.0, 1.0)
    _write_wav(in_dir / "clean.wav", clean, sr)
    _write_wav(in_dir / "clipped.wav", clipped, sr)

    rules = {
        "metric_thresholds": {
            "clipping_ratio": {"max": 0.0},
        },
        "validity_flags": {"clipping": False},
    }
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")

    code = main(
        [
            "validate",
            str(in_dir),
            "--out",
            str(out_dir),
            "--rules",
            str(rules_path),
        ]
    )
    assert code == 2

    payload = json.loads((out_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert int(payload["files_checked"]) == 2
    assert int(payload["files_failed"]) >= 1
