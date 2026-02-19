from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.cli.main import main


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data.astype(np.float32), sr)


def test_analyze_profile_mode_writes_profile_index_and_runs(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    wav = tmp_path / "profile_in.wav"
    _write_wav(wav, x, sr)

    profile = {
        "profiles": [
            {"name": "short", "frame_size": 1024, "hop_size": 256, "metrics": ["rms_dbfs", "snr_db"]},
            {"name": "long", "frame_size": 4096, "hop_size": 1024, "metrics": ["rms_dbfs", "snr_db"]},
        ]
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    code = main(
        [
            "analyze",
            str(wav),
            "--profile",
            str(profile_path),
            "--out-dir",
            str(tmp_path),
            "--verbosity",
            "0",
        ]
    )
    assert code == 0

    index_path = tmp_path / "profile_in_profile.json"
    assert index_path.exists()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert int(payload["created_runs"]) == 2
    run_files = [Path(run["json"]) for run in payload["runs"]]
    assert all(p.exists() for p in run_files)


def test_stream_mode_generates_alerts(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x1 = 0.05 * np.sin(2.0 * np.pi * 220.0 * t[: sr // 2])
    x2 = 0.7 * np.sin(2.0 * np.pi * 220.0 * t[sr // 2 :])
    x = np.concatenate([x1, x2])
    wav = tmp_path / "stream_in.wav"
    _write_wav(wav, x, sr)

    rules = {"metric_thresholds": {"rms_dbfs": {"max": -8.0}}}
    rules_path = tmp_path / "stream_rules.json"
    rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")

    out_dir = tmp_path / "stream_out"
    code = main(
        [
            "stream",
            str(wav),
            "--out",
            str(out_dir),
            "--chunk-size",
            "8000",
            "--metrics",
            "rms_dbfs",
            "--rules",
            str(rules_path),
            "--verbosity",
            "0",
        ]
    )
    assert code == 0

    report_path = out_dir / "stream_report.json"
    alerts_csv = out_dir / "stream_alerts.csv"
    assert report_path.exists()
    assert alerts_csv.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(payload["chunks_processed"]) >= 2
    assert int(payload["alert_count"]) >= 1


def test_spatial_analyze_writes_json_and_beam_map(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    mono = 0.2 * np.sin(2.0 * np.pi * 400.0 * t)
    delayed = np.roll(mono, 2)
    stereo = np.stack([mono, delayed], axis=1)
    wav = tmp_path / "spatial.wav"
    _write_wav(wav, stereo, sr)

    array_cfg = {"geometry": "stereo", "mic_spacing_m": 0.18}
    array_cfg_path = tmp_path / "array.json"
    array_cfg_path.write_text(json.dumps(array_cfg, indent=2), encoding="utf-8")

    code = main(
        [
            "spatial",
            "analyze",
            str(wav),
            "--out-dir",
            str(tmp_path),
            "--array-config",
            str(array_cfg_path),
            "--beam-map",
            "--verbosity",
            "0",
        ]
    )
    assert code == 0

    result_path = tmp_path / "spatial_spatial.json"
    beam_map_path = tmp_path / "spatial_beam_map.csv"
    assert result_path.exists()
    assert beam_map_path.exists()

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert "doa_azimuth_proxy_deg" in payload["metrics"]
    spatial_meta = payload["metadata"].get("spatial", {})
    assert spatial_meta.get("array_config", {}).get("mic_spacing_m") == pytest.approx(0.18)


def test_calibrate_check_writes_report_and_history(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    tone = 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)
    wav = tmp_path / "tone.wav"
    _write_wav(wav, tone, sr)
    expected_dbfs = float(20.0 * np.log10(0.1 / np.sqrt(2.0)))

    report_path = tmp_path / "cal_check.json"
    history_path = tmp_path / "cal_history.csv"
    code_ok = main(
        [
            "calibrate",
            "check",
            "--tone",
            str(wav),
            "--dbfs-reference",
            str(expected_dbfs),
            "--max-drift-db",
            "0.5",
            "--device-id",
            "recorder_07",
            "--history",
            str(history_path),
            "--out",
            str(report_path),
        ]
    )
    assert code_ok == 0
    assert report_path.exists()
    assert history_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert bool(payload["within_tolerance"]) is True

    code_fail = main(
        [
            "calibrate",
            "check",
            "--tone",
            str(wav),
            "--dbfs-reference",
            "-30.0",
            "--max-drift-db",
            "0.5",
            "--out",
            str(tmp_path / "cal_check_fail.json"),
        ]
    )
    assert code_fail == 2
