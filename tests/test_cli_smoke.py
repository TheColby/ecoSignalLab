from pathlib import Path

import numpy as np
import os
import pytest
import soundfile as sf
import subprocess
import sys

from esl.cli.main import main
from esl.schema import SCHEMA_VERSION


def test_cli_analyze_smoke(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    sf.write(wav, x, sr)

    code = main(["analyze", str(wav), "--out-dir", str(tmp_path), "--verbosity", "0"])
    assert code == 0
    assert (tmp_path / "in.json").exists()


def test_cli_help_contains_output_and_debug_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--help"])
    assert int(exc.value.code) == 0
    help_text = capsys.readouterr().out
    assert "--verbosity" in help_text
    assert "--debug" in help_text
    assert "--device" in help_text
    assert "--json" in help_text
    assert "--csv" in help_text
    assert "--parquet" in help_text
    assert "--hdf5" in help_text
    assert "--mat" in help_text
    assert "--frame-seconds" in help_text
    assert "--hop-seconds" in help_text
    assert "--chunk-minutes" in help_text
    assert "--chunk-hours" in help_text
    assert "--chunk-days" in help_text
    assert "--summary-only" in help_text
    assert "--streamable-only" in help_text
    assert "--frame-table-csv" in help_text
    assert "--checkpoint-dir" in help_text
    assert "--resume" in help_text


def test_cli_batch_help_contains_report_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["batch", "--help"])
    assert int(exc.value.code) == 0
    help_text = capsys.readouterr().out
    assert "--report-metrics" in help_text


def test_cli_schema_reports_schema_version(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["schema"])
    assert code == 0
    captured = capsys.readouterr()
    assert f"schema_version: {SCHEMA_VERSION}" in captured.err


def test_python_module_help_entrypoint() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.run([sys.executable, "-m", "esl", "--help"], capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0
    assert "doctor" in proc.stdout
    assert "ecoSignalLab CLI" in proc.stdout
    assert "features" in proc.stdout
    assert "moments" in proc.stdout
    assert "similar" in proc.stdout
    assert "simple" in proc.stdout
    assert "quickstart" in proc.stdout


def test_cli_quickstart_outputs_recipes(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["quickstart"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ecoSignalLab Quickstart" in out
    assert "esl doctor" in out
    assert "esl analyze input.wav" in out
    assert "esl moments extract" in out


def test_cli_quickstart_goal_long(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["quickstart", "--goal", "long", "--input", "day.wav"])
    assert code == 0
    out = capsys.readouterr().out
    assert "esl doctor day.wav" in out
    assert "--chunk-minutes 10" in out


def test_cli_benchmark_device_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["benchmark", "device", "--device", "cpu", "--frames", "128", "--features", "32", "--iters", "2"])
    assert code == 0
    out = capsys.readouterr().out
    assert "summary:" in out


def test_cli_missing_file_shows_friendly_hint(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["analyze", "this_file_does_not_exist.wav"])
    assert code == 1
    err = capsys.readouterr().err
    assert "verify the file/path exists" in err


def test_cli_doctor_and_simple_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sr = 8000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    doctor_json = tmp_path / "doctor.json"
    simple_json = tmp_path / "simple.json"
    sf.write(wav, x, sr)

    code_doctor = main(["doctor", str(wav), "--json-out", str(doctor_json)])
    assert code_doctor == 0
    out_doctor = capsys.readouterr().out
    assert "status:" in out_doctor
    assert "recommendations:" in out_doctor
    assert doctor_json.exists()

    code_simple = main(["simple", str(wav), "--json-out", str(simple_json)])
    assert code_simple == 0
    out_simple = capsys.readouterr().out
    assert "summary:" in out_simple
    assert "rms_dbfs" in out_simple
    assert simple_json.exists()


def test_cli_batch_report_metrics_columns(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    sr = 8000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    wav = in_dir / "tone.wav"
    sf.write(wav, x, sr)

    code = main(
        [
            "batch",
            str(in_dir),
            "--out",
            str(out_dir),
            "--metrics",
            "rms_dbfs,snr_db,novelty_curve",
            "--report-metrics",
            "snr_db,novelty_curve",
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    idx = out_dir / "batch_index.csv"
    assert idx.exists()
    header = idx.read_text(encoding="utf-8").splitlines()[0]
    assert "snr_db_mean" in header
    assert "novelty_curve_mean" in header
