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
    assert "--json" in help_text
    assert "--csv" in help_text
    assert "--parquet" in help_text
    assert "--hdf5" in help_text
    assert "--mat" in help_text


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
    assert "ecoSignalLab CLI" in proc.stdout
    assert "features" in proc.stdout
    assert "moments" in proc.stdout
