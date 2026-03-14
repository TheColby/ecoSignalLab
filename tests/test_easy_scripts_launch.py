from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]


def _write_wav(path: Path, sr: int = 8000, seconds: float = 0.25) -> Path:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(path, x, sr)
    return path


def _script_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_easy_doctor_script_runs(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "in.wav")
    proc = subprocess.run(
        ["bash", "scripts/easy/00_doctor.sh", str(wav)],
        cwd=ROOT,
        env=_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "status:" in proc.stdout


def test_easy_simple_summary_script_runs(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "in.wav")
    proc = subprocess.run(
        ["bash", "scripts/easy/18_simple_summary.sh", str(wav)],
        cwd=ROOT,
        env=_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "summary:" in proc.stdout
