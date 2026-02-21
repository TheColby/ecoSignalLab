from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def test_compare_time_stretch_kpis_script_scipy_only(tmp_path: Path) -> None:
    sr = 16_000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
    x = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    sf.write(wav, x, sr, subtype="PCM_24")

    out_dir = tmp_path / "kpi"
    cmd = [
        ".venv/bin/python",
        "scripts/compare_time_stretch_kpis.py",
        "--input",
        str(wav),
        "--out-dir",
        str(out_dir),
        "--methods",
        "scipy_resample",
        "--factor",
        "1.5",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout

    csv_path = out_dir / "kpi_summary.csv"
    json_path = out_dir / "kpi_summary.json"
    assert csv_path.exists()
    assert json_path.exists()

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["method"] == "scipy_resample"
    assert row["status"] == "ok"
    assert float(row["kpi_score"]) >= 0.0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["input"].endswith("in.wav")
    assert payload["methods"] == ["scipy_resample"]
    assert isinstance(payload["results"], list) and len(payload["results"]) == 1
