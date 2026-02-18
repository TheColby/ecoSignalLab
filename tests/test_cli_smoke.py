from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main


def test_cli_analyze_smoke(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    sf.write(wav, x, sr)

    code = main(["analyze", str(wav), "--out-dir", str(tmp_path), "--verbosity", "0"])
    assert code == 0
    assert (tmp_path / "in.json").exists()
