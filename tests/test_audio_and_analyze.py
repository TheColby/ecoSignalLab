from pathlib import Path

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.core.audio import read_audio


def _write_wav(path: Path, data: np.ndarray, sr: int = 48000) -> None:
    sf.write(path, data, sr)


def test_read_audio_multichannel(tmp_path: Path) -> None:
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    x1 = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    x2 = 0.1 * np.sin(2 * np.pi * 880.0 * t)
    x = np.stack([x1, x2], axis=1).astype(np.float32)
    wav = tmp_path / "stereo.wav"
    _write_wav(wav, x, sr)

    buf = read_audio(wav)
    assert buf.channels == 2
    assert buf.sample_rate == sr
    assert buf.num_samples == sr


def test_analyze_outputs_metrics(tmp_path: Path) -> None:
    sr = 48000
    n = sr
    t = np.arange(n) / sr

    # Synthetic IR-like signal: impulse + decaying tail.
    x = np.zeros(n, dtype=np.float32)
    x[0] = 1.0
    x += (0.2 * np.exp(-t / 0.4) * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
    wav = tmp_path / "ir.wav"
    _write_wav(wav, x, sr)

    cfg = AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0)
    result = analyze(cfg)

    assert "metrics" in result
    assert "rms_dbfs" in result["metrics"]
    assert "rt60_s" in result["metrics"]
    assert result["metadata"]["channels"] == 1
