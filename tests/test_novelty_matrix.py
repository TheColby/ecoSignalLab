from pathlib import Path

import numpy as np
import soundfile as sf

from esl.viz import compute_novelty_matrix, plot_novelty_matrix


def test_compute_novelty_matrix_shapes_and_range(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)

    # Two-section synthetic signal to induce a novelty transition.
    x = np.zeros_like(t, dtype=np.float32)
    half = len(t) // 2
    x[:half] = 0.08 * np.sin(2 * np.pi * 220.0 * t[:half])
    x[half:] = 0.08 * np.sin(2 * np.pi * 880.0 * t[half:])

    wav = tmp_path / "novelty_input.wav"
    sf.write(wav, x, sr)

    data = compute_novelty_matrix(wav)
    ssm = data["ssm"]
    kernel = data["kernel"]
    novelty = data["novelty"]

    assert ssm.ndim == 2
    assert ssm.shape[0] == ssm.shape[1]
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]
    assert novelty.ndim == 1
    assert novelty.shape[0] == ssm.shape[0]
    assert float(np.min(novelty)) >= 0.0
    assert float(np.max(novelty)) <= 1.0 + 1e-9


def test_plot_novelty_matrix_writes_file(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0, 0.4, int(0.4 * sr), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 330.0 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, x, sr)

    out = plot_novelty_matrix(wav, tmp_path)
    assert out.exists()
    assert out.stat().st_size > 0
