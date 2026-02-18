from pathlib import Path

import importlib.util
import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.io import save_csv, save_hdf5, save_json, save_mat


def test_export_writers(tmp_path: Path) -> None:
    sr = 16000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, x, sr)

    result = analyze(AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0))

    p_json = save_json(result, tmp_path / "out.json")
    p_csv = save_csv(result, tmp_path / "out.csv")
    optional_outputs = []
    if importlib.util.find_spec("h5py") is not None:
        optional_outputs.append(save_hdf5(result, tmp_path / "out.h5"))
    if importlib.util.find_spec("scipy") is not None:
        optional_outputs.append(save_mat(result, tmp_path / "out.mat"))

    for p in (p_json, p_csv, *optional_outputs):
        assert p.exists()
        assert p.stat().st_size > 0
