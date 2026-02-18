from pathlib import Path

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze


def test_analyzer_emits_runtime_metadata(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, x, sr)

    result = analyze(AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0))

    assert "analysis_time_local" in result
    meta = result["metadata"]
    assert meta.get("channel_layout_hint") in {"mono", "stereo", "multichannel", "ambisonic_b_format"}
    assert isinstance(meta.get("runtime"), dict)
    assert "python" in meta["runtime"]
