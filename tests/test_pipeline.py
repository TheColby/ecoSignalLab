from pathlib import Path

import numpy as np
import soundfile as sf

from esl.pipeline import PipelineRunConfig, read_pipeline_status, run_pipeline


def test_pipeline_run_and_status(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    x = (0.1 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    wav = in_dir / "tone.wav"
    sf.write(wav, x, sr)

    cfg = PipelineRunConfig(
        input_dir=in_dir,
        output_dir=out_dir,
        stages=["analyze", "digest"],
    )
    manifest_path, manifest = run_pipeline(cfg)

    assert manifest_path.exists()
    assert manifest["status"] == "completed"
    assert manifest["stages"]["analyze"]["status"] == "completed"
    assert manifest["stages"]["digest"]["status"] == "completed"

    status = read_pipeline_status(manifest_path)
    assert status["pipeline_id"] == manifest["pipeline_id"]
    assert (out_dir / "pipeline_digest.csv").exists()
    assert (out_dir / "pipeline_digest.json").exists()
