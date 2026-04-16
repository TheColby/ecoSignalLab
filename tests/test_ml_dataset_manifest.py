from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.ml import build_dataset_manifest_from_ml_metadata, export_ml_features


def test_export_ml_features_writes_dataset_manifest(tmp_path: Path) -> None:
    sr = 8000
    t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, x, sr)

    result = analyze(AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0))
    artifacts = export_ml_features(result, output_dir=tmp_path / "ml", prefix="demo", seed=123)
    assert "dataset_manifest_json" in artifacts
    payload = json.loads(Path(artifacts["dataset_manifest_json"]).read_text(encoding="utf-8"))
    assert payload["sample_id"] == "demo"
    assert payload["frame_table_version"]


def test_build_dataset_manifest_from_ml_metadata(tmp_path: Path) -> None:
    for idx, freq in enumerate((220.0, 440.0, 880.0), start=1):
        sr = 8000
        t = np.linspace(0.0, 0.25, int(sr * 0.25), endpoint=False)
        x = (0.1 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        wav = tmp_path / f"tone_{idx}.wav"
        sf.write(wav, x, sr)
        result = analyze(AnalysisConfig(input_path=wav, output_dir=tmp_path, verbosity=0))
        export_ml_features(result, output_dir=tmp_path / f"ml_{idx}", prefix=f"demo_{idx}", seed=123)

    out_path, manifest = build_dataset_manifest_from_ml_metadata(tmp_path, tmp_path / "dataset_manifest.json")
    assert out_path.exists()
    assert int(manifest["num_samples"]) == 3
    assert sum(int(v) for v in manifest["split_counts"].values()) == 3
