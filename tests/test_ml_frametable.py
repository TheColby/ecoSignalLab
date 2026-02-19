from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.ml import (
    FRAMETABLE_VERSION,
    build_frame_table,
    export_ml_features,
    frame_table_tensor,
    frame_wide_table,
)


def test_frametable_contract_and_tensor_layout(tmp_path: Path) -> None:
    sr = 16_000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    sf.write(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "spectral_centroid_hz", "novelty_curve"],
            verbosity=0,
        )
    )

    frame_table = build_frame_table(result)
    assert frame_table.metadata["version"] == FRAMETABLE_VERSION
    assert frame_table.values.ndim == 2
    assert frame_table.timestamps_s.ndim == 1
    assert frame_table.values.shape[0] == frame_table.timestamps_s.shape[0]
    assert frame_table.values.shape[1] == len(frame_table.feature_names)
    assert frame_table.metadata["tensor_layout"] == "[channels, frames, features]"

    tensor, labels, mode = frame_table_tensor(frame_table, source_channels=2, replicate_aggregate=False)
    assert tensor.ndim == 3
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == frame_table.values.shape[0]
    assert tensor.shape[2] == frame_table.values.shape[1]
    assert labels == ["mix"]
    assert mode == "aggregate_mixdown"

    ts, cols, mat = frame_wide_table(result)
    assert mat.shape == frame_table.values.shape
    assert cols == frame_table.feature_names
    assert np.allclose(ts, frame_table.timestamps_s)


def test_export_ml_features_writes_frametable_artifacts(tmp_path: Path) -> None:
    sr = 8_000
    t = np.linspace(0.0, 0.6, int(0.6 * sr), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    sf.write(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "snr_db", "novelty_curve"],
            verbosity=0,
        )
    )
    artifacts = export_ml_features(result, output_dir=tmp_path / "ml", prefix="demo", seed=123)

    for key in ["frame_csv", "frame_npy", "frame_table_csv", "frame_tensor_npy", "ml_metadata_json"]:
        assert key in artifacts
        assert Path(artifacts[key]).exists()

    meta = json.loads(Path(artifacts["ml_metadata_json"]).read_text(encoding="utf-8"))
    assert meta["frame_table_version"] == FRAMETABLE_VERSION
    assert meta["frame_table"]["tensor_layout"] == "[channels, frames, features]"
    assert meta["frame_table"]["tensor_mode"] == "aggregate_mixdown"
    assert meta["frame_table"]["tensor_channel_labels"] == ["mix"]
