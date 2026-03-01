from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.ml import export_ml_features, resolve_compute_device


def test_resolve_compute_device_auto_and_cpu() -> None:
    auto_info = resolve_compute_device("auto", strict=False)
    assert auto_info.requested == "auto"
    assert auto_info.resolved in {"cpu", "cuda", "mps"}

    cpu_info = resolve_compute_device("cpu", strict=True)
    assert cpu_info.resolved == "cpu"


def test_resolve_compute_device_strict_unavailable_paths() -> None:
    info = resolve_compute_device("auto", strict=False)
    if not info.cuda_available:
        with pytest.raises(RuntimeError):
            resolve_compute_device("cuda", strict=True)
    else:
        assert resolve_compute_device("cuda", strict=True).resolved == "cuda"

    if not info.mps_available:
        with pytest.raises(RuntimeError):
            resolve_compute_device("mps", strict=True)
    else:
        assert resolve_compute_device("mps", strict=True).resolved == "mps"


def test_export_ml_features_records_compute_device_metadata(tmp_path: Path) -> None:
    sr = 8_000
    t = np.linspace(0.0, 0.5, int(0.5 * sr), endpoint=False)
    x = (0.1 * np.sin(2.0 * np.pi * 330.0 * t)).astype(np.float32)
    wav = tmp_path / "in.wav"
    sf.write(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "novelty_curve"],
            verbosity=0,
            compute_device="cpu",
        )
    )
    artifacts = export_ml_features(
        result,
        output_dir=tmp_path / "ml",
        prefix="dev",
        seed=7,
        device="cpu",
        strict_device=True,
    )

    meta = json.loads(Path(artifacts["ml_metadata_json"]).read_text(encoding="utf-8"))
    dev = meta.get("compute_device", {})
    assert dev.get("requested") == "cpu"
    assert dev.get("resolved") == "cpu"
