from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.cli.main import main
from esl.viz.feature_vectors import extract_feature_vectors, load_feature_vectors, save_feature_vectors


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, data.astype(np.float32), sr)


def test_extract_core_feature_vectors_shape(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    wav = tmp_path / "core.wav"
    _write_wav(wav, x, sr)

    fv = extract_feature_vectors(wav, feature_set="core", frame_size=1024, hop_size=256)
    assert fv.matrix.ndim == 2
    assert fv.matrix.shape[0] > 0
    assert fv.matrix.shape[1] > 32
    assert len(fv.feature_names) == fv.matrix.shape[1]
    assert fv.backend == "scipy-core"


def test_feature_vectors_roundtrip_npz_and_csv(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.2 * np.sin(2.0 * np.pi * 220.0 * t)
    wav = tmp_path / "rt.wav"
    _write_wav(wav, x, sr)

    fv = extract_feature_vectors(wav, feature_set="core", frame_size=1024, hop_size=256)
    npz_path = save_feature_vectors(fv, tmp_path / "fv.npz")
    csv_path = save_feature_vectors(fv, tmp_path / "fv.csv")

    fv_npz = load_feature_vectors(npz_path)
    fv_csv = load_feature_vectors(csv_path)
    assert fv_npz.matrix.shape == fv.matrix.shape
    assert fv_csv.matrix.shape == fv.matrix.shape
    assert fv_npz.feature_names[:5] == fv.feature_names[:5]
    assert fv_csv.feature_names[:5] == fv.feature_names[:5]


def test_cli_features_extract_and_plot_with_feature_vectors(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.2 * np.sin(2.0 * np.pi * 330.0 * t)
    wav = tmp_path / "in.wav"
    _write_wav(wav, x, sr)

    vec_path = tmp_path / "vectors.npz"
    meta_path = tmp_path / "vectors_meta.json"
    code_fv = main(
        [
            "features",
            "extract",
            str(wav),
            "--out",
            str(vec_path),
            "--feature-set",
            "core",
            "--meta-json",
            str(meta_path),
        ]
    )
    assert code_fv == 0
    assert vec_path.exists()
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert int(meta["features"]) > 0
    assert int(meta["frames"]) > 0

    code_an = main(["analyze", str(wav), "--out-dir", str(tmp_path), "--verbosity", "0"])
    assert code_an == 0
    result_json = tmp_path / "in.json"
    assert result_json.exists()

    out_dir = tmp_path / "plots"
    code_plot = main(
        [
            "plot",
            str(result_json),
            "--out",
            str(out_dir),
            "--audio",
            str(wav),
            "--similarity-matrix",
            "--novelty-matrix",
            "--feature-vectors",
            str(vec_path),
            "--no-spectral",
        ]
    )
    assert code_plot == 0
    assert (out_dir / "similarity_matrix.png").exists()
    assert (out_dir / "novelty_matrix.png").exists()


def test_librosa_feature_set_if_available(tmp_path: Path) -> None:
    pytest.importorskip("librosa")
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.2 * np.sin(2.0 * np.pi * 500.0 * t)
    wav = tmp_path / "librosa.wav"
    _write_wav(wav, x, sr)

    fv = extract_feature_vectors(wav, feature_set="librosa", frame_size=1024, hop_size=256)
    assert fv.matrix.shape[1] > 64
    assert "librosa" in fv.backend
