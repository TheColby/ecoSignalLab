from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main


def _write_wav(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    sf.write(path, x.astype(np.float32), sr)


def test_cli_similar_feature_mode_default(tmp_path: Path) -> None:
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr

    query = 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    close = 0.2 * np.sin(2.0 * np.pi * 440.0 * t + 0.05)
    far = 0.2 * np.sin(2.0 * np.pi * 880.0 * t)

    query_path = tmp_path / "query.wav"
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    close_path = corpus / "close.wav"
    far_path = corpus / "far.wav"

    _write_wav(query_path, query, sr)
    _write_wav(close_path, close, sr)
    _write_wav(far_path, far, sr)

    out_json = tmp_path / "similarity.json"
    out_csv = tmp_path / "similarity.csv"
    code = main(
        [
            "similar",
            str(query_path),
            str(corpus),
            "--top-k",
            "2",
            "--json",
            str(out_json),
            "--csv",
            str(out_csv),
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    assert out_json.exists()
    assert out_csv.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["mode_used"] == "feature"
    assert payload["results"]
    assert Path(payload["results"][0]["path"]).name == "close.wav"


def test_cli_similar_metric_mode_single_metric(tmp_path: Path) -> None:
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr

    query = 0.10 * np.sin(2.0 * np.pi * 300.0 * t)
    close = 0.11 * np.sin(2.0 * np.pi * 300.0 * t)
    far = 0.35 * np.sin(2.0 * np.pi * 300.0 * t)

    query_path = tmp_path / "query.wav"
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    close_path = corpus / "close_rms.wav"
    far_path = corpus / "far_rms.wav"

    _write_wav(query_path, query, sr)
    _write_wav(close_path, close, sr)
    _write_wav(far_path, far, sr)

    out_json = tmp_path / "similarity_metric.json"
    code = main(
        [
            "similar",
            str(query_path),
            str(corpus),
            "--mode",
            "metric",
            "--metric",
            "rms_dbfs",
            "--top-k",
            "1",
            "--json",
            str(out_json),
            "--verbosity",
            "0",
        ]
    )
    assert code == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["mode_used"] == "metric"
    assert payload["results"]
    assert Path(payload["results"][0]["path"]).name == "close_rms.wav"
