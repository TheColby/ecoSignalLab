from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.core.moments import MomentsExtractConfig, run_moments_extract
from esl.core.streaming import StreamRunConfig, run_stream_analysis


def _write_wav(path: Path, data: np.ndarray, sr: int = 16000) -> Path:
    sf.write(path, data.astype(np.float32), sr)
    return path


def test_chunked_analyze_avoids_full_read_for_streamable_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    wav = _write_wav(tmp_path / "in.wav", x, sr)

    def _forbid_read_audio(*args: object, **kwargs: object) -> object:
        raise AssertionError("read_audio should not be called for streamable chunked analysis")

    monkeypatch.setattr("esl.core.analyzer.read_audio", _forbid_read_audio)
    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "novelty_curve"],
            chunk_size=4000,
            summary_only=True,
            verbosity=0,
        )
    )
    assert result["analysis_mode"] == "streaming"
    assert result["metadata"]["analysis_strategy"]["out_of_core"] is True
    assert result["metrics"]["rms_dbfs"]["series"] == []


def test_chunked_analyze_rejects_non_streaming_metrics_without_allow_full_read(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    wav = _write_wav(tmp_path / "in.wav", x, sr)

    with pytest.raises(RuntimeError, match="require full-file context"):
        analyze(
            AnalysisConfig(
                input_path=wav,
                output_dir=tmp_path,
                metrics=["rt60_s"],
                chunk_size=4000,
                verbosity=0,
            )
        )


def test_chunked_analyze_writes_frame_table_csv_and_checkpoint(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    wav = _write_wav(tmp_path / "in.wav", x, sr)
    frame_csv = tmp_path / "frame_table.csv"
    checkpoint_dir = tmp_path / "ckpt"

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "novelty_curve"],
            chunk_size=4000,
            summary_only=True,
            frame_table_csv=frame_csv,
            checkpoint_dir=checkpoint_dir,
            verbosity=0,
        )
    )
    assert frame_csv.exists()
    assert (tmp_path / "frame_table.csv.meta.json").exists()
    checkpoint_file = checkpoint_dir / "analysis_state.json"
    assert checkpoint_file.exists()
    assert result["artifacts"]["frame_table_csv"] == str(frame_csv.resolve())


def test_stream_resume_appends_jsonl_and_moments_can_read_it(tmp_path: Path) -> None:
    sr = 16_000
    t = np.arange(2 * sr, dtype=np.float64) / sr
    x = np.concatenate(
        [
            0.05 * np.sin(2.0 * np.pi * 220.0 * t[:sr]),
            0.8 * np.sin(2.0 * np.pi * 220.0 * t[sr:]),
        ]
    )
    wav = _write_wav(tmp_path / "stream.wav", x, sr)
    rules = {"metric_thresholds": {"rms_dbfs": {"max": -8.0}}}
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    stream_out = tmp_path / "stream_out"
    checkpoint_dir = tmp_path / "stream_ckpt"

    run_stream_analysis(
        StreamRunConfig(
            input_path=wav,
            output_dir=stream_out,
            metrics=["rms_dbfs"],
            chunk_size=8000,
            rules_path=str(rules_path),
            max_chunks=1,
            checkpoint_dir=checkpoint_dir,
        )
    )
    report_path, report = run_stream_analysis(
        StreamRunConfig(
            input_path=wav,
            output_dir=stream_out,
            metrics=["rms_dbfs"],
            chunk_size=8000,
            rules_path=str(rules_path),
            checkpoint_dir=checkpoint_dir,
            resume=True,
        )
    )
    assert int(report["chunks_processed"]) >= 2
    chunks_jsonl = Path(report["artifacts"]["chunks_jsonl"])
    assert chunks_jsonl.exists()
    assert len(chunks_jsonl.read_text(encoding="utf-8").splitlines()) >= 2

    moments_out = tmp_path / "moments"
    _, moments = run_moments_extract(
        MomentsExtractConfig(
            input_path=wav,
            output_dir=moments_out,
            stream_report_path=str(report_path),
            pre_roll_s=0.0,
            post_roll_s=0.0,
            merge_gap_s=0.0,
            selection_mode="single",
        )
    )
    assert int(moments["clips_written"]) >= 1
