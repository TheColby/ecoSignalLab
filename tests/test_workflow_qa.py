"""Recorder QA regressions, including dropout continuity across block boundaries."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.workflows.qa import MAX_BLOCK_SAMPLES, recorder_qa, register


def test_faults_and_duplicate_groups(tmp_path: Path) -> None:
    samples = np.full((1000, 4), 0.02)
    samples[:, 0] = np.random.default_rng(3).normal(0, 0.1, 1000)
    samples[100:400, 0] = 0
    samples[600, 0] = 1.0
    samples[:, 1] = samples[:, 0]
    samples[:, 3] = 0
    source = tmp_path / "faults.wav"
    sf.write(source, samples, 1000, subtype="DOUBLE")
    out = tmp_path / "qa"
    report = recorder_qa(source, out, block_frames=128)
    assert report["schema_version"] == "1.0.0"
    assert report["esl_version"]
    assert report["status"] == "review"
    assert report["duplicate_channel_groups"] == [[0, 1]]
    assert report["channels"][0]["clipped_samples"] == 1
    assert report["channels"][0]["longest_dropout_frames"] == 300
    assert report["channels"][0]["dropout_flag"]
    assert report["channels"][2]["dc_mean_fs"] == pytest.approx(0.02)
    assert report["channels"][2]["dc_flag"]
    assert report["channels"][3]["longest_dropout_frames"] == 1000
    assert report["clock_drift"]["status"] == "not_assessed"
    assert json.loads((out / "recorder_qa.json").read_text())["artifact_kind"] == "recorder_qa"
    assert len((out / "recorder_qa.csv").read_text().splitlines()) == 5
    other = recorder_qa(source, tmp_path / "other", block_frames=31)
    assert other["provenance"]["decoded_sha256"] == report["provenance"]["decoded_sha256"]
    assert other["channels"][0]["longest_dropout_frames"] == 300


def test_clean_and_exit_policy(tmp_path: Path) -> None:
    source = tmp_path / "clean.wav"
    wave = np.tile([0.1, -0.1], 500)
    sf.write(source, wave, 1000, subtype="DOUBLE")
    assert recorder_qa(source, tmp_path / "out")["status"] == "no_flags"
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    argv = ["qa", str(source), "--out-dir", str(tmp_path / "cli"), "--fail-on-flags"]
    assert parser.parse_args(argv).func(parser.parse_args(argv)) == 0
    sf.write(source, np.zeros(500), 1000)
    args = parser.parse_args(argv)
    assert args.func(args) == 2
    args = parser.parse_args(argv[:-1])
    assert args.func(args) == 0


def test_empty_and_nonfinite(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.zeros((0, 2)), 1000)
    empty = recorder_qa(source, tmp_path / "empty")
    assert empty["status"] == "review"
    assert empty["channels"][0]["dc_mean_fs"] is None
    assert empty["duplicate_channel_groups"] == []
    samples = np.array([[np.nan, np.nan], [np.inf, np.inf], [0.1, 0.1]])
    sf.write(source, samples, 1000, subtype="DOUBLE")
    report = recorder_qa(source, tmp_path / "nan", block_frames=1)
    assert report["channels"][0]["nonfinite_samples"] == 2
    assert report["channels"][0]["dc_mean_fs"] == pytest.approx(0.1)
    assert report["duplicate_channel_groups"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_frames": 0},
        {"block_frames": MAX_BLOCK_SAMPLES + 1},
        {"clip_threshold": 0},
        {"dropout_threshold": 1},
        {"dc_threshold": -1},
        {"min_dropout_seconds": 0},
        {"clip_threshold": float("inf")},
    ],
)
def test_invalid_config(tmp_path: Path, kwargs: dict) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.zeros(10), 1000)
    with pytest.raises(ValueError):
        recorder_qa(source, tmp_path / "out", **kwargs)


def test_bounded_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.zeros((1234, 6)), 1000)
    original = sf.SoundFile.read
    sizes = []

    def read(
        recording: sf.SoundFile,
        frames: int = -1,
        *args: object,
        **kwargs: object,
    ) -> np.ndarray:
        sizes.append(frames)
        assert 0 < frames <= 100
        return original(recording, frames, *args, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", read)
    report = recorder_qa(source, tmp_path / "out", block_frames=100)
    assert len(sizes) == 13
    assert report["duplicate_channel_groups"] == [list(range(6))]
    assert all(row["longest_dropout_frames"] == 1234 for row in report["channels"])
