"""Synthetic spatial timeline regressions and bounded-read contracts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.workflows.spatial_timeline import MAX_WINDOW_SAMPLES, register, spatial_timeline


@pytest.mark.parametrize("convention", ["acn-sn3d", "acn-n3d", "fuma"])
def test_plane_wave_conventions(tmp_path: Path, convention: str) -> None:
    pressure = np.random.default_rng(31).normal(0, 0.1, 205)
    zeros = np.zeros_like(pressure)
    if convention == "fuma":
        samples = np.column_stack([pressure / np.sqrt(2), pressure, zeros, zeros])
    else:
        gain = np.sqrt(3) if convention == "acn-n3d" else 1
        samples = np.column_stack([pressure, zeros, zeros, pressure * gain])
    source = tmp_path / "plane.wav"
    sf.write(source, samples, 1000, subtype="DOUBLE")
    out = tmp_path / "output"
    report = spatial_timeline(source, out, window_seconds=0.1, ambisonics=convention, plot=True)
    assert report["schema_version"] == "1.0.0"
    assert report["esl_version"]
    assert len(report["provenance"]["decoded_sha256"]) == 64
    assert [row["frames"] for row in report["timeline"]] == [100, 100, 5]
    for row in report["timeline"]:
        assert row["azimuth_deg"] == pytest.approx(0)
        assert row["diffuseness"] == pytest.approx(0, abs=1e-12)
        assert row["direction_x"] == pytest.approx(1)
    assert json.loads((out / "spatial_timeline.json").read_text())["artifact_kind"]
    with (out / "spatial_timeline.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    assert (out / "spatial_timeline.png").stat().st_size > 100


def test_itd_sign_ild_and_tail(tmp_path: Path) -> None:
    samples = np.zeros((100, 3))
    samples[40, 0], samples[43, 1] = 0.4, 0.2
    source = tmp_path / "pair.wav"
    sf.write(source, samples, 1000, subtype="DOUBLE")
    row = spatial_timeline(source, tmp_path / "out", max_itd_ms=5)["timeline"][0]
    assert row["itd_s"] == pytest.approx(0.003)
    assert row["ild_db"] == pytest.approx(20 * np.log10(2))
    assert row["diffuseness"] is None
    reverse = spatial_timeline(source, tmp_path / "rev", pair=(1, 0), max_itd_ms=5)
    assert reverse["timeline"][0]["itd_s"] == pytest.approx(-0.003)


def test_silence_and_empty(tmp_path: Path) -> None:
    source = tmp_path / "silence.wav"
    sf.write(source, np.zeros((20, 4)), 1000)
    row = spatial_timeline(source, tmp_path / "out", ambisonics="acn-sn3d")["timeline"][0]
    assert row["energy_fs2"] == 0
    assert row["diffuseness"] is None
    assert row["itd_s"] is None
    sf.write(source, np.zeros((0, 4)), 1000)
    assert spatial_timeline(source, tmp_path / "empty", plot=True)["timeline"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_seconds": float("nan")},
        {"window_seconds": 0},
        {"pair": (0, 0)},
        {"pair": (0, 4)},
        {"ambisonics": "guess"},
        {"max_itd_ms": -1},
        {"max_windows": 0},
        {"window_seconds": 0.001, "max_windows": 1},
        {"window_seconds": MAX_WINDOW_SAMPLES},
    ],
)
def test_invalid_config(tmp_path: Path, kwargs: dict) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.zeros((20, 2)), 1000)
    with pytest.raises(ValueError):
        spatial_timeline(source, tmp_path / "out", **kwargs)


def test_actual_read_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.zeros((321, 2)), 1000)
    original = sf.SoundFile.read
    sizes = []

    def read(recording: sf.SoundFile, frames: int = -1, **kwargs: object) -> np.ndarray:
        sizes.append(frames)
        assert 0 < frames <= 100
        return original(recording, frames, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", read)
    spatial_timeline(source, tmp_path / "out", window_seconds=0.1)
    assert len(sizes) == 4


def test_registered_handler(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.zeros((10, 2)), 1000)
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(["spatial-timeline", str(source), "--out-dir", str(tmp_path / "out")])
    assert args.func(args) == 0


def test_nonfinite_and_ambisonic_channel_validation(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    sf.write(source, np.full((10, 2), np.nan), 1000, subtype="DOUBLE")
    with pytest.raises(ValueError, match="Non-finite"):
        spatial_timeline(source, tmp_path / "out")
    with pytest.raises(ValueError, match="four channels"):
        spatial_timeline(source, tmp_path / "out", ambisonics="fuma")
