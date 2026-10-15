"""Sample-exact conservative masking and auditable output tests."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.workflows.privacy import (
    MAX_BLOCK_SAMPLES,
    output_format,
    redact_audio,
    register,
    sample_spans,
)


def test_large_output_selects_rf64_before_reading() -> None:
    boundary_frames = (2**32 - 65536) // (8 * 8)
    assert output_format(boundary_frames - 1, 8) == "WAV"
    assert output_format(boundary_frames, 8) == "RF64"


def test_block_budget_and_rf64_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "input.wav"
    samples = np.ones((20, 2)) * 0.2
    sf.write(source, samples, 10, subtype="DOUBLE")
    output, audit = tmp_path / "out.wav", tmp_path / "audit.json"
    with pytest.raises(ValueError, match="samples"):
        redact_audio(source, output, audit, [], block_frames=MAX_BLOCK_SAMPLES)
    assert not output.exists()
    assert not audit.exists()
    monkeypatch.setattr("esl.workflows.privacy.output_format", lambda frames, channels: "RF64")
    report = redact_audio(source, output, audit, [{"start": 0.5, "end": 1.0}])
    assert sf.info(output).format == "RF64"
    assert report["config"]["encoding"] == "RF64/DOUBLE"
    decoded, rate = sf.read(output, always_2d=True)
    assert rate == 10
    np.testing.assert_array_equal(decoded[:5], samples[:5])
    assert np.all(decoded[5:10] == 0)


def test_decimal_boundaries_and_union() -> None:
    assert sample_spans(
        [{"start": "0.1", "end": "0.3"}, {"start": "0.25", "end": "0.41"}], 10, 10
    ) == [(1, 5)]
    assert sample_spans(
        [{"start": "0.100000000000000000000000000001", "end": "0.200000000000000000000000000001"}],
        10,
        10,
    ) == [(1, 3)]
    assert sample_spans([], 3, 1) == []


@pytest.mark.parametrize(
    "interval",
    [
        {"start": -1, "end": 1},
        {"start": 1, "end": 1},
        {"start": 1, "end": 0},
        {"start": 0, "end": "NaN"},
        {"start": 0, "end": "Infinity"},
        {"start": 0, "end": "oops"},
        {"start": 0, "end": 2},
        {"start": 0},
        {"start": 0, "end": 1, "channel": 0},
    ],
)
def test_invalid_intervals(interval: dict) -> None:
    with pytest.raises(ValueError):
        sample_spans([interval], 10, 10)


def test_multichannel_mask_and_hashes(tmp_path: Path) -> None:
    source, output, audit = (tmp_path / name for name in ("in.wav", "out.wav", "audit.json"))
    values = np.arange(40).reshape(20, 2) / 40
    sf.write(source, values, 10, subtype="DOUBLE")
    report = redact_audio(
        source, output, audit, [{"start": Decimal("0.3"), "end": Decimal("0.7")}], block_frames=4
    )
    actual, rate = sf.read(output, always_2d=True)
    values[3:7] = 0
    np.testing.assert_array_equal(actual, values)
    assert rate == 10
    assert report["redacted_sample_spans"] == [[3, 7]]
    assert report["redacted_frames"] == 4
    assert report == json.loads(audit.read_text())
    assert report["provenance"]["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report["schema_version"] == "1.0.0" and report["esl_version"]
    assert sf.info(output).subtype == "DOUBLE"
    with pytest.raises(ValueError):
        redact_audio(source, output, audit, [])


def test_detector_union_and_no_input_mutation(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    sf.write(source, np.ones((10, 3)), 10, subtype="DOUBLE")
    original = source.read_bytes()

    def detector(path: Path) -> list[dict]:
        assert path == source
        return [{"start": "0.4", "end": "0.6"}]

    report = redact_audio(
        source,
        tmp_path / "out.wav",
        tmp_path / "audit.json",
        [{"start": "0.1", "end": "0.2"}],
        detector=detector,
        detector_id="test:detector",
    )
    assert report["redacted_sample_spans"] == [[1, 2], [4, 6]]
    assert source.read_bytes() == original


def test_detector_source_mutation_fails_closed(tmp_path: Path) -> None:
    source, output, audit = (tmp_path / name for name in ("in.wav", "out.wav", "audit.json"))
    sf.write(source, np.ones(10), 10)

    def detector(path: Path) -> list:
        sf.write(path, np.zeros(10), 10)
        return []

    with pytest.raises(ValueError, match="source changed"):
        redact_audio(source, output, audit, [], detector=detector)
    assert not output.exists() and not audit.exists()
    assert not list(tmp_path.glob(".esl-*"))


def test_cli_and_explicit_intervals_required(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    source = tmp_path / "in.wav"
    sf.write(source, np.ones(10), 10)
    intervals = tmp_path / "intervals.json"
    intervals.write_text('[{"start":0.1,"end":0.3}]')
    command = [
        "privacy",
        "redact",
        str(source),
        str(tmp_path / "out.wav"),
        "--audit",
        str(tmp_path / "audit.json"),
    ]
    args = parser.parse_args(command)
    assert args.func(args) == 2
    args = parser.parse_args([*command, "--intervals", str(intervals)])
    assert args.func(args) == 0


def test_reject_source_alias_and_bad_output(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    sf.write(source, np.ones(10), 10)
    with pytest.raises(ValueError):
        redact_audio(source, source, tmp_path / "audit", [])
    with pytest.raises(ValueError):
        redact_audio(source, tmp_path / "out.flac", tmp_path / "audit", [])
