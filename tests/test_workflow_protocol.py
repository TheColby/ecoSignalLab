"""Strict protocol schema and standalone command integration."""

import argparse
import copy
import json
from pathlib import Path

import pytest

from esl.workflows.protocol import PROTOCOL_SCHEMA, load_protocol, register, validate_protocol


def invoke(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(arguments)
    return args.func(args)


@pytest.fixture
def manifest(tmp_path: Path) -> dict:
    path = tmp_path / "protocol.json"
    assert (
        invoke(
            [
                "protocol",
                "init",
                str(path),
                "--name",
                "Park",
                "--author",
                "A",
                "--channels",
                "2",
                "--site",
                "Park east",
                "--deployment",
                "2026-10",
                "--microphone",
                "matched pair",
                "--channel-label",
                "left",
                "--channel-label",
                "right",
                "--calibration",
                "94 dB check before capture",
                "--consent",
                "Approved restricted deployment",
            ]
        )
        == 0
    )
    return load_protocol(path)


def test_context_and_validation(manifest: dict) -> None:
    assert manifest["context"]["channel_layout"] == ["left", "right"]
    assert manifest["context"]["site"] == "Park east"
    assert validate_protocol(manifest) is manifest
    assert PROTOCOL_SCHEMA["additionalProperties"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("channels", True),
        ("sample_rate", 0),
        ("sample_rate", 1.5),
        ("window_seconds", float("nan")),
        ("hop_seconds", float("inf")),
        ("hop_seconds", 2),
        ("hop_seconds", 1e-10),
        ("metrics", []),
        ("metrics", ["rms", "rms"]),
        ("metrics", [True]),
        ("metrics", [{}]),
        ("privacy_policy", "automatic"),
    ],
)
def test_invalid_config(manifest: dict, field: str, value: object) -> None:
    manifest["config"][field] = value
    with pytest.raises(ValueError):
        validate_protocol(manifest)


def test_missing_unknown_nested_and_layout(manifest: dict) -> None:
    for update in (
        {"unexpected": 1},
        {"context": {"channel_layout": ["only-one"]}},
        {"context": {"site": "Park", "unknown": "x"}},
    ):
        broken = copy.deepcopy(manifest)
        broken.update(update)
        with pytest.raises(ValueError):
            validate_protocol(broken)
    del manifest["provenance"]
    with pytest.raises(ValueError):
        validate_protocol(manifest)


def test_duplicate_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"name": "a", "name": "b"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_protocol(path)


def test_cli_no_overwrite_validate_and_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "p.json"
    command = ["protocol", "init", str(path), "--name", "Study", "--author", "Me"]
    assert invoke(command) == 0
    original = path.read_bytes()
    assert invoke(command) == 2
    assert path.read_bytes() == original
    capsys.readouterr()
    assert invoke(["protocol", "validate", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] and report["schema_version"] == "1.0.0"
    assert report["esl_version"] and report["provenance"] and report["config"]
    assert invoke(["protocol", "schema"]) == 0
    assert json.loads(capsys.readouterr().out) == PROTOCOL_SCHEMA


def test_init_bad_context_leaves_no_output(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    assert (
        invoke(
            [
                "protocol",
                "init",
                str(path),
                "--name",
                "N",
                "--author",
                "A",
                "--channels",
                "2",
                "--channel-label",
                "one",
            ]
        )
        == 2
    )
    assert not path.exists()
