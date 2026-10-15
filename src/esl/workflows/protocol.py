"""Strict, dependency-free protocol manifests and their JSON Schema contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from esl import __version__

PROTOCOL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "artifact_kind", "esl_version", "name", "config", "provenance"],
    "properties": {
        "schema_version": {"const": "1.0.0"},
        "artifact_kind": {"const": "measurement_protocol"},
        "esl_version": {"type": "string", "minLength": 1},
        "name": {"type": "string", "minLength": 1},
        "context": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "site": {"type": "string", "minLength": 1},
                "deployment": {"type": "string", "minLength": 1},
                "microphones": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "channel_layout": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "calibration": {"type": "string", "minLength": 1},
                "consent": {"type": "string", "minLength": 1},
            },
        },
        "provenance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["author", "description"],
            "properties": {
                "author": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
            },
        },
        "config": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "sample_rate",
                "channels",
                "window_seconds",
                "hop_seconds",
                "metrics",
                "privacy_policy",
            ],
            "properties": {
                "sample_rate": {"type": "integer", "minimum": 1},
                "channels": {"type": "integer", "minimum": 1},
                "window_seconds": {"type": "number", "exclusiveMinimum": 0},
                "hop_seconds": {"type": "number", "exclusiveMinimum": 0},
                "metrics": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
                "privacy_policy": {"enum": ["none", "supplied-intervals", "detector-review"]},
            },
        },
    },
}


def _validate(value: Any, schema: dict[str, Any], location: str) -> None:
    """Evaluate the deliberately small JSON Schema vocabulary used above."""
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{location}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location}: expected one of {schema['enum']}")
    kind = schema.get("type")
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: type(value) is int,
        "number": lambda: type(value) in (int, float) and math.isfinite(value),
    }
    if kind and not checks[kind]():
        raise ValueError(f"{location}: expected {kind}")
    if kind == "object":
        missing = set(schema.get("required", [])) - value.keys()
        extra = value.keys() - schema["properties"].keys()
        if missing or extra:
            raise ValueError(f"{location}: missing {sorted(missing)}, unknown {sorted(extra)}")
        for key, item in value.items():
            _validate(item, schema["properties"][key], f"{location}.{key}")
    if kind == "string" and len(value) < schema.get("minLength", 0):
        raise ValueError(f"{location}: must not be empty")
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{location}: below minimum")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(f"{location}: must be positive")
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{location}: requires at least one item")
        for index, item in enumerate(value):
            _validate(item, schema["items"], f"{location}[{index}]")
        if schema.get("uniqueItems") and len(set(value)) != len(value):
            raise ValueError(f"{location}: duplicate items")


def validate_protocol(manifest: Any) -> dict[str, Any]:
    """Validate strict structure and window semantics; return the supplied manifest."""
    _validate(manifest, PROTOCOL_SCHEMA, "protocol")
    config = manifest["config"]
    layout = manifest.get("context", {}).get("channel_layout")
    if layout is not None and len(layout) != config["channels"]:
        raise ValueError("context.channel_layout must contain one label per channel")
    if config["hop_seconds"] > config["window_seconds"]:
        raise ValueError("config.hop_seconds must not exceed window_seconds")
    if config["hop_seconds"] * config["sample_rate"] < 1:
        raise ValueError("config.hop_seconds must span at least one sample")
    return manifest


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Read JSON without silently accepting repeated object keys."""
    return validate_protocol(json.loads(Path(path).read_text(), object_pairs_hook=_unique_object))


def _run(args: argparse.Namespace) -> int:
    try:
        if args.protocol_action == "schema":
            print(json.dumps(PROTOCOL_SCHEMA, indent=2))
        elif args.protocol_action == "init":
            manifest = validate_protocol(
                {
                    "schema_version": "1.0.0",
                    "artifact_kind": "measurement_protocol",
                    "esl_version": __version__,
                    "name": args.name,
                    "provenance": {"author": args.author, "description": ""},
                    "config": {
                        "sample_rate": args.sample_rate,
                        "channels": args.channels,
                        "window_seconds": 1.0,
                        "hop_seconds": 0.5,
                        "metrics": ["rms_dbfs"],
                        "privacy_policy": "none",
                    },
                }
            )
            context = {
                key: getattr(args, key)
                for key in (
                    "site",
                    "deployment",
                    "microphones",
                    "channel_layout",
                    "calibration",
                    "consent",
                )
                if getattr(args, key) is not None
            }
            if context:
                manifest["context"] = context
                validate_protocol(manifest)
            with Path(args.path).open("x") as stream:
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
        else:
            manifest = load_protocol(args.path)
            print(
                json.dumps(
                    {
                        "schema_version": "1.0.0",
                        "artifact_kind": "protocol_validation",
                        "esl_version": __version__,
                        "valid": True,
                        "provenance": {"manifest": str(Path(args.path).resolve())},
                        "config": manifest["config"],
                    }
                )
            )
    except (OSError, ValueError) as exc:
        print(f"protocol: {exc}")
        return 2
    return 0


def register(subparsers: Any) -> None:
    """Register protocol init, validate, and schema commands."""
    parser = subparsers.add_parser("protocol", help="Create and validate strict study protocols")
    actions = parser.add_subparsers(dest="protocol_action", required=True)
    init = actions.add_parser("init", help="Create a protocol without overwriting an existing file")
    init.add_argument("path")
    init.add_argument("--name", required=True)
    init.add_argument("--author", required=True)
    init.add_argument("--sample-rate", type=int, default=48000)
    init.add_argument("--channels", type=int, default=1)
    for field in ("site", "deployment", "calibration", "consent"):
        init.add_argument(f"--{field}")
    init.add_argument("--microphone", dest="microphones", action="append")
    init.add_argument("--channel-label", dest="channel_layout", action="append")
    validate = actions.add_parser("validate")
    validate.add_argument("path")
    actions.add_parser("schema")
    parser.set_defaults(func=_run)
