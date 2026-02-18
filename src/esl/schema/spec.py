"""JSON schema definition for esl analysis outputs."""

from __future__ import annotations


def analysis_output_schema() -> dict:
    """Return JSON Schema (Draft 2020-12) for esl result documents."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://ecosignallab.dev/schema/analysis-output-0.1.0.json",
        "title": "ecoSignalLab analysis output",
        "type": "object",
        "required": ["esl_version", "analysis_time_utc", "config_hash", "metadata", "metrics"],
        "properties": {
            "esl_version": {"type": "string"},
            "analysis_time_utc": {"type": "string", "format": "date-time"},
            "config_hash": {"type": "string"},
            "analysis_mode": {"type": "string", "enum": ["full", "streaming"]},
            "metadata": {
                "type": "object",
                "required": [
                    "input_path",
                    "sample_rate",
                    "num_samples",
                    "channels",
                    "duration_s",
                    "frame_size",
                    "hop_size",
                ],
                "properties": {
                    "input_path": {"type": "string"},
                    "sample_rate": {"type": "integer", "minimum": 1},
                    "num_samples": {"type": "integer", "minimum": 0},
                    "channels": {"type": "integer", "minimum": 1},
                    "duration_s": {"type": "number", "minimum": 0},
                    "format_name": {"type": ["string", "null"]},
                    "subtype": {"type": ["string", "null"]},
                    "backend": {"type": ["string", "null"]},
                    "frame_size": {"type": "integer", "minimum": 1},
                    "hop_size": {"type": "integer", "minimum": 1},
                    "seed": {"type": "integer"},
                    "project": {"type": ["string", "null"]},
                    "variant": {"type": ["string", "null"]},
                    "calibration": {"type": ["object", "null"]},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
            "metrics": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "required": ["units", "summary", "confidence", "spec"],
                    "properties": {
                        "units": {"type": "string"},
                        "summary": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "series": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "timestamps_s": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "extra": {"type": "object"},
                        "spec": {
                            "type": "object",
                            "required": [
                                "name",
                                "category",
                                "units",
                                "window_size",
                                "hop_size",
                                "streaming_capable",
                                "calibration_dependency",
                                "ml_compatible",
                                "confidence_logic",
                            ],
                            "properties": {
                                "name": {"type": "string"},
                                "category": {"type": "string"},
                                "units": {"type": "string"},
                                "window_size": {"type": "integer", "minimum": 0},
                                "hop_size": {"type": "integer", "minimum": 0},
                                "streaming_capable": {"type": "boolean"},
                                "calibration_dependency": {"type": "boolean"},
                                "ml_compatible": {"type": "boolean"},
                                "confidence_logic": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
    }
