"""Configuration models for esl analyze/batch flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


Weighting = Literal["A", "C", "Z"]


@dataclass(slots=True)
class CalibrationProfile:
    """Calibration metadata used to map dBFS to SPL-like units."""

    dbfs_reference: float = 0.0
    spl_reference_db: float = 94.0
    weighting: Weighting = "Z"
    mic_sensitivity_mv_pa: float | None = None
    calibration_tone_file: str | None = None


@dataclass(slots=True)
class AnalysisConfig:
    """Config controlling analysis behavior and reproducibility."""

    input_path: Path
    output_dir: Path = Path(".")
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int | None = None
    metrics: list[str] = field(default_factory=list)
    calibration: CalibrationProfile | None = None
    project: str | None = None
    variant: str | None = None
    verbosity: int = 1
    debug: int = 0
    seed: int = 42
    make_plots: bool = False
    ml_export: bool = False


@dataclass(slots=True)
class BatchConfig:
    """Batch analysis configuration for directory-driven processing."""

    input_dir: Path
    output_dir: Path
    recursive: bool = True
    glob_patterns: tuple[str, ...] = (
        "*.wav",
        "*.flac",
        "*.aiff",
        "*.aif",
        "*.rf64",
        "*.caf",
        "*.mp3",
        "*.aac",
        "*.ogg",
        "*.opus",
        "*.wma",
        "*.alac",
        "*.m4a",
        "*.sofa",
    )
    analysis_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestConfig:
    """Online ingest job configuration."""

    source: Literal["freesound", "huggingface", "http"]
    query: str
    limit: int = 20
    output_dir: Path = Path("ingest")
    auto_analyze: bool = False
