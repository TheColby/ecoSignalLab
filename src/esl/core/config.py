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
    preamp_gain_db: float | None = None
    adc_full_scale_vrms: float | None = None
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
    compute_device: str = "auto"
    make_plots: bool = False
    ml_export: bool = False
    summary_only: bool = False
    streamable_only: bool = False
    allow_full_read: bool = False
    max_series_points: int | None = None
    frame_table_csv: Path | None = None
    frame_table_parquet_dir: Path | None = None
    frame_table_hdf5: Path | None = None
    checkpoint_dir: Path | None = None
    resume: bool = False
    spatial_metadata_sidecar: Path | None = None
    channel_pair: tuple[int, int] | None = None
    """Zero-based channel indices used by paired-channel metrics.

    When unset, paired metrics fall back to channels (0, 1) and record that the
    pair was inferred from channel count rather than declared.
    """

    channel_semantics: str = "unspecified"
    """How the declared channel pair should be interpreted.

    ``"binaural"`` asserts that the pair really is a left/right ear signal, which
    is what licenses the interaural reading of IACC, ILD and ITD.  Anything else
    leaves them as interchannel quantities.
    """

    stream_continuity: bool = True
    """Keep the streaming frame grid aligned with full-file framing.

    When true, chunked analysis re-blocks the stream so every frame starts on
    the same global grid a full-file run would use, carrying
    ``frame_size - hop_size`` samples of overlap across chunk boundaries.  Set
    false to reproduce the legacy behaviour in which each chunk is framed
    independently from its own sample 0.
    """

    stream_context_frames: int = 1
    """Frames of already-emitted context prepended to each chunk.

    One frame is enough for first-difference metrics (spectral flux and the
    novelty curves built on it) to see their predecessor across a boundary.
    Ignored when ``stream_continuity`` is false.
    """


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
