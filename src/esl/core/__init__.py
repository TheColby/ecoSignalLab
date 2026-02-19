"""Core analysis engine components."""

from .audio import read_audio, stream_audio
from .calibration import load_calibration
from .config import AnalysisConfig, BatchConfig, CalibrationProfile, IngestConfig


def analyze(*args, **kwargs):
    """Lazy import wrapper to avoid core<->metrics import cycles."""
    from .analyzer import analyze as _analyze

    return _analyze(*args, **kwargs)


__all__ = [
    "analyze",
    "read_audio",
    "stream_audio",
    "load_calibration",
    "AnalysisConfig",
    "BatchConfig",
    "CalibrationProfile",
    "IngestConfig",
]
