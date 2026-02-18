"""Analysis context shared by metric plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .audio import AudioBuffer, detect_signal_layout
from .config import AnalysisConfig, CalibrationProfile


@dataclass(slots=True)
class AnalysisContext:
    """Immutable-ish runtime context used by metrics and exporters."""

    audio: AudioBuffer
    config: AnalysisConfig
    calibration: CalibrationProfile | None
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def signal(self) -> np.ndarray:
        return self.audio.samples

    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @property
    def channels(self) -> int:
        return self.audio.channels

    @property
    def duration_s(self) -> float:
        return self.audio.duration_s

    @property
    def frame_size(self) -> int:
        return self.config.frame_size

    @property
    def hop_size(self) -> int:
        return self.config.hop_size

    @property
    def layout(self) -> str:
        return detect_signal_layout(self.channels, self.audio.source_path)

    def mono(self) -> np.ndarray:
        """Downmix multichannel signal to mono by channel average."""
        if self.channels == 1:
            return self.signal[:, 0]
        return np.mean(self.signal, axis=1)
