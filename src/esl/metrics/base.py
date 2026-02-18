"""Metric plugin base contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from esl.core.context import AnalysisContext


@dataclass(slots=True)
class MetricSpec:
    """Declarative metadata for one metric plugin."""

    name: str
    category: str
    units: str
    window_size: int
    hop_size: int
    streaming_capable: bool
    calibration_dependency: bool
    ml_compatible: bool
    confidence_logic: str
    description: str = ""


@dataclass(slots=True)
class MetricResult:
    """Output payload for a metric computation."""

    name: str
    units: str
    summary: dict[str, float]
    series: list[float] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)
    confidence: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


class MetricPlugin(Protocol):
    """Metric plugin interface."""

    spec: MetricSpec

    def compute(self, ctx: AnalysisContext) -> MetricResult:
        """Compute metric from analysis context."""
        ...
