"""Structured spatial and Ambisonics metadata inference.

This module promotes channel-layout guesses into a stable metadata object so
CLI outputs and downstream tooling can reason about multichannel and
Ambisonics-aware inputs consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_FOA_TOKENS = ("ambi", "ambisonic", "bformat", "b_format", "foa", "wxyz")


@dataclass(slots=True)
class AmbisonicsMetadata:
    order: int
    component_order: str
    normalization: str
    channels_expected: int


@dataclass(slots=True)
class SpatialMetadata:
    layout_family: str
    layout_hint: str
    channels: int
    channel_labels: list[str]
    source_channel_layout: str | None = None
    ambisonics: AmbisonicsMetadata | None = None
    array_geometry: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.ambisonics is None:
            payload["ambisonics"] = None
        return payload


def _default_labels(channels: int) -> list[str]:
    return [f"ch{i + 1}" for i in range(max(int(channels), 0))]


def infer_spatial_metadata(
    channels: int,
    source_path: str | Path,
    *,
    source_channel_layout: str | None = None,
    array_geometry: dict[str, Any] | None = None,
) -> SpatialMetadata:
    """Infer structured spatial metadata from channel count and source hints."""
    ch = max(int(channels), 0)
    path_text = str(source_path).lower()
    channel_layout = str(source_channel_layout) if source_channel_layout else None

    if ch <= 1:
        return SpatialMetadata(
            layout_family="mono",
            layout_hint="mono",
            channels=max(ch, 1),
            channel_labels=["ch1"],
            source_channel_layout=channel_layout,
            array_geometry=array_geometry,
        )

    if ch == 2:
        if str(source_path).lower().endswith(".sofa"):
            return SpatialMetadata(
                layout_family="spatial_ir",
                layout_hint="binaural",
                channels=2,
                channel_labels=["left", "right"],
                source_channel_layout=channel_layout,
                array_geometry=array_geometry,
            )
        return SpatialMetadata(
            layout_family="stereo",
            layout_hint="stereo",
            channels=2,
            channel_labels=["left", "right"],
            source_channel_layout=channel_layout,
            array_geometry=array_geometry,
        )

    if ch == 4 and any(token in path_text for token in _FOA_TOKENS):
        return SpatialMetadata(
            layout_family="ambisonic",
            layout_hint="ambisonic_b_format",
            channels=4,
            channel_labels=["W", "X", "Y", "Z"],
            source_channel_layout=channel_layout,
            ambisonics=AmbisonicsMetadata(
                order=1,
                component_order="WXYZ",
                normalization="unknown",
                channels_expected=4,
            ),
            array_geometry=array_geometry,
        )

    return SpatialMetadata(
        layout_family="multichannel",
        layout_hint="multichannel",
        channels=ch,
        channel_labels=_default_labels(ch),
        source_channel_layout=channel_layout,
        array_geometry=array_geometry,
    )

