"""Structured spatial and Ambisonics metadata inference.

This module promotes channel-layout guesses into a stable metadata object so
CLI outputs and downstream tooling can reason about multichannel and
Ambisonics-aware inputs consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_FOA_TOKENS = ("ambi", "ambisonic", "bformat", "b_format", "foa", "wxyz", "ambix", "acn", "fuma")
_ACN_TOKENS = ("ambix", "acn")
_FUMA_TOKENS = ("fuma", "wxyz", "bformat", "b_format", "foa")
_SN3D_TOKENS = ("sn3d",)
_N3D_TOKENS = ("n3d",)
_MAXN_TOKENS = ("maxn", "fuma")


@dataclass(slots=True)
class AmbisonicsMetadata:
    order: int
    component_order: str
    normalization: str
    channels_expected: int
    format_hint: str | None = None
    convention_confidence: float = 0.5
    complete_set: bool = True


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


def _perfect_square_order(channels: int) -> int | None:
    if channels < 1:
        return None
    root = int(round(channels**0.5))
    if root * root != channels:
        return None
    order = root - 1
    return order if order >= 0 else None


def _foa_labels(component_order: str) -> list[str]:
    if component_order.upper() == "ACN":
        # ACN index order for FOA is 0,1,2,3 -> W,Y,Z,X.
        return ["W", "Y", "Z", "X"]
    return ["W", "X", "Y", "Z"]


def _infer_ambisonics_convention(path_text: str) -> tuple[str, str, str, float]:
    component_order = "unknown"
    normalization = "unknown"
    format_hint = "b_format"
    confidence = 0.45
    if any(token in path_text for token in _ACN_TOKENS):
        component_order = "ACN"
        format_hint = "ambix"
        confidence = 0.85
    elif any(token in path_text for token in _FUMA_TOKENS):
        component_order = "FuMa"
        format_hint = "b_format"
        confidence = 0.8
    if any(token in path_text for token in _SN3D_TOKENS):
        normalization = "SN3D"
        confidence = max(confidence, 0.9)
    elif any(token in path_text for token in _N3D_TOKENS):
        normalization = "N3D"
        confidence = max(confidence, 0.9)
    elif any(token in path_text for token in _MAXN_TOKENS):
        normalization = "maxN"
        confidence = max(confidence, 0.75)
    elif component_order == "FuMa":
        normalization = "maxN"
    return component_order, normalization, format_hint, confidence


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
        component_order, normalization, format_hint, confidence = _infer_ambisonics_convention(path_text)
        return SpatialMetadata(
            layout_family="ambisonic",
            layout_hint="ambisonic_b_format",
            channels=4,
            channel_labels=_foa_labels(component_order),
            source_channel_layout=channel_layout,
            ambisonics=AmbisonicsMetadata(
                order=1,
                component_order=component_order,
                normalization=normalization,
                channels_expected=4,
                format_hint=format_hint,
                convention_confidence=confidence,
                complete_set=True,
            ),
            array_geometry=array_geometry,
        )

    inferred_order = _perfect_square_order(ch)
    if inferred_order is not None and inferred_order >= 1 and any(token in path_text for token in _FOA_TOKENS):
        component_order, normalization, format_hint, confidence = _infer_ambisonics_convention(path_text)
        labels = _foa_labels(component_order) if inferred_order == 1 else [f"ambi_{i}" for i in range(ch)]
        return SpatialMetadata(
            layout_family="ambisonic",
            layout_hint="ambisonic_higher_order",
            channels=ch,
            channel_labels=labels,
            source_channel_layout=channel_layout,
            ambisonics=AmbisonicsMetadata(
                order=inferred_order,
                component_order=component_order,
                normalization=normalization,
                channels_expected=(inferred_order + 1) ** 2,
                format_hint=format_hint,
                convention_confidence=confidence,
                complete_set=((inferred_order + 1) ** 2 == ch),
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
