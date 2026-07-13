"""Structured spatial and Ambisonics metadata inference.

This module promotes channel-layout guesses into a stable metadata object so
CLI outputs and downstream tooling can reason about multichannel and
Ambisonics-aware inputs consistently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
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
    standards_profile: str = "unknown"
    normalization_scale: str = "unknown"
    channel_map: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None


@dataclass(slots=True)
class SpatialMetadata:
    layout_family: str
    layout_hint: str
    channels: int
    channel_labels: list[str]
    source_channel_layout: str | None = None
    ambisonics: AmbisonicsMetadata | None = None
    array_geometry: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None

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


def _ceil_ambisonic_order(channels: int) -> int | None:
    if channels < 1:
        return None
    order = 0
    while (order + 1) ** 2 < channels:
        order += 1
    return order if order >= 1 else None


def _acn_index(l_degree: int, m_order: int) -> int:
    return int(l_degree * l_degree + l_degree + m_order)


def _ambisonic_channel_map(order: int, component_order: str, channels: int) -> list[dict[str, Any]]:
    if str(component_order).upper() == "ACN":
        rows: list[dict[str, Any]] = []
        for l_degree in range(order + 1):
            for m_order in range(-l_degree, l_degree + 1):
                acn = _acn_index(l_degree, m_order)
                rows.append(
                    {
                        "index": int(acn),
                        "label": f"Y_{l_degree}_{m_order}",
                        "degree_l": int(l_degree),
                        "order_m": int(m_order),
                        "acn": int(acn),
                    }
                )
        return rows[:channels]
    if order == 1 and channels == 4:
        labels = _foa_labels(component_order)
        return [
            {
                "index": idx,
                "label": label,
                "degree_l": None,
                "order_m": None,
                "acn": None,
            }
            for idx, label in enumerate(labels)
        ]
    return [
        {
            "index": idx,
            "label": f"ambi_{idx}",
            "degree_l": None,
            "order_m": None,
            "acn": None,
        }
        for idx in range(channels)
    ]


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


def _normalization_scale(normalization: str) -> str:
    norm = normalization.upper()
    if norm == "SN3D":
        return "semi_normalized"
    if norm == "N3D":
        return "orthonormal"
    if norm == "MAXN":
        return "legacy_maxn"
    return "unknown"


def _standards_profile(component_order: str, normalization: str, format_hint: str | None) -> str:
    comp = component_order.lower()
    norm = normalization.lower()
    if comp == "acn" and norm in {"sn3d", "n3d"}:
        return f"ambix_acn_{norm}"
    if comp == "fuma" and norm == "maxn":
        return "fuma_wxyz_maxn"
    return str(format_hint or "unknown")


def _ambisonics_warnings(channels: int, expected: int, component_order: str, normalization: str) -> list[str]:
    warnings: list[str] = []
    if channels != expected:
        warnings.append(
            f"Channel count {channels} is not a complete Ambisonics set; expected {expected} channels."
        )
    if component_order == "unknown":
        warnings.append("Ambisonics component order is unknown; use filename tokens like acn, ambix, or fuma.")
    if normalization == "unknown":
        warnings.append("Ambisonics normalization is unknown; use filename tokens like sn3d, n3d, or maxn.")
    return warnings


def _ambisonics_metadata(
    *,
    order: int,
    channels: int,
    component_order: str,
    normalization: str,
    format_hint: str | None,
    confidence: float,
) -> AmbisonicsMetadata:
    expected = int((order + 1) ** 2)
    warnings = _ambisonics_warnings(channels, expected, component_order, normalization)
    return AmbisonicsMetadata(
        order=int(order),
        component_order=component_order,
        normalization=normalization,
        channels_expected=expected,
        format_hint=format_hint,
        convention_confidence=confidence,
        complete_set=(channels == expected),
        standards_profile=_standards_profile(component_order, normalization, format_hint),
        normalization_scale=_normalization_scale(normalization),
        channel_map=_ambisonic_channel_map(order, component_order, channels),
        warnings=warnings,
    )


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
        ambi = _ambisonics_metadata(
            order=1,
            channels=4,
            component_order=component_order,
            normalization=normalization,
            format_hint=format_hint,
            confidence=confidence,
        )
        return SpatialMetadata(
            layout_family="ambisonic",
            layout_hint="ambisonic_b_format",
            channels=4,
            channel_labels=_foa_labels(component_order),
            source_channel_layout=channel_layout,
            ambisonics=ambi,
            array_geometry=array_geometry,
        )

    has_ambisonic_hint = any(token in path_text for token in _FOA_TOKENS)
    inferred_order = _perfect_square_order(ch)
    if inferred_order is None and has_ambisonic_hint:
        inferred_order = _ceil_ambisonic_order(ch)
    if inferred_order is not None and inferred_order >= 1 and has_ambisonic_hint:
        component_order, normalization, format_hint, confidence = _infer_ambisonics_convention(path_text)
        labels = (
            _foa_labels(component_order)
            if inferred_order == 1 and ch == 4
            else [row["label"] for row in _ambisonic_channel_map(inferred_order, component_order, ch)]
        )
        ambi = _ambisonics_metadata(
            order=inferred_order,
            channels=ch,
            component_order=component_order,
            normalization=normalization,
            format_hint=format_hint,
            confidence=confidence,
        )
        return SpatialMetadata(
            layout_family="ambisonic",
            layout_hint=(
                "ambisonic_higher_order"
                if inferred_order > 1
                else "ambisonic_incomplete_or_nonstandard"
            ),
            channels=ch,
            channel_labels=labels,
            source_channel_layout=channel_layout,
            ambisonics=ambi,
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


def _load_sidecar_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON/YAML spatial sidecar as an object."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional YAML dependency
            raise RuntimeError("YAML spatial sidecars require PyYAML; use JSON or install pyyaml.") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Spatial sidecar must contain an object: {path}")
    nested = value.get("spatial_metadata", value)
    if not isinstance(nested, dict):
        raise ValueError(f"spatial_metadata must be an object: {path}")
    return nested


def apply_spatial_metadata_sidecar(
    inferred: SpatialMetadata,
    sidecar_path: str | Path,
) -> SpatialMetadata:
    """Apply a validated, explicit spatial/Ambisonics metadata override.

    Sidecars intentionally cannot change the decoder-observed channel count. This
    preserves a useful invariant: a metadata override may clarify interpretation,
    but it cannot invent audio channels that do not exist.
    """
    path = Path(sidecar_path)
    if not path.exists():
        raise FileNotFoundError(f"Spatial metadata sidecar not found: {path}")
    override = _load_sidecar_mapping(path)
    allowed = {
        "layout_family",
        "layout_hint",
        "channels",
        "channel_labels",
        "source_channel_layout",
        "array_geometry",
        "ambisonics",
    }
    unknown = sorted(set(override) - allowed)
    if unknown:
        raise ValueError(f"Unsupported spatial sidecar fields in {path}: {', '.join(unknown)}")
    if "channels" in override and int(override["channels"]) != inferred.channels:
        raise ValueError(
            f"Spatial sidecar channels={override['channels']} does not match decoded channels={inferred.channels}."
        )

    payload = inferred.to_dict()
    changed: list[str] = []
    for key in ("layout_family", "layout_hint", "source_channel_layout", "array_geometry"):
        if key in override:
            payload[key] = override[key]
            changed.append(key)
    if "channel_labels" in override:
        labels = override["channel_labels"]
        if not isinstance(labels, list) or len(labels) != inferred.channels or not all(isinstance(x, str) for x in labels):
            raise ValueError("spatial sidecar channel_labels must be one string per decoded channel.")
        payload["channel_labels"] = labels
        changed.append("channel_labels")

    if "ambisonics" in override:
        raw_ambi = override["ambisonics"]
        if not isinstance(raw_ambi, dict):
            raise ValueError("spatial sidecar ambisonics must be an object.")
        required = {"order", "component_order", "normalization"}
        missing = sorted(required - set(raw_ambi))
        if missing:
            raise ValueError(f"spatial sidecar ambisonics is missing: {', '.join(missing)}")
        order = int(raw_ambi["order"])
        if order < 1:
            raise ValueError("spatial sidecar ambisonics.order must be >= 1.")
        component_order = str(raw_ambi["component_order"])
        normalization = str(raw_ambi["normalization"])
        format_hint = str(raw_ambi.get("format_hint") or "sidecar")
        confidence = float(raw_ambi.get("convention_confidence", 1.0))
        rebuilt = _ambisonics_metadata(
            order=order,
            channels=inferred.channels,
            component_order=component_order,
            normalization=normalization,
            format_hint=format_hint,
            confidence=max(0.0, min(confidence, 1.0)),
        )
        if "channel_map" in raw_ambi:
            if not isinstance(raw_ambi["channel_map"], list) or len(raw_ambi["channel_map"]) != inferred.channels:
                raise ValueError("spatial sidecar ambisonics.channel_map must contain one entry per decoded channel.")
            rebuilt.channel_map = raw_ambi["channel_map"]
        payload["ambisonics"] = asdict(rebuilt)
        changed.append("ambisonics")

    return SpatialMetadata(
        layout_family=str(payload["layout_family"]),
        layout_hint=str(payload["layout_hint"]),
        channels=inferred.channels,
        channel_labels=list(payload["channel_labels"]),
        source_channel_layout=(str(payload["source_channel_layout"]) if payload.get("source_channel_layout") else None),
        ambisonics=(AmbisonicsMetadata(**payload["ambisonics"]) if isinstance(payload.get("ambisonics"), dict) else None),
        array_geometry=payload.get("array_geometry"),
        provenance={
            "source": "sidecar",
            "sidecar_path": str(path.resolve()),
            "overridden_fields": changed,
        },
    )
