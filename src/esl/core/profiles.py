"""Multi-resolution analysis profile loading and application."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esl.core.config import AnalysisConfig


@dataclass(slots=True)
class ResolutionProfile:
    """Single profile entry for multi-resolution analysis."""

    name: str
    frame_size: int | None = None
    hop_size: int | None = None
    sample_rate: int | None = None
    chunk_size: int | None = None
    metrics: list[str] | None = None


def _load_profile_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML profiles require pyyaml to be installed.") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Profile file must contain an object: {path}")
    return payload


def load_resolution_profiles(path: str | Path) -> list[ResolutionProfile]:
    """Load resolution profiles from YAML/JSON."""
    profile_path = Path(path)
    payload = _load_profile_payload(profile_path)
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise RuntimeError(f"Profile file must define a non-empty 'profiles' list: {profile_path}")

    out: list[ResolutionProfile] = []
    for idx, item in enumerate(raw_profiles):
        if not isinstance(item, dict):
            raise RuntimeError(f"Profile entry at index {idx} must be an object")
        name_raw = item.get("name")
        name = str(name_raw).strip() if name_raw is not None else f"profile_{idx + 1}"
        metrics_raw = item.get("metrics")
        metrics: list[str] | None = None
        if metrics_raw is not None:
            if not isinstance(metrics_raw, list):
                raise RuntimeError(f"Profile '{name}' field 'metrics' must be a list")
            metrics = [str(m).strip() for m in metrics_raw if str(m).strip()]

        def _opt_int(field: str) -> int | None:
            val = item.get(field)
            return None if val is None else int(val)

        out.append(
            ResolutionProfile(
                name=name,
                frame_size=_opt_int("frame_size"),
                hop_size=_opt_int("hop_size"),
                sample_rate=_opt_int("sample_rate"),
                chunk_size=_opt_int("chunk_size"),
                metrics=metrics,
            )
        )
    return out


def with_resolution_profile(base: AnalysisConfig, profile: ResolutionProfile) -> AnalysisConfig:
    """Create a derived analysis config with profile overrides."""
    return AnalysisConfig(
        input_path=base.input_path,
        output_dir=base.output_dir,
        frame_size=profile.frame_size if profile.frame_size is not None else base.frame_size,
        hop_size=profile.hop_size if profile.hop_size is not None else base.hop_size,
        sample_rate=profile.sample_rate if profile.sample_rate is not None else base.sample_rate,
        chunk_size=profile.chunk_size if profile.chunk_size is not None else base.chunk_size,
        metrics=list(profile.metrics) if profile.metrics is not None else list(base.metrics),
        calibration=base.calibration,
        project=base.project,
        variant=base.variant,
        verbosity=base.verbosity,
        debug=base.debug,
        seed=base.seed,
        make_plots=base.make_plots,
        ml_export=base.ml_export,
    )
