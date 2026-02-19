"""Spatial analysis helpers and wrappers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from esl.core import AnalysisConfig, analyze
from esl.core.audio import read_audio


SPEED_OF_SOUND_M_S = 343.0

SPATIAL_DEFAULT_METRICS: list[str] = [
    "interchannel_coherence",
    "iacc",
    "ild_db",
    "ipd_rad",
    "itd_s",
    "doa_azimuth_proxy_deg",
    "ambisonic_diffuseness",
    "ambisonic_energy_vector_azimuth_deg",
    "ambisonic_energy_vector_elevation_deg",
]


def load_array_config(path: str | None) -> dict[str, Any] | None:
    """Load optional array configuration from JSON or YAML."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Array config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("YAML array config requires pyyaml") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Array config must be an object: {p}")
    return payload


def _scalar_summary(value: float) -> dict[str, float]:
    return {
        "mean": value,
        "std": 0.0,
        "min": value,
        "max": value,
        "p50": value,
        "p95": value,
    }


def _doa_azimuth_from_itd(itd_s: float, mic_spacing_m: float) -> float:
    arg = float(np.clip(itd_s * SPEED_OF_SOUND_M_S / max(float(mic_spacing_m), 1e-12), -1.0, 1.0))
    return float(np.degrees(np.arcsin(arg)))


def run_spatial_analysis(
    cfg: AnalysisConfig,
    array_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run analysis with spatial-focused metadata enrichment."""
    result = analyze(cfg)
    meta = result.setdefault("metadata", {})
    spatial_meta = {
        "array_config": array_config,
        "selected_spatial_metrics": list(cfg.metrics),
    }
    meta["spatial"] = spatial_meta

    if array_config and "doa_azimuth_proxy_deg" in result.get("metrics", {}):
        spacing = array_config.get("mic_spacing_m")
        if isinstance(spacing, (int, float)) and spacing > 0:
            doa = result["metrics"]["doa_azimuth_proxy_deg"]
            extra = doa.get("extra", {}) if isinstance(doa, dict) else {}
            mean_itd = extra.get("mean_itd_s") if isinstance(extra, dict) else None
            if isinstance(mean_itd, (int, float)):
                az = _doa_azimuth_from_itd(float(mean_itd), float(spacing))
                doa["series"] = [az]
                doa["timestamps_s"] = [0.0]
                doa["summary"] = _scalar_summary(az)
                doa_extra = doa.setdefault("extra", {})
                if isinstance(doa_extra, dict):
                    doa_extra["mic_spacing_m"] = float(spacing)
                    doa_extra["azimuth_recomputed_from_array_config"] = True
    return result


def stereo_beam_map(
    audio_path: Path,
    mic_spacing_m: float = 0.2,
    azimuth_step_deg: int = 5,
    target_sr: int | None = None,
) -> list[dict[str, float]]:
    """Compute a simple stereo delay-and-sum azimuth score map."""
    audio = read_audio(audio_path, target_sr=target_sr)
    x = audio.samples
    if x.ndim != 2 or x.shape[1] < 2:
        raise RuntimeError("Beam map requires at least two channels.")
    left = x[:, 0].astype(np.float64)
    right = x[:, 1].astype(np.float64)
    n = left.shape[0]
    if n == 0:
        return []

    l0 = left - np.mean(left)
    r0 = right - np.mean(right)
    denom = float(np.linalg.norm(l0) * np.linalg.norm(r0) + 1e-12)

    out: list[dict[str, float]] = []
    for az in range(-90, 91, max(1, int(azimuth_step_deg))):
        tau_s = float((mic_spacing_m / SPEED_OF_SOUND_M_S) * np.sin(np.deg2rad(float(az))))
        shift = int(round(tau_s * audio.sample_rate))
        if shift >= 0:
            a = l0[shift:]
            b = r0[: n - shift]
        else:
            sh = abs(shift)
            a = l0[: n - sh]
            b = r0[sh:]
        if a.size == 0 or b.size == 0:
            score = 0.0
        else:
            score = float(np.dot(a, b) / denom)
        out.append({"azimuth_deg": float(az), "score": score})

    scores = np.array([row["score"] for row in out], dtype=np.float64)
    if scores.size and np.max(np.abs(scores)) > 0.0:
        m = float(np.max(np.abs(scores)))
        for row in out:
            row["score"] = float(row["score"] / m)
    return out


def write_beam_map_csv(rows: list[dict[str, float]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["azimuth_deg", "score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path
