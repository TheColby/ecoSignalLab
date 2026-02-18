"""Calibration and weighting utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import CalibrationProfile, Weighting


def load_calibration(path: str | Path) -> CalibrationProfile:
    """Load calibration profile from YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {p}")
    raw: dict[str, Any]
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to load YAML calibration files.") from exc
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif p.suffix.lower() == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    else:
        raise ValueError("Calibration file must be YAML or JSON.")

    return CalibrationProfile(
        dbfs_reference=float(raw.get("dbfs_reference", raw.get("dbfs_ref", 0.0))),
        spl_reference_db=float(raw.get("spl_reference_db", raw.get("spl_ref_db", 94.0))),
        weighting=str(raw.get("weighting", "Z")).upper(),
        mic_sensitivity_mv_pa=(
            None if raw.get("mic_sensitivity_mv_pa") is None else float(raw["mic_sensitivity_mv_pa"])
        ),
        calibration_tone_file=raw.get("calibration_tone_file"),
    )


def calibration_to_dict(profile: CalibrationProfile | None) -> dict[str, Any] | None:
    """Serialize calibration profile for result metadata."""
    if profile is None:
        return None
    return asdict(profile)


def db(value: np.ndarray | float, floor: float = 1e-12) -> np.ndarray | float:
    """Safe decibel conversion."""
    if isinstance(value, np.ndarray):
        return 20.0 * np.log10(np.maximum(np.abs(value), floor))
    return float(20.0 * np.log10(max(abs(value), floor)))


def power_db(value: np.ndarray | float, floor: float = 1e-12) -> np.ndarray | float:
    """Safe power decibel conversion."""
    if isinstance(value, np.ndarray):
        return 10.0 * np.log10(np.maximum(np.abs(value), floor))
    return float(10.0 * np.log10(max(abs(value), floor)))


def dbfs_to_spl(dbfs: float | np.ndarray, profile: CalibrationProfile | None) -> float | np.ndarray:
    """Map dBFS values to SPL space based on user calibration reference."""
    if profile is None:
        return dbfs
    offset = profile.spl_reference_db - profile.dbfs_reference
    return dbfs + offset


def spl_to_dbfs(spl: float | np.ndarray, profile: CalibrationProfile | None) -> float | np.ndarray:
    """Map SPL values back to dBFS reference space."""
    if profile is None:
        return spl
    offset = profile.spl_reference_db - profile.dbfs_reference
    return spl - offset


def weighting_db(f_hz: np.ndarray, weighting: Weighting = "Z") -> np.ndarray:
    """Return IEC-style A/C/Z weighting values in dB for frequencies."""
    w = weighting.upper()
    f = np.maximum(f_hz.astype(np.float64), 1e-6)
    if w == "Z":
        return np.zeros_like(f)

    f2 = f * f
    if w == "A":
        ra_num = (12200.0**2) * (f2**2)
        ra_den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200.0**2)
        ra = ra_num / np.maximum(ra_den, 1e-20)
        return 20.0 * np.log10(np.maximum(ra, 1e-20)) + 2.0
    if w == "C":
        rc_num = (12200.0**2) * f2
        rc_den = (f2 + 20.6**2) * (f2 + 12200.0**2)
        rc = rc_num / np.maximum(rc_den, 1e-20)
        return 20.0 * np.log10(np.maximum(rc, 1e-20)) + 0.06
    raise ValueError(f"Unsupported weighting: {weighting}")


def weighted_rms(signal: np.ndarray, sample_rate: int, weighting: Weighting = "Z") -> np.ndarray:
    """Compute frequency-domain weighted RMS per channel."""
    if signal.ndim != 2:
        raise ValueError("Signal must be 2-D [samples, channels].")
    if weighting.upper() == "Z":
        return np.sqrt(np.mean(np.square(signal), axis=0))

    n = signal.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    w_lin = np.power(10.0, weighting_db(freqs, weighting) / 20.0)
    out = np.zeros(signal.shape[1], dtype=np.float64)
    for c in range(signal.shape[1]):
        spec = np.fft.rfft(signal[:, c])
        weighted = spec * w_lin
        weighted_t = np.fft.irfft(weighted, n=n)
        out[c] = np.sqrt(np.mean(np.square(weighted_t)))
    return out
