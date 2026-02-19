"""Calibration drift check utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from esl.core.audio import read_audio
from esl.core.calibration import db
from esl.core.config import CalibrationProfile


@dataclass(slots=True)
class CalibrationCheckConfig:
    tone_path: Path
    output_path: Path
    dbfs_reference: float
    spl_reference_db: float = 94.0
    weighting: str = "Z"
    mic_sensitivity_mv_pa: float | None = None
    calibration_profile: CalibrationProfile | None = None
    device_id: str | None = None
    history_csv: Path | None = None
    max_drift_db: float = 1.0
    sample_rate: int | None = None


def run_calibration_check(cfg: CalibrationCheckConfig) -> tuple[Path, dict[str, Any], bool]:
    """Compute calibration drift against expected dBFS reference."""
    tone = read_audio(cfg.tone_path, target_sr=cfg.sample_rate)
    rms = float(np.sqrt(np.mean(np.square(tone.samples))))
    measured_dbfs = float(db(rms))
    drift_db = float(measured_dbfs - cfg.dbfs_reference)
    within_tolerance = bool(abs(drift_db) <= float(cfg.max_drift_db))
    offset = float(cfg.spl_reference_db - cfg.dbfs_reference)
    spl_estimate_db = float(measured_dbfs + offset)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tone_path": str(cfg.tone_path.resolve()),
        "sample_rate": int(tone.sample_rate),
        "channels": int(tone.channels),
        "weighting": str(cfg.weighting).upper(),
        "device_id": cfg.device_id,
        "dbfs_reference": float(cfg.dbfs_reference),
        "spl_reference_db": float(cfg.spl_reference_db),
        "mic_sensitivity_mv_pa": (
            None if cfg.mic_sensitivity_mv_pa is None else float(cfg.mic_sensitivity_mv_pa)
        ),
        "measured_rms": rms,
        "measured_dbfs": measured_dbfs,
        "drift_db": drift_db,
        "max_drift_db": float(cfg.max_drift_db),
        "within_tolerance": within_tolerance,
        "spl_estimate_db": spl_estimate_db,
        "assumptions": [
            "Drift compares measured tone RMS (dBFS) against configured dbfs_reference.",
            "SPL estimate is a reference offset mapping, not a compliance measurement.",
        ],
        "profile": (
            {
                "dbfs_reference": cfg.calibration_profile.dbfs_reference,
                "spl_reference_db": cfg.calibration_profile.spl_reference_db,
                "weighting": cfg.calibration_profile.weighting,
                "mic_sensitivity_mv_pa": cfg.calibration_profile.mic_sensitivity_mv_pa,
                "calibration_tone_file": cfg.calibration_profile.calibration_tone_file,
            }
            if cfg.calibration_profile is not None
            else None
        ),
    }

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if cfg.history_csv:
        cfg.history_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not cfg.history_csv.exists()
        with cfg.history_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "created_utc",
                    "device_id",
                    "tone_path",
                    "sample_rate",
                    "measured_dbfs",
                    "dbfs_reference",
                    "drift_db",
                    "max_drift_db",
                    "within_tolerance",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "created_utc": report["created_utc"],
                    "device_id": cfg.device_id or "",
                    "tone_path": report["tone_path"],
                    "sample_rate": report["sample_rate"],
                    "measured_dbfs": report["measured_dbfs"],
                    "dbfs_reference": report["dbfs_reference"],
                    "drift_db": report["drift_db"],
                    "max_drift_db": report["max_drift_db"],
                    "within_tolerance": report["within_tolerance"],
                }
            )

    return cfg.output_path, report, within_tolerance
