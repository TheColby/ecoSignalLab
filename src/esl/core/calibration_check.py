"""Calibration drift check utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import soundfile as sf

from esl.core.audio import read_audio
from esl.core.calibration import db, dbfs_to_pa, pa_to_dbfs, precision_chain_available
from esl.core.config import CalibrationProfile


@dataclass(slots=True)
class CalibrationCheckConfig:
    tone_path: Path
    output_path: Path
    dbfs_reference: float
    spl_reference_db: float = 94.0
    weighting: str = "Z"
    mic_sensitivity_mv_pa: float | None = None
    preamp_gain_db: float | None = None
    adc_full_scale_vrms: float | None = None
    calibration_profile: CalibrationProfile | None = None
    device_id: str | None = None
    history_csv: Path | None = None
    max_drift_db: float = 1.0
    sample_rate: int | None = None


REFERENCE_FIXTURES: dict[str, dict[str, float | str]] = {
    "sine_1khz_minus20dbfs": {
        "frequency_hz": 1000.0,
        "duration_s": 1.0,
        "sample_rate": 48000.0,
        "dbfs_rms": -20.0,
        "weighting": "Z",
    },
    "sine_1khz_minus26dbfs": {
        "frequency_hz": 1000.0,
        "duration_s": 1.0,
        "sample_rate": 48000.0,
        "dbfs_rms": -26.0,
        "weighting": "Z",
    },
}


@dataclass(slots=True)
class CalibrationVerifyConfig:
    fixture: str
    output_path: Path
    calibration_profile: CalibrationProfile | None = None
    max_abs_error_db: float = 0.25
    write_tone_path: Path | None = None


def _fixture_signal(dbfs_rms: float, frequency_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
    n = max(int(round(duration_s * sample_rate)), 1)
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    rms_lin = float(np.power(10.0, float(dbfs_rms) / 20.0))
    peak = min(float(np.sqrt(2.0) * rms_lin), 0.999999)
    return (peak * np.sin(2.0 * np.pi * float(frequency_hz) * t)).astype(np.float32)


def run_calibration_check(cfg: CalibrationCheckConfig) -> tuple[Path, dict[str, Any], bool]:
    """Compute calibration drift against expected dBFS reference."""
    tone = read_audio(cfg.tone_path, target_sr=cfg.sample_rate)
    rms = float(np.sqrt(np.mean(np.square(tone.samples))))
    measured_dbfs = float(db(rms))
    drift_db = float(measured_dbfs - cfg.dbfs_reference)
    within_tolerance = bool(abs(drift_db) <= float(cfg.max_drift_db))
    offset = float(cfg.spl_reference_db - cfg.dbfs_reference)
    spl_estimate_db = float(measured_dbfs + offset)

    profile_for_pressure: CalibrationProfile | None = cfg.calibration_profile
    if profile_for_pressure is None:
        profile_for_pressure = CalibrationProfile(
            dbfs_reference=float(cfg.dbfs_reference),
            spl_reference_db=float(cfg.spl_reference_db),
            weighting=str(cfg.weighting).upper(),
            mic_sensitivity_mv_pa=cfg.mic_sensitivity_mv_pa,
            preamp_gain_db=cfg.preamp_gain_db,
            adc_full_scale_vrms=cfg.adc_full_scale_vrms,
            calibration_tone_file=None,
        )
    pressure_supported = precision_chain_available(profile_for_pressure)
    measured_pa_rms: float | None = None
    measured_db_spl_from_pa: float | None = None
    if pressure_supported:
        measured_pa_rms = float(dbfs_to_pa(measured_dbfs, profile_for_pressure))
        measured_db_spl_from_pa = float(20.0 * np.log10(max(measured_pa_rms / 20e-6, 1e-18)))

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
        "preamp_gain_db": None if cfg.preamp_gain_db is None else float(cfg.preamp_gain_db),
        "adc_full_scale_vrms": None if cfg.adc_full_scale_vrms is None else float(cfg.adc_full_scale_vrms),
        "measured_rms": rms,
        "measured_dbfs": measured_dbfs,
        "drift_db": drift_db,
        "max_drift_db": float(cfg.max_drift_db),
        "within_tolerance": within_tolerance,
        "spl_estimate_db": spl_estimate_db,
        "pressure_chain_supported": pressure_supported,
        "measured_pa_rms": measured_pa_rms,
        "measured_db_spl_from_pa": measured_db_spl_from_pa,
        "assumptions": [
            "Drift compares measured tone RMS (dBFS) against configured dbfs_reference.",
            "SPL estimate is a reference offset mapping, not a compliance measurement.",
            "Precision Pa<->dBFS conversion requires mic_sensitivity_mv_pa, preamp_gain_db, and adc_full_scale_vrms.",
        ],
        "profile": (
            {
                "dbfs_reference": cfg.calibration_profile.dbfs_reference,
                "spl_reference_db": cfg.calibration_profile.spl_reference_db,
                "weighting": cfg.calibration_profile.weighting,
                "mic_sensitivity_mv_pa": cfg.calibration_profile.mic_sensitivity_mv_pa,
                "preamp_gain_db": cfg.calibration_profile.preamp_gain_db,
                "adc_full_scale_vrms": cfg.calibration_profile.adc_full_scale_vrms,
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


def run_calibration_verify(cfg: CalibrationVerifyConfig) -> tuple[Path, dict[str, Any], bool]:
    """Verify the calibration/check path against a deterministic synthetic reference fixture."""
    fixture = REFERENCE_FIXTURES.get(cfg.fixture)
    if fixture is None:
        raise ValueError(f"Unknown calibration reference fixture: {cfg.fixture}")

    dbfs_rms = float(fixture["dbfs_rms"])
    sr = int(fixture["sample_rate"])
    signal = _fixture_signal(
        dbfs_rms=dbfs_rms,
        frequency_hz=float(fixture["frequency_hz"]),
        duration_s=float(fixture["duration_s"]),
        sample_rate=sr,
    )

    tone_path = cfg.write_tone_path
    cleanup_path: Path | None = None
    if tone_path is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="esl_cal_verify_"))
        cleanup_path = tmp_dir
        tone_path = tmp_dir / f"{cfg.fixture}.wav"
    tone_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(tone_path, signal, sr, subtype="FLOAT")

    profile = cfg.calibration_profile or CalibrationProfile(
        dbfs_reference=dbfs_rms,
        spl_reference_db=94.0,
        weighting=str(fixture.get("weighting", "Z")).upper(),
    )
    check_report_path = cfg.output_path.with_name(cfg.output_path.stem + ".check.json")
    _, check_report, within = run_calibration_check(
        CalibrationCheckConfig(
            tone_path=tone_path,
            output_path=check_report_path,
            dbfs_reference=float(profile.dbfs_reference),
            spl_reference_db=float(profile.spl_reference_db),
            weighting=str(profile.weighting),
            mic_sensitivity_mv_pa=profile.mic_sensitivity_mv_pa,
            preamp_gain_db=profile.preamp_gain_db,
            adc_full_scale_vrms=profile.adc_full_scale_vrms,
            calibration_profile=profile,
            max_drift_db=float(cfg.max_abs_error_db),
            sample_rate=sr,
        )
    )

    measured_dbfs = float(check_report["measured_dbfs"])
    abs_error_db = abs(measured_dbfs - dbfs_rms)
    pressure_chain_error_db: float | None = None
    if precision_chain_available(profile):
        pa_ref = float(dbfs_to_pa(dbfs_rms, profile))
        dbfs_back = float(pa_to_dbfs(pa_ref, profile))
        pressure_chain_error_db = abs(dbfs_back - dbfs_rms)

    verify_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": cfg.fixture,
        "tone_path": str(tone_path.resolve()),
        "expected_dbfs_rms": dbfs_rms,
        "measured_dbfs_rms": measured_dbfs,
        "abs_error_db": abs_error_db,
        "max_abs_error_db": float(cfg.max_abs_error_db),
        "within_tolerance": bool(abs_error_db <= float(cfg.max_abs_error_db) and within),
        "pressure_chain_error_db": pressure_chain_error_db,
        "calibration_profile": {
            "dbfs_reference": profile.dbfs_reference,
            "spl_reference_db": profile.spl_reference_db,
            "weighting": profile.weighting,
            "mic_sensitivity_mv_pa": profile.mic_sensitivity_mv_pa,
            "preamp_gain_db": profile.preamp_gain_db,
            "adc_full_scale_vrms": profile.adc_full_scale_vrms,
        },
        "check_report_path": str(check_report_path.resolve()),
    }
    cfg.output_path.write_text(json.dumps(verify_report, indent=2), encoding="utf-8")
    if cleanup_path is not None and cfg.write_tone_path is None:
        # Keep the synthesized fixture only when the user asked for it.
        try:
            tone_path.unlink(missing_ok=True)
            cleanup_path.rmdir()
        except Exception:
            pass
    return cfg.output_path, verify_report, bool(verify_report["within_tolerance"])
