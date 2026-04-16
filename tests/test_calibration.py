import pytest

from esl.core.calibration import dbfs_to_pa, dbfs_to_spl, pa_to_dbfs, precision_chain_available, spl_to_dbfs
from esl.core.calibration_check import CalibrationVerifyConfig, run_calibration_verify
from esl.core.config import CalibrationProfile


def test_dbfs_spl_roundtrip() -> None:
    profile = CalibrationProfile(dbfs_reference=-20.0, spl_reference_db=74.0, weighting="A")
    spl = dbfs_to_spl(-12.0, profile)
    assert abs(spl - 82.0) < 1e-6
    dbfs = spl_to_dbfs(spl, profile)
    assert abs(dbfs - (-12.0)) < 1e-6


def test_pa_dbfs_roundtrip_with_precision_chain() -> None:
    profile = CalibrationProfile(
        dbfs_reference=-20.0,
        spl_reference_db=74.0,
        weighting="A",
        mic_sensitivity_mv_pa=12.5,
        preamp_gain_db=34.0,
        adc_full_scale_vrms=1.0,
    )
    assert precision_chain_available(profile) is True
    pa = 1.0
    dbfs = pa_to_dbfs(pa, profile)
    pa_back = dbfs_to_pa(dbfs, profile)
    assert abs(pa_back - pa) < 1e-9


def test_pa_dbfs_requires_precision_chain() -> None:
    profile = CalibrationProfile(
        dbfs_reference=-20.0,
        spl_reference_db=74.0,
        weighting="A",
        mic_sensitivity_mv_pa=12.5,
        preamp_gain_db=None,
        adc_full_scale_vrms=1.0,
    )
    assert precision_chain_available(profile) is False
    with pytest.raises(ValueError):
        _ = pa_to_dbfs(1.0, profile)
    with pytest.raises(ValueError):
        _ = dbfs_to_pa(-20.0, profile)


def test_calibration_verify_fixture_passes(tmp_path) -> None:
    report_path, report, ok = run_calibration_verify(
        CalibrationVerifyConfig(
            fixture="sine_1khz_minus20dbfs",
            output_path=tmp_path / "verify.json",
            max_abs_error_db=0.25,
        )
    )
    assert report_path.exists()
    assert ok is True
    assert report["fixture"] == "sine_1khz_minus20dbfs"
    assert abs(float(report["expected_dbfs_rms"]) - float(report["measured_dbfs_rms"])) < 0.25


def test_calibration_verify_precision_chain_fixture_reports_pressure_error(tmp_path) -> None:
    report_path, report, ok = run_calibration_verify(
        CalibrationVerifyConfig(
            fixture="sine_1khz_minus20dbfs_precision_chain",
            output_path=tmp_path / "verify_precision.json",
            max_abs_error_db=0.25,
        )
    )
    assert report_path.exists()
    assert ok is True
    assert report["fixture"] == "sine_1khz_minus20dbfs_precision_chain"
    assert report["pressure_chain_error_db"] is not None
