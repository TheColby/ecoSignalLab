import pytest

from esl.core.calibration import dbfs_to_pa, dbfs_to_spl, pa_to_dbfs, precision_chain_available, spl_to_dbfs
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
