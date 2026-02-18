from esl.core.calibration import dbfs_to_spl, spl_to_dbfs
from esl.core.config import CalibrationProfile


def test_dbfs_spl_roundtrip() -> None:
    profile = CalibrationProfile(dbfs_reference=-20.0, spl_reference_db=74.0, weighting="A")
    spl = dbfs_to_spl(-12.0, profile)
    assert abs(spl - 82.0) < 1e-6
    dbfs = spl_to_dbfs(spl, profile)
    assert abs(dbfs - (-12.0)) < 1e-6
