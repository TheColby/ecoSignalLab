"""Monitoring contracts without depending on sibling workflow registrations."""

import argparse
import copy
import json

import pytest

from esl.workflows.monitor import _assignments, compare_reports, register


def report(value=10):
    return {
        "schema_version": "0.2.0",
        "metadata": {"calibration": {"offset": 94}},
        "metrics": {"rms_dbfs": {"units": "dBFS", "summary": {"mean": value}}},
    }


def test_limits_and_relative():
    assert compare_reports(report(), report(12), {"rms_dbfs": 2})["status"] == "pass"
    result = compare_reports(
        report(), report(12), {"rms_dbfs": 3}, relative_thresholds={"rms_dbfs": 0.1}
    )
    assert result["status"] == "alert"
    assert result["metrics"][0]["delta"] == 2
    assert result["schema_version"] == "1.0.0"
    assert result["esl_version"]


@pytest.mark.parametrize(
    "field,value", [("calibration", None), ("frame_size", 4096), ("sample_rate", 48000)]
)
def test_incompatible_metadata(field, value):
    current = report()
    current["metadata"][field] = value
    with pytest.raises(ValueError, match="Incompatible"):
        compare_reports(report(), current, {"rms_dbfs": 1})


def test_units_spec_missing_and_nonfinite():
    for field, value in [("units", "Pa"), ("spec", {"window_size": 4})]:
        current = report()
        current["metrics"]["rms_dbfs"][field] = value
        with pytest.raises(ValueError, match="Incompatible"):
            compare_reports(report(), current, {"rms_dbfs": 1})
    with pytest.raises(ValueError, match="Missing selected"):
        compare_reports(report(), report(), {"unknown": 1})
    with pytest.raises(ValueError, match="finite"):
        compare_reports(report(), report(float("nan")), {"rms_dbfs": 1})
    with pytest.raises(ValueError, match="threshold"):
        compare_reports(report(), report(), {})


def test_uncalibrated_and_zero_baseline():
    baseline = {"metrics": {"x": 0}}
    current = {"metrics": {"x": 1}}
    with pytest.raises(ValueError, match="Calibration missing"):
        compare_reports(baseline, current, {"x": 1})
    result = compare_reports(
        baseline, current, {}, relative_thresholds={"x": 1}, allow_uncalibrated=True
    )
    assert result["status"] == "alert"
    assert result["metrics"][0]["relative_change"] is None
    assert (
        compare_reports(
            baseline, baseline, {}, relative_thresholds={"x": 0}, allow_uncalibrated=True
        )["status"]
        == "pass"
    )


def test_calibration_dependent_cannot_bypass():
    baseline = report()
    baseline["metadata"]["calibration"] = None
    baseline["metrics"]["rms_dbfs"]["spec"] = {"calibration_dependency": True}
    with pytest.raises(ValueError, match="Calibration required"):
        compare_reports(baseline, copy.deepcopy(baseline), {"rms_dbfs": 1}, allow_uncalibrated=True)


@pytest.mark.parametrize("items", [["x=-1"], ["x=nan"], ["x=1", "x=2"], ["oops"]])
def test_bad_assignments(items):
    with pytest.raises(ValueError):
        _assignments(items)


def test_cli(tmp_path):
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    baseline, current, output = [tmp_path / name for name in ("b.json", "c.json", "o.json")]
    baseline.write_text(json.dumps(report()))
    current.write_text(json.dumps(report(13)))
    args = parser.parse_args(
        [
            "monitor",
            "compare",
            str(baseline),
            str(current),
            "--threshold",
            "rms_dbfs=2",
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 2
    result = json.loads(output.read_text())
    assert result["provenance"]["baseline"] == str(baseline)
    assert len(result["provenance"]["baseline_sha256"]) == 64


def test_path_dependent_hash_not_used():
    baseline, current = report(), report()
    baseline["pipeline_hash"] = "path-a"
    current["pipeline_hash"] = "path-b"
    assert compare_reports(baseline, current, {"rms_dbfs": 1})["status"] == "pass"


def test_cannot_overwrite_input(tmp_path):
    source = tmp_path / "b.json"
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    args = parser.parse_args(
        [
            "monitor",
            "compare",
            str(source),
            str(source),
            "--threshold",
            "x=1",
            "--output",
            str(source),
        ]
    )
    with pytest.raises(ValueError, match="overwrite"):
        args.func(args)
