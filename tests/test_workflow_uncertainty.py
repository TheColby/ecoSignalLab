"""Reproducible bootstrap, shared-offset propagation, and audio path tests."""

import argparse
import json

import numpy as np
import pytest
import soundfile as sf

from esl.workflows.uncertainty import _audio_series, bootstrap_mean, register


def test_deterministic_and_sensitivity():
    result = bootstrap_mean([1, 2, 3, 4], resamples=200, seed=4, sensitivity_step=2)
    assert result == bootstrap_mean([1, 2, 3, 4], resamples=200, seed=4, sensitivity_step=2)
    assert result["estimate"] == 2.5
    assert result["sampling_interval"][0] <= 2.5 <= result["sampling_interval"][1]
    assert result["sensitivity"]["minus"] == 0.5
    assert result["sensitivity"]["plus"] == 4.5
    assert result["sampling_interval"] == result["propagated_interval"]


def test_systematic_does_not_shrink_with_sample_count():
    result = bootstrap_mean([3.0] * 200, resamples=2000, offset_std=2)
    assert result["sampling_interval"] == [3.0, 3.0]
    assert 1.8 < result["propagated_standard_error"] < 2.2
    assert result["propagated_interval"][1] - result["propagated_interval"][0] > 7


def test_block_bootstrap():
    values = np.repeat([0.0, 10.0], 40)
    iid = bootstrap_mean(values, resamples=400)
    blocks = bootstrap_mean(values, block_size=10, resamples=400)
    assert blocks["sampling_standard_error"] > iid["sampling_standard_error"]


@pytest.mark.parametrize(
    "values,kwargs",
    [
        ([1], {}),
        ([1, float("nan")], {}),
        ([1, 2], {"confidence": 1}),
        ([1, 2], {"block_size": 3}),
        ([1, 2], {"offset_std": -1}),
        ([1, 2], {"seed": -1}),
        ([1, 2], {"resamples": 99}),
        ([1, 2], {"sensitivity_step": float("inf")}),
        ([1] * 1000, {"resamples": 100000}),
    ],
)
def test_invalid(values, kwargs):
    with pytest.raises(ValueError):
        bootstrap_mean(values, **kwargs)


def test_audio(tmp_path):
    path = tmp_path / "constant.wav"
    sf.write(path, np.column_stack([np.ones(250) * 0.5, np.zeros(250)]), 100, subtype="FLOAT")
    report = _audio_series(path, 1, 0)
    assert report["metrics"]["rms_dbfs"]["series"] == pytest.approx([-6.0206, -6.0206])
    assert report["metadata"]["discarded_tail_samples"] == 50
    assert _audio_series(path, 1, 1)["metrics"]["rms_dbfs"]["series"] == [-240.0, -240.0]
    with pytest.raises(ValueError, match="channel"):
        _audio_series(path, 1, 2)


def parser():
    result = argparse.ArgumentParser()
    register(result.add_subparsers(required=True))
    return result


def test_report_cli_and_selection_stability(tmp_path):
    source, output = tmp_path / "r.json", tmp_path / "out.json"
    source.write_text(
        json.dumps(
            {
                "metrics": {
                    "x": {"units": "dB", "series": [1, 2, 3]},
                    "y": {"units": "Hz", "series": [4, 5, 6]},
                }
            }
        )
    )
    argv = ["uncertainty", str(source), "--resamples", "100", "--output", str(output)]
    args = parser().parse_args(argv)
    assert args.func(args) == 0
    first = json.loads(output.read_text())
    args = parser().parse_args(argv + ["--metric", "x"])
    assert args.func(args) == 0
    second = json.loads(output.read_text())
    assert first["metrics"]["x"] == second["metrics"]["x"]
    assert first["schema_version"] == "1.0.0"
    assert first["esl_version"]
    assert len(first["provenance"]["sha256"]) == 64


def test_summary_only_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text('{"metrics": {"x": {"summary": {"mean": 3}}}}')
    args = parser().parse_args(["uncertainty", str(path)])
    with pytest.raises(ValueError, match="summary-only"):
        args.func(args)


def test_audio_cli(tmp_path, capsys):
    path = tmp_path / "a.wav"
    sf.write(path, np.ones(200) * 0.25, 100, subtype="FLOAT")
    args = parser().parse_args(
        ["uncertainty", str(path), "--resamples", "100", "--offset-std", "rms_dbfs=0.5"]
    )
    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["metrics"]["rms_dbfs"]["offset_std"] == 0.5


def test_cannot_overwrite_input(tmp_path):
    source = tmp_path / "a.json"
    args = parser().parse_args(["uncertainty", str(source), "--output", str(source)])
    with pytest.raises(ValueError, match="overwrite"):
        args.func(args)
