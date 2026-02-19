from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze


GOLDEN_PATH = Path(__file__).parent / "golden" / "metric_expectations.json"


def _golden() -> dict[str, dict[str, float]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    sf.write(path, x.astype(np.float32), sr)


def _metric_mean(result: dict[str, object], metric_name: str) -> float:
    metrics = result["metrics"]  # type: ignore[index]
    payload = metrics[metric_name]  # type: ignore[index]
    summary = payload["summary"]  # type: ignore[index]
    return float(summary["mean"])  # type: ignore[index]


def test_basic_metric_golden_sine(tmp_path: Path) -> None:
    g = _golden()["sine_amp_0_5_1khz_48k"]
    sr = 48_000
    t = np.arange(sr * 2, dtype=np.float64) / sr
    x = 0.5 * np.sin(2.0 * np.pi * 1000.0 * t)
    wav = tmp_path / "sine.wav"
    _write_wav(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            frame_size=4800,
            hop_size=4800,
            metrics=["rms_dbfs", "peak_dbfs", "crest_factor_db", "dc_offset", "clipping_ratio"],
            verbosity=0,
        )
    )

    assert abs(_metric_mean(result, "rms_dbfs") - float(g["rms_dbfs"])) <= float(g["tolerance_db"])
    assert abs(_metric_mean(result, "peak_dbfs") - float(g["peak_dbfs"])) <= float(g["tolerance_db"])
    assert abs(_metric_mean(result, "crest_factor_db") - float(g["crest_factor_db"])) <= float(g["tolerance_db"])
    assert abs(_metric_mean(result, "dc_offset") - float(g["dc_offset"])) <= float(g["tolerance_dc"])

    flags = result["metadata"]["validity_flags"]  # type: ignore[index]
    assert flags["clipping"] is False  # type: ignore[index]


def test_clipping_validity_flag(tmp_path: Path) -> None:
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = np.clip(1.2 * np.sin(2.0 * np.pi * 440.0 * t), -1.0, 1.0)
    wav = tmp_path / "clipped.wav"
    _write_wav(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["clipping_ratio", "rms_dbfs"],
            verbosity=0,
        )
    )
    flags = result["metadata"]["validity_flags"]  # type: ignore[index]
    assert flags["clipping"] is True  # type: ignore[index]
    assert float(flags["clipping_ratio"]) > 0.0  # type: ignore[index]
    assert _metric_mean(result, "clipping_ratio") > 0.0


def test_loudness_golden_tone(tmp_path: Path) -> None:
    g = _golden()["loudness_tone_amp_0_1_1khz_48k"]
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    x = 0.1 * np.sin(2.0 * np.pi * 1000.0 * t)
    wav = tmp_path / "loudness.wav"
    _write_wav(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["integrated_lufs", "short_term_lufs", "momentary_lufs", "loudness_range_lu"],
            verbosity=0,
        )
    )

    integrated = _metric_mean(result, "integrated_lufs")
    assert abs(integrated - float(g["integrated_lufs"])) <= float(g["tolerance_lufs"])
    assert _metric_mean(result, "loudness_range_lu") <= float(g["lra_max"])

    assert result["metrics"]["short_term_lufs"]["series"]  # type: ignore[index]
    assert result["metrics"]["momentary_lufs"]["series"]  # type: ignore[index]


def test_ir_metrics_rt60_edt_and_fit_quality(tmp_path: Path) -> None:
    g = _golden()["ir_exponential_tau_0_3"]
    sr = 48_000
    duration_s = 3.0
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    tau = float(g["tau_s"])
    x = 0.99 * np.exp(-t / tau) * np.cos(2.0 * np.pi * 400.0 * t)
    wav = tmp_path / "ir_decay.wav"
    _write_wav(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rt60_s", "edt_s", "snr_db"],
            verbosity=0,
        )
    )

    rt60 = _metric_mean(result, "rt60_s")
    edt = _metric_mean(result, "edt_s")
    assert abs(rt60 - float(g["rt60_expected_s"])) <= float(g["tolerance_s"])
    assert abs(edt - float(g["rt60_expected_s"])) <= float(g["tolerance_s"])

    fit = result["metrics"]["rt60_s"]["extra"]["fit"]  # type: ignore[index]
    assert float(fit["r2"]) >= float(g["fit_r2_min"])  # type: ignore[index]
    assert int(fit["num_points"]) >= 8  # type: ignore[index]

    flags = result["metadata"]["validity_flags"]  # type: ignore[index]
    assert flags["ir_detected"] is True  # type: ignore[index]
    assert flags["ir_tail_low_snr"] is False  # type: ignore[index]


def test_ir_validity_low_tail_snr_flag(tmp_path: Path) -> None:
    sr = 48_000
    n = sr * 2
    rng = np.random.default_rng(123)
    x = 0.03 * rng.standard_normal(n)
    x[0] = 0.95
    wav = tmp_path / "ir_noisy.wav"
    _write_wav(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rt60_s", "snr_db"],
            verbosity=0,
        )
    )
    flags = result["metadata"]["validity_flags"]  # type: ignore[index]
    assert flags["ir_tail_low_snr"] is True  # type: ignore[index]


def test_ndsi_band_sanity_sign_and_scale(tmp_path: Path) -> None:
    sr = 48_000
    t = np.arange(sr * 2, dtype=np.float64) / sr

    def _run(anthro_amp: float, bio_amp: float, name: str) -> float:
        anthro = anthro_amp * np.sin(2.0 * np.pi * 1500.0 * t)  # anthropophony band [1,2] kHz
        bio = bio_amp * np.sin(2.0 * np.pi * 4000.0 * t)  # biophony band [2,11] kHz
        wav = tmp_path / f"{name}.wav"
        _write_wav(wav, anthro + bio, sr)
        result = analyze(AnalysisConfig(input_path=wav, output_dir=tmp_path, metrics=["ndsi"], verbosity=0))
        return _metric_mean(result, "ndsi")

    ndsi_bio = _run(anthro_amp=0.05, bio_amp=0.25, name="ndsi_bio")
    ndsi_anthro = _run(anthro_amp=0.25, bio_amp=0.05, name="ndsi_anthro")
    ndsi_balanced = _run(anthro_amp=0.15, bio_amp=0.15, name="ndsi_balanced")

    assert -1.0 <= ndsi_bio <= 1.0
    assert -1.0 <= ndsi_anthro <= 1.0
    assert ndsi_bio > 0.2
    assert ndsi_anthro < -0.2
    assert abs(ndsi_balanced) < 0.2


def test_multichannel_aggregation_semantics(tmp_path: Path) -> None:
    sr = 48_000
    t = np.arange(sr, dtype=np.float64) / sr
    ch1 = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    ch2 = 0.4 * np.sin(2.0 * np.pi * 440.0 * t)
    x = np.stack([ch1, ch2], axis=1)
    wav = tmp_path / "stereo_diff.wav"
    _write_wav(wav, x, sr)

    result = analyze(
        AnalysisConfig(
            input_path=wav,
            output_dir=tmp_path,
            metrics=["rms_dbfs", "peak_dbfs"],
            verbosity=0,
        )
    )

    channel_metrics = result["metadata"]["channel_metrics"]  # type: ignore[index]
    channels = channel_metrics["channels"]  # type: ignore[index]
    aggregate = channel_metrics["aggregate"]  # type: ignore[index]

    assert len(channels) == 2
    ch1_rms = float(channels[0]["rms_dbfs"])  # type: ignore[index]
    ch2_rms = float(channels[1]["rms_dbfs"])  # type: ignore[index]
    assert ch1_rms != ch2_rms
    assert ch1_rms < ch2_rms

    rms_linear = np.array([10.0 ** (ch1_rms / 20.0), 10.0 ** (ch2_rms / 20.0)], dtype=np.float64)
    expected_rms_dbfs = float(20.0 * np.log10(np.sqrt(np.mean(np.square(rms_linear)))))
    assert abs(float(aggregate["rms_dbfs"]) - expected_rms_dbfs) <= 1e-9  # type: ignore[index]

    expected_peak = max(float(channels[0]["peak_dbfs"]), float(channels[1]["peak_dbfs"]))  # type: ignore[index]
    assert abs(float(aggregate["peak_dbfs"]) - expected_peak) <= 1e-9  # type: ignore[index]
