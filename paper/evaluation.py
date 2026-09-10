#!/usr/bin/env python3
"""Deterministic evaluation used by the JAES engineering-report draft.

Run from the repository root with ``uv run python paper/evaluation.py``.
The script regenerates every table row and figure cited in the manuscript.

Design note: fixtures here are chosen so that they *can* fail.  Closed-form
signals confirm that the arithmetic is right, but every experiment that is used
to support a claim in the paper also sweeps the conditions under which the
claim is supposed to hold -- framing, commensurability, noise floor, decay
shape, chunk size, channel correlation -- so that the reported numbers describe
a working range rather than a single comfortable operating point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf

from esl.core import AnalysisConfig, analyze
from esl.core.calibration_check import CalibrationVerifyConfig, run_calibration_verify
from esl.core.config import CalibrationProfile
from esl.metrics.registry import create_registry


PAPER_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PAPER_DIR / "results"
FIGURES_DIR = PAPER_DIR / "figures"
FRAME_SIZE = 2048
HOP_SIZE = 512
SEED = 20260906
SR = 48_000

STREAM_METRICS = [
    "rms_dbfs",
    "peak_dbfs",
    "spectral_centroid_hz",
    "spectral_flatness",
    "novelty_curve",
    "interchannel_correlation_zerolag",
]

# Metrics whose values are levels in decibels.  A percentage difference on these
# is meaningless -- it depends on the reference and diverges through 0 dB -- so
# agreement is always reported in decibels for this set and as a relative
# difference only for the ratio-scale metrics.
DB_VALUED = {"rms_dbfs", "peak_dbfs", "leq_db", "spl_a_db", "spl_c_db", "spl_z_db", "l90_db", "l10_db", "lmax_db"}


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_mean(result: dict[str, Any], name: str) -> float:
    return float(result["metrics"][name]["summary"]["mean"])


def _metric_extra(result: dict[str, Any], name: str) -> dict[str, Any]:
    extra = result["metrics"][name].get("extra")
    return extra if isinstance(extra, dict) else {}


def _analyze(
    path: Path,
    metrics: list[str],
    *,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE,
    chunk_size: int | None = None,
    stream_continuity: bool = True,
    calibration: CalibrationProfile | None = None,
    channel_pair: tuple[int, int] | None = None,
    channel_semantics: str = "unspecified",
) -> dict[str, Any]:
    return analyze(
        AnalysisConfig(
            input_path=path,
            output_dir=RESULTS_DIR,
            frame_size=frame_size,
            hop_size=hop_size,
            metrics=metrics,
            seed=SEED,
            chunk_size=chunk_size,
            stream_continuity=stream_continuity,
            calibration=calibration,
            channel_pair=channel_pair,
            channel_semantics=channel_semantics,
            verbosity=0,
        )
    )


def _write_float_wav(path: Path, samples: np.ndarray, sample_rate: int = SR) -> None:
    sf.write(path, np.asarray(samples, dtype=np.float32), sample_rate, subtype="FLOAT")


def _rss_mib() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and most BSDs report KiB.
    return raw / (1024.0 * 1024.0) if sys.platform == "darwin" else raw / 1024.0


def _linfit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for an ordinary least-squares line."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(xa, ya, deg=1)
    pred = slope * xa + intercept
    ss_res = float(np.sum(np.square(ya - pred)))
    ss_tot = float(np.sum(np.square(ya - np.mean(ya))))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


# --------------------------------------------------------------------------- #
# signal fixtures
# --------------------------------------------------------------------------- #


def _synthesize_long_stereo(path: Path, duration_s: int, sample_rate: int = SR) -> None:
    """Write a long stereo probe without ever holding it all in memory."""
    rng = np.random.default_rng(SEED + duration_s)
    block = sample_rate
    with sf.SoundFile(path, mode="w", samplerate=sample_rate, channels=2, subtype="FLOAT", format="WAV") as handle:
        for start in range(0, duration_s * sample_rate, block):
            count = min(block, duration_s * sample_rate - start)
            t = (start + np.arange(count, dtype=np.float64)) / sample_rate
            envelope = 0.65 + 0.25 * np.sin(2.0 * np.pi * 0.071 * t)
            tonal = 0.11 * envelope * np.sin(2.0 * np.pi * 440.0 * t)
            tonal += 0.035 * np.sin(2.0 * np.pi * 1730.0 * t)
            noise = 0.008 * rng.standard_normal(count)
            left = tonal + noise
            right = 0.92 * (0.11 * envelope * np.sin(2.0 * np.pi * 440.0 * (t - 0.00025)))
            right += 0.035 * np.sin(2.0 * np.pi * 1730.0 * (t - 0.00025))
            right += 0.008 * rng.standard_normal(count)
            handle.write(np.column_stack([left, right]).astype(np.float32))


def _synthesize_scene(path: Path, duration_s: float = 30.0, sample_rate: int = SR) -> None:
    """A broadband, nonstationary stereo scene for the provenance sweep.

    Deliberately not a pure tone: a tilted-spectrum noise bed, a low-frequency
    rumble, intermittent mid-band events and a tonal component, so that the
    weighting, framing and aggregation axes all have something to bite on.
    """
    rng = np.random.default_rng(SEED)
    n = int(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate

    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    tilt = 1.0 / np.sqrt(np.maximum(freqs, 1.0))  # pink-ish bed
    bed = np.fft.irfft(spectrum * tilt, n=n)
    bed /= np.max(np.abs(bed)) + 1e-12

    rumble = 0.30 * np.sin(2.0 * np.pi * 42.0 * t) * (0.6 + 0.4 * np.sin(2.0 * np.pi * 0.03 * t))
    tonal = 0.08 * np.sin(2.0 * np.pi * 1000.0 * t)

    events = np.zeros(n, dtype=np.float64)
    for onset in rng.uniform(1.0, duration_s - 1.0, size=18):
        start = int(onset * sample_rate)
        length = int(rng.uniform(0.05, 0.35) * sample_rate)
        stop = min(n, start + length)
        env = np.exp(-np.linspace(0.0, 6.0, stop - start))
        carrier = np.sin(2.0 * np.pi * rng.uniform(700.0, 3500.0) * t[start:stop])
        events[start:stop] += 0.55 * env * carrier

    left = 0.22 * bed + rumble + tonal + events
    right = 0.22 * np.roll(bed, 7) + 0.95 * rumble + tonal + 0.9 * np.roll(events, 11)
    peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
    left = 0.5 * left / peak
    right = 0.5 * right / peak
    _write_float_wav(path, np.column_stack([left, right]), sample_rate)


# --------------------------------------------------------------------------- #
# 1. analytic fixtures at the default configuration
# --------------------------------------------------------------------------- #


def analytic_accuracy(temp_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Level and spectral accuracy at the *pipeline default* framing.

    The earlier revision of this evaluation used a frame length chosen so that
    every test tone completed an integer number of periods per frame, which
    removes spectral leakage and reports an accuracy no user running the default
    configuration will ever see.  Here the default 2048/512 framing is used, and
    tone frequencies are additionally split into bin-centred (commensurate) and
    deliberately off-bin cases.
    """
    rows: list[dict[str, Any]] = []
    bin_width = SR / FRAME_SIZE  # 23.4375 Hz at 48 kHz with a 2048-point frame
    tones = [
        (round(250.0 / bin_width) * bin_width, "commensurate"),
        (round(1000.0 / bin_width) * bin_width, "commensurate"),
        (round(4000.0 / bin_width) * bin_width, "commensurate"),
        (250.0 + 0.5 * bin_width, "off_bin"),
        (1000.0 + 0.5 * bin_width, "off_bin"),
        (4000.0 + 0.5 * bin_width, "off_bin"),
    ]
    for frequency_hz, kind in tones:
        for amplitude in (0.05, 0.1, 0.25, 0.5):
            t = np.arange(2 * SR, dtype=np.float64) / SR
            samples = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
            path = temp_dir / f"sine_{int(round(frequency_hz))}_{kind}_{amplitude:.2f}.wav"
            _write_float_wav(path, samples)
            result = _analyze(path, ["rms_dbfs", "peak_dbfs", "crest_factor_db", "spectral_centroid_hz"])
            expected_rms = 20.0 * math.log10(amplitude / math.sqrt(2.0))
            expected_peak = 20.0 * math.log10(amplitude)
            expected_crest = 20.0 * math.log10(math.sqrt(2.0))
            measured_rms = _metric_mean(result, "rms_dbfs")
            measured_peak = _metric_mean(result, "peak_dbfs")
            measured_crest = _metric_mean(result, "crest_factor_db")
            measured_centroid = _metric_mean(result, "spectral_centroid_hz")
            rows.append(
                {
                    "frequency_hz": frequency_hz,
                    "bin_alignment": kind,
                    "amplitude": amplitude,
                    "frame_size": FRAME_SIZE,
                    "hop_size": HOP_SIZE,
                    "rms_expected_dbfs": expected_rms,
                    "rms_measured_dbfs": measured_rms,
                    "rms_abs_error_db": abs(measured_rms - expected_rms),
                    "peak_expected_dbfs": expected_peak,
                    "peak_measured_dbfs": measured_peak,
                    "peak_abs_error_db": abs(measured_peak - expected_peak),
                    "crest_expected_db": expected_crest,
                    "crest_measured_db": measured_crest,
                    "crest_abs_error_db": abs(measured_crest - expected_crest),
                    "centroid_expected_hz": frequency_hz,
                    "centroid_measured_hz": measured_centroid,
                    "centroid_abs_error_hz": abs(measured_centroid - frequency_hz),
                }
            )

    # Centroid error as a function of frame length, which is the parameter the
    # accuracy actually depends on.
    framing_rows: list[dict[str, Any]] = []
    for frame_size in (512, 1024, 2048, 4096, 8192):
        width = SR / frame_size
        for nominal in (250.0, 1000.0, 4000.0):
            for kind in ("commensurate", "off_bin"):
                freq = round(nominal / width) * width
                if kind == "off_bin":
                    freq += 0.5 * width
                t = np.arange(2 * SR, dtype=np.float64) / SR
                path = temp_dir / f"cent_{frame_size}_{int(round(freq))}_{kind}.wav"
                _write_float_wav(path, 0.25 * np.sin(2.0 * np.pi * freq * t))
                result = _analyze(
                    path,
                    ["spectral_centroid_hz"],
                    frame_size=frame_size,
                    hop_size=max(1, frame_size // 4),
                )
                measured = _metric_mean(result, "spectral_centroid_hz")
                framing_rows.append(
                    {
                        "frame_size": frame_size,
                        "bin_width_hz": width,
                        "nominal_hz": nominal,
                        "bin_alignment": kind,
                        "expected_hz": freq,
                        "measured_hz": measured,
                        "abs_error_hz": abs(measured - freq),
                    }
                )
    return rows, framing_rows


# --------------------------------------------------------------------------- #
# 2. decay fixtures with noise floors and a double slope
# --------------------------------------------------------------------------- #


def decay_accuracy(temp_dir: Path) -> list[dict[str, Any]]:
    """T20/T30 against exponential decays with a controlled noise floor.

    A noiseless exponential is the case a Schroeder integrator cannot get wrong.
    Real decays sit on a noise floor and sometimes have two slopes, and the
    implementation performs no Lundeby-style truncation, so this sweep is where
    the estimator is expected to break.
    """
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for tau_s in (0.15, 0.30, 0.60):
        expected = 3.0 * math.log(10.0) * tau_s
        for floor_db in (None, 45.0, 30.0, 20.0):
            for shape in ("single", "double"):
                duration_s = max(4.0, expected * 2.0)
                t = np.arange(int(round(duration_s * SR)), dtype=np.float64) / SR
                envelope = np.exp(-t / tau_s)
                if shape == "double":
                    # Late-decaying second mode at a third of the level, three
                    # times slower: the classic coupled-volume signature.
                    envelope = envelope + 0.33 * np.exp(-t / (3.0 * tau_s))
                    envelope /= envelope[0]
                samples = 0.99 * envelope * np.cos(2.0 * np.pi * 400.0 * t)
                if floor_db is not None:
                    noise_rms = 10.0 ** (-float(floor_db) / 20.0)
                    samples = samples + noise_rms * rng.standard_normal(samples.size)
                    samples = samples / (np.max(np.abs(samples)) + 1e-12) * 0.99
                label = "inf" if floor_db is None else f"{int(floor_db)}"
                path = temp_dir / f"decay_{tau_s:.2f}_{shape}_{label}.wav"
                _write_float_wav(path, samples)
                result = _analyze(path, ["rt60_s", "t20_s", "t30_s", "edt_s"])
                fit = _metric_extra(result, "rt60_s").get("fit", {})
                measured_t30 = _metric_mean(result, "t30_s")
                measured_t20 = _metric_mean(result, "t20_s")
                # A double-slope decay has no single true RT; the early slope is
                # the reference for T20 and the paper reports it as such.
                reference = expected if shape == "single" else float("nan")
                rows.append(
                    {
                        "tau_s": tau_s,
                        "decay_shape": shape,
                        "noise_floor_db": "none" if floor_db is None else floor_db,
                        "expected_rt60_s": reference,
                        "t20_s": measured_t20,
                        "t30_s": measured_t30,
                        "edt_s": _metric_mean(result, "edt_s"),
                        "t30_abs_error_s": abs(measured_t30 - reference) if reference == reference else float("nan"),
                        "t30_rel_error_percent": (
                            100.0 * abs(measured_t30 - reference) / reference if reference == reference else float("nan")
                        ),
                        "t30_valid": bool(_metric_extra(result, "t30_s").get("valid", True)),
                        "t20_valid": bool(_metric_extra(result, "t20_s").get("valid", True)),
                        "rt60_valid": bool(_metric_extra(result, "rt60_s").get("valid", True)),
                        "fit_r2": float(fit.get("r2", float("nan"))),
                        "fit_points": int(fit.get("num_points", 0)),
                        "fit_range_db": str(fit.get("fit_range_db", "")),
                        "dynamic_range_db": float(_metric_extra(result, "rt60_s").get("dynamic_range_db", float("nan"))),
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# 3. spatial fixtures, now with a real coherence and a validity gate
# --------------------------------------------------------------------------- #


SPATIAL_METRICS = [
    "interchannel_correlation_zerolag",
    "interchannel_coherence_msc",
    "iacc",
    "ild_db",
    "itd_s",
]


def spatial_accuracy(temp_dir: Path) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    duration_s = 2
    base = 0.12 * rng.standard_normal(SR * duration_s)
    independent = 0.12 * rng.standard_normal(SR * duration_s)
    delay_samples = 12
    delayed = np.zeros_like(base)
    delayed[delay_samples:] = base[:-delay_samples]
    partial = 0.7 * base + 0.7 * independent  # correlated but not a pure copy

    cases = {
        # The estimator reports the lag of channel 1 relative to channel 2, so a
        # delayed second channel gives a negative signed interchannel delay.
        "identical": (base, base, 0.0, 0.0),
        "second_channel_delayed_12_samples": (base, delayed, -delay_samples / SR, 0.0),
        "second_channel_minus_6_db": (base, 0.5 * base, 0.0, 20.0 * math.log10(2.0)),
        "partially_correlated": (base, partial, 0.0, float("nan")),
        "independent_noise": (base, independent, float("nan"), 0.0),
    }
    rows: list[dict[str, Any]] = []
    for name, (left, right, expected_itd, expected_ild) in cases.items():
        path = temp_dir / f"spatial_{name}.wav"
        _write_float_wav(path, np.column_stack([left, right]))
        result = _analyze(path, SPATIAL_METRICS, channel_pair=(0, 1), channel_semantics="unspecified")
        measured_itd = _metric_mean(result, "itd_s")
        itd_extra = _metric_extra(result, "itd_s")
        rows.append(
            {
                "case": name,
                "expected_itd_us": expected_itd * 1e6 if math.isfinite(expected_itd) else float("nan"),
                "measured_itd_us": measured_itd * 1e6,
                "itd_abs_error_us": (
                    abs(measured_itd - expected_itd) * 1e6 if math.isfinite(expected_itd) else float("nan")
                ),
                "itd_valid": bool(itd_extra.get("valid", True)),
                "itd_median_psr": float(itd_extra.get("median_peak_to_sidelobe_ratio", float("nan"))),
                "expected_ild_db": expected_ild,
                "measured_ild_db": _metric_mean(result, "ild_db"),
                "correlation_zerolag": _metric_mean(result, "interchannel_correlation_zerolag"),
                "coherence_msc": _metric_mean(result, "interchannel_coherence_msc"),
                "iacc": _metric_mean(result, "iacc"),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# 4. does the validity contract actually fire?
# --------------------------------------------------------------------------- #


def validity_matrix(temp_dir: Path) -> list[dict[str, Any]]:
    """Present degenerate inputs and record whether validity fired.

    The result contract claims that an unavailable measurement is distinguishable
    from a real one.  That claim is only worth anything if it is tested against
    inputs on which each metric has nothing to say.
    """
    rng = np.random.default_rng(SEED)
    n = SR * 2

    inputs: dict[str, np.ndarray] = {
        "digital_silence": np.zeros((n, 2), dtype=np.float64),
        "dc_offset_only": np.full((n, 2), 0.4, dtype=np.float64),
        "hard_clipped": np.clip(4.0 * rng.standard_normal((n, 2)), -1.0, 1.0),
        "mono_single_channel": (0.2 * rng.standard_normal((n, 1))),
        "independent_channels": np.column_stack(
            [0.12 * rng.standard_normal(n), 0.12 * rng.standard_normal(n)]
        ),
        "shorter_than_one_frame": 0.2 * rng.standard_normal((1024, 2)),
    }
    probes = ["itd_s", "interchannel_coherence_msc", "iacc", "rms_dbfs", "spectral_centroid_hz", "t30_s"]

    # What *should* happen, stated in advance. A metric is expected to declare
    # itself unavailable only where the input genuinely denies it an answer;
    # elsewhere returning a number is the correct behaviour and flagging it
    # would be a false alarm.
    expect_invalid: dict[tuple[str, str], bool] = {
        ("digital_silence", "itd_s"): True,
        ("digital_silence", "interchannel_coherence_msc"): True,
        ("digital_silence", "iacc"): True,
        ("digital_silence", "rms_dbfs"): False,
        ("digital_silence", "spectral_centroid_hz"): True,
        ("digital_silence", "t30_s"): True,
        # Both channels carry the same constant, so a zero delay is the correct
        # answer rather than a missing one.
        ("dc_offset_only", "itd_s"): False,
        ("dc_offset_only", "interchannel_coherence_msc"): True,
        ("dc_offset_only", "iacc"): False,
        ("dc_offset_only", "rms_dbfs"): False,
        ("dc_offset_only", "spectral_centroid_hz"): False,
        ("dc_offset_only", "t30_s"): True,
        ("hard_clipped", "itd_s"): True,
        ("hard_clipped", "interchannel_coherence_msc"): False,
        ("hard_clipped", "iacc"): False,
        ("hard_clipped", "rms_dbfs"): False,
        ("hard_clipped", "spectral_centroid_hz"): False,
        ("hard_clipped", "t30_s"): True,
        ("mono_single_channel", "itd_s"): True,
        ("mono_single_channel", "interchannel_coherence_msc"): True,
        ("mono_single_channel", "iacc"): True,
        ("mono_single_channel", "rms_dbfs"): False,
        ("mono_single_channel", "spectral_centroid_hz"): False,
        ("mono_single_channel", "t30_s"): True,
        ("independent_channels", "itd_s"): True,
        ("independent_channels", "interchannel_coherence_msc"): False,
        ("independent_channels", "iacc"): False,
        ("independent_channels", "rms_dbfs"): False,
        ("independent_channels", "spectral_centroid_hz"): False,
        ("independent_channels", "t30_s"): True,
        ("shorter_than_one_frame", "itd_s"): True,
        ("shorter_than_one_frame", "interchannel_coherence_msc"): True,
        ("shorter_than_one_frame", "iacc"): False,
        ("shorter_than_one_frame", "rms_dbfs"): False,
        ("shorter_than_one_frame", "spectral_centroid_hz"): False,
        ("shorter_than_one_frame", "t30_s"): True,
    }

    rows: list[dict[str, Any]] = []
    for label, data in inputs.items():
        path = temp_dir / f"degenerate_{label}.wav"
        _write_float_wav(path, data)
        for metric in probes:
            try:
                result = _analyze(path, [metric])
            except Exception as exc:  # pragma: no cover - recorded as an outcome
                rows.append(
                    {
                        "input": label,
                        "metric": metric,
                        "outcome": "raised",
                        "value": "",
                        "flagged_invalid": True,
                        "detail": type(exc).__name__,
                    }
                )
                continue
            value = _metric_mean(result, metric)
            extra = _metric_extra(result, metric)
            flagged = bool(extra.get("valid") is False)
            confidence = float(result["metrics"][metric].get("confidence", 1.0))
            if math.isnan(value):
                outcome = "nan"
            elif flagged:
                outcome = "flagged_invalid"
            elif confidence < 0.5:
                outcome = "low_confidence"
            else:
                outcome = "returned_value"
            unavailable = flagged or math.isnan(value) or confidence < 0.5
            expected = expect_invalid.get((label, metric))
            if expected is None:
                verdict = "unclassified"
            elif expected and unavailable:
                verdict = "correctly_unavailable"
            elif expected and not unavailable:
                verdict = "missed"
            elif not expected and unavailable:
                verdict = "false_alarm"
            else:
                verdict = "correctly_reported"
            rows.append(
                {
                    "input": label,
                    "metric": metric,
                    "expected_unavailable": expected,
                    "outcome": outcome,
                    "verdict": verdict,
                    "value": value,
                    "flagged_invalid": unavailable,
                    "detail": str(extra.get("invalid_reason") or "")[:120],
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# 5. event localisation off the hop grid, with noise
# --------------------------------------------------------------------------- #


def event_localization(temp_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Randomised boundary times and an SNR sweep.

    Placing a transition exactly on a hop boundary guarantees zero localisation
    error and measures nothing.  Boundaries here are drawn uniformly, so the
    quantisation error of the hop grid is part of the reported distribution.
    """
    from scipy.signal import find_peaks

    sample_rate = 16_000
    hop = 256
    frame = 1024
    duration_s = 12.0
    rng = np.random.default_rng(SEED)
    trials = 40

    rows: list[dict[str, Any]] = []
    example: dict[str, Any] | None = None

    for transition in ("tone_to_tone", "tone_to_noise", "level_step"):
        for snr_db in (float("inf"), 20.0, 10.0, 0.0):
            errors: list[float] = []
            detected_count = 0
            for trial in range(trials):
                boundary = float(rng.uniform(3.0, 9.0))
                t = np.arange(int(duration_s * sample_rate), dtype=np.float64) / sample_rate
                before = t < boundary
                after = ~before
                samples = np.empty_like(t)
                samples[before] = 0.10 * np.sin(2.0 * np.pi * 220.0 * t[before])
                if transition == "tone_to_tone":
                    samples[after] = 0.10 * np.sin(2.0 * np.pi * 1900.0 * t[after])
                elif transition == "tone_to_noise":
                    samples[after] = 0.10 * rng.standard_normal(int(np.count_nonzero(after)))
                else:  # level_step
                    samples[after] = 0.30 * np.sin(2.0 * np.pi * 220.0 * t[after])
                if math.isfinite(snr_db):
                    signal_rms = float(np.sqrt(np.mean(np.square(samples))))
                    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
                    samples = samples + noise_rms * rng.standard_normal(samples.size)

                path = temp_dir / "event_probe.wav"
                _write_float_wav(path, samples, sample_rate)
                result = _analyze(path, ["novelty_curve"], frame_size=frame, hop_size=hop)
                novelty = np.asarray(result["metrics"]["novelty_curve"]["series"], dtype=np.float64)
                times = np.asarray(result["metrics"]["novelty_curve"]["timestamps_s"], dtype=np.float64)
                valid = (times > 1.0) & (times < duration_s - 1.0)
                idx = np.flatnonzero(valid)
                if idx.size == 0:
                    continue
                peaks, _ = find_peaks(novelty[valid], distance=max(1, int(0.5 * sample_rate / hop)))
                if peaks.size == 0:
                    continue
                candidates = idx[peaks]
                best = int(candidates[int(np.argmax(novelty[candidates]))])
                error_ms = 1000.0 * abs(float(times[best]) - boundary)
                errors.append(error_ms)
                # A detection is counted as correct if it lands within one frame
                # of the true boundary.
                if error_ms <= 1000.0 * frame / sample_rate:
                    detected_count += 1
                if example is None and transition == "tone_to_tone" and not math.isfinite(snr_db):
                    example = {
                        "times_s": times.tolist(),
                        "novelty": novelty.tolist(),
                        "boundary_s": boundary,
                        "detected_s": float(times[best]),
                    }
            if not errors:
                continue
            arr = np.asarray(errors, dtype=np.float64)
            rows.append(
                {
                    "transition": transition,
                    "snr_db": "inf" if not math.isfinite(snr_db) else snr_db,
                    "trials": len(errors),
                    "detection_rate": detected_count / len(errors),
                    "median_abs_error_ms": float(np.median(arr)),
                    "p95_abs_error_ms": float(np.percentile(arr, 95)),
                    "max_abs_error_ms": float(np.max(arr)),
                    "hop_duration_ms": 1000.0 * hop / sample_rate,
                }
            )
    return rows, (example or {})


# --------------------------------------------------------------------------- #
# 6. streaming: frame accounting and agreement, legacy vs continuity
# --------------------------------------------------------------------------- #


def streaming_agreement(temp_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare full-file, independently chunked, and frame-aligned streaming."""
    path = temp_dir / "stream_probe.wav"
    _synthesize_long_stereo(path, 60)

    full = _analyze(path, STREAM_METRICS)
    frame_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []

    for chunk_s in (0.5, 1.0, 4.0):
        chunk = int(chunk_s * SR)
        legacy = _analyze(path, STREAM_METRICS, chunk_size=chunk, stream_continuity=False)
        aligned = _analyze(path, STREAM_METRICS, chunk_size=chunk, stream_continuity=True)
        for label, res in (("independent_chunks", legacy), ("aligned_overlap", aligned)):
            for metric in STREAM_METRICS:
                ref = np.asarray(full["metrics"][metric]["series"], dtype=np.float64)
                got = np.asarray(res["metrics"][metric]["series"], dtype=np.float64)
                frame_rows.append(
                    {
                        "chunk_seconds": chunk_s,
                        "strategy": label,
                        "metric": metric,
                        "full_file_frames": int(ref.size),
                        "streaming_frames": int(got.size),
                        "frames_missing": int(ref.size - got.size),
                        "frames_missing_percent": 100.0 * (ref.size - got.size) / max(1, ref.size),
                        "series_identical": bool(ref.shape == got.shape and np.array_equal(ref, got)),
                        "max_abs_frame_difference": (
                            float(np.max(np.abs(ref - got))) if ref.shape == got.shape else float("nan")
                        ),
                    }
                )
                full_mean = float(full["metrics"][metric]["summary"]["mean"])
                stream_mean = float(res["metrics"][metric]["summary"]["mean"])
                diff = abs(stream_mean - full_mean)
                agreement_rows.append(
                    {
                        "chunk_seconds": chunk_s,
                        "strategy": label,
                        "metric": metric,
                        "units": "dB" if metric in DB_VALUED else str(full["metrics"][metric].get("units", "")),
                        "full_mean": full_mean,
                        "streaming_mean": stream_mean,
                        "absolute_difference": diff,
                        # Percentages are only reported for ratio-scale metrics.
                        "relative_difference_percent": (
                            "" if metric in DB_VALUED else 100.0 * diff / max(abs(full_mean), 1e-12)
                        ),
                    }
                )
    return frame_rows, agreement_rows


# --------------------------------------------------------------------------- #
# 7. memory scaling
# --------------------------------------------------------------------------- #


def _worker(audio_path: Path, mode: str, duration_s: float) -> int:
    started = time.perf_counter()
    result = analyze(
        AnalysisConfig(
            input_path=audio_path,
            output_dir=RESULTS_DIR,
            frame_size=FRAME_SIZE,
            hop_size=HOP_SIZE,
            chunk_size=SR if mode == "streaming" else None,
            metrics=list(STREAM_METRICS),
            seed=SEED,
            summary_only=mode == "streaming",
            verbosity=0,
        )
    )
    elapsed_s = time.perf_counter() - started
    payload = {
        "mode": mode,
        "duration_s": duration_s,
        "elapsed_s": elapsed_s,
        "max_rss_mib": _rss_mib(),
        "metric_summaries": {name: result["metrics"][name]["summary"] for name in STREAM_METRICS},
        "series_points": {name: len(result["metrics"][name]["series"]) for name in STREAM_METRICS},
        "processed_chunks": int(result["metadata"]["analysis_strategy"].get("processed_chunks", 1)),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _run_worker(audio_path: Path, mode: str, duration_s: float) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(audio_path), mode, str(duration_s)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": str(SEED)},
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _baseline_rss() -> float:
    """Peak RSS of an import-only process, i.e. the fixed interpreter floor."""
    code = (
        "import resource,sys,json;import numpy,scipy,soundfile;import esl.core;"
        "raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
        "print(json.dumps({'rss': raw/(1024*1024) if sys.platform=='darwin' else raw/1024}))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return float(json.loads(out.stdout.strip().splitlines()[-1])["rss"])


def streaming_benchmark(temp_dir: Path, durations: tuple[int, ...], repeats: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for duration_s in durations:
        path = temp_dir / f"long_stereo_{duration_s}s.wav"
        _synthesize_long_stereo(path, duration_s)
        for mode in ("full", "streaming"):
            for repeat in range(repeats):
                payload = _run_worker(path, mode, float(duration_s))
                rows.append(
                    {
                        "duration_s": duration_s,
                        "mode": mode,
                        "repeat": repeat,
                        "elapsed_s": float(payload["elapsed_s"]),
                        "max_rss_mib": float(payload["max_rss_mib"]),
                        "processed_chunks": int(payload["processed_chunks"]),
                        "stored_series_points": int(sum(payload["series_points"].values())),
                    }
                )
        path.unlink(missing_ok=True)
    return rows


def memory_budget(benchmark_rows: list[dict[str, Any]], baseline_mib: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Explain the measured full-file slope from the arrays the pipeline holds."""
    # Peak RSS is not linear from zero: below some duration the interpreter and
    # library allocation floor dominates completely and the measurement carries
    # no information about the analysis at all.  Fitting through those points
    # biases the slope toward zero, so the floor is measured first and the line
    # is fitted only where the process has actually grown past it.
    floor = min(float(r["max_rss_mib"]) for r in benchmark_rows)
    fits: dict[str, Any] = {}
    for mode in ("full", "streaming"):
        subset = [r for r in benchmark_rows if r["mode"] == mode]
        above = [r for r in subset if float(r["max_rss_mib"]) > floor * 1.05]
        if len(above) >= 2:
            slope, intercept, r2 = _linfit(
                [float(r["duration_s"]) for r in above], [float(r["max_rss_mib"]) for r in above]
            )
            floor_bound_below_s = max(
                (float(r["duration_s"]) for r in subset if float(r["max_rss_mib"]) <= floor * 1.05),
                default=0.0,
            )
        else:
            # Never left the floor over the measured range.
            slope, intercept, r2 = 0.0, floor, float("nan")
            floor_bound_below_s = max(float(r["duration_s"]) for r in subset)
        fits[mode] = {
            "slope_mib_per_s": slope,
            "intercept_mib": intercept,
            "r2": r2,
            "floor_mib": floor,
            "floor_dominates_below_s": floor_bound_below_s,
            "points_used_for_fit": len(above),
        }

    def _predict(mode: str, seconds: float) -> float:
        f = fits[mode]
        return max(float(f["floor_mib"]), float(f["intercept_mib"]) + float(f["slope_mib_per_s"]) * seconds)

    frames_per_s = SR / HOP_SIZE
    bins = FRAME_SIZE // 2 + 1
    mib = 1024.0 * 1024.0
    components = [
        {
            "component": "decoded waveform (float64, 2 ch)",
            "predicted_mib_per_s": SR * 2 * 8 / mib,
            "note": "read_audio promotes the decoded float32 buffer to float64",
        },
        {
            "component": "complex STFT (complex128, 1 ch)",
            "predicted_mib_per_s": frames_per_s * bins * 16 / mib,
            "note": f"{frames_per_s:.4g} frames/s x {bins} bins",
        },
        {
            "component": "magnitude spectrogram (float64, 1 ch)",
            "predicted_mib_per_s": frames_per_s * bins * 8 / mib,
            "note": "cached alongside the complex transform",
        },
        {
            "component": "framed time-domain buffer (float32, 2 ch)",
            "predicted_mib_per_s": frames_per_s * FRAME_SIZE * 2 * 4 / mib,
            "note": "overlapping frames are materialised, not viewed",
        },
        {
            "component": "retained metric series (float64)",
            "predicted_mib_per_s": frames_per_s * len(STREAM_METRICS) * 8 / mib,
            "note": f"{len(STREAM_METRICS)} metrics at one value per frame",
        },
    ]
    predicted_total = sum(c["predicted_mib_per_s"] for c in components)
    spectral_subset = sum(
        c["predicted_mib_per_s"] for c in components if "STFT" in c["component"] or "magnitude" in c["component"] or "waveform" in c["component"]
    )
    components.append(
        {
            "component": "predicted total (all arrays live at once)",
            "predicted_mib_per_s": predicted_total,
            "note": "upper bound: peak RSS reflects the largest coexisting set, not the sum",
        }
    )
    components.append(
        {
            "component": "predicted subset (transform + magnitude + waveform)",
            "predicted_mib_per_s": spectral_subset,
            "note": "the arrays that plausibly coexist at the peak",
        }
    )
    components.append(
        {
            "component": "measured full-file slope",
            "predicted_mib_per_s": fits["full"]["slope_mib_per_s"],
            "note": f"least-squares fit, r2 = {fits['full']['r2']:.4f}",
        }
    )
    components.append(
        {
            "component": "raw stereo float32 data rate",
            "predicted_mib_per_s": SR * 2 * 4 / mib,
            "note": "for reference: what the audio itself costs",
        }
    )

    summary = {
        "baseline_import_only_mib": baseline_mib,
        "measured_floor_mib": floor,
        "full": fits["full"],
        "streaming": fits["streaming"],
        "predicted_full_slope_mib_per_s": predicted_total,
        "predicted_full_slope_subset_mib_per_s": spectral_subset,
        "extrapolated_full_1h_gib": _predict("full", 3600.0) / 1024.0,
        "extrapolated_streaming_1h_gib": _predict("streaming", 3600.0) / 1024.0,
        "extrapolated_full_24h_gib": _predict("full", 86400.0) / 1024.0,
        "full_slope_multiple_of_data_rate": fits["full"]["slope_mib_per_s"] / (SR * 2 * 4 / mib),
    }
    return components, summary


# --------------------------------------------------------------------------- #
# 8. the provenance experiment: what an undocumented level is worth
# --------------------------------------------------------------------------- #


def provenance_dispersion(temp_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Spread of a reported level across individually defensible choices.

    Every combination below is something a competent engineer might do without
    documenting it: a different frame length, a different weighting, a different
    time aggregation, a different channel policy.  The dispersion of the single
    number they would all call "the level" is what the provenance record exists
    to remove.
    """
    scene = temp_dir / "provenance_scene.wav"
    _synthesize_scene(scene, duration_s=30.0)

    # A single fixed calibration so the spread is not merely a calibration error.
    calibration = CalibrationProfile(dbfs_reference=-20.0, spl_reference_db=94.0, weighting="Z")

    channel_files = {"stereo_downmix": scene}
    data, _sr = sf.read(scene, always_2d=True)
    for idx, label in ((0, "channel_1_only"), (1, "channel_2_only")):
        single = temp_dir / f"provenance_{label}.wav"
        _write_float_wav(single, data[:, idx : idx + 1])
        channel_files[label] = single

    # Every entry here is a calibrated sound-pressure level in dB re 20 uPa.
    # Mixing in a dBFS metric would make the spread a units error rather than a
    # measurement of how much the analysis choices matter.
    aggregations = {
        "mean_of_frame_levels": "spl_z_db",
        "energy_average_leq": "leq_db",
        "background_l90": "l90_db",
        "maximum_lmax": "lmax_db",
    }
    weightings = {"A": "spl_a_db", "C": "spl_c_db", "Z": "spl_z_db"}
    frame_sizes = (512, 1024, 2048, 8192)

    rows: list[dict[str, Any]] = []
    for channel_label, path in channel_files.items():
        for frame_size in frame_sizes:
            metrics = sorted(set(list(aggregations.values()) + list(weightings.values())))
            result = _analyze(
                path,
                metrics,
                frame_size=frame_size,
                hop_size=max(1, frame_size // 4),
                calibration=calibration,
            )
            for agg_label, metric in aggregations.items():
                rows.append(
                    {
                        "channel_policy": channel_label,
                        "frame_size": frame_size,
                        "axis": "time_aggregation",
                        "choice": agg_label,
                        "metric": metric,
                        "reported_level_db": _metric_mean(result, metric),
                    }
                )
            for weight_label, metric in weightings.items():
                rows.append(
                    {
                        "channel_policy": channel_label,
                        "frame_size": frame_size,
                        "axis": "frequency_weighting",
                        "choice": weight_label,
                        "metric": metric,
                        "reported_level_db": _metric_mean(result, metric),
                    }
                )

    values = np.asarray([r["reported_level_db"] for r in rows], dtype=np.float64)
    values = values[np.isfinite(values)]

    # Per-axis contribution: hold everything else at the pipeline default and
    # vary one axis at a time.
    default_channel = "stereo_downmix"
    default_frame = FRAME_SIZE
    default_weighting = "Z"
    default_aggregation = "energy_average_leq"

    def _pick(channel: str, frame: int, axis: str, choice: str) -> float | None:
        for row in rows:
            if (
                row["channel_policy"] == channel
                and row["frame_size"] == frame
                and row["axis"] == axis
                and row["choice"] == choice
            ):
                return float(row["reported_level_db"])
        return None

    axis_rows: list[dict[str, Any]] = []

    weighting_vals = [
        v for w in weightings if (v := _pick(default_channel, default_frame, "frequency_weighting", w)) is not None
    ]
    aggregation_vals = [
        v for a in aggregations if (v := _pick(default_channel, default_frame, "time_aggregation", a)) is not None
    ]
    frame_vals = [
        v
        for f in frame_sizes
        if (v := _pick(default_channel, f, "time_aggregation", default_aggregation)) is not None
    ]
    channel_vals = [
        v
        for c in channel_files
        if (v := _pick(c, default_frame, "time_aggregation", default_aggregation)) is not None
    ]

    for label, vals in (
        ("Frequency weighting (A / C / Z)", weighting_vals),
        ("Time aggregation (mean / Leq / L90 / Lmax)", aggregation_vals),
        ("Frame length (512 - 8192)", frame_vals),
        ("Channel policy (downmix / ch1 / ch2)", channel_vals),
    ):
        arr = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
        if arr.size < 2:
            continue
        axis_rows.append(
            {
                "axis": label,
                "levels": int(arr.size),
                "min_db": float(np.min(arr)),
                "max_db": float(np.max(arr)),
                "range_db": float(np.max(arr) - np.min(arr)),
            }
        )
    axis_rows.sort(key=lambda r: r["range_db"], reverse=True)

    # The conservative figure: hold the time statistic fixed at Leq -- so nobody
    # can object that Lmax and L90 are simply different quantities -- and vary
    # only the choices that are supposed to be immaterial.
    conservative = np.asarray(
        [
            r["reported_level_db"]
            for r in rows
            if (r["axis"] == "frequency_weighting")
            or (r["axis"] == "time_aggregation" and r["choice"] == default_aggregation)
        ],
        dtype=np.float64,
    )
    conservative = conservative[np.isfinite(conservative)]

    summary = {
        "combinations": int(values.size),
        "min_db": float(np.min(values)),
        "max_db": float(np.max(values)),
        "total_range_db": float(np.max(values) - np.min(values)),
        "conservative_combinations": int(conservative.size),
        "conservative_range_db": float(np.max(conservative) - np.min(conservative)) if conservative.size else float("nan"),
        "std_db": float(np.std(values)),
        "p5_db": float(np.percentile(values, 5)),
        "p95_db": float(np.percentile(values, 95)),
        "interdecile_range_db": float(np.percentile(values, 95) - np.percentile(values, 5)),
        "largest_single_axis": axis_rows[0]["axis"] if axis_rows else None,
        "largest_single_axis_range_db": axis_rows[0]["range_db"] if axis_rows else None,
    }
    return rows, axis_rows, summary


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

COLUMN_WIDTH_IN = 3.25  # single JAES column; figures are never scaled in LaTeX
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"


def _save(fig, name: str) -> None:
    for suffix in ("eps", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{suffix}", dpi=400)


def _save_figures(
    dispersion_rows: list[dict[str, Any]],
    axis_rows: list[dict[str, Any]],
    event_example: dict[str, Any],
    event_rows: list[dict[str, Any]],
    benchmark_rows: list[dict[str, Any]],
    framing_rows: list[dict[str, Any]],
    memory_summary: dict[str, Any],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PAPER_DIR / "build" / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    # --- Fig. 1: provenance dispersion -----------------------------------
    values = np.asarray([r["reported_level_db"] for r in dispersion_rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    fig, (ax_hist, ax_axis) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH_IN, 3.6), gridspec_kw={"height_ratios": [1.0, 1.15]}
    )
    ax_hist.hist(values, bins=24, color=BLUE, edgecolor="black", linewidth=0.4)
    ax_hist.set(xlabel="Reported level (dB)", ylabel="Combinations")
    ax_hist.grid(True, axis="y", alpha=0.25)
    ax_hist.set_title(f"span {np.max(values) - np.min(values):.1f} dB over {values.size} choices", fontsize=8)

    labels = [r["axis"].split(" (")[0] for r in axis_rows]
    ranges = [r["range_db"] for r in axis_rows]
    ypos = np.arange(len(labels))
    ax_axis.barh(ypos, ranges, color=ORANGE, edgecolor="black", linewidth=0.4)
    ax_axis.set_yticks(ypos, labels)
    ax_axis.invert_yaxis()
    ax_axis.set_xlabel("Range of reported level (dB)")
    ax_axis.grid(True, axis="x", alpha=0.25)
    for y, value in zip(ypos, ranges, strict=True):
        label = f"{value:.1f}" if value >= 0.1 else f"{value:.3f}"
        ax_axis.text(value, y, f" {label}", va="center", fontsize=7)
    ax_axis.set_xlim(0, max(ranges) * 1.25 if ranges else 1.0)
    fig.tight_layout()
    _save(fig, "fig_provenance_dispersion")
    plt.close(fig)

    # --- Fig. 2: memory scaling ------------------------------------------
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    for mode, marker, style, color in (("full", "o", "-", BLUE), ("streaming", "s", "--", ORANGE)):
        subset = sorted((r for r in benchmark_rows if r["mode"] == mode), key=lambda r: r["duration_s"])
        xs = sorted({float(r["duration_s"]) for r in subset})
        ys = [float(np.mean([r["max_rss_mib"] for r in subset if float(r["duration_s"]) == x])) for x in xs]
        errs = [float(np.ptp([r["max_rss_mib"] for r in subset if float(r["duration_s"]) == x])) for x in xs]
        ax.errorbar(xs, ys, yerr=errs, marker=marker, linestyle=style, color=color, label=mode, capsize=2, linewidth=1.0)
    slope = memory_summary["full"]["slope_mib_per_s"]
    ax.set(xlabel="Audio duration (s)", ylabel="Peak resident memory (MiB)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.97,
        0.05,
        f"full: {slope:.2f} MiB/s\nstreaming: {memory_summary['streaming']['slope_mib_per_s']:.3f} MiB/s",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()
    _save(fig, "fig_memory_scaling")
    plt.close(fig)

    # --- Fig. 3: event localisation --------------------------------------
    fig, (ax_curve, ax_err) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH_IN, 3.4))
    if event_example:
        times = np.asarray(event_example["times_s"], dtype=np.float64)
        novelty = np.asarray(event_example["novelty"], dtype=np.float64)
        ax_curve.plot(times, novelty, color="black", linewidth=0.7)
        ax_curve.axvline(event_example["boundary_s"], color=BLUE, linestyle="--", linewidth=1.0, label="true boundary")
        ax_curve.axvline(event_example["detected_s"], color=ORANGE, linestyle=":", linewidth=1.2, label="detected peak")
        ax_curve.set(xlabel="Time (s)", ylabel="Novelty (a.u.)", xlim=(0, 12))
        ax_curve.legend(frameon=False, loc="upper right")
        ax_curve.grid(True, alpha=0.25)

    transitions = sorted({r["transition"] for r in event_rows})
    snrs = ["inf", 20.0, 10.0, 0.0]
    width = 0.8 / max(1, len(transitions))
    for i, transition in enumerate(transitions):
        xs = []
        ys = []
        for j, snr in enumerate(snrs):
            match = [r for r in event_rows if r["transition"] == transition and str(r["snr_db"]) == str(snr)]
            if not match:
                continue
            xs.append(j + i * width - 0.4 + width / 2)
            ys.append(float(match[0]["median_abs_error_ms"]))
        ax_err.bar(xs, ys, width=width, label=transition.replace("_", " "), edgecolor="black", linewidth=0.3)
    ax_err.set_xticks(range(len(snrs)), ["no noise", "20 dB", "10 dB", "0 dB"])
    ax_err.set(xlabel="Signal-to-noise ratio", ylabel="Median |error| (ms)")
    ax_err.set_yscale("log")
    ax_err.grid(True, axis="y", alpha=0.25)
    ax_err.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    _save(fig, "fig_event_localization")
    plt.close(fig)

    # --- Fig. 4: centroid error vs frame length --------------------------
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.3))
    for kind, marker, color in (("commensurate", "o", GREEN), ("off_bin", "s", ORANGE)):
        frames = sorted({int(r["frame_size"]) for r in framing_rows})
        ys = []
        for frame in frames:
            errs = [float(r["abs_error_hz"]) for r in framing_rows if int(r["frame_size"]) == frame and r["bin_alignment"] == kind]
            ys.append(float(np.max(errs)) if errs else float("nan"))
        ax.plot(frames, ys, marker=marker, color=color, label=kind.replace("_", "-"), linewidth=1.0)
    ax.set(xlabel="Frame length (samples)", ylabel="Max |centroid error| (Hz)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "fig_centroid_framing")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _tex_escape(text: str) -> str:
    """Escape the characters that would otherwise be markup in a LaTeX macro body."""
    for char, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(char, replacement)
    return text


class TeX(str):
    """A string that is already LaTeX and must not be escaped again."""


def _tex_sci(value: float, digits: int = 1) -> TeX:
    """Render a small quantity in LaTeX scientific notation, e.g. $8.7\\times10^{-7}$."""
    if value == 0 or not math.isfinite(value):
        return TeX(f"{value:g}")
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    if -3 <= exponent <= 3:
        return TeX(f"{value:.{max(0, digits - exponent)}f}".rstrip("0").rstrip("."))
    return TeX(rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$")


def _tex_macro(name: str, value: Any, fmt: str = "{:.4g}", *, group: bool = False) -> str:
    """Render one value as a LaTeX macro.

    ``group`` adds thin-space digit grouping; it is off by default because
    identifiers such as seeds, frame lengths and commit hashes must not be
    typeset as though they were quantities.
    """
    if isinstance(value, TeX):
        rendered = str(value)
    elif isinstance(value, (int, np.integer)):
        rendered = f"{int(value):,}".replace(",", r"\,") if group else str(int(value))
    elif isinstance(value, float):
        rendered = fmt.format(value)
    else:
        rendered = _tex_escape(str(value))
    return rf"\newcommand{{\{name}}}{{{rendered}}}"


def _fmt(value: str, spec: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "---"
    if not math.isfinite(number):
        return "---"
    text = format(number, spec)
    # A leading ASCII hyphen sets as a hyphen, not a minus sign.
    return f"${text}$" if text.startswith("-") else text


def write_result_tables() -> list[Path]:
    """Emit the data-bearing table bodies as LaTeX, straight from the results.

    The manuscript inputs these rather than restating the rows, for the same
    reason it inputs numbers.tex: a table typed by hand is a table that can
    disagree with the run it claims to report.
    """
    written: list[Path] = []

    def read(name: str) -> list[dict[str, str]]:
        path = RESULTS_DIR / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    labels = {
        "identical": "Identical",
        "second_channel_delayed_12_samples": "Channel 2 delayed 12 samples",
        "second_channel_minus_6_db": "Channel 2 at $-6.02$ dB",
        "partially_correlated": "Partially correlated",
        "independent_noise": "Independent noise",
    }
    lines = []
    for row in read("spatial_accuracy.csv"):
        valid = str(row.get("itd_valid", "")).lower() == "true"
        lines.append(
            " & ".join(
                [
                    labels.get(row["case"], row["case"].replace("_", " ")),
                    _fmt(row["expected_itd_us"], ".1f"),
                    _fmt(row["measured_itd_us"], ".1f"),
                    _fmt(row["measured_ild_db"], ".4f"),
                    _fmt(row["correlation_zerolag"], ".4f"),
                    _fmt(row["coherence_msc"], ".4f"),
                    _fmt(row["iacc"], ".4f"),
                    "yes" if valid else r"\textbf{no}",
                ]
            )
            + r"\\"
        )
    path = RESULTS_DIR / "table_spatial.tex"
    path.write_text("\n".join(lines) + "%\n", encoding="utf-8")
    written.append(path)

    lines = []
    for row in read("provenance_axes.csv"):
        label = (
            row["axis"]
            .replace("Leq", "$L_{\\mathrm{eq}}$")
            .replace("L90", "$L_{90}$")
            .replace("Lmax", "$L_{\\mathrm{max}}$")
            .replace("ch1", "ch.\\ 1")
            .replace("ch2", "ch.\\ 2")
            .replace(" - ", "--")
        )
        lines.append(f"{label} & {row['levels']} & {_fmt(row['range_db'], '.3g')}" + r"\\")
    path = RESULTS_DIR / "table_axes.tex"
    path.write_text("\n".join(lines) + "%\n", encoding="utf-8")
    written.append(path)

    components: list[str] = []
    totals: list[str] = []
    for row in read("memory_budget.csv"):
        name = row["component"].replace("_", " ")
        value = _fmt(row["predicted_mib_per_s"], ".3g")
        if name.startswith("measured"):
            totals.append(rf"\textbf{{Measured slope}} & \textbf{{{value}}}\\")
        elif name.startswith("predicted total"):
            totals.append(rf"All arrays live at once & {value}\\")
        elif name.startswith("predicted subset"):
            totals.append(rf"Transform, magnitude, waveform only & {value}\\")
        elif name.startswith("raw stereo"):
            totals.append(rf"Raw stereo float32 data rate & {value}\\")
        else:
            components.append(rf"{name} & {value}\\")
    # Split into two files: a \midrule is a \noalign command and cannot be
    # reached through \input from inside a tabular.
    for suffix, block in (("components", components), ("totals", totals)):
        path = RESULTS_DIR / f"table_budget_{suffix}.tex"
        path.write_text("\n".join(block) + "%\n", encoding="utf-8")
        written.append(path)
    return written


def _refresh_figures() -> None:
    """Redraw the figures from stored results, without recomputing anything."""
    summary = json.loads((RESULTS_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))

    def read(name: str) -> list[dict[str, Any]]:
        path = RESULTS_DIR / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            out = []
            for row in csv.DictReader(handle):
                converted: dict[str, Any] = {}
                for key, value in row.items():
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        converted[key] = value
                out.append(converted)
            return out

    example_path = RESULTS_DIR / "event_example.json"
    example = json.loads(example_path.read_text(encoding="utf-8")) if example_path.exists() else {}
    _save_figures(
        read("provenance_dispersion.csv"),
        read("provenance_axes.csv"),
        example,
        read("event_localization.csv"),
        read("streaming_benchmark.csv"),
        read("centroid_framing.csv"),
        summary["memory"],
    )


def write_numbers_tex() -> Path:
    """Emit every figure quoted in the manuscript as a LaTeX macro.

    The manuscript inputs this file rather than restating values, so a number in
    the paper cannot drift from the run that produced it.  This is the same
    argument the paper makes about measurement provenance, applied to the paper.
    """
    summary = json.loads((RESULTS_DIR / "evaluation_summary.json").read_text(encoding="utf-8"))

    def rows(name: str) -> list[dict[str, str]]:
        path = RESULTS_DIR / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    disp = summary["provenance_dispersion"]
    mem = summary["memory"]
    val = summary["validity"]
    stream = summary["streaming"]
    ana = summary["analytic_accuracy"]
    decay = summary["decay_accuracy"]
    cat = summary["metric_catalog"]
    cal = summary["calibration"]

    bench = rows("streaming_benchmark.csv")
    durations = sorted({int(float(r["duration_s"])) for r in bench}) if bench else []
    if bench:
        # Recompute the memory model from the stored benchmark so that this file
        # always reflects the current analysis of the recorded runs.
        budget_rows, mem = memory_budget(
            [{"duration_s": float(r["duration_s"]), "mode": r["mode"], "max_rss_mib": float(r["max_rss_mib"])} for r in bench],
            float(summary["memory"].get("baseline_import_only_mib", float("nan"))),
        )
        _write_csv(RESULTS_DIR / "memory_budget.csv", budget_rows)
        summary["memory"] = mem
        (RESULTS_DIR / "evaluation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
    frame_rows = rows("streaming_frames.csv")
    legacy_missing = [float(r["frames_missing_percent"]) for r in frame_rows if r["strategy"] == "independent_chunks"]
    event_rows = rows("event_localization.csv")

    def event_stat(transition: str, snr: str, field: str) -> float:
        for r in event_rows:
            if r["transition"] == transition and str(r["snr_db"]) == snr:
                return float(r[field])
        return float("nan")

    lines = [
        "% Generated by paper/evaluation.py -- do not edit by hand.",
        _tex_macro("eslCommit", str(summary["repository_commit"])[:12]),
        _tex_macro("eslCommitFull", str(summary["repository_commit"])),
        _tex_macro("eslSeed", str(summary["seed"])),
        _tex_macro("eslPython", summary["software"]["python"]),
        _tex_macro("eslNumpy", summary["software"]["numpy"]),
        _tex_macro("eslSoundfile", summary["software"]["soundfile"]),
        _tex_macro("eslPlatform", summary["software"]["platform"]),
        _tex_macro("eslMetricTotal", int(cat["total"])),
        _tex_macro("eslMetricStreaming", int(cat["streaming_capable"])),
        # provenance dispersion
        _tex_macro("dispCombos", int(disp["combinations"])),
        _tex_macro("dispRange", float(disp["total_range_db"]), "{:.1f}"),
        _tex_macro("dispMin", float(disp["min_db"]), "{:.1f}"),
        _tex_macro("dispMax", float(disp["max_db"]), "{:.1f}"),
        _tex_macro("dispStd", float(disp["std_db"]), "{:.1f}"),
        _tex_macro("dispConservativeCombos", int(disp["conservative_combinations"])),
        _tex_macro("dispConservativeRange", float(disp["conservative_range_db"]), "{:.1f}"),
        _tex_macro("dispTopAxis", str(disp["largest_single_axis"]).split(" (")[0]),
        _tex_macro("dispTopAxisRange", float(disp["largest_single_axis_range_db"]), "{:.1f}"),
        # analytic
        _tex_macro("anaCases", int(ana["cases"])),
        _tex_macro("anaFrame", str(ana["frame_size"])),
        _tex_macro("anaRmsErr", _tex_sci(float(ana["max_rms_error_db"]))),
        _tex_macro("anaPeakErr", _tex_sci(float(ana["max_peak_error_db"]))),
        _tex_macro("anaCrestErr", _tex_sci(float(ana["max_crest_error_db"]))),
        _tex_macro("anaCentroidOn", float(ana["max_centroid_error_hz_commensurate"]), "{:.3g}"),
        _tex_macro("anaCentroidOff", float(ana["max_centroid_error_hz_off_bin"]), "{:.3g}"),
        _tex_macro(
            "anaCentroidRatio",
            float(ana["max_centroid_error_hz_off_bin"]) / max(float(ana["max_centroid_error_hz_commensurate"]), 1e-12),
            "{:.0f}",
        ),
        # decay
        _tex_macro("decayCases", int(decay["cases"])),
        _tex_macro("decayCleanErr", _tex_sci(float(decay["noiseless_single_slope_max_t30_error_s"]))),
        _tex_macro("decayWorstPct", float(decay["worst_noisy_single_slope_t30_error_percent"]), "{:.0f}"),
        _tex_macro("decayDegraded", int(decay["degraded_cases"])),
        _tex_macro("decayDegradedFlagged", int(decay["degraded_cases_flagged_invalid"])),
        # validity
        _tex_macro("valCases", int(val["cases"])),
        _tex_macro("valExpected", int(val["expected_unavailable"])),
        _tex_macro("valCorrect", int(val["correctly_unavailable"])),
        _tex_macro("valMissed", int(val["missed"])),
        _tex_macro("valFalseAlarms", int(val["false_alarms"])),
        _tex_macro("valCorrectReported", int(val["correctly_reported"])),
        # streaming
        _tex_macro("streamChecked", int(stream["aligned_metrics_checked"])),
        _tex_macro("streamIdentical", "yes" if stream["aligned_series_identical"] else "no"),
        _tex_macro("streamLegacyMissing", float(stream["legacy_max_frames_missing_percent"]), "{:.1f}"),
        _tex_macro("streamLegacyMissingMin", float(min(legacy_missing)) if legacy_missing else 0.0, "{:.1f}"),
        _tex_macro("streamLegacyWorstRel", float(stream["legacy_worst_relative_difference_percent"]), "{:.2f}"),
        # memory
        _tex_macro("memBaseline", float(mem["baseline_import_only_mib"]), "{:.0f}"),
        _tex_macro("memFloor", float(mem["measured_floor_mib"]), "{:.0f}"),
        _tex_macro("memFloorBelow", float(mem["full"]["floor_dominates_below_s"]), "{:.0f}"),
        _tex_macro("memFitPoints", int(mem["full"]["points_used_for_fit"])),
        _tex_macro("memPredictedSubset", float(mem["predicted_full_slope_subset_mib_per_s"]), "{:.2f}"),
        _tex_macro("memFullSlope", float(mem["full"]["slope_mib_per_s"]), "{:.2f}"),
        _tex_macro("memFullIntercept", float(mem["full"]["intercept_mib"]), "{:.0f}"),
        _tex_macro("memFullRsq", float(mem["full"]["r2"]), "{:.4f}"),
        _tex_macro("memStreamSlope", float(mem["streaming"]["slope_mib_per_s"]), "{:.3f}"),
        _tex_macro("memStreamIntercept", float(mem["streaming"]["intercept_mib"]), "{:.0f}"),
        _tex_macro("memPredicted", float(mem["predicted_full_slope_mib_per_s"]), "{:.2f}"),
        _tex_macro("memDataRateMultiple", float(mem["full_slope_multiple_of_data_rate"]), "{:.0f}"),
        _tex_macro("memFullHour", float(mem["extrapolated_full_1h_gib"]), "{:.1f}"),
        _tex_macro("memStreamHour", float(mem["extrapolated_streaming_1h_gib"]), "{:.2f}"),
        _tex_macro("memFullDay", float(mem["extrapolated_full_24h_gib"]), "{:.0f}"),
        _tex_macro("memMaxDuration", int(max(durations)) if durations else 0),
        _tex_macro("memDurations", ", ".join(str(d) for d in durations)),
        # calibration
        _tex_macro("calFixtures", int(cal["fixture_count"] or 0)),
        _tex_macro("calTolerance", float(cal["tolerance_db"]), "{:.3f}"),
        _tex_macro("calAchieved", _tex_sci(float(cal.get("max_achieved_abs_error_db", float("nan"))))),
        # event localisation
        _tex_macro("evtCleanMedian", event_stat("tone_to_tone", "inf", "median_abs_error_ms"), "{:.1f}"),
        _tex_macro("evtCleanPninetyfive", event_stat("tone_to_tone", "inf", "p95_abs_error_ms"), "{:.1f}"),
        _tex_macro("evtNoisyMedian", event_stat("tone_to_tone", "0.0", "median_abs_error_ms"), "{:.1f}"),
        _tex_macro("evtNoisyPninetyfive", event_stat("tone_to_tone", "0.0", "p95_abs_error_ms") / 1000.0, "{:.1f}"),
        _tex_macro("evtNoiseRateStep", 100.0 * event_stat("level_step", "0.0", "detection_rate"), "{:.0f}"),
        _tex_macro("evtNoiseMedianStep", event_stat("level_step", "0.0", "median_abs_error_ms") / 1000.0, "{:.1f}"),
        _tex_macro("evtCleanMedianStep", event_stat("level_step", "inf", "median_abs_error_ms"), "{:.1f}"),
        _tex_macro("evtCleanMedianNoise", event_stat("tone_to_noise", "inf", "median_abs_error_ms"), "{:.1f}"),
        _tex_macro("evtCleanRate", 100.0 * event_stat("tone_to_tone", "inf", "detection_rate"), "{:.0f}"),
        _tex_macro("evtNoisyRate", 100.0 * event_stat("tone_to_tone", "0.0", "detection_rate"), "{:.0f}"),
        _tex_macro("evtHopMs", event_stat("tone_to_tone", "inf", "hop_duration_ms"), "{:.0f}"),
        _tex_macro("evtTrials", int(event_stat("tone_to_tone", "inf", "trials")) if event_rows else 0),
    ]
    out = RESULTS_DIR / "numbers.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=3, metavar=("PATH", "MODE", "DURATION"))
    parser.add_argument("--quick", action="store_true", help="short duration sweep for smoke testing")
    parser.add_argument(
        "--numbers-only",
        action="store_true",
        help="regenerate results/numbers.tex from an existing evaluation_summary.json",
    )
    args, _unknown = parser.parse_known_args()
    if args.worker:
        return _worker(Path(args.worker[0]), args.worker[1], float(args.worker[2]))
    if args.numbers_only:
        print(write_numbers_tex())
        for path in write_result_tables():
            print(path)
        _refresh_figures()
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    durations = (15, 30, 60) if args.quick else (15, 30, 60, 120, 240, 480)
    repeats = 1 if args.quick else 3

    with tempfile.TemporaryDirectory(prefix="esl_jaes_eval_") as tmp:
        temp_dir = Path(tmp)
        analytic_rows, framing_rows = analytic_accuracy(temp_dir)
        decay_rows = decay_accuracy(temp_dir)
        spatial_rows = spatial_accuracy(temp_dir)
        validity_rows = validity_matrix(temp_dir)
        event_rows, event_example = event_localization(temp_dir)
        frame_rows, agreement_rows = streaming_agreement(temp_dir)
        dispersion_rows, axis_rows, dispersion_summary = provenance_dispersion(temp_dir)
        baseline_mib = _baseline_rss()
        benchmark_rows = streaming_benchmark(temp_dir, durations, repeats)

    budget_rows, memory_summary = memory_budget(benchmark_rows, baseline_mib)

    _, calibration_report, calibration_ok = run_calibration_verify(
        CalibrationVerifyConfig(
            fixture="all",
            output_path=RESULTS_DIR / "calibration_suite.json",
            # Tightened from 0.25 dB: the previous tolerance was six orders of
            # magnitude looser than the achieved error, so it could not fail.
            max_abs_error_db=0.001,
        )
    )

    _write_csv(RESULTS_DIR / "analytic_accuracy.csv", analytic_rows)
    _write_csv(RESULTS_DIR / "centroid_framing.csv", framing_rows)
    _write_csv(RESULTS_DIR / "decay_accuracy.csv", decay_rows)
    _write_csv(RESULTS_DIR / "spatial_accuracy.csv", spatial_rows)
    _write_csv(RESULTS_DIR / "validity_matrix.csv", validity_rows)
    _write_csv(RESULTS_DIR / "event_localization.csv", event_rows)
    _write_csv(RESULTS_DIR / "streaming_frames.csv", frame_rows)
    _write_csv(RESULTS_DIR / "streaming_agreement.csv", agreement_rows)
    _write_csv(RESULTS_DIR / "streaming_benchmark.csv", benchmark_rows)
    _write_csv(RESULTS_DIR / "memory_budget.csv", budget_rows)
    _write_csv(RESULTS_DIR / "provenance_dispersion.csv", dispersion_rows)
    _write_csv(RESULTS_DIR / "provenance_axes.csv", axis_rows)
    (RESULTS_DIR / "event_example.json").write_text(json.dumps(event_example), encoding="utf-8")

    _save_figures(
        dispersion_rows,
        axis_rows,
        event_example,
        event_rows,
        benchmark_rows,
        framing_rows,
        memory_summary,
    )

    registry = create_registry(with_external=False)
    specs = registry.specs()
    categories: dict[str, int] = {}
    for spec in specs:
        categories[spec.category] = categories.get(spec.category, 0) + 1

    commit = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, cwd=PAPER_DIR
        ).stdout.strip()
    except Exception:
        pass

    aligned = [r for r in frame_rows if r["strategy"] == "aligned_overlap"]
    legacy = [r for r in frame_rows if r["strategy"] == "independent_chunks"]

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_elapsed_s": time.perf_counter() - started,
        "repository_commit": commit,
        "seed": SEED,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "soundfile": sf.__version__,
            "platform": platform.platform(),
        },
        "metric_catalog": {
            "total": len(specs),
            "streaming_capable": sum(1 for s in specs if s.streaming_capable),
            "categories": categories,
            "deprecated_aliases": registry.aliases(),
        },
        "analytic_accuracy": {
            "cases": len(analytic_rows),
            "frame_size": FRAME_SIZE,
            "max_rms_error_db": max(r["rms_abs_error_db"] for r in analytic_rows),
            "max_peak_error_db": max(r["peak_abs_error_db"] for r in analytic_rows),
            "max_crest_error_db": max(r["crest_abs_error_db"] for r in analytic_rows),
            "max_centroid_error_hz_commensurate": max(
                r["centroid_abs_error_hz"] for r in analytic_rows if r["bin_alignment"] == "commensurate"
            ),
            "max_centroid_error_hz_off_bin": max(
                r["centroid_abs_error_hz"] for r in analytic_rows if r["bin_alignment"] == "off_bin"
            ),
        },
        "decay_accuracy": {
            "cases": len(decay_rows),
            "noiseless_single_slope_max_t30_error_s": max(
                (r["t30_abs_error_s"] for r in decay_rows if r["decay_shape"] == "single" and r["noise_floor_db"] == "none"),
                default=float("nan"),
            ),
            "worst_noisy_single_slope_t30_error_percent": max(
                (
                    r["t30_rel_error_percent"]
                    for r in decay_rows
                    if r["decay_shape"] == "single" and r["noise_floor_db"] != "none"
                ),
                default=float("nan"),
            ),
            "degraded_cases": sum(
                1
                for r in decay_rows
                if r["decay_shape"] == "single"
                and r["t30_rel_error_percent"] == r["t30_rel_error_percent"]
                and r["t30_rel_error_percent"] > 5.0
            ),
            "degraded_cases_flagged_invalid": sum(
                1
                for r in decay_rows
                if r["decay_shape"] == "single"
                and r["t30_rel_error_percent"] == r["t30_rel_error_percent"]
                and r["t30_rel_error_percent"] > 5.0
                and not r["t30_valid"]
            ),
        },
        "spatial": spatial_rows,
        "validity": {
            "cases": len(validity_rows),
            "expected_unavailable": sum(1 for r in validity_rows if r["expected_unavailable"]),
            "correctly_unavailable": sum(1 for r in validity_rows if r["verdict"] == "correctly_unavailable"),
            "missed": sum(1 for r in validity_rows if r["verdict"] == "missed"),
            "false_alarms": sum(1 for r in validity_rows if r["verdict"] == "false_alarm"),
            "correctly_reported": sum(1 for r in validity_rows if r["verdict"] == "correctly_reported"),
            "missed_cases": [f"{r['input']}/{r['metric']}" for r in validity_rows if r["verdict"] == "missed"],
            "false_alarm_cases": [
                f"{r['input']}/{r['metric']}" for r in validity_rows if r["verdict"] == "false_alarm"
            ],
        },
        "event_localization": event_rows,
        "streaming": {
            "aligned_series_identical": all(r["series_identical"] for r in aligned),
            "aligned_metrics_checked": len(aligned),
            "legacy_max_frames_missing_percent": max((r["frames_missing_percent"] for r in legacy), default=0.0),
            "legacy_worst_relative_difference_percent": max(
                (
                    float(r["relative_difference_percent"])
                    for r in agreement_rows
                    if r["strategy"] == "independent_chunks" and r["relative_difference_percent"] != ""
                ),
                default=0.0,
            ),
        },
        "memory": memory_summary,
        "provenance_dispersion": dispersion_summary,
        "provenance_axes": axis_rows,
        "calibration": {
            "all_passed": bool(calibration_ok),
            "fixture_count": int(calibration_report.get("fixture_count", 0))
            if isinstance(calibration_report, dict)
            else None,
            "max_achieved_abs_error_db": max(
                (float(f.get("abs_error_db", 0.0)) for f in calibration_report.get("fixture_results", [])),
                default=float("nan"),
            )
            if isinstance(calibration_report, dict)
            else None,
            "tolerance_db": 0.001,
        },
    }
    (RESULTS_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    write_numbers_tex()
    write_result_tables()
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
