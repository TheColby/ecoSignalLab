"""Bounded bootstrap of metric-series means and explicit additive systematic offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from esl.version import __version__
from esl.workflows.monitor import _assignments, _load

MAX_OBSERVATIONS = 1_000_000
MAX_RESAMPLE_VALUES = 50_000_000


def bootstrap_mean(
    values: list[float] | np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 2000,
    confidence: float = 0.95,
    block_size: int = 1,
    offset_std: float = 0.0,
    sensitivity_step: float = 1.0,
) -> dict[str, Any]:
    """Return percentile intervals; offset is shared by all samples within a replicate."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not 2 <= data.size <= MAX_OBSERVATIONS:
        raise ValueError("Series must contain 2 to 1,000,000 one-dimensional observations")
    if not np.isfinite(data).all():
        raise ValueError("Series contains nonfinite observations; no silent filtering is performed")
    if not 100 <= resamples <= 100_000 or data.size * resamples > MAX_RESAMPLE_VALUES:
        raise ValueError("Require 100..100000 resamples and at most 50,000,000 resampled values")
    if not 0 < confidence < 1 or not 1 <= block_size <= data.size:
        raise ValueError("Require 0 < confidence < 1 and 1 <= block-size <= series length")
    if seed < 0 or any(not math.isfinite(v) or v < 0 for v in (offset_std, sensitivity_step)):
        raise ValueError("Seed must be nonnegative; offset standard deviation and step finite >= 0")
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    blocks = math.ceil(data.size / block_size)
    offsets = np.arange(block_size)
    for index in range(resamples):
        starts = rng.integers(0, data.size, size=blocks)
        indices = ((starts[:, None] + offsets) % data.size).ravel()[: data.size]
        estimates[index] = np.mean(data[indices])
    if not np.isfinite(estimates).all():
        raise ValueError("Numeric overflow computing bootstrap means")
    alpha = (1 - confidence) / 2
    sampling = np.quantile(estimates, [alpha, 1 - alpha]).tolist()
    propagated = estimates + rng.normal(0, offset_std, size=resamples)
    estimate = float(np.mean(data))
    warnings = ["Interval describes the mean of supplied observations, not individual events."]
    if block_size == 1:
        warnings.append(
            "IID resampling assumes independent observations; audio often violates this."
        )
    if data.size // block_size < 10:
        warnings.append("Fewer than ten blocks: interval coverage may be unreliable.")
    if np.ptp(data) == 0:
        warnings.append("Constant observations produce a degenerate sampling interval.")
    result = {
        "estimate": estimate,
        "observations": int(data.size),
        "sampling_interval": sampling,
        "propagated_interval": np.quantile(propagated, [alpha, 1 - alpha]).tolist(),
        "sampling_standard_error": float(np.std(estimates, ddof=1)),
        "propagated_standard_error": float(np.std(propagated, ddof=1)),
        "offset_std": offset_std,
        "sensitivity": {
            "model": "additive shared offset in metric units",
            "derivative": 1.0,
            "step": sensitivity_step,
            "minus": estimate - sensitivity_step,
            "plus": estimate + sensitivity_step,
        },
        "warnings": warnings,
    }
    # Fail before emitting a JSON result if extreme input caused arithmetic overflow.
    json.dumps(result, allow_nan=False)
    return result


def _audio_series(path: Path, window_seconds: float, channel: int) -> dict[str, Any]:
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window-seconds must be finite and positive")
    with sf.SoundFile(path) as audio:
        if not 0 <= channel < audio.channels:
            raise ValueError("channel must be a valid zero-based channel index")
        window = round(audio.samplerate * window_seconds)
        if window < 1 or window * audio.channels > 10_000_000:
            raise ValueError("Audio window must contain 1..10,000,000 channel samples")
        count = len(audio) // window
        if not 2 <= count <= MAX_OBSERVATIONS:
            raise ValueError("Audio must contain 2..1,000,000 complete windows")
        values = []
        for _ in range(count):
            samples = audio.read(window, dtype="float64", always_2d=True)[:, channel]
            if not np.isfinite(samples).all():
                raise ValueError("Audio contains nonfinite samples")
            power = float(np.mean(samples**2))
            values.append(10 * math.log10(max(power, 1e-24)))
        return {
            "metadata": {
                "sample_rate": audio.samplerate,
                "channels": audio.channels,
                "channel": channel,
                "window_samples": window,
                "discarded_tail_samples": len(audio) % window,
                "calibration": None,
                "silence_floor_dbfs": -240,
            },
            "metrics": {"rms_dbfs": {"units": "dBFS", "series": values}},
        }


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> int:
    if args.output and args.output.resolve() == args.input.resolve():
        raise ValueError("Output must not overwrite input")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    source_hash = _digest(args.input)
    report = (
        _load(args.input)
        if args.input.suffix.lower() == ".json"
        else _audio_series(args.input, args.window_seconds, args.channel)
    )
    names = sorted(set(args.metric or report["metrics"].keys()))
    if not names:
        raise ValueError("No metrics selected")
    deviations = _assignments(args.offset_std)
    steps = _assignments(args.sensitivity_step)
    if (deviations.keys() | steps.keys()) - set(names):
        raise ValueError("Offset and sensitivity options must name selected metrics")
    results = {}
    for name in names:
        payload = report["metrics"].get(name)
        if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
            raise ValueError(f"Metric {name!r} needs a numeric series, not summary-only data")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in payload["series"]):
            raise ValueError(f"Metric {name!r} series must contain numbers")
        # A name-derived seed makes results independent of metric selection/order.
        name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
        results[name] = bootstrap_mean(
            payload["series"],
            seed=args.seed + name_seed,
            resamples=args.resamples,
            confidence=args.confidence,
            block_size=args.block_size,
            offset_std=deviations.get(name, 0),
            sensitivity_step=steps.get(name, 1),
        )
        results[name]["units"] = payload.get("units")
    if _digest(args.input) != source_hash:
        raise ValueError("Input changed during uncertainty analysis")
    output = {
        "schema_version": "1.0.0",
        "artifact_kind": "esl.uncertainty.mean_intervals",
        "esl_version": __version__,
        "workflow": "uncertainty",
        "method": "circular_block_percentile_bootstrap_mean",
        "provenance": {
            "input": str(args.input.resolve()),
            "sha256": source_hash,
            "numpy_version": np.__version__,
            "source_metadata": report.get("metadata", {}),
        },
        "config": {
            "seed": args.seed,
            "resamples": args.resamples,
            "confidence": args.confidence,
            "block_size": args.block_size,
            "metrics": names,
            "offset_std": deviations,
            "sensitivity_step": steps,
            "window_seconds": args.window_seconds,
            "channel": args.channel,
        },
        "limitations": [
            "No automatic calibration, model-error or cross-metric propagation.",
            "Offsets are independent Gaussian shared errors, in metric units.",
            "Coverage depends on stationarity and suitable block length.",
            "Mean dB is not an energy-averaged equivalent sound level.",
        ],
        "metrics": results,
    }
    rendered = json.dumps(output, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``esl uncertainty INPUT``."""
    parser = subparsers.add_parser("uncertainty", help="Bootstrap metric-series means or audio RMS")
    parser.add_argument("input", type=Path)
    parser.add_argument("--metric", action="append", help="Repeat for selected metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--offset-std", action="append", default=[], metavar="METRIC=VALUE")
    parser.add_argument("--sensitivity-step", action="append", default=[], metavar="METRIC=VALUE")
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.set_defaults(func=_run)
