"""Bounded-window spatial diagnostics, without implicit microphone conventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from esl.version import __version__

MAX_WINDOW_SAMPLES = 4_194_304
MAX_WINDOWS = 100_000
FIELDS = [
    "start_s",
    "end_s",
    "frames",
    "ild_db",
    "itd_s",
    "correlation_peak",
    "energy_fs2",
    "direction_x",
    "direction_y",
    "direction_z",
    "azimuth_deg",
    "elevation_deg",
    "diffuseness",
]


def _spatial_window(
    samples: np.ndarray,
    sample_rate: int,
    pair: tuple[int, int],
    convention: str | None,
    max_itd_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = dict.fromkeys(FIELDS[3:])
    left, right = samples[:, pair[0]], samples[:, pair[1]]
    powers = np.mean(np.square(samples[:, pair]), axis=0)
    if np.all(powers > 0):
        row["ild_db"] = float(10 * np.log10(powers[0] / powers[1]))
    left = left - left.mean()
    right = right - right.mean()
    norm = float(np.linalg.norm(left) * np.linalg.norm(right))
    if norm > 0 and len(samples) > 1:
        values = correlate(right, left, mode="full", method="fft") / norm
        lags = correlation_lags(len(right), len(left), mode="full")
        selected = np.abs(lags) <= int(math.floor(max_itd_s * sample_rate))
        selected_values, selected_lags = values[selected], lags[selected]
        peak = int(np.argmax(selected_values))
        row["itd_s"] = float(selected_lags[peak] / sample_rate)
        row["correlation_peak"] = float(np.clip(selected_values[peak], -1, 1))
    if convention is None:
        return row
    if convention == "fuma":
        pressure = samples[:, 0] * np.sqrt(2)
        velocity = samples[:, 1:4]
    else:
        pressure = samples[:, 0]
        velocity = samples[:, [3, 1, 2]]
        if convention == "acn-n3d":
            velocity = velocity / np.sqrt(3)
    energy = float(np.mean(pressure**2 + np.sum(velocity**2, axis=1)))
    row["energy_fs2"] = energy
    if energy == 0:
        return row
    intensity = np.mean(pressure[:, None] * velocity, axis=0)
    magnitude = float(np.linalg.norm(intensity))
    row["diffuseness"] = float(np.clip(1 - 2 * magnitude / energy, 0, 1))
    if magnitude > energy * 1e-12:
        direction = intensity / magnitude
        row.update(
            zip(
                ("direction_x", "direction_y", "direction_z"),
                map(float, direction),
                strict=True,
            )
        )
        row["azimuth_deg"] = float(np.degrees(np.arctan2(direction[1], direction[0])))
        row["elevation_deg"] = float(np.degrees(np.arcsin(np.clip(direction[2], -1, 1))))
    return row


def spatial_timeline(
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    window_seconds: float = 1.0,
    pair: tuple[int, int] = (0, 1),
    ambisonics: str | None = None,
    max_itd_ms: float = 1.0,
    max_windows: int = 10_000,
    plot: bool = False,
) -> dict[str, Any]:
    """Export CSV, JSON and optional PNG; reject oversized jobs before reading audio."""
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    if not math.isfinite(max_itd_ms) or not 0 <= max_itd_ms <= 100:
        raise ValueError("max_itd_ms must be finite and between 0 and 100")
    if ambisonics not in {None, "acn-sn3d", "acn-n3d", "fuma"}:
        raise ValueError("Unsupported Ambisonics convention")
    if not 1 <= max_windows <= MAX_WINDOWS:
        raise ValueError(f"max_windows must be between 1 and {MAX_WINDOWS}")
    source, destination = Path(audio_path).resolve(), Path(out_dir).resolve()
    stat = source.stat()
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    with sf.SoundFile(source) as recording:
        rate, channels, frames = recording.samplerate, recording.channels, recording.frames
        if len(pair) != 2 or pair[0] == pair[1] or min(pair) < 0 or max(pair) >= channels:
            raise ValueError("pair must contain two distinct, valid zero-based channel indices")
        if ambisonics is not None and channels < 4:
            raise ValueError("Ambisonics requires at least four channels")
        window = max(1, round(window_seconds * rate))
        if window * channels > MAX_WINDOW_SAMPLES:
            raise ValueError("Window exceeds bounded sample budget; reduce window_seconds")
        if (frames + window - 1) // window > max_windows:
            raise ValueError("Timeline exceeds max_windows; increase window_seconds or max_windows")
        provenance = {
            "source": str(source),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sample_rate": rate,
            "channels": channels,
            "frames": frames,
            "format": recording.format,
            "subtype": recording.subtype,
            "hash_encoding": "decoded float64 little-endian, interleaved frames",
        }
        offset = 0
        while offset < frames:
            block = recording.read(window, dtype="float64", always_2d=True)
            if not len(block):
                raise ValueError("Unexpected end of recording")
            if not np.isfinite(block).all():
                raise ValueError("Non-finite audio samples; use qa to inspect the recording")
            digest.update(block.astype("<f8", copy=False).tobytes())
            row = {
                "start_s": offset / rate,
                "end_s": (offset + len(block)) / rate,
                "frames": len(block),
            }
            row.update(_spatial_window(block, rate, pair, ambisonics, max_itd_ms / 1000))
            rows.append(row)
            offset += len(block)
    if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
        raise ValueError("Source changed during analysis")
    provenance["decoded_sha256"] = digest.hexdigest()
    report = {
        "schema_version": "1.0.0",
        "artifact_kind": "spatial_timeline",
        "esl_version": __version__,
        "provenance": provenance,
        "config": {
            "window_seconds": window_seconds,
            "window_frames": window,
            "pair": list(pair),
            "ambisonics": ambisonics,
            "max_itd_ms": max_itd_ms,
            "max_windows": max_windows,
            "plot": plot,
        },
        "limitations": [
            "Integer-sample cross-correlation ITD; positive means second channel is delayed.",
            "ILD is first/second channel power ratio; silence yields null, not infinity.",
            "First-order intensity direction and diffuseness proxies, not calibrated sound power.",
            "No array geometry, propagation DOA, clock drift or perceptual localization inference.",
            "Higher-order Ambisonics channels are ignored for directional metrics.",
        ],
        "timeline": rows,
    }
    destination.mkdir(parents=True, exist_ok=True)
    outputs = [destination / "spatial_timeline.json", destination / "spatial_timeline.csv"]
    if source in outputs or (plot and source == destination / "spatial_timeline.png"):
        raise ValueError("Output must not overwrite source")
    outputs[0].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with outputs[1].open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        _plot(rows, destination / "spatial_timeline.png")
    return report


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 12), constrained_layout=True)
    FigureCanvasAgg(figure)
    times = [(row["start_s"] + row["end_s"]) / 2 for row in rows]
    fields = ["ild_db", "itd_s", "energy_fs2", "azimuth_deg", "elevation_deg", "diffuseness"]
    for axis, field in zip(figure.subplots(len(fields), 1, sharex=True), fields, strict=True):
        values = [np.nan if row[field] is None else row[field] for row in rows]
        axis.plot(times, values, linewidth=0.8)
        axis.set_ylabel(field)
        axis.grid(alpha=0.25)
    axis.set_xlabel("Time (s)")
    figure.savefig(path, dpi=140)
    figure.clear()


def _run(args: argparse.Namespace) -> int:
    spatial_timeline(
        args.audio,
        args.out_dir,
        window_seconds=args.window_seconds,
        pair=tuple(args.pair),
        ambisonics=args.ambisonics,
        max_itd_ms=args.max_itd_ms,
        max_windows=args.max_windows,
        plot=args.plot,
    )
    return 0


def register(subparsers: Any) -> None:
    """Register the spatial-timeline command on an argparse subparser collection."""
    parser = subparsers.add_parser("spatial-timeline", help=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--pair", nargs=2, type=int, default=[0, 1], metavar=("FIRST", "SECOND"))
    parser.add_argument("--ambisonics", choices=["acn-sn3d", "acn-n3d", "fuma"])
    parser.add_argument("--max-itd-ms", type=float, default=1.0)
    parser.add_argument("--max-windows", type=int, default=10_000)
    parser.add_argument("--plot", action="store_true")
    parser.set_defaults(func=_run)
