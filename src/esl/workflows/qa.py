"""Streaming recorder QA with explicit thresholds and source provenance."""

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

from esl.version import __version__

MAX_BLOCK_SAMPLES = 4_194_304


def _runs(mask: np.ndarray, previous: int) -> tuple[int, int]:
    """Return longest and trailing true run, including the preceding block tail."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    starts, ends = edges[::2], edges[1::2]
    if not len(starts):
        return 0, 0
    lengths = ends - starts
    if starts[0] == 0:
        lengths[0] += previous
    trailing = int(lengths[-1]) if ends[-1] == len(mask) else 0
    return int(lengths.max()), trailing


def recorder_qa(
    audio_path: str | Path,
    out_dir: str | Path,
    *,
    block_frames: int = 65_536,
    clip_threshold: float = 0.999,
    dropout_threshold: float = 1e-6,
    min_dropout_seconds: float = 0.1,
    dc_threshold: float = 0.01,
) -> dict[str, Any]:
    """Write per-channel CSV and a versioned QA report using constant-time-span memory."""
    if not isinstance(block_frames, int) or block_frames < 1:
        raise ValueError("block_frames must be a positive integer")
    thresholds = [clip_threshold, dropout_threshold, min_dropout_seconds, dc_threshold]
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("Thresholds must be finite")
    if not 0 < clip_threshold <= 1 or not 0 <= dropout_threshold < clip_threshold:
        raise ValueError("Require 0 <= dropout_threshold < clip_threshold <= 1")
    if min_dropout_seconds <= 0 or dc_threshold < 0:
        raise ValueError("min_dropout_seconds must be positive; dc_threshold nonnegative")
    source, destination = Path(audio_path).resolve(), Path(out_dir).resolve()
    if source in {destination / "recorder_qa.json", destination / "recorder_qa.csv"}:
        raise ValueError("Output must not overwrite source")
    stat = source.stat()
    digest = hashlib.sha256()
    with sf.SoundFile(source) as recording:
        channels, rate, frames = recording.channels, recording.samplerate, recording.frames
        if block_frames * channels > MAX_BLOCK_SAMPLES:
            raise ValueError("Block exceeds bounded sample budget; reduce block_frames")
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
        counts = np.zeros((4, channels), dtype=np.int64)
        sums = np.zeros(channels)
        peak = np.zeros(channels)
        longest = np.zeros(channels, dtype=np.int64)
        trailing = np.zeros(channels, dtype=np.int64)
        hashes = [hashlib.sha256() for _ in range(channels)]
        total = 0
        for block in recording.blocks(blocksize=block_frames, dtype="float64", always_2d=True):
            digest.update(block.astype("<f8", copy=False).tobytes())
            finite = np.isfinite(block)
            counts[0] += finite.sum(axis=0)
            counts[1] += (~finite).sum(axis=0)
            counts[2] += (finite & (np.abs(block) >= clip_threshold)).sum(axis=0)
            silence = finite & (np.abs(block) <= dropout_threshold)
            counts[3] += silence.sum(axis=0)
            safe = np.where(finite, block, 0)
            sums += safe.sum(axis=0)
            peak = np.maximum(peak, np.max(np.abs(safe), axis=0))
            for channel in range(channels):
                run, tail = _runs(silence[:, channel], int(trailing[channel]))
                longest[channel] = max(longest[channel], run)
                trailing[channel] = tail
                canonical = block[:, channel].copy()
                canonical[canonical == 0] = 0.0
                hashes[channel].update(canonical.astype("<f8", copy=False).tobytes())
            total += len(block)
    if total != frames:
        raise ValueError("Decoded frame count does not match source header")
    if source.stat().st_mtime_ns != stat.st_mtime_ns or source.stat().st_size != stat.st_size:
        raise ValueError("Source changed during analysis")
    provenance["decoded_sha256"] = digest.hexdigest()
    rows = []
    for channel in range(channels):
        finite_count = int(counts[0, channel])
        mean = float(sums[channel] / finite_count) if finite_count else None
        rows.append(
            {
                "channel": channel,
                "finite_samples": finite_count,
                "nonfinite_samples": int(counts[1, channel]),
                "clipped_samples": int(counts[2, channel]),
                "clipped_fraction": float(counts[2, channel] / frames) if frames else None,
                "near_zero_samples": int(counts[3, channel]),
                "longest_dropout_frames": int(longest[channel]),
                "longest_dropout_s": float(longest[channel] / rate),
                "dropout_flag": bool(longest[channel] >= math.ceil(min_dropout_seconds * rate)),
                "dc_mean_fs": mean,
                "dc_flag": bool(mean is not None and abs(mean) > dc_threshold),
                "peak_fs": float(peak[channel]) if finite_count else None,
            }
        )
    groups: dict[str, list[int]] = {}
    for channel, channel_hash in enumerate(hashes):
        if frames and not counts[1, channel]:
            groups.setdefault(channel_hash.hexdigest(), []).append(channel)
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    issues = (
        any(
            row["clipped_samples"]
            or row["nonfinite_samples"]
            or row["dropout_flag"]
            or row["dc_flag"]
            for row in rows
        )
        or bool(duplicate_groups)
        or not frames
    )
    report = {
        "schema_version": "1.0.0",
        "artifact_kind": "recorder_qa",
        "esl_version": __version__,
        "provenance": provenance,
        "config": {
            "block_frames": block_frames,
            "clip_threshold": clip_threshold,
            "dropout_threshold": dropout_threshold,
            "min_dropout_seconds": min_dropout_seconds,
            "dc_threshold": dc_threshold,
        },
        "status": "review" if issues else "no_flags",
        "channels": rows,
        "duplicate_channel_groups": duplicate_groups,
        "clock_drift": {"status": "not_assessed", "reason": "No external timing reference"},
        "limitations": [
            "Flags are heuristics, not confirmed recorder faults; intentional silence may flag.",
            "Clipping is a sample-level threshold, not intersample/true-peak measurement.",
            "DC is the whole-recording finite-sample mean; transient offsets may cancel.",
            "Duplication means identical decoded samples with signed zero normalized.",
            "Silent channels may duplicate; scaled, delayed or near-duplicate channels do not.",
            "Non-finite samples are counted and excluded from DC/peak and duplicate grouping.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "recorder_qa.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (destination / "recorder_qa.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def _run(args: argparse.Namespace) -> int:
    report = recorder_qa(
        args.audio,
        args.out_dir,
        block_frames=args.block_frames,
        clip_threshold=args.clip_threshold,
        dropout_threshold=args.dropout_threshold,
        min_dropout_seconds=args.min_dropout_seconds,
        dc_threshold=args.dc_threshold,
    )
    return 2 if args.fail_on_flags and report["status"] == "review" else 0


def register(subparsers: Any) -> None:
    """Register the qa command; normal completion returns zero even with review flags."""
    parser = subparsers.add_parser("qa", help=__doc__)
    parser.add_argument("audio")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--block-frames", type=int, default=65_536)
    parser.add_argument("--clip-threshold", type=float, default=0.999)
    parser.add_argument("--dropout-threshold", type=float, default=1e-6)
    parser.add_argument("--min-dropout-seconds", type=float, default=0.1)
    parser.add_argument("--dc-threshold", type=float, default=0.01)
    parser.add_argument("--fail-on-flags", action="store_true")
    parser.set_defaults(func=_run)
