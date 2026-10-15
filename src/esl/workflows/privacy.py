"""Explicit interval redaction, preserving decoded multichannel samples outside spans."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

import soundfile as sf

from esl import __version__

MAX_BLOCK_SAMPLES = 1_048_576


def output_format(frames: int, channels: int) -> str:
    """Choose RF64 before a double-precision WAV would exceed RIFF's 32-bit size."""
    return "RF64" if frames * channels * 8 + 65536 >= 2**32 else "WAV"


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_spans(
    intervals: Iterable[dict[str, Any]],
    sample_rate: int,
    frames: int,
) -> list[tuple[int, int]]:
    """Conservatively cover [start,end) using exact decimal seconds, then merge spans."""
    if type(sample_rate) is not int or sample_rate <= 0 or type(frames) is not int or frames < 0:
        raise ValueError("sample_rate must be positive and frames nonnegative integers")
    spans = []
    for interval in intervals:
        if not isinstance(interval, dict) or set(interval) != {"start", "end"}:
            raise ValueError("each interval requires exactly start and end in seconds")
        try:
            start, end = (Decimal(str(interval[key])) for key in ("start", "end"))
        except InvalidOperation as exc:
            raise ValueError("interval endpoints must be finite decimal seconds") from exc
        if not start.is_finite() or not end.is_finite() or start < 0 or end <= start:
            raise ValueError("intervals require finite 0 <= start < end")
        start, end = Fraction(start), Fraction(end)
        if end > Fraction(frames, sample_rate):
            raise ValueError("interval extends beyond recording duration")
        left = math.floor(start * sample_rate)
        right = math.ceil(end * sample_rate)
        spans.append((left, min(right, frames)))
    merged: list[tuple[int, int]] = []
    for left, right in sorted(spans):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return merged


def redact_audio(
    source: str | Path,
    output: str | Path,
    audit: str | Path,
    intervals: list[dict[str, Any]],
    *,
    detector: Callable[[Path], Iterable[dict[str, Any]]] | None = None,
    detector_id: str | None = None,
    block_frames: int = 65536,
) -> dict[str, Any]:
    """Zero all channels in supplied/detected spans and publish audio plus audit exclusively.

    A detector is trusted caller-supplied code, not built-in speech recognition.
    Outputs use double-precision WAV to preserve decoded samples outside redactions.
    """
    source, output, audit = (Path(path).resolve() for path in (source, output, audit))
    if len({source, output, audit}) != 3 or output.exists() or audit.exists():
        raise ValueError("source, output, audit must be distinct; outputs must not exist")
    if type(block_frames) is not int or block_frames <= 0:
        raise ValueError("block_frames must be a positive integer")
    if output.suffix.lower() != ".wav":
        raise ValueError("redacted output must be .wav (lossless DOUBLE encoding)")
    source_hash = _hash(source)
    all_intervals = list(intervals)
    if detector is not None:
        all_intervals.extend(detector(source))
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        with sf.SoundFile(source) as reader:
            if block_frames * reader.channels > MAX_BLOCK_SAMPLES:
                raise ValueError(
                    f"block_frames * channels must not exceed {MAX_BLOCK_SAMPLES} samples"
                )
            encoding = output_format(reader.frames, reader.channels)
            spans = sample_spans(all_intervals, reader.samplerate, reader.frames)
            for destination in (output, audit):
                descriptor, filename = tempfile.mkstemp(prefix=".esl-", dir=destination.parent)
                os.close(descriptor)
                temporary.append(Path(filename))
            with sf.SoundFile(
                temporary[0],
                "w",
                samplerate=reader.samplerate,
                channels=reader.channels,
                format=encoding,
                subtype="DOUBLE",
            ) as writer:
                offset = 0
                span_index = 0
                for block in reader.blocks(blocksize=block_frames, dtype="float64", always_2d=True):
                    limit = offset + len(block)
                    while span_index < len(spans) and spans[span_index][1] <= offset:
                        span_index += 1
                    for left, right in spans[span_index:]:
                        if left >= limit:
                            break
                        block[max(left - offset, 0) : min(right - offset, len(block)), :] = 0
                    writer.write(block)
                    offset = limit
            if source_hash != _hash(source):
                raise ValueError("source changed during redaction; outputs not published")
            report = {
                "schema_version": "1.0.0",
                "artifact_kind": "privacy_redaction",
                "esl_version": __version__,
                "provenance": {
                    "source": str(source),
                    "source_sha256": source_hash,
                    "output": str(output),
                    "output_sha256": _hash(temporary[0]),
                    "detector": detector_id or ("python-callable" if detector else None),
                },
                "config": {
                    "mode": "zero-all-channels",
                    "sample_rate": reader.samplerate,
                    "channels": reader.channels,
                    "frames": reader.frames,
                    "encoding": f"{encoding}/DOUBLE",
                    "block_frames": block_frames,
                    "rounding": "floor-start/ceil-end; half-open sample spans",
                    "requested_intervals": [
                        {key: str(value) for key, value in item.items()} for item in all_intervals
                    ],
                },
                "redacted_sample_spans": [list(span) for span in spans],
                "redacted_frames": sum(right - left for left, right in spans),
                "warning": "Interval masking does not guarantee speech removal or anonymization.",
            }
        temporary[1].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Hard links publish without replacing another process's output.
        for staged, destination in zip(temporary, (output, audit), strict=True):
            os.link(staged, destination)
            published.append(destination)
        return report
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for staged in temporary:
            staged.unlink(missing_ok=True)


def _run(args: argparse.Namespace) -> int:
    try:
        intervals = []
        if args.intervals:
            intervals = json.loads(Path(args.intervals).read_text(), parse_float=Decimal)
            if not isinstance(intervals, list):
                raise ValueError("interval JSON must be a list of start/end objects")
        detector = None
        if args.detector:
            module, separator, name = args.detector.partition(":")
            if not separator:
                raise ValueError("--detector requires module:function (trusted Python code)")
            detector = getattr(importlib.import_module(module), name)
            if not callable(detector):
                raise ValueError("detector must be callable")
        if not args.intervals and detector is None:
            raise ValueError("supply --intervals or --detector; no implicit speech detection")
        report = redact_audio(
            args.source,
            args.output,
            args.audit,
            intervals,
            detector=detector,
            detector_id=args.detector,
            block_frames=args.block_frames,
        )
        print(json.dumps(report))
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError) as exc:
        print(f"privacy: {exc}")
        return 2
    return 0


def register(subparsers: Any) -> None:
    """Register explicit privacy redaction."""
    parser = subparsers.add_parser("privacy", help="Audited interval masking of audio")
    actions = parser.add_subparsers(dest="privacy_action", required=True)
    redact = actions.add_parser("redact")
    redact.add_argument("source")
    redact.add_argument("output")
    redact.add_argument("--audit", required=True)
    redact.add_argument("--intervals", help="JSON list of {start, end}, decimal seconds")
    redact.add_argument("--block-frames", type=int, default=65536)
    redact.add_argument("--detector", help="Trusted optional module:function(Path) interval hook")
    parser.set_defaults(func=_run)
