"""Audio decoding, streaming, and format handling.

References:
- Polyphase resampling foundation:
  Crochiere & Rabiner (1983), \"Multirate Digital Signal Processing\".
- Practical implementation API:
  SciPy `signal.resample_poly` documentation.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Generator, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from esl.core.spatial_metadata import infer_spatial_metadata


SUPPORTED_NATIVE_EXT = {".wav", ".flac", ".aiff", ".aif", ".rf64", ".caf"}
SUPPORTED_COMPRESSED_EXT = {".mp3", ".aac", ".ogg", ".opus", ".wma", ".alac", ".m4a"}
SUPPORTED_SPATIAL_EXT = {".sofa"}


@dataclass(slots=True)
class AudioBuffer:
    """Decoded audio buffer."""

    samples: np.ndarray  # [num_samples, num_channels]
    sample_rate: int
    source_path: str
    format_name: str
    subtype: str | None
    source_backend: str
    decoder_provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return float(self.num_samples / self.sample_rate)


@dataclass(slots=True)
class AudioChunk:
    """Chunk of streaming audio."""

    index: int
    start_sample: int
    sample_rate: int
    samples: np.ndarray  # [chunk_samples, num_channels]
    context_frames: int = 0
    """Leading frames carried from the previous chunk purely as analysis context.

    Consumers must discard the first ``context_frames`` entries of any
    per-frame metric series computed on this chunk; those frames were already
    emitted (or exist only to seed first-difference metrics such as spectral
    flux) and would otherwise be double counted.
    """

    new_samples: int = -1
    """Samples in this chunk not already seen in a previous chunk.

    ``-1`` means "all of them" (the non-overlapping legacy behaviour).
    Whole-signal accumulators must use this to avoid double counting the
    overlap region.
    """


@dataclass(slots=True)
class SofaIR:
    """SOFA impulse response representation."""

    ir: np.ndarray  # [num_samples, num_channels]
    sample_rate: int
    source_path: str


def _resample_if_needed(samples: np.ndarray, src_sr: int, dst_sr: int | None) -> tuple[np.ndarray, int]:
    if dst_sr is None or dst_sr == src_sr:
        return samples, src_sr
    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    # Polyphase resampling (multirate DSP standard approach).
    out = np.zeros((int(np.ceil(samples.shape[0] * dst_sr / src_sr)), samples.shape[1]), dtype=np.float32)
    for c in range(samples.shape[1]):
        out[:, c] = resample_poly(samples[:, c], up, down).astype(np.float32)
    return out, dst_sr


def _read_native(path: Path, target_sr: int | None = None) -> AudioBuffer:
    info = sf.info(str(path))
    samples, sr = sf.read(str(path), always_2d=True, dtype="float32")
    samples, sr = _resample_if_needed(samples, sr, target_sr)
    return AudioBuffer(
        samples=samples,
        sample_rate=sr,
        source_path=str(path),
        format_name=info.format,
        subtype=info.subtype,
        source_backend="soundfile",
        decoder_provenance={
            "decoder_used": "soundfile",
            "ffmpeg_version": None,
            "ffprobe": None,
        },
    )


def _ffmpeg_version() -> str | None:
    cmd = ["ffmpeg", "-version"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").splitlines()
    return first[0].strip() if first else None


def _ffprobe_summary(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=index,codec_name,codec_type,sample_rate,channels,channel_layout,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe executable not found on PATH") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no streams for {path}")
    stream0 = next((s for s in streams if str(s.get("codec_type", "")).lower() == "audio"), streams[0])
    sr = int(stream0.get("sample_rate", 48000))
    ch = int(stream0.get("channels", 1))
    return {
        "sample_rate": sr,
        "channels": ch,
        "codec_name": stream0.get("codec_name"),
        "codec_type": stream0.get("codec_type"),
        "channel_layout": stream0.get("channel_layout"),
        "duration_s": (
            float(stream0.get("duration"))
            if stream0.get("duration") is not None
            else float(payload.get("format", {}).get("duration"))
            if payload.get("format", {}).get("duration") is not None
            else None
        ),
        "stream_index": stream0.get("index"),
    }


def _read_ffmpeg(path: Path, target_sr: int | None = None) -> AudioBuffer:
    probe = _ffprobe_summary(path)
    src_sr = int(probe["sample_rate"])
    channels = int(probe["channels"])
    sr = int(target_sr or src_sr)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed for {path}: {stderr}")
    raw = np.frombuffer(proc.stdout, dtype=np.float32)
    if channels <= 0:
        raise RuntimeError(f"Invalid channel count for {path}: {channels}")
    if raw.size % channels != 0:
        raw = raw[: raw.size - (raw.size % channels)]
    samples = raw.reshape(-1, channels)
    return AudioBuffer(
        samples=samples,
        sample_rate=sr,
        source_path=str(path),
        format_name=path.suffix.lower().lstrip("."),
        subtype=None,
        source_backend="ffmpeg",
        decoder_provenance={
            "decoder_used": "ffmpeg",
            "ffmpeg_version": _ffmpeg_version(),
            "ffprobe": probe,
        },
    )


def load_sofa(path: str | Path) -> SofaIR:
    """Load SOFA IRs and return first-measurement channels as sample-major matrix."""
    p = Path(path)
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for SOFA support.") from exc

    with h5py.File(p, "r") as h5:
        if "Data" not in h5 or "IR" not in h5["Data"]:
            raise RuntimeError(f"SOFA file missing Data.IR dataset: {p}")
        ir = np.array(h5["Data"]["IR"], dtype=np.float32)
        sr_ds = h5["Data"].get("SamplingRate")
        sr = int(np.array(sr_ds)[0] if sr_ds is not None else 48000)

    if ir.ndim == 3:
        # [M, R, N] -> first measurement, sample-major [N, R]
        ir2 = np.transpose(ir[0], (1, 0))
    elif ir.ndim == 2:
        # [R, N] -> [N, R]
        ir2 = np.transpose(ir, (1, 0))
    else:
        raise RuntimeError(f"Unsupported SOFA IR rank {ir.ndim} for {p}")

    return SofaIR(ir=ir2, sample_rate=sr, source_path=str(p))


def read_audio(path: str | Path, target_sr: int | None = None) -> AudioBuffer:
    """Read audio from native formats or ffmpeg-backed compressed formats."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    ext = p.suffix.lower()
    if ext in SUPPORTED_SPATIAL_EXT:
        sofa = load_sofa(p)
        samples, sr = _resample_if_needed(sofa.ir, sofa.sample_rate, target_sr)
        return AudioBuffer(
            samples=samples,
            sample_rate=sr,
            source_path=str(p),
            format_name="SOFA",
            subtype=None,
            source_backend="h5py",
            decoder_provenance={
                "decoder_used": "h5py",
                "ffmpeg_version": None,
                "ffprobe": None,
            },
        )

    try:
        return _read_native(p, target_sr)
    except Exception:
        return _read_ffmpeg(p, target_sr)


def probe_sample_rate(path: str | Path) -> int:
    """Return source sample rate without decoding the full file when possible."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")
    try:
        info = sf.info(str(p))
        return int(info.samplerate)
    except Exception:
        probe = _ffprobe_summary(p)
        return int(probe["sample_rate"])


def probe_audio_metadata(path: str | Path) -> dict[str, Any]:
    """Return best-effort file metadata without decoding the full signal."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    size_bytes = int(p.stat().st_size)
    ext = p.suffix.lower()
    payload: dict[str, Any] = {
        "path": str(p.resolve()),
        "name": p.name,
        "extension": ext,
        "size_bytes": size_bytes,
        "size_gb": float(size_bytes / 1_000_000_000.0),
    }

    if ext in SUPPORTED_SPATIAL_EXT:
        sofa = load_sofa(p)
        spatial = infer_spatial_metadata(
            int(sofa.ir.shape[1]),
            p,
            source_channel_layout=None,
        ).to_dict()
        payload.update(
            {
                "sample_rate": int(sofa.sample_rate),
                "channels": int(sofa.ir.shape[1]),
                "num_samples": int(sofa.ir.shape[0]),
                "duration_s": float(sofa.ir.shape[0] / max(sofa.sample_rate, 1)),
                "format_name": "SOFA",
                "subtype": None,
                "backend": "h5py",
                "codec_name": None,
                "channel_layout": None,
                "decoder_provenance": {
                    "decoder_used": "h5py",
                    "ffmpeg_version": None,
                    "ffprobe": None,
                },
                "channel_layout_hint": spatial["layout_hint"],
                "spatial_metadata": spatial,
            }
        )
        return payload

    try:
        info = sf.info(str(p))
        spatial = infer_spatial_metadata(
            int(info.channels),
            p,
            source_channel_layout=None,
        ).to_dict()
        payload.update(
            {
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "num_samples": int(info.frames),
                "duration_s": float(info.frames / max(info.samplerate, 1)),
                "format_name": info.format,
                "subtype": info.subtype,
                "backend": "soundfile",
                "codec_name": None,
                "channel_layout": None,
                "decoder_provenance": {
                    "decoder_used": "soundfile",
                    "ffmpeg_version": None,
                    "ffprobe": None,
                },
                "channel_layout_hint": spatial["layout_hint"],
                "spatial_metadata": spatial,
            }
        )
        return payload
    except Exception:
        probe = _ffprobe_summary(p)
        spatial = infer_spatial_metadata(
            int(probe["channels"]),
            p,
            source_channel_layout=(str(probe.get("channel_layout")) if probe.get("channel_layout") is not None else None),
        ).to_dict()
        payload.update(
            {
                "sample_rate": int(probe["sample_rate"]),
                "channels": int(probe["channels"]),
                "num_samples": (
                    int(round(float(probe["duration_s"]) * float(probe["sample_rate"])))
                    if probe.get("duration_s") is not None
                    else None
                ),
                "duration_s": float(probe["duration_s"]) if probe.get("duration_s") is not None else None,
                "format_name": ext.lstrip(".").upper() or "unknown",
                "subtype": None,
                "backend": "ffprobe",
                "codec_name": probe.get("codec_name"),
                "channel_layout": probe.get("channel_layout"),
                "decoder_provenance": {
                    "decoder_used": "ffmpeg",
                    "ffmpeg_version": _ffmpeg_version(),
                    "ffprobe": probe,
                },
                "channel_layout_hint": spatial["layout_hint"],
                "spatial_metadata": spatial,
            }
        )
        return payload


def _stream_ffmpeg(
    path: Path,
    *,
    chunk_size: int,
    target_sr: int | None,
) -> Generator[AudioChunk, None, None]:
    """Stream decoded float32 frames from ffmpeg stdout.

    This avoids full-buffer decode for long compressed recordings and any file
    that SoundFile cannot stream directly.
    """
    probe = _ffprobe_summary(path)
    channels = int(probe["channels"])
    sample_rate = int(target_sr or probe["sample_rate"])
    bytes_per_frame = 4 * channels
    read_size = max(int(chunk_size), 1) * bytes_per_frame
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable not found on PATH") from exc

    assert proc.stdout is not None
    assert proc.stderr is not None
    start = 0
    idx = 0
    pending = b""
    while True:
        block = proc.stdout.read(read_size)
        if not block:
            break
        pending += block
        frames = len(pending) // bytes_per_frame
        usable = frames * bytes_per_frame
        if frames <= 0:
            continue
        raw = pending[:usable]
        pending = pending[usable:]
        samples = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).copy()
        yield AudioChunk(index=idx, start_sample=start, sample_rate=sample_rate, samples=samples)
        start += int(samples.shape[0])
        idx += 1

    stderr = proc.stderr.read().decode("utf-8", errors="ignore").strip()
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg streaming decode failed for {path}: {stderr}")


def stream_audio(
    path: str | Path,
    chunk_size: int = 131072,
    target_sr: int | None = None,
) -> Generator[AudioChunk, None, None]:
    """Yield audio chunks for streaming-friendly analysis."""
    p = Path(path)
    if p.suffix.lower() == ".sofa":
        buf = read_audio(p, target_sr=target_sr)
        start = 0
        idx = 0
        while start < buf.num_samples:
            end = min(start + chunk_size, buf.num_samples)
            yield AudioChunk(index=idx, start_sample=start, sample_rate=buf.sample_rate, samples=buf.samples[start:end])
            start = end
            idx += 1
        return

    try:
        with sf.SoundFile(str(p), mode="r") as f:
            src_sr = int(f.samplerate)
            idx = 0
            start = 0
            for block in f.blocks(blocksize=chunk_size, dtype="float32", always_2d=True):
                out, sr = _resample_if_needed(block, src_sr, target_sr)
                yield AudioChunk(index=idx, start_sample=start, sample_rate=sr, samples=out)
                start += int(out.shape[0])
                idx += 1
            return
    except Exception:
        pass

    try:
        yield from _stream_ffmpeg(p, chunk_size=chunk_size, target_sr=target_sr)
        return
    except Exception:
        pass

    # Last-resort fallback when streaming decode is unavailable anywhere.
    buf = read_audio(p, target_sr=target_sr)
    idx = 0
    for start in range(0, buf.num_samples, chunk_size):
        end = min(start + chunk_size, buf.num_samples)
        yield AudioChunk(index=idx, start_sample=start, sample_rate=buf.sample_rate, samples=buf.samples[start:end])
        idx += 1


def stream_audio_frame_aligned(
    path: str | Path,
    *,
    frame_size: int,
    hop_size: int,
    chunk_size: int = 131072,
    target_sr: int | None = None,
    context_frames: int = 1,
) -> Generator[AudioChunk, None, None]:
    """Yield hop-aligned, overlapping chunks whose framing matches full-file framing.

    ``stream_audio`` cuts the signal at fixed sample boundaries and each chunk is
    then framed from its own sample 0.  Unless ``chunk_size`` happens to be a
    multiple of ``hop_size``, that re-anchors the analysis grid at every chunk,
    and frames that would straddle a boundary are dropped entirely.  This
    generator instead tracks the global frame grid: every emitted chunk starts on
    a multiple of ``hop_size``, carries ``frame_size - hop_size`` samples of
    overlap so boundary-straddling frames are still computed, and optionally
    prepends ``context_frames`` already-emitted frames so that first-difference
    metrics (spectral flux and friends) see their predecessor.

    Each yielded chunk reports how many leading frames are context
    (:attr:`AudioChunk.context_frames`) and how many of its samples are new
    (:attr:`AudioChunk.new_samples`).
    """
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if hop_size > frame_size:
        raise ValueError("hop_size must not exceed frame_size")
    context_frames = max(0, int(context_frames))

    buf: np.ndarray | None = None
    buf_start = 0  # global index of buf[0]
    next_frame = 0  # global start of the next frame not yet emitted
    emitted_end = 0  # global sample index one past the last frame emitted
    idx = 0
    sample_rate = 0
    total_samples = 0
    channels = 0

    def _emit(
        buffer: np.ndarray,
        buffer_start: int,
        first_frame: int,
        n_frames: int,
        chunk_index: int,
        prior_end: int,
    ) -> AudioChunk:
        lead = min(context_frames, (first_frame - buffer_start) // hop_size)
        start = first_frame - lead * hop_size
        stop = first_frame + (n_frames - 1) * hop_size + frame_size
        lo = start - buffer_start
        hi = stop - buffer_start
        window = buffer[lo:hi]
        return AudioChunk(
            index=chunk_index,
            start_sample=start,
            sample_rate=sample_rate,
            samples=window,
            context_frames=lead,
            new_samples=max(0, stop - max(prior_end, start)),
        )

    for raw in stream_audio(path, chunk_size=chunk_size, target_sr=target_sr):
        sample_rate = int(raw.sample_rate)
        channels = int(raw.samples.shape[1]) if raw.samples.ndim == 2 else 1
        total_samples += int(raw.samples.shape[0])
        buf = raw.samples if buf is None else np.concatenate([buf, raw.samples], axis=0)
        available = buf_start + int(buf.shape[0])

        span = available - frame_size - next_frame
        n_new = (span // hop_size) + 1 if span >= 0 else 0
        if n_new > 0:
            chunk = _emit(buf, buf_start, next_frame, n_new, idx, emitted_end)
            yield chunk
            idx += 1
            emitted_end = next_frame + (n_new - 1) * hop_size + frame_size
            next_frame += n_new * hop_size

        # Retain enough history for the next chunk's overlap and context frames.
        keep_from = max(0, next_frame - context_frames * hop_size)
        if keep_from > buf_start:
            buf = buf[keep_from - buf_start :]
            buf_start = keep_from

    if buf is None or sample_rate == 0:
        return

    if idx == 0:
        # Signal shorter than one frame: full-file framing zero-pads it to a
        # single frame, so emit it whole and let the framing helper pad.
        if int(buf.shape[0]) > 0:
            yield AudioChunk(
                index=0,
                start_sample=buf_start,
                sample_rate=sample_rate,
                samples=buf,
                context_frames=0,
                new_samples=int(buf.shape[0]),
            )
        return

    # Reproduce full-file framing's trailing flush-right frame.
    if total_samples >= frame_size and (total_samples - frame_size) % hop_size != 0:
        start = total_samples - frame_size
        if start >= buf_start:
            window = buf[start - buf_start : start - buf_start + frame_size]
        else:  # pragma: no cover - only if the tail was trimmed too aggressively
            window = np.zeros((frame_size, channels), dtype=buf.dtype)
            window[: buf.shape[0]] = buf[:frame_size]
        yield AudioChunk(
            index=idx,
            start_sample=start,
            sample_rate=sample_rate,
            samples=window,
            context_frames=0,
            new_samples=max(0, total_samples - max(emitted_end, start)),
        )


def detect_signal_layout(channels: int, source_path: str | Path) -> str:
    """Classify high-level channel layout hints."""
    return infer_spatial_metadata(channels, source_path).layout_hint


def iter_supported_files(root: str | Path, patterns: Iterable[str], recursive: bool = True) -> list[Path]:
    """Collect supported input files from a directory."""
    r = Path(root)
    if not r.exists():
        return []
    files: list[Path] = []
    for pattern in patterns:
        if recursive:
            files.extend(r.rglob(pattern))
        else:
            files.extend(r.glob(pattern))
    unique = sorted({f.resolve() for f in files if f.is_file()})
    return list(unique)
