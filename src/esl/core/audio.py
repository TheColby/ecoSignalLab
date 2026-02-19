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
                start += block.shape[0]
                idx += 1
            return
    except Exception:
        pass

    # Fallback when streaming decode is unavailable.
    buf = read_audio(p, target_sr=target_sr)
    idx = 0
    for start in range(0, buf.num_samples, chunk_size):
        end = min(start + chunk_size, buf.num_samples)
        yield AudioChunk(index=idx, start_sample=start, sample_rate=buf.sample_rate, samples=buf.samples[start:end])
        idx += 1


def detect_signal_layout(channels: int, source_path: str | Path) -> str:
    """Classify high-level channel layout hints."""
    if channels == 1:
        return "mono"
    p = str(source_path).lower()
    if channels == 4 and any(token in p for token in ("ambi", "bformat", "ambisonic")):
        return "ambisonic_b_format"
    if channels > 2:
        return "multichannel"
    return "stereo"


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
