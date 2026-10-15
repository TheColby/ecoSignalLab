"""Local, provenance-addressed audio embedding backends."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly, welch

from esl import __version__


def canonical_json(value: Any) -> str:
    """Serialize configuration without NaN or ordering ambiguity."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def representation_id(config: dict[str, Any]) -> str:
    """Name a representation by its complete canonical configuration."""
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def unit_vector(vector: Any, dimensions: int) -> np.ndarray:
    """Validate and normalize a nonzero finite vector."""
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (dimensions,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite vector of shape ({dimensions},)")
    scale = float(np.max(np.abs(result)))
    if scale == 0:
        raise ValueError("Zero vectors (including silent spectral audio) are unsupported")
    result = result / scale
    return result / np.linalg.norm(result)


class EmbeddingBackend(Protocol):
    """Plugins describe all representation-affecting settings, then embed mono audio."""

    @property
    def config(self) -> dict[str, Any]: ...

    def embed(self, audio: np.ndarray) -> np.ndarray: ...


class SpectralBackend:
    """Deterministic square-root linear-band Welch energy, not a semantic model."""

    def __init__(self, sample_rate: int = 16000, dimensions: int = 64) -> None:
        if not 1000 <= sample_rate <= 192000 or not 2 <= dimensions <= 512:
            raise ValueError("sample_rate must be 1000..192000; dimensions must be 2..512")
        self.sample_rate = sample_rate
        self.dimensions = dimensions

    @property
    def config(self) -> dict[str, Any]:
        return {
            "backend": "spectral",
            "version": 1,
            "sample_rate": self.sample_rate,
            "dimensions": self.dimensions,
            "nfft": 2048,
            "window": "hann",
            "overlap": 1024,
            "pooling": "sqrt-linear-band-energy",
            "normalization": "l2",
            "preprocessing": "mono-mean-resample-poly-v1",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
        }

    def embed(self, audio: np.ndarray) -> np.ndarray:
        audio = _validate_audio(audio)
        audio = np.pad(audio, (0, max(0, 2048 - len(audio))))
        frequencies, power = welch(
            audio,
            fs=self.sample_rate,
            window="hann",
            nperseg=2048,
            noverlap=1024,
            nfft=2048,
            detrend=False,
            scaling="density",
            average="mean",
        )
        bands = np.minimum(
            (frequencies / (self.sample_rate / 2) * self.dimensions).astype(int),
            self.dimensions - 1,
        )
        energy = np.bincount(bands, weights=power, minlength=self.dimensions)
        return unit_vector(np.sqrt(energy), self.dimensions)


def _validate_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty finite mono samples")
    return audio


class TorchScriptBackend:
    """Trusted local TorchScript: float32 [1, samples] -> [1, dimensions] or [dimensions]."""

    def __init__(self, model: Path, sample_rate: int, dimensions: int) -> None:
        if not 1000 <= sample_rate <= 192000 or not 1 <= dimensions <= 65536:
            raise ValueError("Invalid model sample rate or dimensions")
        try:
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ValueError(
                "TorchScript requires optional torch; install ecosignallab[ml]"
            ) from exc
        # Hash and load the same open file, so a path replacement cannot change provenance.
        with Path(model).open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            self._sha256 = digest.hexdigest()
            stream.seek(0)
            self._model = self._torch.jit.load(stream, map_location="cpu").eval()
        self.sample_rate = sample_rate
        self.dimensions = dimensions

    @property
    def config(self) -> dict[str, Any]:
        return {
            "backend": "torchscript",
            "version": 1,
            "model_sha256": self._sha256,
            "sample_rate": self.sample_rate,
            "dimensions": self.dimensions,
            "torch": self._torch.__version__,
            "normalization": "l2",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "preprocessing": "mono-mean-resample-poly-v1",
            "device": "cpu",
        }

    def embed(self, audio: np.ndarray) -> np.ndarray:
        audio = _validate_audio(audio)
        tensor = self._torch.from_numpy(audio.astype(np.float32)[None, :])
        with self._torch.inference_mode():
            output = self._model(tensor).detach().cpu().numpy()
        if output.shape == (1, self.dimensions):
            output = output[0]
        return unit_vector(output, self.dimensions)


def embed_file(
    path: Path,
    backend: EmbeddingBackend,
    max_seconds: float = 300.0,
) -> dict[str, Any]:
    """Reject oversized audio, downmix and resample, then return a portable record."""
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be positive and finite")
    config = backend.config
    rate = int(config["sample_rate"])
    if not 1000 <= rate <= 192000:
        raise ValueError("Backend sample_rate must be in 1000..192000")
    with sf.SoundFile(path) as stream:
        if stream.frames > max_seconds * stream.samplerate or stream.frames == 0:
            raise ValueError("Audio is empty or exceeds max_seconds; segment it first")
        if stream.channels > 64 or stream.frames * stream.channels > 50_000_000:
            raise ValueError("Audio exceeds the 64-channel or 50-million decoded-sample limit")
        if math.ceil(stream.frames * rate / stream.samplerate) > 50_000_000:
            raise ValueError("Audio exceeds the 50-million resampled-sample limit")
        audio = stream.read(dtype="float64", always_2d=True).mean(axis=1)
        source_rate = stream.samplerate
    divisor = math.gcd(rate, source_rate)
    if source_rate != rate:
        audio = resample_poly(audio, rate // divisor, source_rate // divisor)
    vector = unit_vector(backend.embed(audio), int(config["dimensions"]))
    return {
        "schema_version": "1.0.0",
        "artifact_kind": "audio-embedding",
        "esl_version": __version__,
        "representation_id": representation_id(config),
        "representation": config,
        "source": str(Path(path).resolve()),
        "vector": vector.tolist(),
    }


def add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared backend selection for embedding and index commands."""
    parser.add_argument("--backend", choices=("spectral", "torchscript"), default="spectral")
    parser.add_argument(
        "--model", type=Path, help="Trusted local TorchScript file; never downloaded"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=300.0)


def backend_from_args(args: argparse.Namespace) -> EmbeddingBackend:
    if args.backend == "torchscript":
        if args.model is None:
            raise ValueError("--model is required for torchscript")
        return TorchScriptBackend(args.model, args.sample_rate, args.dimensions)
    if args.model is not None:
        raise ValueError("--model requires --backend torchscript")
    return SpectralBackend(args.sample_rate, args.dimensions)


def _run(args: argparse.Namespace) -> int:
    result = embed_file(args.audio, backend_from_args(args), args.max_seconds)
    encoded = canonical_json(result) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


def register(subparsers: Any) -> None:
    """Register ``esl embeddings AUDIO [--output JSON]``."""
    parser = subparsers.add_parser("embeddings", help="Embed local audio with explicit provenance")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path)
    add_backend_arguments(parser)
    parser.set_defaults(func=_run)
