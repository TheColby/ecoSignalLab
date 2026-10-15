"""Embedding compatibility, validation, local inference and standalone parser tests."""

import argparse
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.workflows import embeddings


def tone(path: Path, frequency: float = 440, rate: int = 16000) -> Path:
    samples = np.sin(2 * np.pi * frequency * np.arange(rate) / rate)
    sf.write(path, samples, rate, subtype="FLOAT")
    return path


def test_spectral_deterministic_and_discriminative(tmp_path):
    backend = embeddings.SpectralBackend()
    low = tone(tmp_path / "low.wav", 440)
    high = tone(tmp_path / "high.wav", 4000)
    first = embeddings.embed_file(low, backend)
    second = embeddings.embed_file(low, backend)
    other = embeddings.embed_file(high, backend)
    assert first == second
    assert first["schema_version"] == "1.0.0"
    assert first["artifact_kind"] == "audio-embedding"
    assert first["esl_version"]
    assert len(first["vector"]) == 64
    assert np.linalg.norm(first["vector"]) == pytest.approx(1)
    assert np.dot(first["vector"], other["vector"]) < 0.1
    assert embeddings.representation_id({"a": 1, "b": 2}) == (
        embeddings.representation_id({"b": 2, "a": 1})
    )
    assert first["representation_id"] != embeddings.representation_id(
        embeddings.SpectralBackend(dimensions=32).config
    )


def test_preprocess_stereo_resample_and_limits(tmp_path):
    path = tmp_path / "stereo.wav"
    wave = np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    sf.write(path, np.stack((wave, wave), axis=1), 8000)
    record = embeddings.embed_file(path, embeddings.SpectralBackend())
    assert len(record["vector"]) == 64
    with pytest.raises(ValueError, match="exceeds max_seconds"):
        embeddings.embed_file(path, embeddings.SpectralBackend(), max_seconds=0.5)
    with pytest.raises(ValueError, match="positive and finite"):
        embeddings.embed_file(path, embeddings.SpectralBackend(), max_seconds=float("nan"))


@pytest.mark.parametrize("vector", [[0, 0], [1, float("nan")], [[1, 2]], [1]])
def test_invalid_vectors(vector):
    with pytest.raises(ValueError):
        embeddings.unit_vector(vector, 2)


def test_silence_and_bad_audio():
    backend = embeddings.SpectralBackend()
    for samples in [np.zeros(2048), np.array([]), np.array([float("inf")])]:
        with pytest.raises(ValueError):
            backend.embed(samples)
    with pytest.raises(ValueError):
        embeddings.SpectralBackend(dimensions=0)


def test_embedding_cli_without_global_registration(tmp_path):
    parser = argparse.ArgumentParser()
    embeddings.register(parser.add_subparsers(required=True))
    output = tmp_path / "embedding.json"
    args = parser.parse_args(["embeddings", str(tone(tmp_path / "a.wav")), "--output", str(output)])
    assert args.func(args) == 0
    assert json.loads(output.read_text())["representation"]["backend"] == "spectral"
    args.model = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="requires --backend"):
        embeddings.backend_from_args(args)
    args.backend = "torchscript"
    args.model = None
    with pytest.raises(ValueError, match="--model is required"):
        embeddings.backend_from_args(args)


def test_missing_optional_torch(monkeypatch, tmp_path):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(embeddings.importlib, "import_module", missing)
    with pytest.raises(ValueError, match="optional torch"):
        embeddings.TorchScriptBackend(tmp_path / "model.pt", 16000, 2)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="optional torch not installed"
)
def test_torchscript_local_model_provenance(tmp_path):
    import torch

    class Model(torch.nn.Module):
        def forward(self, samples):
            return torch.stack((samples.mean(dim=1), samples.square().mean(dim=1)), dim=1)

    model = torch.jit.trace(Model(), torch.ones(1, 100))
    path = tmp_path / "model.pt"
    model.save(str(path))
    backend = embeddings.TorchScriptBackend(path, 16000, 2)
    vector = backend.embed(np.ones(100))
    assert vector == pytest.approx([2**-0.5, 2**-0.5])
    assert len(backend.config["model_sha256"]) == 64
    moved = tmp_path / "moved.pt"
    moved.write_bytes(path.read_bytes())
    assert backend.config == embeddings.TorchScriptBackend(moved, 16000, 2).config
    with pytest.raises(ValueError, match="shape"):
        embeddings.TorchScriptBackend(path, 16000, 3).embed(np.ones(100))


def test_python310_parse():
    for module in [embeddings]:
        ast.parse(Path(module.__file__).read_text(), feature_version=(3, 10))
