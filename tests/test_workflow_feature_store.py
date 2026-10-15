"""Persistence, exact ranking, metadata filters and independent CLI integration."""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.workflows import embeddings, feature_store

CONFIG = {"backend": "test", "dimensions": 2, "version": 1}
REPRESENTATION = embeddings.representation_id(CONFIG)


def test_persistence_replace_and_ranking(tmp_path):
    path = tmp_path / "vectors.sqlite"
    with feature_store.FeatureStore(path, CONFIG) as store:
        for item, vector in [("b", [1, 0]), ("a", [1, 0]), ("c", [0, 1])]:
            store.upsert(item, vector, {"site": "park"}, representation=REPRESENTATION)
        assert store.info()["count"] == 3
    with feature_store.FeatureStore(path) as store:
        matches = store.query([2, 0], representation=REPRESENTATION, top_k=2)
        assert [match["id"] for match in matches] == ["a", "b"]
        assert matches[0]["score"] == pytest.approx(1)
        store.upsert("a", [-1, 0], {"site": "road"}, representation=REPRESENTATION)
        assert store.info()["count"] == 3
        matches = store.query([1, 0], representation=REPRESENTATION)
        assert [match["id"] for match in matches] == ["b", "c", "a"]
        assert matches[-1]["score"] == pytest.approx(-1)


def test_filters_exact_json_types_and_no_sql_injection(tmp_path):
    with feature_store.FeatureStore(tmp_path / "db", CONFIG) as store:
        store.upsert(
            "a",
            [1, 0],
            {"site": "O'Reilly", "flag": True, "x": None},
            representation=REPRESENTATION,
        )
        assert (
            len(
                store.query(
                    [1, 0], representation=REPRESENTATION, filters={"site": "O'Reilly", "x": None}
                )
            )
            == 1
        )
        for filters in [{"flag": 1}, {"missing": None}, {"site": "' OR 1=1 --"}]:
            assert store.query([1, 0], representation=REPRESENTATION, filters=filters) == []


def test_compatibility_and_validation(tmp_path):
    path = tmp_path / "db"
    with feature_store.FeatureStore(path, CONFIG) as store:
        with pytest.raises(ValueError, match="Representation mismatch"):
            store.upsert("a", [1, 0], {}, representation="wrong")
        with pytest.raises(ValueError, match="Representation mismatch"):
            store.query([1, 0], representation="wrong")
        for vector in [[1], [0, 0], [float("inf"), 1]]:
            with pytest.raises(ValueError):
                store.upsert("a", vector, {}, representation=REPRESENTATION)
        with pytest.raises(ValueError):
            store.upsert(" ", [1, 0], {}, representation=REPRESENTATION)
        with pytest.raises(ValueError):
            store.upsert("a", [1, 0], {"x": float("nan")}, representation=REPRESENTATION)
        for count in [0, -1, 10001, True]:
            with pytest.raises(ValueError, match="top_k"):
                store.query([1, 0], representation=REPRESENTATION, top_k=count)
        assert store.info()["count"] == 0
    with pytest.raises(ValueError, match="Representation mismatch"):
        feature_store.FeatureStore(path, {**CONFIG, "version": 2})
    with pytest.raises(ValueError, match="does not exist"):
        feature_store.FeatureStore(tmp_path / "missing")


def test_standalone_cli_round_trip(tmp_path, capsys):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    embeddings.register(subparsers)
    feature_store.register(subparsers)
    audio = tmp_path / "tone.wav"
    sf.write(audio, np.sin(2 * np.pi * 440 * np.arange(16000) / 16000), 16000)
    database = tmp_path / "index.sqlite"

    def run(argv):
        args = parser.parse_args(argv)
        assert args.func(args) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["schema_version"] == "1.0.0"
        assert report["esl_version"]
        assert report["artifact_kind"]
        return report

    for _ in range(2):
        report = run(
            ["index", "ingest", str(database), str(audio), "--metadata", '{"site":"park"}']
        )
        assert report["count"] == 1
    record = tmp_path / "record.json"
    args = parser.parse_args(["embeddings", str(audio), "--output", str(record)])
    assert args.func(args) == 0
    for source in [["--record", str(record)], ["--audio", str(audio)]]:
        result = run(["index", "query", str(database), *source, "--filters", '{"site":"park"}'])
        assert result["results"][0]["id"] == str(audio.resolve())
        assert result["results"][0]["score"] == pytest.approx(1)
    assert run(["index", "info", str(database)])["count"] == 1
    record_data = json.loads(record.read_text())
    record_data["representation"]["dimensions"] = 12
    record.write_text(json.dumps(record_data))
    args = parser.parse_args(["index", "query", str(database), "--record", str(record)])
    with pytest.raises(ValueError, match="fingerprint"):
        args.func(args)


def test_python310_parse():
    ast.parse(Path(feature_store.__file__).read_text(), feature_version=(3, 10))
