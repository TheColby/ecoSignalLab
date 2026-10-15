"""Persistent single-representation SQLite vectors with exact cosine retrieval."""

from __future__ import annotations

import argparse
import heapq
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from esl import __version__
from esl.workflows.embeddings import (
    add_backend_arguments,
    backend_from_args,
    canonical_json,
    embed_file,
    representation_id,
    unit_vector,
)


class FeatureStore:
    """A context-managed index. One database has exactly one immutable representation.

    Supply a representation to create or validate a store; omit it to open an existing store.
    Upserts commit individually and duplicate IDs replace the vector and metadata atomically.
    """

    def __init__(self, path: Path, representation: dict[str, Any] | None = None) -> None:
        path = Path(path)
        if representation is None and not path.is_file():
            raise ValueError(f"Index does not exist: {path}")
        if representation is not None:
            canonical_json(representation)
            dimensions = representation.get("dimensions")
            if type(dimensions) is not int or not 1 <= dimensions <= 65536:
                raise ValueError("Representation dimensions must be an integer in 1..65536")
        self._connection = sqlite3.connect(path, timeout=30)
        try:
            if representation is not None:
                with self._connection:
                    self._connection.execute(
                        "CREATE TABLE IF NOT EXISTS index_config "
                        "(singleton INTEGER PRIMARY KEY CHECK(singleton=1), "
                        "schema_version INTEGER NOT NULL, representation TEXT NOT NULL)"
                    )
                    self._connection.execute(
                        "INSERT OR IGNORE INTO index_config VALUES (1, 1, ?)",
                        (canonical_json(representation),),
                    )
                    self._connection.execute(
                        "CREATE TABLE IF NOT EXISTS vectors "
                        "(id TEXT PRIMARY KEY, vector BLOB NOT NULL, metadata TEXT NOT NULL)"
                    )
            row = self._connection.execute(
                "SELECT schema_version, representation FROM index_config WHERE singleton=1"
            ).fetchone()
            if row is None or row[0] != 1:
                raise ValueError("Unsupported or missing feature store schema")
            self.representation = json.loads(row[1])
            self.representation_id = representation_id(self.representation)
            self.dimensions = self.representation["dimensions"]
            if representation is not None and representation_id(representation) != (
                self.representation_id
            ):
                raise ValueError(
                    "Representation mismatch: use a separate index or identical backend"
                )
        except BaseException:
            self._connection.close()
            raise

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _check_representation(self, supplied: str) -> None:
        if supplied != self.representation_id:
            raise ValueError("Representation mismatch: cannot mix or query incompatible vectors")

    def upsert(
        self,
        item_id: str,
        vector: Any,
        metadata: dict[str, Any],
        *,
        representation: str,
    ) -> None:
        """Insert or replace a stable ID; caller must explicitly identify its representation."""
        self._check_representation(representation)
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("item_id must be a nonempty string")
        if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
            raise ValueError("metadata must be a JSON object with string keys")
        encoded = canonical_json(metadata)
        normalized = unit_vector(vector, self.dimensions).astype("<f8").tobytes()
        with self._connection:
            self._connection.execute(
                "INSERT INTO vectors VALUES (?, ?, ?) ON CONFLICT(id) DO UPDATE SET "
                "vector=excluded.vector, metadata=excluded.metadata",
                (item_id, normalized, encoded),
            )

    def query(
        self,
        vector: Any,
        *,
        representation: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Exact cosine search; filters AND top-level JSON key/value equalities."""
        self._check_representation(representation)
        if type(top_k) is not int or not 1 <= top_k <= 10000:
            raise ValueError("top_k must be an integer in 1..10000")
        filters = {} if filters is None else filters
        if not isinstance(filters, dict) or any(not isinstance(key, str) for key in filters):
            raise ValueError("filters must be a JSON object with string keys")
        canonical_json(filters)
        query = unit_vector(vector, self.dimensions)

        def candidates() -> Any:
            cursor = self._connection.execute("SELECT id, vector, metadata FROM vectors")
            try:
                for item_id, blob, encoded_metadata in cursor:
                    metadata = json.loads(encoded_metadata)
                    if any(
                        key not in metadata
                        or canonical_json(metadata[key]) != canonical_json(value)
                        for key, value in filters.items()
                    ):
                        continue
                    stored = unit_vector(np.frombuffer(blob, dtype="<f8"), self.dimensions)
                    score = float(np.clip(np.dot(query, stored), -1, 1))
                    yield {"id": item_id, "score": score, "metadata": metadata}
            finally:
                cursor.close()

        return heapq.nsmallest(top_k, candidates(), key=lambda row: (-row["score"], row["id"]))

    def info(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "artifact_kind": "feature-index",
            "esl_version": __version__,
            "representation_id": self.representation_id,
            "representation": self.representation,
            "count": self._connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
            "metric": "cosine",
            "search": "exact-linear-scan",
        }


def _object(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    canonical_json(result)
    return result


def _run(args: argparse.Namespace) -> int:
    if args.index_action == "info":
        with FeatureStore(args.database) as store:
            print(canonical_json(store.info()))
        return 0
    if args.index_action == "ingest":
        backend = backend_from_args(args)
        metadata = _object(args.metadata)
        with FeatureStore(args.database, backend.config) as store:
            count = 0
            for path in args.audio:
                record = embed_file(path, backend, args.max_seconds)
                # Resolved paths give stable rerun IDs, replacing moved-in content explicitly.
                store.upsert(
                    record["source"],
                    record["vector"],
                    {**metadata, "source": record["source"]},
                    representation=record["representation_id"],
                )
                count += 1
            print(canonical_json({**store.info(), "ingested": count}))
        return 0
    filters = _object(args.filters)
    if args.record:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != "1.0.0"
            or record.get("artifact_kind") != "audio-embedding"
            or not isinstance(record.get("representation"), dict)
            or "vector" not in record
            or representation_id(record["representation"]) != record.get("representation_id")
        ):
            raise ValueError("Invalid embedding record schema or representation fingerprint")
    else:
        record = embed_file(args.audio, backend_from_args(args), args.max_seconds)
    with FeatureStore(args.database) as store:
        results = store.query(
            record["vector"],
            representation=record["representation_id"],
            top_k=args.top_k,
            filters=filters,
        )
        print(
            canonical_json(
                {
                    **store.info(),
                    "artifact_kind": "feature-index-query",
                    "filters": filters,
                    "top_k": args.top_k,
                    "results": results,
                }
            )
        )
    return 0


def register(subparsers: Any) -> None:
    """Register ``index ingest``, ``index query`` and ``index info``."""
    parser = subparsers.add_parser("index", help="Persistent SQLite audio vector index")
    actions = parser.add_subparsers(dest="index_action", required=True)
    ingest = actions.add_parser("ingest", help="Upsert audio using resolved paths as stable IDs")
    ingest.add_argument("database", type=Path)
    ingest.add_argument("audio", type=Path, nargs="+")
    ingest.add_argument("--metadata", default="{}", help="JSON object shared by this ingestion")
    add_backend_arguments(ingest)
    ingest.set_defaults(func=_run)
    query = actions.add_parser("query", help="Exact cosine nearest neighbors")
    query.add_argument("database", type=Path)
    source = query.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--record", type=Path, help="JSON emitted by embeddings")
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--filters", default="{}", help="AND of exact top-level metadata matches")
    add_backend_arguments(query)
    query.set_defaults(func=_run)
    info = actions.add_parser("info", help="Show count and immutable representation provenance")
    info.add_argument("database", type=Path)
    info.set_defaults(func=_run)
