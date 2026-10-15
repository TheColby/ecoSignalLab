"""Claims, fencing, bounded shard execution, and opt-in PostgreSQL integration."""

import argparse
import contextlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from esl.workflows.archive_jobs import (
    JobStore,
    LeaseLostError,
    PostgresJobStore,
    _PostgresConnection,
    execute_shard,
    load_manifest,
    materialize,
    register,
    run_worker,
)


def test_idempotent_atomic_concurrent_claims(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    jobs = [{"uri": f"/audio/{index}.wav"} for index in range(20)]
    ids = store.enqueue(jobs)
    assert store.enqueue(jobs) == ids
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda index: store.claim(str(index)), range(24)))
    claimed = [item["id"] for item in claims if item is not None]
    assert len(claimed) == len(set(claimed)) == 20
    assert len(store.status()) == 20


def test_expiry_reclaim_fencing_and_renewal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.enqueue([{"uri": "/test.wav"}])
    first = store.claim("first", lease_seconds=10, now=100)
    assert first is not None
    assert store.claim("second", now=109) is None
    store.renew(first["id"], first["token"], lease_seconds=10, now=109)
    assert store.claim("second", now=118) is None
    second = store.claim("second", lease_seconds=10, now=119)
    assert second is not None and first["token"] != second["token"]
    for action in (store.complete, store.fail):
        with pytest.raises(LeaseLostError):
            action(first["id"], first["token"], {} if action == store.complete else "x", now=120)
    with pytest.raises(LeaseLostError):
        store.renew(first["id"], first["token"], now=120)
    store.complete(second["id"], second["token"], {"ok": True}, now=120)
    assert store.status()[0]["result"] == {"ok": True}
    assert store.claim("third", now=130) is None


def test_exhausted_expiry_and_expired_completion(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    store.enqueue([{"uri": "/test.wav"}], max_attempts=1)
    first = store.claim("first", lease_seconds=1, now=0)
    assert first is not None
    with pytest.raises(LeaseLostError):
        store.complete(first["id"], first["token"], {}, now=1)
    assert store.claim("next", now=1) is None
    assert store.status()[0]["state"] == "failed"


def test_shard_execution_and_worker_failure(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    data = np.array([[0.25, -0.5], [0.5, 0.25], [-0.75, 0.125]])
    sf.write(source, data, 8000, subtype="DOUBLE")
    result = execute_shard({"uri": str(source), "start_frame": 1, "stop_frame": 3})
    np.testing.assert_allclose(result["rms"], np.sqrt(np.mean(data[1:] ** 2, axis=0)))
    np.testing.assert_array_equal(result["peak"], np.max(np.abs(data[1:]), axis=0))
    assert result["schema_version"] == "1.0.0" and result["esl_version"]
    store = JobStore(tmp_path / "jobs.sqlite")
    store.enqueue([{"uri": str(source)}, {"uri": str(tmp_path / "missing.wav")}])
    report = run_worker(store, "test", max_jobs=3, lease_seconds=1)
    assert report["counts"] == {"complete": 1, "failed": 1, "lease_lost": 0}
    assert {row["state"] for row in store.status()} == {"complete", "failed"}


@pytest.mark.parametrize(
    "job",
    [
        {},
        {"uri": ""},
        {"uri": "/x", "unknown": 1},
        {"uri": "/x", "start_frame": True},
        {"uri": "/x", "stop_frame": 0},
        {"uri": "/x", "sha256": "bad"},
    ],
)
def test_invalid_jobs(tmp_path: Path, job: dict) -> None:
    store = JobStore(tmp_path / "q")
    with pytest.raises(ValueError):
        store.enqueue([{"uri": "/valid"}, job])
    assert store.status() == []


def test_manifest_and_standalone_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    source = tmp_path / "in.wav"
    sf.write(source, np.ones((10, 2)), 10)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_kind": "archive_job_manifest",
                "jobs": [{"uri": "in.wav"}],
            }
        )
    )
    assert load_manifest(str(path))[0]["uri"] == str(source)
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(required=True))
    for command in (
        ["enqueue", str(tmp_path / "q"), str(path)],
        ["worker", str(tmp_path / "q"), "--owner", "test"],
        ["status", str(tmp_path / "q")],
    ):
        args = parser.parse_args(["archive-jobs", *command])
        assert args.func(args) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["esl_version"] and report["provenance"] and report["config"]
    assert report["jobs"][0]["state"] == "complete"


def test_remote_opt_in_and_download_byte_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [b"abc", b"def"]

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="allow-remote"), materialize("https://example.org/audio"):
        pass
    with materialize("https://example.org/audio", allow_remote=True, max_bytes=6) as path:
        assert path.read_bytes() == b"abcdef"
        staged = path
    assert not staged.exists()
    with (
        pytest.raises(ValueError, match="exceeds max_bytes"),
        materialize(
            "https://example.org/audio",
            allow_remote=True,
            max_bytes=5,
        ),
    ):
        pass


def test_hash_and_range_validation(tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    sf.write(source, np.ones(10), 10)
    with pytest.raises(ValueError, match="SHA-256"):
        execute_shard({"uri": str(source), "sha256": "0" * 64})
    with pytest.raises(ValueError, match="outside recording"):
        execute_shard({"uri": str(source), "stop_frame": 11})
    with pytest.raises(ValueError, match="max_bytes"):
        execute_shard({"uri": str(source)}, max_bytes=1)


def test_postgres_adapter_enqueue_claim_status_and_fencing() -> None:
    calls: list[tuple[str, tuple]] = []

    class Cursor:
        rowcount = 1

        def fetchone(self) -> dict:
            return {"id": "job", "payload": '{"uri":"/a"}', "attempts": 1, "lease_until": 123.0}

        def fetchall(self) -> list[dict]:
            return [{"id": "job", "payload": '{"uri":"/a"}', "result": '{"done":true}'}]

    class Connection:
        def execute(self, sql: str, parameters: tuple = ()) -> Cursor:
            calls.append((sql, parameters))
            return Cursor()

    store = PostgresJobStore.__new__(PostgresJobStore)

    @contextlib.contextmanager
    def connection() -> Any:
        yield _PostgresConnection(Connection())

    store._connection = connection
    assert len(store.enqueue([{"uri": "/a"}])) == 1
    assert "ON CONFLICT(id) DO NOTHING" in calls[-1][0]
    assert "INSERT OR IGNORE" not in calls[-1][0]
    assert "%s" in calls[-1][0] and "?" not in calls[-1][0]
    claim = store.claim("worker")
    assert claim is not None
    assert "FOR UPDATE SKIP LOCKED" in calls[-1][0]
    assert "clock_timestamp()" in calls[-1][0]
    store.renew(claim["id"], claim["token"])
    assert "token=%s" in calls[-1][0] and "clock_timestamp()" in calls[-1][0]
    store.complete(claim["id"], claim["token"], {"done": True})
    assert "token=%s" in calls[-1][0] and "clock_timestamp()" in calls[-1][0]
    assert store.status()[0]["result"] == {"done": True}
    assert "rowid" not in calls[-1][0]


def test_postgres_live_claim_fencing_and_execution(tmp_path: Path) -> None:
    """CI can opt in; isolate a unique schema and never delete shared queue tables."""
    dsn = os.environ.get("ESL_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("set ESL_TEST_POSTGRES_DSN to run live PostgreSQL integration")
    psycopg = pytest.importorskip("psycopg")
    schema = "esl_test_" + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema}"')

    class IsolatedStore(PostgresJobStore):
        @contextlib.contextmanager
        def _connection(self) -> Any:
            with super()._connection() as connection:
                connection.execute(f'SET search_path TO "{schema}"')
                yield connection

    try:
        store = IsolatedStore(dsn)
        source = tmp_path / f"{uuid.uuid4().hex}.wav"
        sf.write(source, np.ones((20, 2)) * 0.25, 10, subtype="DOUBLE")
        jobs = [
            {"uri": str(source), "start_frame": index, "stop_frame": index + 1}
            for index in range(12)
        ]
        identifiers = store.enqueue(jobs)
        assert store.enqueue(jobs) == identifiers
        with ThreadPoolExecutor(max_workers=6) as pool:
            claims = list(pool.map(lambda index: store.claim(str(index)), range(12)))
        assert all(claim is not None for claim in claims)
        assert len({claim["id"] for claim in claims}) == 12
        for claim in claims:
            store.complete(claim["id"], claim["token"], {"test": True})
        assert all(row["state"] == "complete" for row in store.status())
        stale_id = store.enqueue([{"uri": str(source), "start_frame": 12, "stop_frame": 14}])[0]
        first = store.claim("stale", lease_seconds=30)
        assert first is not None and first["id"] == stale_id
        store.renew(first["id"], first["token"], lease_seconds=30)
        with store._connection() as connection:
            connection.execute(
                "UPDATE esl_archive_jobs SET lease_until=0 WHERE id=?",
                (stale_id,),
            )
        second = store.claim("replacement")
        assert second is not None and second["id"] == stale_id
        with pytest.raises(LeaseLostError):
            store.complete(first["id"], first["token"], {})
        store.complete(second["id"], second["token"], execute_shard(second["payload"]))
        store.enqueue([{"uri": str(source), "start_frame": 14, "stop_frame": 18}])
        report = run_worker(store, "live-runner")
        assert report["counts"]["complete"] == 1
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
