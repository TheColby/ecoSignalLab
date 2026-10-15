"""Lease-fenced SQLite (same-host) or PostgreSQL (multi-host) audio shard workers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sqlite3
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf

from esl import __version__


class LeaseLostError(RuntimeError):
    """A claim expired or was replaced before this worker completed."""


class JobStore:
    """Atomic claims for processes sharing a host and a local SQLite file.

    Each method opens its own connection. Timestamps use the same host wall clock;
    lease tokens fence stale workers, but cannot provide exactly-once execution.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).resolve())
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS esl_archive_jobs (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
                owner TEXT, token TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL, result TEXT, error TEXT)""")

    @contextlib.contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def enqueue(self, jobs: list[dict[str, Any]], *, max_attempts: int = 3) -> list[str]:
        """Insert validated shard payloads idempotently in one transaction."""
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        payloads = [_validate_job(job) for job in jobs]
        identifiers = []
        with self._connection() as connection:
            for job in payloads:
                payload = json.dumps(job, sort_keys=True, separators=(",", ":"), allow_nan=False)
                identifier = hashlib.sha256(payload.encode()).hexdigest()
                connection.execute(
                    """INSERT INTO esl_archive_jobs(id,payload,max_attempts) VALUES(?,?,?)
                    ON CONFLICT(id) DO NOTHING""",
                    (identifier, payload, max_attempts),
                )
                identifiers.append(identifier)
        return identifiers

    def claim(
        self,
        owner: str,
        *,
        lease_seconds: float = 60,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Claim one queued/expired job with a fresh fencing token."""
        _positive(lease_seconds, "lease_seconds")
        if not owner:
            raise ValueError("owner must not be empty")
        now = time.time() if now is None else now
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE esl_archive_jobs SET state='failed', error='lease attempts exhausted'
                WHERE state='running' AND lease_until<=? AND attempts>=max_attempts""",
                (now,),
            )
            row = connection.execute(
                """SELECT * FROM esl_archive_jobs WHERE attempts<max_attempts AND
                (state='queued' OR (state='running' AND lease_until<=?)) ORDER BY rowid LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            connection.execute(
                """UPDATE esl_archive_jobs SET state='running',owner=?,token=?,lease_until=?,
                attempts=attempts+1,error=NULL WHERE id=?""",
                (owner, token, now + lease_seconds, row["id"]),
            )
            return {
                "id": row["id"],
                "token": token,
                "payload": json.loads(row["payload"]),
                "attempt": row["attempts"] + 1,
                "lease_until": now + lease_seconds,
            }

    def renew(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: float = 60,
        now: float | None = None,
    ) -> None:
        """Only an unexpired matching claim may extend a lease."""
        _positive(lease_seconds, "lease_seconds")
        now = time.time() if now is None else now
        with self._connection() as connection:
            changed = connection.execute(
                """UPDATE esl_archive_jobs SET lease_until=? WHERE id=? AND token=?
                AND state='running' AND lease_until>?""",
                (now + lease_seconds, job_id, token, now),
            ).rowcount
            if not changed:
                raise LeaseLostError(job_id)

    def complete(
        self,
        job_id: str,
        token: str,
        result: dict[str, Any],
        *,
        now: float | None = None,
    ) -> None:
        """Commit a result only if the completing worker still owns the lease."""
        self._finish(job_id, token, "complete", json.dumps(result, allow_nan=False), None, now)

    def fail(self, job_id: str, token: str, error: str, *, now: float | None = None) -> None:
        """Record a terminal processing failure; expired claims are retried separately."""
        self._finish(job_id, token, "failed", None, error, now)

    def _finish(
        self,
        job_id: str,
        token: str,
        state: str,
        result: str | None,
        error: str | None,
        now: float | None,
    ) -> None:
        now = time.time() if now is None else now
        with self._connection() as connection:
            changed = connection.execute(
                """UPDATE esl_archive_jobs SET state=?,result=?,error=?,lease_until=NULL
                WHERE id=? AND token=? AND state='running' AND lease_until>?""",
                (state, result, error, job_id, token, now),
            ).rowcount
            if not changed:
                raise LeaseLostError(job_id)

    def status(self) -> list[dict[str, Any]]:
        """Return durable payloads/results; expired leases become reclaimable on claim."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT id,payload,state,owner,lease_until,attempts,max_attempts,result,error
                FROM esl_archive_jobs ORDER BY id""",
            ).fetchall()
        jobs = [dict(row) for row in rows]
        for job in jobs:
            job["payload"] = json.loads(job["payload"])
            if job["result"] is not None:
                job["result"] = json.loads(job["result"])
        return jobs


class _PostgresConnection:
    """Adapt this module's positional SQL parameters to psycopg's parameter style."""

    def __init__(self, connection: Any) -> None:
        self.connection = connection

    def execute(self, sql: str, parameters: tuple[Any, ...] = ()) -> Any:
        return self.connection.execute(sql.replace("?", "%s"), parameters)


class PostgresJobStore(JobStore):
    """Multi-host queue using PostgreSQL row locks and database-clock lease checks.

    Requires optional psycopg v3 and a provisioned PostgreSQL server. No SQLite
    files are shared between machines. All workers must use this same database.
    """

    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ValueError("PostgreSQL requires optional psycopg[binary]>=3") from exc
        self._driver = psycopg
        self._row_factory = dict_row
        self._dsn = dsn
        parsed = urlparse(dsn)
        self.path = f"postgresql://{parsed.hostname or 'server'}{parsed.path}"
        with self._connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS esl_archive_jobs (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'queued',
                owner TEXT, token TEXT, lease_until DOUBLE PRECISION,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL, result TEXT, error TEXT)""")

    @contextlib.contextmanager
    def _connection(self) -> Iterator[Any]:
        try:
            with self._driver.connect(
                self._dsn,
                row_factory=self._row_factory,
                connect_timeout=10,
                options="-c statement_timeout=30000 -c lock_timeout=10000",
            ) as connection:
                yield _PostgresConnection(connection)
        except self._driver.Error as exc:
            # Do not expose credentials/DSNs via connection exceptions or worker reports.
            raise RuntimeError(f"PostgreSQL operation failed ({type(exc).__name__})") from exc

    def claim(
        self,
        owner: str,
        *,
        lease_seconds: float = 60,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        _positive(lease_seconds, "lease_seconds")
        if not owner or now is not None:
            raise ValueError("owner required; PostgreSQL always uses the database clock")
        token = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute("""UPDATE esl_archive_jobs
                SET state='failed',error='lease attempts exhausted'
                WHERE id IN (SELECT id FROM esl_archive_jobs WHERE state='running'
                AND lease_until<=EXTRACT(EPOCH FROM clock_timestamp())
                AND attempts>=max_attempts FOR UPDATE SKIP LOCKED)""")
            row = connection.execute(
                """WITH candidate AS (
                    SELECT id FROM esl_archive_jobs WHERE attempts<max_attempts AND
                    (state='queued' OR (state='running'
                    AND lease_until<=EXTRACT(EPOCH FROM clock_timestamp())))
                    ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED
                ) UPDATE esl_archive_jobs AS j SET state='running',owner=?,token=?,
                lease_until=EXTRACT(EPOCH FROM clock_timestamp())+?,
                attempts=j.attempts+1,error=NULL FROM candidate WHERE j.id=candidate.id
                RETURNING j.id,j.payload,j.attempts,j.lease_until""",
                (owner, token, lease_seconds),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "token": token,
                "payload": json.loads(row["payload"]),
                "attempt": row["attempts"],
                "lease_until": row["lease_until"],
            }

    def renew(
        self,
        job_id: str,
        token: str,
        *,
        lease_seconds: float = 60,
        now: float | None = None,
    ) -> None:
        _positive(lease_seconds, "lease_seconds")
        if now is not None:
            raise ValueError("PostgreSQL always uses the database clock")
        with self._connection() as connection:
            changed = connection.execute(
                """UPDATE esl_archive_jobs
                SET lease_until=EXTRACT(EPOCH FROM clock_timestamp())+?
                WHERE id=? AND token=? AND state='running'
                AND lease_until>EXTRACT(EPOCH FROM clock_timestamp())""",
                (lease_seconds, job_id, token),
            ).rowcount
            if not changed:
                raise LeaseLostError(job_id)

    def _finish(
        self,
        job_id: str,
        token: str,
        state: str,
        result: str | None,
        error: str | None,
        now: float | None,
    ) -> None:
        if now is not None:
            raise ValueError("PostgreSQL always uses the database clock")
        with self._connection() as connection:
            changed = connection.execute(
                """UPDATE esl_archive_jobs SET state=?,result=?,error=?,lease_until=NULL
                WHERE id=? AND token=? AND state='running'
                AND lease_until>EXTRACT(EPOCH FROM clock_timestamp())""",
                (state, result, error, job_id, token),
            ).rowcount
            if not changed:
                raise LeaseLostError(job_id)


def open_store(database: str) -> JobStore:
    """Select PostgreSQL for a postgres[ql] URI, otherwise a same-host SQLite file."""
    if urlparse(database).scheme in ("postgres", "postgresql"):
        return PostgresJobStore(database)
    return JobStore(database)


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_job(job: Any) -> dict[str, Any]:
    if not isinstance(job, dict) or "uri" not in job:
        raise ValueError("job requires uri")
    if set(job) - {"uri", "start_frame", "stop_frame", "sha256"}:
        raise ValueError("unknown job fields")
    if not isinstance(job["uri"], str) or not job["uri"]:
        raise ValueError("uri must be a nonempty string")
    start, stop = job.get("start_frame", 0), job.get("stop_frame")
    if type(start) is not int or start < 0:
        raise ValueError("start_frame must be a nonnegative integer")
    if stop is not None and (type(stop) is not int or stop <= start):
        raise ValueError("stop_frame must be an integer greater than start_frame")
    digest = job.get("sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ValueError("sha256 must contain 64 lowercase hexadecimal characters")
    return {**job, "start_frame": start, "stop_frame": stop}


@contextlib.contextmanager
def materialize(
    uri: str,
    *,
    allow_remote: bool = False,
    max_bytes: int = 1024 * 1024 * 1024,
) -> Iterator[Path]:
    """Stage bounded HTTP(S)/S3 objects, or use a local path without copying."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise ValueError("file URI must refer to the local host")
        path = Path(unquote(parsed.path) if parsed.scheme else uri).resolve()
        if path.stat().st_size > max_bytes:
            raise ValueError("input exceeds max_bytes")
        yield path
        return
    if parsed.scheme not in ("http", "https", "s3"):
        raise ValueError("supported URI schemes: local, file, http, https, s3")
    if not allow_remote:
        raise ValueError("remote URI requires explicit --allow-remote")
    with tempfile.TemporaryDirectory(prefix="esl-archive-") as directory:
        path = Path(directory) / "object"
        with contextlib.ExitStack() as stack:
            if parsed.scheme == "s3":
                try:
                    import boto3
                except ImportError as exc:
                    raise ValueError("S3 requires optional boto3; install it separately") from exc
                client = boto3.client("s3")
                stack.callback(client.close)
                response = client.get_object(
                    Bucket=parsed.netloc,
                    Key=unquote(parsed.path.lstrip("/")),
                )
                body = response["Body"]
                stack.callback(body.close)
                blocks = iter(lambda: body.read(1024 * 1024), b"")
            else:
                import requests

                response = stack.enter_context(requests.get(uri, stream=True, timeout=(10, 30)))
                response.raise_for_status()
                blocks = response.iter_content(chunk_size=1024 * 1024)
            total = 0
            with path.open("wb") as stream:
                for block in blocks:
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError("remote object exceeds max_bytes")
                    stream.write(block)
        yield path


def load_manifest(
    uri: str,
    *,
    allow_remote: bool = False,
    max_bytes: int = 16 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Read a strict manifest; relative local sources resolve against a local manifest."""
    with materialize(uri, allow_remote=allow_remote, max_bytes=max_bytes) as path:
        manifest = json.loads(path.read_text())
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "artifact_kind",
            "jobs",
        }:
            raise ValueError("manifest requires exactly schema_version, artifact_kind, jobs")
        if (
            manifest["schema_version"] != "1.0.0"
            or manifest["artifact_kind"] != "archive_job_manifest"
        ):
            raise ValueError("unsupported archive manifest schema/kind")
        if not isinstance(manifest["jobs"], list) or not manifest["jobs"]:
            raise ValueError("manifest jobs must be a nonempty list")
        jobs = [_validate_job(job) for job in manifest["jobs"]]
        remote_manifest = urlparse(uri).scheme in ("http", "https", "s3")
        for job in jobs:
            if not urlparse(job["uri"]).scheme and not Path(job["uri"]).is_absolute():
                if remote_manifest:
                    raise ValueError("remote manifests require absolute paths or object URIs")
                job["uri"] = str((path.parent / job["uri"]).resolve())
        return jobs


def execute_shard(
    job: dict[str, Any],
    *,
    allow_remote: bool = False,
    max_bytes: int = 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Execute a bounded-memory per-channel RMS/peak summary for a frame range."""
    job = _validate_job(job)
    with materialize(job["uri"], allow_remote=allow_remote, max_bytes=max_bytes) as path:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        source_hash = digest.hexdigest()
        if job.get("sha256") and job["sha256"] != source_hash:
            raise ValueError("source SHA-256 mismatch")
        with sf.SoundFile(path) as reader:
            start, stop = job["start_frame"], job["stop_frame"]
            stop = reader.frames if stop is None else stop
            if not 0 <= start < stop <= reader.frames:
                raise ValueError("shard frame range is empty or outside recording")
            reader.seek(start)
            squares = np.zeros(reader.channels, dtype=np.float64)
            peak = np.zeros(reader.channels, dtype=np.float64)
            remaining = stop - start
            while remaining:
                data = reader.read(min(65536, remaining), dtype="float64", always_2d=True)
                if not len(data) or not np.isfinite(data).all():
                    raise ValueError("source is truncated or contains nonfinite samples")
                squares += np.sum(data * data, axis=0)
                peak = np.maximum(peak, np.max(np.abs(data), axis=0))
                remaining -= len(data)
            return {
                "schema_version": "1.0.0",
                "artifact_kind": "archive_shard_summary",
                "esl_version": __version__,
                "provenance": {"uri": job["uri"], "source_sha256": source_hash},
                "config": {
                    "start_frame": start,
                    "stop_frame": stop,
                    "sample_rate": reader.samplerate,
                    "channels": reader.channels,
                    "units": "decoded sample amplitude; not calibrated SPL",
                },
                "frames": stop - start,
                "rms": np.sqrt(squares / (stop - start)).tolist(),
                "peak": peak.tolist(),
            }


def run_worker(
    store: JobStore,
    owner: str,
    *,
    max_jobs: int = 1,
    lease_seconds: float = 60,
    allow_remote: bool = False,
    max_bytes: int = 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Consume jobs until the queue is presently empty or max_jobs claims were handled."""
    _positive(lease_seconds, "lease_seconds")
    if type(max_jobs) is not int or max_jobs < 1:
        raise ValueError("max_jobs must be a positive integer")
    counts = {"complete": 0, "failed": 0, "lease_lost": 0}
    for _ in range(max_jobs):
        claim = store.claim(owner, lease_seconds=lease_seconds)
        if claim is None:
            break
        stop = threading.Event()
        lost = threading.Event()

        def heartbeat(
            claim: dict[str, Any] = claim,
            stop: threading.Event = stop,
            lost: threading.Event = lost,
        ) -> None:
            while not stop.wait(lease_seconds / 3):
                try:
                    store.renew(claim["id"], claim["token"], lease_seconds=lease_seconds)
                except (LeaseLostError, sqlite3.Error, RuntimeError):
                    lost.set()
                    return

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            result = execute_shard(claim["payload"], allow_remote=allow_remote, max_bytes=max_bytes)
            if lost.is_set():
                raise LeaseLostError(claim["id"])
            store.complete(claim["id"], claim["token"], result)
            counts["complete"] += 1
        except LeaseLostError:
            counts["lease_lost"] += 1
        except Exception as exc:  # Record a job failure; do not stop processing other shards.
            try:
                store.fail(claim["id"], claim["token"], f"{type(exc).__name__}: {exc}")
                counts["failed"] += 1
            except LeaseLostError:
                counts["lease_lost"] += 1
        finally:
            stop.set()
            thread.join()
    return {
        "schema_version": "1.0.0",
        "artifact_kind": "archive_worker_run",
        "esl_version": __version__,
        "provenance": {"database": store.path, "owner": owner},
        "config": {
            "max_jobs": max_jobs,
            "lease_seconds": lease_seconds,
            "allow_remote": allow_remote,
            "max_bytes": max_bytes,
        },
        "counts": counts,
    }


def _run(args: argparse.Namespace) -> int:
    try:
        store = open_store(args.database)
        if args.archive_action == "enqueue":
            jobs = load_manifest(
                args.manifest, allow_remote=args.allow_remote, max_bytes=args.max_bytes
            )
            report = {
                "schema_version": "1.0.0",
                "artifact_kind": "archive_enqueue",
                "esl_version": __version__,
                "provenance": {"manifest": args.manifest, "database": store.path},
                "config": {"max_attempts": args.max_attempts},
                "job_ids": store.enqueue(jobs, max_attempts=args.max_attempts),
            }
        elif args.archive_action == "worker":
            report = run_worker(
                store,
                args.owner,
                max_jobs=args.max_jobs,
                lease_seconds=args.lease_seconds,
                allow_remote=args.allow_remote,
                max_bytes=args.max_bytes,
            )
        else:
            report = {
                "schema_version": "1.0.0",
                "artifact_kind": "archive_queue_status",
                "esl_version": __version__,
                "provenance": {"database": store.path},
                "config": {
                    "backend": "postgresql"
                    if isinstance(store, PostgresJobStore)
                    else "same-host local SQLite"
                },
                "jobs": store.status(),
            }
        print(json.dumps(report, allow_nan=False))
        return int(
            args.archive_action == "worker"
            and bool(report["counts"]["failed"] or report["counts"]["lease_lost"])
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(f"archive-jobs: {exc}")
        return 2


def register(subparsers: Any) -> None:
    """Register enqueue, executable worker, and durable queue status commands."""
    parser = subparsers.add_parser(
        "archive-jobs", help="Lease-fenced SQLite/local or PostgreSQL/multi-host shard workers"
    )
    actions = parser.add_subparsers(dest="archive_action", required=True)
    enqueue = actions.add_parser("enqueue")
    enqueue.add_argument("database")
    enqueue.add_argument("manifest")
    enqueue.add_argument("--max-attempts", type=int, default=3)
    worker = actions.add_parser("worker")
    worker.add_argument("database")
    worker.add_argument("--owner", required=True)
    worker.add_argument("--max-jobs", type=int, default=1)
    worker.add_argument("--lease-seconds", type=float, default=60)
    for action in (enqueue, worker):
        action.add_argument("--allow-remote", action="store_true")
        action.add_argument("--max-bytes", type=int, default=1024 * 1024 * 1024)
    status = actions.add_parser("status")
    status.add_argument("database")
    parser.set_defaults(func=_run)
