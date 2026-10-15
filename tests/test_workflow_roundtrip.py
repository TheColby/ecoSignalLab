"""Exercise all ten workflows together through the public command dispatcher."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main


def test_public_workflow_roundtrip(tmp_path: Path) -> None:
    rate = 8000
    times = np.arange(rate * 2) / rate
    tone = 0.2 * np.sin(2 * np.pi * 440 * times)
    audio = tmp_path / "recording.wav"
    sf.write(audio, np.column_stack((tone, tone, tone, tone)), rate, subtype="FLOAT")
    report = tmp_path / "analysis.json"
    assert (
        main(
            [
                "analyze",
                str(audio),
                "--metrics",
                "rms_dbfs",
                "--verbosity",
                "0",
                "--json",
                str(report),
                "--out-dir",
                str(tmp_path / "analysis"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "monitor",
                "compare",
                str(report),
                str(report),
                "--threshold",
                "rms_dbfs=0",
                "--allow-uncalibrated",
                "--output",
                str(tmp_path / "comparison.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "uncertainty",
                str(report),
                "--metric",
                "rms_dbfs",
                "--resamples",
                "100",
                "--output",
                str(tmp_path / "uncertainty.json"),
            ]
        )
        == 0
    )
    assert main(["qa", str(audio), "--out-dir", str(tmp_path / "qa")]) == 0
    assert (
        main(
            [
                "spatial-timeline",
                str(audio),
                "--out-dir",
                str(tmp_path / "spatial"),
                "--ambisonics",
                "acn-sn3d",
            ]
        )
        == 0
    )
    embedding = tmp_path / "embedding.json"
    database = tmp_path / "vectors.sqlite"
    assert main(["embeddings", str(audio), "--output", str(embedding)]) == 0
    assert main(["index", "ingest", str(database), str(audio)]) == 0
    assert main(["index", "query", str(database), "--record", str(embedding)]) == 0
    protocol = tmp_path / "protocol.json"
    assert (
        main(
            [
                "protocol",
                "init",
                str(protocol),
                "--name",
                "roundtrip",
                "--author",
                "esl tests",
                "--sample-rate",
                str(rate),
                "--channels",
                "4",
            ]
        )
        == 0
    )
    assert main(["protocol", "validate", str(protocol)]) == 0
    intervals = tmp_path / "intervals.json"
    intervals.write_text('[{"start": 0.5, "end": 0.75}]', encoding="utf-8")
    redacted = tmp_path / "redacted.wav"
    assert (
        main(
            [
                "privacy",
                "redact",
                str(audio),
                str(redacted),
                "--audit",
                str(tmp_path / "privacy.json"),
                "--intervals",
                str(intervals),
            ]
        )
        == 0
    )
    samples, _ = sf.read(redacted, always_2d=True)
    assert np.all(samples[rate // 2 : 3 * rate // 4] == 0)
    assert np.max(np.abs(samples[: rate // 2])) > 0
    moments = tmp_path / "moments.csv"
    with moments.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "clip_id",
                "start_s",
                "end_s",
                "rank_score",
                "wav_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "clip_id": "event1",
                "start_s": 0,
                "end_s": 2,
                "rank_score": 1,
                "wav_path": str(audio),
            }
        )
    assert main(["moments", "review", str(moments), "--out", str(tmp_path / "review")]) == 0
    assert (tmp_path / "review/index.html").exists()
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "artifact_kind": "archive_job_manifest",
                "jobs": [{"uri": str(audio)}],
            }
        ),
        encoding="utf-8",
    )
    queue = str(tmp_path / "jobs.sqlite")
    assert main(["archive-jobs", "enqueue", queue, str(jobs)]) == 0
    assert main(["archive-jobs", "worker", queue, "--owner", "roundtrip"]) == 0
    from esl.workflows.archive_jobs import JobStore

    assert JobStore(queue).status()[0]["state"] == "complete"
