from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from esl.version import __version__
from esl.workflows import review


def _moments(tmp_path: Path) -> Path:
    source = tmp_path / "native"
    source.mkdir()
    rate = 8000
    data = np.column_stack((np.full(rate, 0.2), np.full(rate, 0.4)))
    sf.write(source / "clip.wav", data, rate, subtype="FLOAT")
    with (source / "moments.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["clip_id", "start_s", "end_s", "rank_score", "wav_path"],
        )
        writer.writeheader()
        for key, score in (("low", "0.1"), ("<script>alert(1)</script>", "0.9")):
            writer.writerow(
                dict(clip_id=key, start_s=42, end_s=43, rank_score=score, wav_path="clip.wav")
            )
    report = source / "moments_report.json"
    report.write_text(json.dumps({"csv_path": "moments.csv"}), encoding="utf-8")
    return report


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_native_pack_ranking_portability_and_audio_bound(tmp_path: Path) -> None:
    report = _moments(tmp_path)
    output = tmp_path / "pack"
    page = review.build_review_pack(report, output, max_items=1, preview_seconds=0.125)
    document = page.read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<script>alert(1)</script>" not in document
    assert str(tmp_path) not in document
    assert "Download labels.csv" in document
    manifest = json.loads((output / "review_manifest.json").read_text())
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["artifact_kind"] == "review_pack"
    assert manifest["esl_version"] == __version__
    assert manifest["candidate_count"] == 2
    assert manifest["item_count"] == 1
    assert manifest["items"][0]["clip_id"] == "<script>alert(1)</script>"
    assert manifest["provenance"]["moments_manifest"] == str(report)
    samples, rate = sf.read(output / manifest["items"][0]["audio"])
    assert rate == 8000
    assert samples.shape == (1000,)
    assert np.allclose(samples, 0.3, atol=1e-4)
    for image in output.glob("assets/*.png"):
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(list(output.glob("assets/*.png"))) == 2
    row = _csv_rows(output / "labels.csv")[0]
    assert row["start_s"] == "42"
    assert row["status"] == "pending"


def test_resume_annotations_csv_and_escaping(tmp_path: Path) -> None:
    report = _moments(tmp_path)
    labels = tmp_path / "annotations.csv"
    with labels.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["clip_id", "status", "label", "notes"])
        writer.writeheader()
        writer.writerow(
            dict(
                clip_id="low",
                status="accepted",
                label="bird, finch",
                notes='</textarea><script>"unsafe"</script>\nnext line',
            )
        )
    output = tmp_path / "resume"
    review.build_review_pack(report.with_name("moments.csv"), output, labels_path=labels)
    row = _csv_rows(output / "labels.csv")[1]
    assert row["status"] == "accepted"
    assert row["label"] == "bird, finch"
    assert "\nnext line" in row["notes"]
    document = (output / "index.html").read_text()
    assert "&lt;/textarea&gt;&lt;script&gt;&quot;unsafe&quot;" in document
    assert "selected>accepted" in document
    second = tmp_path / "resume_again"
    review.build_review_pack(report, second, labels_path=output / "labels.csv")
    assert _csv_rows(second / "labels.csv") == _csv_rows(output / "labels.csv")


def test_failure_does_not_leave_partial_pack(tmp_path: Path) -> None:
    report = _moments(tmp_path)
    report.with_name("clip.wav").unlink()
    output = tmp_path / "broken"
    with pytest.raises((OSError, sf.LibsndfileError)):
        review.build_review_pack(report, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".review-*"))


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), 121])
def test_preview_limits(tmp_path: Path, seconds: float) -> None:
    with pytest.raises(ValueError, match="preview_seconds"):
        review.build_review_pack(
            tmp_path / "missing.csv", tmp_path / "out", preview_seconds=seconds
        )


def test_refuse_overwrite_and_invalid_queue(tmp_path: Path) -> None:
    report = _moments(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        review.build_review_pack(report, output)
    assert marker.read_text() == "keep"
    csv_path = report.with_name("moments.csv")
    content = csv_path.read_text()
    csv_path.write_text(content.replace("0.9", "nan"))
    with pytest.raises(ValueError, match="finite"):
        review.build_review_pack(report, tmp_path / "badscore")
    csv_path.write_text(content.replace("low", "<script>alert(1)</script>"))
    with pytest.raises(ValueError, match="Duplicate"):
        review.build_review_pack(report, tmp_path / "duplicate")


def test_unknown_annotations_rejected(tmp_path: Path) -> None:
    report = _moments(tmp_path)
    labels = tmp_path / "labels.csv"
    labels.write_text("clip_id,status,label,notes\nunknown,accepted,bird,\n")
    with pytest.raises(ValueError, match="Unknown"):
        review.build_review_pack(report, tmp_path / "bad", labels_path=labels)


def test_manifest_byte_and_row_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "moments.csv"
    manifest.write_text("clip_id,wav_path\na,clip.wav\nb,clip.wav\n")
    monkeypatch.setattr(review, "_MAX_BYTES", 4)
    with pytest.raises(ValueError, match="bytes"):
        review.build_review_pack(manifest, tmp_path / "out")
    monkeypatch.setattr(review, "_MAX_BYTES", 1000)
    monkeypatch.setattr(review, "_MAX_ROWS", 1)
    with pytest.raises(ValueError, match="rows"):
        review.build_review_pack(manifest, tmp_path / "out")


def test_read_audio_in_bounded_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _moments(tmp_path)
    calls = []
    original = sf.SoundFile.read

    def tracked_read(self: sf.SoundFile, frames: int, **kwargs: object) -> np.ndarray:
        calls.append(frames)
        return original(self, frames, **kwargs)

    monkeypatch.setattr(sf.SoundFile, "read", tracked_read)
    review.build_review_pack(report, tmp_path / "pack", max_items=1, preview_seconds=0.1)
    assert calls == [800]


@pytest.mark.parametrize("alias", [False, True])
def test_registration_and_handler(tmp_path: Path, alias: bool) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    if alias:
        moments = subparsers.add_parser("moments")
        review.register_moments(moments.add_subparsers(required=True))
    else:
        review.register(subparsers)
    report = _moments(tmp_path)
    argv = ["moments"] if alias else []
    args = parser.parse_args(
        argv
        + [
            "review",
            str(report),
            "--out",
            str(tmp_path / "pack"),
            "--max-items",
            "1",
            "--preview-seconds",
            "0.05",
        ]
    )
    assert args.func(args) == 0
    assert (tmp_path / "pack" / "index.html").exists()


def test_csv_formula_escaping() -> None:
    assert review._safe_cell("  =HYPERLINK(1)").startswith("'")
    assert review._safe_cell("bird") == "bird"


@pytest.mark.parametrize("max_items", [0, -1, 1001])
def test_item_limits(tmp_path: Path, max_items: int) -> None:
    with pytest.raises(ValueError, match="max_items"):
        review.build_review_pack(tmp_path / "missing.csv", tmp_path / "out", max_items=max_items)


def test_native_invocation_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _moments(tmp_path)
    monkeypatch.chdir(tmp_path)
    report.write_text(json.dumps({"csv_path": "native/moments.csv"}))
    csv_path = report.with_name("moments.csv")
    csv_path.write_text(csv_path.read_text().replace("clip.wav", "native/clip.wav"))
    review.build_review_pack(report, tmp_path / "pack", max_items=1, preview_seconds=0.05)
    assert (tmp_path / "pack" / "assets" / "item_0001.wav").exists()


def test_short_and_nonfinite_audio(tmp_path: Path) -> None:
    report = _moments(tmp_path)
    clip = report.with_name("clip.wav")
    sf.write(clip, np.ones(2, dtype=np.float32) * 0.1, 8000, subtype="FLOAT")
    review.build_review_pack(report, tmp_path / "tiny", max_items=1)
    assert len(list((tmp_path / "tiny" / "assets").glob("*.png"))) == 2
    sf.write(clip, np.array([0.0, np.nan, 0.0]), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="Nonfinite"):
        review.build_review_pack(report, tmp_path / "bad", max_items=1)
    assert not (tmp_path / "bad").exists()


def test_failure_during_plotting_is_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _moments(tmp_path)

    def fail(*args: object) -> None:
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(review, "_plots", fail)
    with pytest.raises(RuntimeError, match="renderer"):
        review.build_review_pack(report, tmp_path / "bad", max_items=1)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".review-*"))
