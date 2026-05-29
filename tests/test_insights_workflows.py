from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from esl.cli.main import main


def _write_scene_wav(path: Path, *, stereo: bool = False) -> Path:
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    a = 0.08 * np.sin(2.0 * np.pi * 220.0 * t[: sr // 2])
    b = 0.35 * np.sin(2.0 * np.pi * 1100.0 * t[: sr // 2])
    x = np.concatenate([a, b]).astype(np.float32)
    if stereo:
        y = np.stack([x, np.roll(x, 2)], axis=1)
        sf.write(path, y, sr)
    else:
        sf.write(path, x, sr)
    return path


def test_cli_insights_scene_calmness_occupancy_and_storyboard(tmp_path: Path) -> None:
    wav = _write_scene_wav(tmp_path / "scene.wav")

    scene_dir = tmp_path / "scene_out"
    code_scene = main(
        [
            "insights",
            "scene",
            str(wav),
            "--out",
            str(scene_dir),
            "--feature-set",
            "core",
            "--frame-size",
            "512",
            "--hop-size",
            "128",
            "--threshold-z",
            "0",
        ]
    )
    assert code_scene == 0
    scene_json = scene_dir / "scene_changes.json"
    assert scene_json.exists()
    scene = json.loads(scene_json.read_text(encoding="utf-8"))
    assert scene["insight_kind"] == "scene_changes"
    assert scene["changes"]
    assert (scene_dir / "scene_changes.csv").exists()

    calm_path = tmp_path / "calmness.json"
    code_calm = main(
        [
            "insights",
            "calmness",
            str(wav),
            "--out",
            str(calm_path),
            "--frame-size",
            "512",
            "--hop-size",
            "128",
        ]
    )
    assert code_calm == 0
    calm = json.loads(calm_path.read_text(encoding="utf-8"))
    assert 0.0 <= float(calm["calmness_score"]) <= 1.0
    assert float(calm["diversity_score"]) >= 0.0

    occ_dir = tmp_path / "occupancy"
    code_occ = main(
        [
            "insights",
            "occupancy",
            str(wav),
            "--out",
            str(occ_dir),
            "--bands",
            "low:20-400,high:800-2000",
            "--frame-size",
            "512",
            "--hop-size",
            "256",
            "--threshold-ratio",
            "0.05",
        ]
    )
    assert code_occ == 0
    occ = json.loads((occ_dir / "bio_occupancy.json").read_text(encoding="utf-8"))
    assert "low" in occ["bands"]
    assert "high" in occ["bands"]
    assert (occ_dir / "bio_occupancy.csv").exists()

    story_dir = tmp_path / "story"
    code_story = main(
        [
            "insights",
            "storyboard",
            str(wav),
            "--out",
            str(story_dir),
            "--clips",
            "2",
            "--window",
            "0.1",
            "--feature-set",
            "core",
            "--frame-size",
            "512",
            "--hop-size",
            "128",
        ]
    )
    assert code_story == 0
    story = json.loads((story_dir / "storyboard.json").read_text(encoding="utf-8"))
    assert story["items"]
    assert (story_dir / "storyboard.csv").exists()
    assert any((story_dir / "clips").glob("story_*.wav"))


def test_cli_insights_spatial_embeddings_retrieve_report_and_compare(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    query = _write_scene_wav(tmp_path / "query.wav", stereo=True)
    close = _write_scene_wav(corpus / "close.wav", stereo=True)
    sr = 8000
    t = np.arange(sr, dtype=np.float64) / sr
    far = np.stack(
        [
            (0.12 * np.sin(2.0 * np.pi * 1800.0 * t)).astype(np.float32),
            (0.12 * np.sin(2.0 * np.pi * 700.0 * t)).astype(np.float32),
        ],
        axis=1,
    )
    sf.write(corpus / "far.wav", far, sr)

    spatial_dir = tmp_path / "spatial"
    code_spatial = main(
        [
            "insights",
            "spatial",
            str(query),
            "--out",
            str(spatial_dir),
            "--frame-size",
            "512",
            "--hop-size",
            "256",
        ]
    )
    assert code_spatial == 0
    spatial = json.loads((spatial_dir / "spatial_timeline.json").read_text(encoding="utf-8"))
    assert spatial["channels"] == 2
    assert (spatial_dir / "spatial_timeline.csv").exists()

    emb_dir = tmp_path / "embeddings"
    code_emb = main(
        [
            "insights",
            "embeddings",
            str(corpus),
            "--out",
            str(emb_dir),
            "--feature-set",
            "core",
            "--frame-size",
            "512",
            "--hop-size",
            "256",
            "--device",
            "cpu",
        ]
    )
    assert code_emb == 0
    emb = json.loads((emb_dir / "embeddings_manifest.json").read_text(encoding="utf-8"))
    assert emb["num_files"] == 2
    assert (emb_dir / "embeddings.npz").exists()
    assert (emb_dir / "embeddings.csv").exists()

    ret_dir = tmp_path / "retrieve"
    code_ret = main(
        [
            "insights",
            "retrieve",
            str(close),
            str(corpus),
            "--out",
            str(ret_dir),
            "--top-k",
            "1",
            "--feature-set",
            "core",
            "--frame-size",
            "512",
            "--hop-size",
            "256",
        ]
    )
    assert code_ret == 0
    ret = json.loads((ret_dir / "event_retrieval.json").read_text(encoding="utf-8"))
    assert ret["results"]

    report_a = tmp_path / "a.json"
    report_b = tmp_path / "b.json"
    report_a.write_text(
        json.dumps(
            {
                "metrics": {
                    "rms_dbfs": {"summary": {"mean": -20.0}},
                    "rt60_s": {"summary": {"mean": 1.2}},
                }
            }
        ),
        encoding="utf-8",
    )
    report_b.write_text(
        json.dumps(
            {
                "metrics": {
                    "rms_dbfs": {"summary": {"mean": -18.0}},
                    "rt60_s": {"summary": {"mean": 1.4}},
                }
            }
        ),
        encoding="utf-8",
    )

    drift_path = tmp_path / "drift.json"
    assert main(["insights", "drift", str(report_a), str(report_b), "--out", str(drift_path)]) == 0
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    assert drift["common_metrics"] == ["rms_dbfs", "rt60_s"]
    assert float(drift["drift_score"]) > 0.0

    html_dir = tmp_path / "html_report"
    assert main(["insights", "report", str(report_a), "--out", str(html_dir)]) == 0
    assert (html_dir / "soundscape_report.html").exists()
    html = (html_dir / "soundscape_report.html").read_text(encoding="utf-8")
    assert "mermaid" in html

    cmp_path = tmp_path / "sim_compare.json"
    assert (
        main(
            ["insights", "simulation-compare", str(report_a), str(report_b), "--out", str(cmp_path)]
        )
        == 0
    )
    cmp_report = json.loads(cmp_path.read_text(encoding="utf-8"))
    assert cmp_report["metric_deltas"]


def test_cli_help_lists_insights(capsys) -> None:  # type: ignore[no-untyped-def]
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "insights" in out

    try:
        main(["insights", "--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "scene" in out
    assert "embeddings" in out
    assert "storyboard" in out
