from __future__ import annotations

import json
from pathlib import Path

from esl.cli.main import main
from esl.project import compare_project_variants, record_project_variant


def _mock_result(
    input_path: str,
    rms_dbfs: float,
    snr_db: float,
    clipping: bool,
) -> dict:
    return {
        "schema_version": "0.2.0",
        "esl_version": "0.1.0",
        "analysis_time_utc": "2026-02-19T12:00:00+00:00",
        "analysis_time_local": "2026-02-19T04:00:00-08:00",
        "config_hash": "cfg_hash",
        "pipeline_hash": "pipe_hash",
        "analysis_mode": "full",
        "metadata": {
            "input_path": input_path,
            "duration_s": 1.0,
            "sample_rate": 48000,
            "channels": 1,
            "validity_flags": {"clipping": clipping, "ir_detected": False},
            "warnings": [],
            "assumptions": [],
            "metric_catalog_version": "esl-metrics-1.0.0",
        },
        "metrics": {
            "rms_dbfs": {"summary": {"mean": rms_dbfs}, "confidence": 0.95},
            "snr_db": {"summary": {"mean": snr_db}, "confidence": 0.8},
        },
    }


def test_compare_project_variants_emits_delta_reports(tmp_path: Path) -> None:
    project = "restaurant_design"
    root = tmp_path
    record_project_variant(
        _mock_result("A.wav", rms_dbfs=-20.0, snr_db=14.0, clipping=False),
        project=project,
        variant="A",
        root=root,
    )
    record_project_variant(
        _mock_result("B.wav", rms_dbfs=-16.0, snr_db=10.0, clipping=True),
        project=project,
        variant="B",
        root=root,
    )

    report = compare_project_variants(project=project, root=root, baseline_variant="A")
    assert report["baseline_variant"] == "A"
    assert set(report["variants"]) == {"A", "B"}

    artifact_json = Path(report["artifacts"]["json"])
    artifact_csv = Path(report["artifacts"]["csv"])
    assert artifact_json.exists()
    assert artifact_csv.exists()

    rows = report["comparison_rows"]
    target = next(row for row in rows if row["variant"] == "B" and row["metric"] == "rms_dbfs")
    assert float(target["baseline_value"]) == -20.0
    assert float(target["value"]) == -16.0
    assert float(target["delta"]) == 4.0

    variant_b = next(v for v in report["variant_reports"] if v["variant"] == "B")
    assert variant_b["validity_flags"]["clipping"] is True


def test_cli_project_compare_command(tmp_path: Path) -> None:
    project = "office_refit"
    root = tmp_path
    record_project_variant(
        _mock_result("A.wav", rms_dbfs=-25.0, snr_db=16.0, clipping=False),
        project=project,
        variant="A",
        root=root,
    )
    record_project_variant(
        _mock_result("B.wav", rms_dbfs=-22.0, snr_db=13.0, clipping=False),
        project=project,
        variant="B",
        root=root,
    )

    code = main(
        [
            "project",
            "compare",
            "--project",
            project,
            "--root",
            str(root),
            "--baseline",
            "A",
            "--metrics",
            "rms_dbfs,snr_db",
        ]
    )
    assert code == 0

    report_json = root / "projects" / project / "comparison_report.json"
    assert report_json.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["baseline_variant"] == "A"
