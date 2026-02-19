from __future__ import annotations

import json
from pathlib import Path

from esl.schema import SCHEMA_VERSION, analysis_output_schema


def test_published_schema_artifact_matches_runtime_schema() -> None:
    schema_path = Path("docs") / "schema" / f"analysis-output-{SCHEMA_VERSION}.json"
    assert schema_path.exists(), (
        f"Missing published schema artifact: {schema_path}. "
        "Generate it with: esl schema --out " + str(schema_path)
    )

    published = json.loads(schema_path.read_text(encoding="utf-8"))
    runtime = analysis_output_schema()
    assert published == runtime
