"""Release-version consistency checks."""

from __future__ import annotations

import re
from pathlib import Path

from esl import __version__
from esl.schema import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_project_metadata() -> None:
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = metadata.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version = "([^"]+)"$', project, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == __version__ == "0.3.0"


def test_schema_version_remains_independent_of_package_version() -> None:
    """A package release does not imply a JSON-schema contract change."""
    assert SCHEMA_VERSION == "0.2.0"
