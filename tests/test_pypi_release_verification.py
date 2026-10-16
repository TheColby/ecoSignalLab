"""Reject substituted, corrupt, or mismatched release files before publication."""

from __future__ import annotations

import hashlib
import io
import runpy
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest

verify = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/verify_pypi_release.py")
)["verify_distributions"]


def release_pair(directory: Path, metadata_version: str = "0.3.0") -> dict[str, Any]:
    metadata = f"Name: ecosignallab\nVersion: {metadata_version}\n".encode()
    wheel_path = directory / "ecosignallab-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr("ecosignallab-0.3.0.dist-info/METADATA", metadata)
    source_path = directory / "ecosignallab-0.3.0.tar.gz"
    with tarfile.open(source_path, "w:gz") as source:
        entry = tarfile.TarInfo("ecosignallab-0.3.0/PKG-INFO")
        entry.size = len(metadata)
        source.addfile(entry, io.BytesIO(metadata))
    return {
        "tagName": "v0.3.0",
        "isDraft": False,
        "assets": [
            {
                "name": artifact.name,
                "digest": "sha256:" + hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            for artifact in (wheel_path, source_path)
        ],
    }


def test_verified_pair(tmp_path: Path) -> None:
    assert len(verify(tmp_path, release_pair(tmp_path), "v0.3.0")) == 2


def test_modified_download_rejected(tmp_path: Path) -> None:
    release = release_pair(tmp_path)
    (tmp_path / "ecosignallab-0.3.0.tar.gz").write_bytes(b"substituted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path, release, "v0.3.0")


def test_metadata_version_must_match_tag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identity mismatch"):
        verify(tmp_path, release_pair(tmp_path, "9.0.0"), "v0.3.0")


def test_unexpected_file_rejected(tmp_path: Path) -> None:
    release = release_pair(tmp_path)
    (tmp_path / "unrelated.whl").touch()
    with pytest.raises(ValueError, match="exactly"):
        verify(tmp_path, release, "v0.3.0")


@pytest.mark.parametrize("field,value", [("isDraft", True), ("tagName", "v0.2.0")])
def test_wrong_release_rejected(tmp_path: Path, field: str, value: Any) -> None:
    release = release_pair(tmp_path)
    release[field] = value
    with pytest.raises(ValueError, match="published release"):
        verify(tmp_path, release, "v0.3.0")


def test_missing_asset_digest_rejected(tmp_path: Path) -> None:
    release = release_pair(tmp_path)
    release["assets"][0].pop("digest")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify(tmp_path, release, "v0.3.0")
