"""Verify GitHub release distribution hashes and identities before PyPI publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any


def verify_distributions(directory: Path, release: dict[str, Any], tag: str) -> list[str]:
    """Require exactly the release wheel and sdist, with matching SHA-256 and metadata."""
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", tag) is None:
        raise ValueError("Expected a stable vMAJOR.MINOR.PATCH tag")
    version = tag[1:]
    if release.get("tagName") != tag or release.get("isDraft") is not False:
        raise ValueError("Expected a published release matching the requested tag")
    expected = {
        f"ecosignallab-{version}-py3-none-any.whl",
        f"ecosignallab-{version}.tar.gz",
    }
    if {item.name for item in directory.iterdir()} != expected:
        raise ValueError("Distribution directory must contain exactly one wheel and one sdist")
    assets = release.get("assets", [])
    for name in sorted(expected):
        matching = [asset for asset in assets if asset.get("name") == name]
        if len(matching) != 1:
            raise ValueError(f"Release asset missing or ambiguous: {name}")
        artifact = directory / name
        if not artifact.is_file() or artifact.is_symlink():
            raise ValueError(f"Expected a regular distribution file: {name}")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if matching[0].get("digest") != f"sha256:{digest}":
            raise ValueError(f"Release SHA-256 mismatch: {name}")
        if name.endswith(".whl"):
            with zipfile.ZipFile(artifact) as wheel:
                metadata = wheel.read(f"ecosignallab-{version}.dist-info/METADATA")
        else:
            with tarfile.open(artifact, mode="r:gz") as source:
                entry = source.getmember(f"ecosignallab-{version}/PKG-INFO")
                if not entry.isfile() or entry.size > 16 * 1024 * 1024:
                    raise ValueError("Invalid source-distribution metadata entry")
                stream = source.extractfile(entry)
                if stream is None:
                    raise ValueError("Missing source-distribution metadata")
                with stream:
                    metadata = stream.read()
        parsed = BytesParser().parsebytes(metadata)
        if parsed["Name"] != "ecosignallab" or parsed["Version"] != version:
            raise ValueError(f"Package identity mismatch: {name}")
    return sorted(expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--release-json", type=Path, required=True)
    parser.add_argument("--dist", type=Path, required=True)
    args = parser.parse_args()
    verified = verify_distributions(args.dist, json.loads(args.release_json.read_text()), args.tag)
    print("Verified: " + ", ".join(verified))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
