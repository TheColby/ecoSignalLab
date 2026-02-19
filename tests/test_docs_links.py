from __future__ import annotations

import re
from pathlib import Path


DOC_FILES = (
    Path("README.md"),
    Path("DESIGN.md"),
    Path("ARCHITECTURE.md"),
    *sorted(Path("docs").glob("*.md")),
)

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ABSOLUTE_LOCAL_RE = re.compile(r"^/Users/|^file://|^vscode://")


def _iter_markdown_links(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [m.group(1).strip() for m in MARKDOWN_LINK_RE.finditer(text)]


def test_core_docs_have_no_local_filesystem_links() -> None:
    offenders: list[str] = []
    for doc in DOC_FILES:
        for link in _iter_markdown_links(doc):
            if ABSOLUTE_LOCAL_RE.search(link):
                offenders.append(f"{doc}: {link}")
    assert not offenders, "Found local filesystem links:\n" + "\n".join(offenders)


def test_core_docs_relative_links_resolve() -> None:
    missing: list[str] = []
    for doc in DOC_FILES:
        for link in _iter_markdown_links(doc):
            target = link.split("#", 1)[0].strip()
            if not target:
                continue
            if re.match(r"^(?:https?://|mailto:)", target):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{doc}: {target}")
    assert not missing, "Broken relative links:\n" + "\n".join(missing)
