"""Render the ten workflow guides with the existing Mermaid/MathJax book renderer."""

from __future__ import annotations

import argparse
from pathlib import Path

from esl.docsgen import build_docs

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "ADVANCED_WORKFLOWS",
    "REVIEW_PACKS",
    "MONITORING",
    "SPATIAL_TIMELINES",
    "RECORDER_QA",
    "FEATURE_STORE",
    "UNCERTAINTY",
    "PROTOCOLS",
    "EMBEDDINGS",
    "PRIVACY",
    "ARCHIVE_WORKERS",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "docs/build/workflows"))
    parser.add_argument("--formats", default="html,pdf")
    args = parser.parse_args()
    # Reference links from a focused guide still need their HTML target pages.
    build_docs(root=ROOT, output_root=args.out, formats={"html"})
    report = build_docs(
        root=ROOT,
        output_root=args.out,
        formats=set(args.formats.split(",")),
        title="ecoSignalLab Advanced Workflows",
        docs_files=[ROOT / "docs" / f"{name}.md" for name in GUIDES],
    )
    print(f"HTML: {report.output_root / 'html/ecoSignalLab_docs.html'}")
    if report.pdf_pages:
        print(f"PDF: {report.output_root / 'pdf/ecoSignalLab_docs.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
