#!/usr/bin/env python3
"""Build the dedicated ecoSignalLab user-guide HTML and PDF artifacts.

The guide is intentionally a curated manual rather than a concatenation of all
reference material. It uses the project documentation renderer so Mermaid,
MathJax, internal links, and print styling stay consistent with published docs.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from esl.docsgen import build_docs


ROOT = Path(__file__).resolve().parent.parent
GUIDE_SOURCE = ROOT / "docs" / "USERGUIDE.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ecoSignalLab user guide as HTML and PDF")
    parser.add_argument("--out", default=str(ROOT / "USERGUIDE.pdf"), help="Final PDF output path")
    parser.add_argument(
        "--build-dir",
        default=str(ROOT / "docs" / "build" / "userguide"),
        help="Directory for intermediate HTML/PDF artifacts",
    )
    args = parser.parse_args()

    output_pdf = Path(args.out).resolve()
    report = build_docs(
        root=ROOT,
        output_root=Path(args.build_dir),
        formats={"html", "pdf"},
        title="ecoSignalLab User Guide",
        docs_files=[GUIDE_SOURCE],
    )
    guide_pdf = next((path for path in report.pdf_pages if path.name == "USERGUIDE.pdf"), None)
    if guide_pdf is None:
        raise RuntimeError("Documentation renderer did not produce the user-guide PDF.")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(guide_pdf, output_pdf)
    print(f"markdown: {GUIDE_SOURCE}")
    print(f"html: {report.output_root / 'html' / 'ecoSignalLab_docs.html'}")
    print(f"pdf: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
