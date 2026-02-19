#!/usr/bin/env python3
"""Build esl documentation in HTML/PDF formats.

Usage:
  python scripts/build_docs.py --out docs/build --formats html,pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from esl.docsgen import build_docs


def _parse_formats(raw: str) -> set[str]:
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ecoSignalLab documentation")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--out", default="docs/build", help="Output directory")
    parser.add_argument("--formats", default="html,pdf", help="Comma-separated formats: html,pdf")
    parser.add_argument("--title", default="ecoSignalLab Documentation")
    args = parser.parse_args()

    report = build_docs(
        root=Path(args.root),
        output_root=Path(args.out),
        formats=_parse_formats(args.formats),
        title=args.title,
    )

    print(f"html: {len(report.html_pages)} files -> {report.output_root / 'html'}")
    print(f"pdf: {len(report.pdf_pages)} files -> {report.output_root / 'pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
