from pathlib import Path

from esl.docsgen import build_docs


def test_build_docs_html_with_mermaid_and_links(tmp_path: Path) -> None:
    root = tmp_path
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    readme = root / "README.md"
    ref = docs_dir / "REF.md"

    readme.write_text(
        """
# Test Doc

See [Reference](docs/REF.md).

```mermaid
graph TD
  A[Start] --> B[End]
```
""".strip()
        + "\n",
        encoding="utf-8",
    )
    ref.write_text("# Ref\n\nHello.\n", encoding="utf-8")

    report = build_docs(
        root=root,
        output_root=root / "build",
        formats={"html"},
        title="ecoSignalLab Textbook",
        docs_files=[readme, ref],
    )

    assert report.html_pages
    assert not report.pdf_pages

    readme_html = root / "build" / "html" / "README.html"
    assert readme_html.exists()

    rendered = readme_html.read_text(encoding="utf-8")
    assert 'class="mermaid"' in rendered
    assert 'href="docs/REF.html"' in rendered
    assert '"TeX Gyre Schola"' in rendered
    assert 'assets/fonts/texgyreschola-regular.otf' in rendered
    assert "--code-font" in rendered
    assert "orphans: 3" in rendered
    assert "widows: 3" in rendered
    assert "break-after: avoid-page" in rendered
    assert 'class="textbook-edition"' in rendered
    assert "body.textbook-edition" in rendered
    assert (root / "build" / "html" / "assets" / "fonts" / "texgyreschola-regular.otf").exists()
    assert (root / "build" / "html" / "assets" / "fonts" / "GUST-FONT-LICENSE.txt").exists()

    combined_html = root / "build" / "html" / "ecoSignalLab_docs.html"
    assert combined_html.exists()


def test_build_docs_auto_visual_outline_and_math_runtime(tmp_path: Path) -> None:
    root = tmp_path
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    readme = root / "README.md"
    readme.write_text(
        """
# Math Doc

## Signal Model

This page has TeX math:

$$
E = \\sum_{n=0}^{N-1} x[n]^2
$$
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = build_docs(root=root, output_root=root / "build", formats={"html"}, docs_files=[readme])
    assert report.html_pages

    rendered = (root / "build" / "html" / "README.html").read_text(encoding="utf-8")
    assert "MathJax" in rendered
    assert "tex-svg.js" in rendered
    assert 'class="equation-preview"' in rendered
    assert "this is an accumulation" in rendered
    assert 'class="mermaid"' in rendered
    assert "Visual Outline (Auto-generated)" in rendered
    assert "\\sum_{n=0}^{N-1}" in rendered
