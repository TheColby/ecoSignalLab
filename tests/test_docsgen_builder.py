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

    report = build_docs(root=root, output_root=root / "build", formats={"html"}, docs_files=[readme, ref])

    assert report.html_pages
    assert not report.pdf_pages

    readme_html = root / "build" / "html" / "README.html"
    assert readme_html.exists()

    rendered = readme_html.read_text(encoding="utf-8")
    assert 'class="mermaid"' in rendered
    assert 'href="docs/REF.html"' in rendered

    combined_html = root / "build" / "html" / "ecoSignalLab_docs.html"
    assert combined_html.exists()
