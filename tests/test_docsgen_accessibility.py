from pathlib import Path

from esl.docsgen import build_docs


def test_docsgen_accessibility_features(tmp_path: Path) -> None:
    root = tmp_path
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    readme = root / "README.md"
    other = docs_dir / "OTHER.md"

    readme.write_text("# Welcome\n\nMain content here.\n", encoding="utf-8")
    other.write_text("# Other\n\n```python\nprint('hi')\n```\n", encoding="utf-8")

    build_docs(root=root, output_root=root / "build", formats={"html"}, docs_files=[readme, other])

    content = (root / "build" / "html" / "README.html").read_text(encoding="utf-8")

    assert 'class="skip-link"' in content
    assert 'href="#main-content"' in content
    assert 'nav aria-label="Main Documentation"' in content
    assert 'main id="main-content" tabindex="-1"' in content
    assert 'href="README.html" aria-current="page"' in content
    assert 'href="docs/OTHER.html"' in content
    assert 'href="docs/OTHER.html" aria-current="page"' not in content
    assert 'nav a[aria-current="page"]' in content
    assert 'font-weight: 700;' in content
    assert "copy-btn" in content
    assert "Copy code to clipboard" in content
