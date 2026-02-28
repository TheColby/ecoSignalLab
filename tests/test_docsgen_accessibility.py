from pathlib import Path
from esl.docsgen import build_docs

def test_docsgen_accessibility_features(tmp_path: Path) -> None:
    root = tmp_path
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    readme = root / "README.md"
    readme.write_text("# Welcome\n\nMain content here.\n", encoding="utf-8")

    other = docs_dir / "OTHER.md"
    other.write_text("# Other\n\nOther content.\n", encoding="utf-8")

    report = build_docs(root=root, output_root=root / "build", formats={"html"}, docs_files=[readme, other])

    readme_html = root / "build" / "html" / "README.html"
    content = readme_html.read_text(encoding="utf-8")

    # Check for "Skip to content" link
    assert 'class="skip-link"' in content
    assert 'href="#main-content"' in content
    assert 'Skip to content' in content

    # Check for ARIA landmarks
    assert 'nav aria-label="Main Documentation"' in content
    assert 'main id="main-content" tabindex="-1"' in content

    # Check for aria-current="page" on the active link
    # In README.html, the link to README.html should have aria-current="page"
    assert 'href="README.html" aria-current="page"' in content
    # The link to OTHER.html should NOT have aria-current="page"
    assert 'href="docs/OTHER.html"' in content
    assert 'href="docs/OTHER.html" aria-current="page"' not in content

    # Check for active link styling
    assert 'nav a[aria-current="page"]' in content
    assert 'font-weight: 700;' in content
