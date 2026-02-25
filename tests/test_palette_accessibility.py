from pathlib import Path

from esl.docsgen import build_docs


def test_docs_accessibility_features(tmp_path: Path) -> None:
    root = tmp_path
    readme = root / "README.md"
    readme.write_text("# Test Doc\nContent here.\n", encoding="utf-8")

    build_docs(root=root, output_root=root / "build", formats={"html"}, docs_files=[readme])

    readme_html = root / "build" / "html" / "README.html"
    rendered = readme_html.read_text(encoding="utf-8")

    # Check for skip link
    assert 'class="skip-link"' in rendered
    assert 'href="#main-content"' in rendered

    # Check for nav landmark
    assert '<nav aria-label="Global Documentation">' in rendered

    # Check for main content target
    assert 'id="main-content"' in rendered
    assert 'tabindex="-1"' in rendered

    # Check for aria-current in nav (it's the only page, so it should be current)
    assert 'aria-current="page"' in rendered
