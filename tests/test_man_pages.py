from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_man_page_files_exist() -> None:
    expected = [
        "esl.1",
        "esl-analyze.1",
        "esl-batch.1",
        "esl-moments.1",
        "esl-similar.1",
        "esl-calibrate.1",
    ]
    for name in expected:
        assert (ROOT / "man" / "man1" / name).exists(), name


def test_pyproject_installs_man_pages() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.data-files]" in text
    assert '"share/man/man1"' in text
    assert '"man/man1/esl.1"' in text
    assert '"man/man1/esl-analyze.1"' in text
    assert '"man/man1/esl-batch.1"' in text
    assert '"man/man1/esl-moments.1"' in text
    assert '"man/man1/esl-similar.1"' in text
    assert '"man/man1/esl-calibrate.1"' in text


def test_install_script_copies_man_pages() -> None:
    text = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    assert "share/man/man1" in text
    assert "cp man/man1/*.1" in text
    assert "MANPATH=" in text
