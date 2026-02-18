from pathlib import Path

from esl.viz.spawn import spawn_plot_paths


def test_spawn_plot_paths_with_limit_and_fake_launcher(tmp_path: Path) -> None:
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p3 = tmp_path / "c.png"
    for p in (p1, p2, p3):
        p.write_bytes(b"x")

    opened: list[str] = []

    def fake_launcher(path: Path) -> tuple[bool, str | None]:
        opened.append(str(path))
        return True, None

    summary = spawn_plot_paths([p1, p2, p3], limit=2, launcher=fake_launcher)

    assert summary["existing"] == 3
    assert summary["opened"] == 2
    assert summary["failed"] == 0
    assert summary["skipped_by_limit"] == 1
    assert len(opened) == 2


def test_spawn_plot_paths_handles_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "ok.png"
    missing = tmp_path / "missing.png"
    existing.write_bytes(b"x")

    def fake_launcher(path: Path) -> tuple[bool, str | None]:
        return False, "boom"

    summary = spawn_plot_paths([existing, missing], limit=10, launcher=fake_launcher)

    assert summary["requested"] == 2
    assert summary["existing"] == 1
    assert summary["opened"] == 0
    assert summary["failed"] == 1
