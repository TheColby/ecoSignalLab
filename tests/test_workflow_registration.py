"""All additive workflows must be reachable through the public CLI."""

from __future__ import annotations

import pytest

from esl.cli.main import main


@pytest.mark.parametrize(
    "command",
    [
        "review",
        "monitor",
        "uncertainty",
        "spatial-timeline",
        "qa",
        "index",
        "embeddings",
        "protocol",
        "privacy",
        "archive-jobs",
    ],
)
def test_workflow_help_is_reachable(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
    assert command in capsys.readouterr().out


def test_moments_review_alias(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["moments", "review", "--help"])
    assert exc.value.code == 0
    assert "manifest" in capsys.readouterr().out
