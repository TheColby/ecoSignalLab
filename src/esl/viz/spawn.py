"""Spawn/open generated plot artifacts with the system default viewer."""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence


Launcher = Callable[[Path], tuple[bool, str | None]]


def _system_open(path: Path) -> tuple[bool, str | None]:
    """Open a file/path using the platform default application."""
    try:
        system = platform.system().lower()
        if system == "darwin":
            subprocess.Popen(
                ["open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, None
        if system == "linux":
            subprocess.Popen(
                ["xdg-open", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, None
        if system == "windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True, None
        return False, f"Unsupported platform for auto-open: {system}"
    except Exception as exc:
        return False, str(exc)


def spawn_plot_paths(
    paths: Sequence[str | Path],
    limit: int | None = 12,
    launcher: Launcher | None = None,
) -> dict[str, Any]:
    """Attempt to open generated plot artifacts; returns summary."""
    resolved = [Path(p).expanduser().resolve() for p in paths]
    existing = [p for p in resolved if p.exists()]

    if limit is not None and limit > 0:
        target = existing[:limit]
        skipped = max(0, len(existing) - len(target))
    else:
        target = existing
        skipped = 0

    run = launcher or _system_open
    opened: list[str] = []
    failed: list[dict[str, str]] = []

    for p in target:
        ok, err = run(p)
        if ok:
            opened.append(str(p))
        else:
            failed.append({"path": str(p), "error": err or "unknown"})

    return {
        "requested": len(paths),
        "existing": len(existing),
        "opened": len(opened),
        "failed": len(failed),
        "skipped_by_limit": skipped,
        "opened_paths": opened,
        "failures": failed,
    }
