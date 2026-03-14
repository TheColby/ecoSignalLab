"""Environment and input diagnostics for first-run troubleshooting.

References:
- FFmpeg project documentation: https://ffmpeg.org/documentation.html
- SoundFile API and backend behavior: https://python-soundfile.readthedocs.io/
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

from esl import __version__
from esl.core.audio import (
    SUPPORTED_COMPRESSED_EXT,
    detect_signal_layout,
    probe_audio_metadata,
)
from esl.ml import device_resolution_dict, resolve_compute_device


@dataclass(slots=True)
class DoctorConfig:
    """Configuration for environment and input diagnostics."""

    input_path: Path | None = None
    requested_device: str = "auto"
    strict: bool = False


def _dist_version(name: str) -> str | None:
    try:
        return str(version(name))
    except PackageNotFoundError:
        return None


def _module_status(module_name: str, dist_name: str | None = None) -> dict[str, Any]:
    try:
        mod = import_module(module_name)
    except Exception:
        return {"installed": False, "version": None}
    mod_ver = getattr(mod, "__version__", None)
    return {
        "installed": True,
        "version": str(mod_ver) if mod_ver is not None else _dist_version(dist_name or module_name),
    }


def _tool_status(cmd: str) -> dict[str, Any]:
    exe = shutil.which(cmd)
    if exe is None:
        return {"available": False, "path": None, "version": None}
    try:
        proc = subprocess.run([cmd, "-version"], capture_output=True, text=True, check=False)
        first = (proc.stdout or "").splitlines()
        ver = first[0].strip() if proc.returncode == 0 and first else None
    except Exception:
        ver = None
    return {"available": True, "path": exe, "version": ver}


def _format_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total = int(round(float(seconds)))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _input_recommendations(meta: dict[str, Any], compressed_ready: bool) -> list[str]:
    path = str(meta["path"])
    duration_s = float(meta["duration_s"]) if meta.get("duration_s") is not None else None
    channels = int(meta.get("channels", 1))
    sample_rate = int(meta.get("sample_rate", 48_000))
    size_gb = float(meta.get("size_gb", 0.0))

    recs = [f'esl simple "{path}"', f'esl analyze "{path}" --out-dir out --plot --json out/{Path(path).stem}.json']
    if duration_s is not None and (duration_s >= 3600.0 or size_gb >= 4.0 or channels > 2 or sample_rate >= 96_000):
        recs = [
            f'esl doctor "{path}"',
            f'esl stream "{path}" --out stream_out --frame-seconds 1 --hop-seconds 0.5 --chunk-minutes 10 --metrics spl_a_db,ndsi,novelty_curve',
            f'esl moments extract "{path}" --out out/moments --single --rank-metric novelty_curve --chunk-minutes 10 --event-window 8',
        ]
    if channels > 2:
        recs.append(f'esl spatial analyze "{path}" --out-dir spatial_out')
    if Path(path).suffix.lower() in SUPPORTED_COMPRESSED_EXT and not compressed_ready:
        recs.insert(0, "Install ffmpeg and ffprobe before analyzing this compressed file.")
    return recs


def run_doctor(cfg: DoctorConfig) -> dict[str, Any]:
    """Run environment and optional input diagnostics."""
    ffmpeg = _tool_status("ffmpeg")
    ffprobe = _tool_status("ffprobe")
    soundfile = _module_status("soundfile", "soundfile")
    plotly = _module_status("plotly", "plotly")
    librosa = _module_status("librosa", "librosa")
    pyarrow = _module_status("pyarrow", "pyarrow")
    playwright = _module_status("playwright", "playwright")
    device_info = resolve_compute_device(cfg.requested_device, strict=False)

    blockers: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = [
        ".venv/bin/python -m esl doctor",
        "esl quickstart --goal analyze",
    ]

    if not ffmpeg["available"] or not ffprobe["available"]:
        warnings.append("FFmpeg/ffprobe not fully available; compressed formats may fail to decode.")
        recommendations.append("Install FFmpeg, then rerun `esl doctor`.")

    input_payload: dict[str, Any] | None = None
    if cfg.input_path is not None:
        ext = cfg.input_path.suffix.lower()
        size_gb = float(cfg.input_path.stat().st_size / 1_000_000_000.0)
        try:
            meta = probe_audio_metadata(cfg.input_path)
            meta["layout_hint"] = detect_signal_layout(int(meta["channels"]), str(cfg.input_path))
            meta["duration_human"] = _format_seconds(meta.get("duration_s"))
            input_payload = meta
        except Exception as exc:
            input_payload = {
                "path": str(cfg.input_path.resolve()),
                "name": cfg.input_path.name,
                "extension": ext,
                "size_gb": size_gb,
                "probe_error": str(exc),
            }
            warnings.append(f"Could not fully probe input metadata: {exc}")
            meta = input_payload

        if ext in SUPPORTED_COMPRESSED_EXT and (not ffmpeg["available"] or not ffprobe["available"]):
            blockers.append("Compressed input requires both ffmpeg and ffprobe on PATH.")
        if ext == ".wav" and float(meta.get("size_gb", size_gb)) >= 4.0:
            warnings.append("Large WAV is near or beyond the classic 4 GB limit; RF64 is safer.")
        if meta.get("duration_s") is not None and float(meta["duration_s"]) >= 3600.0:
            warnings.append("Long-duration input detected; prefer `stream` or `moments extract` before full sweeps.")
        recommendations.extend(
            _input_recommendations(meta, compressed_ready=bool(ffmpeg["available"] and ffprobe["available"]))
        )

    status = "ok"
    if blockers:
        status = "fail"
    elif warnings:
        status = "warn"

    deduped_recommendations = list(dict.fromkeys(recommendations))

    return {
        "status": status,
        "esl_version": __version__,
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "python_executable": shutil.which("python3") or shutil.which("python") or "python",
        },
        "core_dependencies": {
            "soundfile": soundfile,
            "ffmpeg": ffmpeg,
            "ffprobe": ffprobe,
        },
        "optional_dependencies": {
            "plotly": plotly,
            "librosa": librosa,
            "pyarrow": pyarrow,
            "playwright": playwright,
        },
        "device": device_resolution_dict(device_info),
        "input": input_payload,
        "blockers": blockers,
        "warnings": warnings,
        "recommendations": deduped_recommendations,
        "strict": bool(cfg.strict),
    }
