"""Interesting-moment extraction from streaming alert reports."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from esl.core.audio import read_audio
from esl.core.config import CalibrationProfile
from esl.core.streaming import StreamRunConfig, run_stream_analysis


@dataclass(slots=True)
class MomentsExtractConfig:
    input_path: Path
    output_dir: Path
    rules_path: str | None = None
    metrics: list[str] | None = None
    calibration: CalibrationProfile | None = None
    frame_size: int = 2048
    hop_size: int = 512
    sample_rate: int | None = None
    chunk_size: int = 131072
    seed: int = 42
    max_chunks: int | None = None
    stream_report_path: str | None = None
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0
    merge_gap_s: float = 2.0
    min_alerts_per_chunk: int = 1
    max_clips: int | None = None
    csv_out: str | None = None
    clips_dir: str | None = None
    report_out: str | None = None


def _sec_to_hms(value_s: float) -> str:
    total_ms = int(round(max(0.0, value_s) * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, check=False)
    except FileNotFoundError:
        return False
    return proc.returncode == 0


def _codec_from_subtype(subtype: str | None) -> str:
    st = str(subtype or "").upper()
    if st == "PCM_16":
        return "pcm_s16le"
    if st == "PCM_24":
        return "pcm_s24le"
    if st == "PCM_32":
        return "pcm_s32le"
    if st == "FLOAT":
        return "pcm_f32le"
    if st == "DOUBLE":
        return "pcm_f64le"
    return "pcm_s24le"


def _clip_with_soundfile(
    input_path: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    target_sr: int | None,
) -> None:
    with sf.SoundFile(str(input_path), mode="r") as f:
        src_sr = int(f.samplerate)
        start = max(0, int(round(start_s * src_sr)))
        end = max(start, int(round(end_s * src_sr)))
        f.seek(start)
        frames = end - start
        data = f.read(frames=frames, dtype="float32", always_2d=True)
        if target_sr and target_sr != src_sr:
            # Fallback path: use read_audio for simple resampling when soundfile-only cut is used.
            buf = read_audio(input_path, target_sr=target_sr)
            sr = int(buf.sample_rate)
            s2 = max(0, int(round(start_s * sr)))
            e2 = max(s2, int(round(end_s * sr)))
            data = buf.samples[s2:e2]
            sf.write(str(output_path), data, sr, subtype="PCM_24")
            return
        sf.write(str(output_path), data, src_sr, subtype=f.subtype)


def _clip_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    codec: str,
    sample_rate: int | None,
    channels: int | None,
) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(input_path),
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-map",
        "0:a:0",
        "-c:a",
        codec,
    ]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    cmd.append(str(output_path))
    proc = subprocess.run(cmd, capture_output=True, check=False)
    return proc.returncode == 0


def _load_stream_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stream report not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid stream report structure: {path}")
    return payload


def _collect_windows(
    chunks: list[dict[str, Any]],
    pre_roll_s: float,
    post_roll_s: float,
    merge_gap_s: float,
    min_alerts_per_chunk: int,
    duration_s: float,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for ch in chunks:
        alerts = ch.get("alerts", [])
        if not isinstance(alerts, list) or len(alerts) < int(min_alerts_per_chunk):
            continue
        start = float(ch.get("start_s", 0.0))
        end = float(ch.get("end_s", start))
        s = max(0.0, start - float(pre_roll_s))
        e = min(float(duration_s), end + float(post_roll_s))
        metrics: set[str] = set()
        for a in alerts:
            if isinstance(a, dict) and "metric" in a:
                metrics.add(str(a["metric"]))
        windows.append(
            {
                "start_s": s,
                "end_s": e,
                "alerts": len(alerts),
                "metrics": metrics,
                "chunk_indices": [int(ch.get("index", -1))],
            }
        )
    if not windows:
        return []

    windows.sort(key=lambda x: float(x["start_s"]))
    merged: list[dict[str, Any]] = [windows[0]]
    for w in windows[1:]:
        prev = merged[-1]
        if float(w["start_s"]) <= float(prev["end_s"]) + float(merge_gap_s):
            prev["end_s"] = max(float(prev["end_s"]), float(w["end_s"]))
            prev["alerts"] = int(prev["alerts"]) + int(w["alerts"])
            prev["metrics"] = set(prev["metrics"]) | set(w["metrics"])
            prev["chunk_indices"] = list(prev["chunk_indices"]) + list(w["chunk_indices"])
        else:
            merged.append(w)
    return merged


def run_moments_extract(cfg: MomentsExtractConfig) -> tuple[Path, dict[str, Any]]:
    """Extract interesting moments and export clips + timestamp CSV."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(cfg.clips_dir) if cfg.clips_dir else (cfg.output_dir / "clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(cfg.csv_out) if cfg.csv_out else (cfg.output_dir / "moments.csv")
    report_path = Path(cfg.report_out) if cfg.report_out else (cfg.output_dir / "moments_report.json")

    if cfg.stream_report_path:
        stream_report_file = Path(cfg.stream_report_path)
        stream_report = _load_stream_report(stream_report_file)
    else:
        stream_cfg = StreamRunConfig(
            input_path=cfg.input_path,
            output_dir=cfg.output_dir / "stream_pass",
            metrics=list(cfg.metrics or []),
            frame_size=cfg.frame_size,
            hop_size=cfg.hop_size,
            sample_rate=cfg.sample_rate,
            chunk_size=cfg.chunk_size,
            calibration=cfg.calibration,
            seed=cfg.seed,
            rules_path=cfg.rules_path,
            max_chunks=cfg.max_chunks,
        )
        stream_report_file, stream_report = run_stream_analysis(stream_cfg)

    chunks = stream_report.get("chunks", [])
    if not isinstance(chunks, list):
        chunks = []
    duration_s = float(stream_report.get("chunks", [{}])[-1].get("end_s", 0.0)) if chunks else 0.0

    windows = _collect_windows(
        chunks=chunks,
        pre_roll_s=cfg.pre_roll_s,
        post_roll_s=cfg.post_roll_s,
        merge_gap_s=cfg.merge_gap_s,
        min_alerts_per_chunk=cfg.min_alerts_per_chunk,
        duration_s=duration_s,
    )
    if cfg.max_clips is not None and cfg.max_clips >= 0:
        windows = windows[: int(cfg.max_clips)]

    info = sf.info(str(cfg.input_path))
    codec = _codec_from_subtype(info.subtype)
    out_sr = int(cfg.sample_rate) if cfg.sample_rate is not None else int(info.samplerate)
    out_ch = int(info.channels)
    use_ffmpeg = _ffmpeg_available()

    rows: list[dict[str, Any]] = []
    for idx, win in enumerate(windows, start=1):
        clip_id = f"moment_{idx:04d}"
        out_wav = clips_dir / f"{clip_id}.wav"
        start_s = float(win["start_s"])
        end_s = float(win["end_s"])
        duration = max(0.0, end_s - start_s)
        wrote = False
        if use_ffmpeg:
            wrote = _clip_with_ffmpeg(
                input_path=cfg.input_path,
                output_path=out_wav,
                start_s=start_s,
                end_s=end_s,
                codec=codec,
                sample_rate=out_sr,
                channels=out_ch,
            )
        if not wrote:
            _clip_with_soundfile(
                input_path=cfg.input_path,
                output_path=out_wav,
                start_s=start_s,
                end_s=end_s,
                target_sr=cfg.sample_rate,
            )

        rows.append(
            {
                "clip_id": clip_id,
                "start_s": f"{start_s:.3f}",
                "end_s": f"{end_s:.3f}",
                "start_hms": _sec_to_hms(start_s),
                "end_hms": _sec_to_hms(end_s),
                "duration_s": f"{duration:.3f}",
                "alerts": int(win["alerts"]),
                "metrics": ";".join(sorted(str(x) for x in set(win["metrics"]))),
                "chunk_indices": ";".join(str(x) for x in win["chunk_indices"]),
                "wav_path": str(out_wav),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "start_s",
                "end_s",
                "start_hms",
                "end_hms",
                "duration_s",
                "alerts",
                "metrics",
                "chunk_indices",
                "wav_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(cfg.input_path.resolve()),
        "stream_report_path": str(Path(stream_report_file).resolve()),
        "rules_path": cfg.rules_path,
        "sample_rate": out_sr,
        "channels": out_ch,
        "codec": codec,
        "pre_roll_s": float(cfg.pre_roll_s),
        "post_roll_s": float(cfg.post_roll_s),
        "merge_gap_s": float(cfg.merge_gap_s),
        "min_alerts_per_chunk": int(cfg.min_alerts_per_chunk),
        "max_clips": cfg.max_clips,
        "windows_selected": len(windows),
        "clips_written": len(rows),
        "csv_path": str(csv_path),
        "clips_dir": str(clips_dir),
        "ffmpeg_used": bool(use_ffmpeg),
        "source_stream_summary": {
            "chunks_processed": stream_report.get("chunks_processed"),
            "alert_count": stream_report.get("alert_count"),
            "metrics": stream_report.get("metrics"),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path, report
