#!/usr/bin/env python3
"""Compare time-stretch algorithms with numerical KPIs on a real input file.

This script is intentionally input-driven: every KPI is computed from the actual
audio file passed by the user (and each method's rendered output), not from
synthetic placeholder values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, resample, stft


EPS = 1e-12


@dataclass(slots=True)
class MethodResult:
    method: str
    status: str
    output_wav: str | None
    runtime_s: float | None
    duration_s: float | None
    target_duration_s: float | None
    duration_error_ms: float | None
    realtime_factor: float | None
    peak_abs: float | None
    clipping_ratio: float | None
    rms_linear: float | None
    crest_factor_db: float | None
    spectral_centroid_hz_mean: float | None
    spectral_flatness_mean: float | None
    spectral_flux_mean: float | None
    transient_count: int | None
    transient_count_ratio: float | None
    centroid_delta_pct: float | None
    flatness_delta_pct: float | None
    kpi_score: float | None
    error: str | None = None


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x[:, None]
    return x


def _read_wav(path: Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = _ensure_2d(np.asarray(data, dtype=np.float32))
    if target_sr is None or int(target_sr) == int(sr):
        return x, int(sr)
    out_len = max(1, int(round(x.shape[0] * (float(target_sr) / float(sr)))))
    out = np.zeros((out_len, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = resample(x[:, ch], out_len).astype(np.float32)
    return out, int(target_sr)


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), sr, subtype="PCM_24")


def _mixdown(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x.astype(np.float64)
    return np.mean(x.astype(np.float64), axis=1)


def _atempo_filter_for_stretch(stretch_factor: float) -> str:
    # ffmpeg atempo uses tempo ratio; slower 2x duration => tempo 0.5
    tempo = 1.0 / float(stretch_factor)
    parts: list[float] = []
    while tempo < 0.5:
        parts.append(0.5)
        tempo /= 0.5
    while tempo > 2.0:
        parts.append(2.0)
        tempo /= 2.0
    parts.append(tempo)
    return ",".join(f"atempo={p:.6f}" for p in parts)


def _render_ffmpeg_atempo(input_path: Path, output_path: Path, stretch_factor: float, sr: int | None = None) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    filt = _atempo_filter_for_stretch(stretch_factor)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(input_path), "-filter:a", filt]
    if sr is not None:
        cmd += ["-ar", str(int(sr))]
    cmd.append(str(output_path))
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg atempo failed")


def _render_scipy_resample(x: np.ndarray, output_path: Path, sr: int, stretch_factor: float) -> None:
    out_len = max(1, int(round(x.shape[0] * float(stretch_factor))))
    out = np.zeros((out_len, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = resample(x[:, ch], out_len).astype(np.float32)
    _write_wav(output_path, out, sr)


def _render_librosa_phase_vocoder(x: np.ndarray, output_path: Path, sr: int, stretch_factor: float) -> None:
    try:
        import librosa
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("librosa is not installed (pip install -e .[features])") from exc

    rate = 1.0 / float(stretch_factor)
    chans: list[np.ndarray] = []
    max_len = 0
    for ch in range(x.shape[1]):
        y = librosa.effects.time_stretch(x[:, ch].astype(np.float64), rate=rate).astype(np.float32)
        chans.append(y)
        max_len = max(max_len, int(y.shape[0]))
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    for ch, y in enumerate(chans):
        out[: y.shape[0], ch] = y
    _write_wav(output_path, out, sr)


def _render_pvx_external(input_path: Path, output_path: Path, stretch_factor: float, template: str) -> None:
    if not template:
        raise RuntimeError("missing --pvx-cmd template")
    cmd = template.format(input=str(input_path), output=str(output_path), factor=f"{stretch_factor:.8f}")
    proc = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or proc.stdout.strip() or "pvx external command failed"
        raise RuntimeError(err)
    if not output_path.exists():
        raise RuntimeError("pvx command completed but output file was not created")


def _spectral_summary(x: np.ndarray, sr: int) -> dict[str, float]:
    mono = _mixdown(x)
    nperseg = min(1024, max(64, int(2 ** math.floor(math.log2(max(64, mono.size // 8))))))
    nperseg = max(64, nperseg)
    noverlap = max(0, int(nperseg * 0.75))
    _, _, z = stft(mono, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    mag = np.abs(z) + EPS
    if mag.size == 0 or mag.shape[1] == 0:
        return {
            "centroid_mean": 0.0,
            "flatness_mean": 0.0,
            "flux_mean": 0.0,
            "transient_count": 0.0,
        }
    freqs = np.linspace(0.0, sr / 2.0, mag.shape[0], dtype=np.float64)
    centroid = (freqs[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + EPS)
    flatness = np.exp(np.mean(np.log(mag), axis=0)) / (np.mean(mag, axis=0) + EPS)
    dmag = np.diff(mag, axis=1)
    flux = np.sum(np.maximum(dmag, 0.0), axis=0) if dmag.size else np.zeros(1, dtype=np.float64)
    zf = (flux - np.mean(flux)) / (np.std(flux) + EPS)
    peaks, _ = find_peaks(zf, height=1.0, distance=max(1, int(sr / max(1, nperseg * 4))))
    return {
        "centroid_mean": float(np.mean(centroid)),
        "flatness_mean": float(np.mean(flatness)),
        "flux_mean": float(np.mean(flux)),
        "transient_count": float(peaks.size),
    }


def _crest_factor_db(x: np.ndarray) -> float:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return float(20.0 * np.log10(max(peak, EPS) / max(rms, EPS)))


def _kpi_score(
    duration_error_ms: float,
    clipping_ratio: float,
    centroid_delta_pct: float,
    flatness_delta_pct: float,
    transient_count_ratio: float,
) -> float:
    # Heuristic score for quick ranking. Higher is better.
    penalty = 0.0
    penalty += abs(duration_error_ms) / 20.0
    penalty += clipping_ratio * 1200.0
    penalty += abs(centroid_delta_pct) * 0.25
    penalty += abs(flatness_delta_pct) * 0.20
    penalty += abs(1.0 - transient_count_ratio) * 18.0
    return float(max(0.0, 100.0 - penalty))


def _compute_result(
    method: str,
    output_wav: Path,
    runtime_s: float,
    target_duration_s: float,
    input_ref: dict[str, float],
) -> MethodResult:
    y, sr_out = _read_wav(output_wav, target_sr=None)
    dur = float(y.shape[0] / float(sr_out))
    dur_err_ms = float((dur - target_duration_s) * 1000.0)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    clip_ratio = float(np.mean(np.abs(y) >= 0.999)) if y.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
    spec = _spectral_summary(y, sr_out)
    trans_in = max(float(input_ref["transient_count"]), 1.0)
    trans_ratio = float(spec["transient_count"] / trans_in)
    centroid_delta = float(
        100.0 * (spec["centroid_mean"] - input_ref["centroid_mean"]) / max(abs(input_ref["centroid_mean"]), EPS)
    )
    flatness_delta = float(
        100.0 * (spec["flatness_mean"] - input_ref["flatness_mean"]) / max(abs(input_ref["flatness_mean"]), EPS)
    )
    score = _kpi_score(
        duration_error_ms=dur_err_ms,
        clipping_ratio=clip_ratio,
        centroid_delta_pct=centroid_delta,
        flatness_delta_pct=flatness_delta,
        transient_count_ratio=trans_ratio,
    )
    rt_factor = float(runtime_s / max(dur, EPS))
    return MethodResult(
        method=method,
        status="ok",
        output_wav=str(output_wav),
        runtime_s=float(runtime_s),
        duration_s=dur,
        target_duration_s=float(target_duration_s),
        duration_error_ms=dur_err_ms,
        realtime_factor=rt_factor,
        peak_abs=peak,
        clipping_ratio=clip_ratio,
        rms_linear=rms,
        crest_factor_db=_crest_factor_db(y),
        spectral_centroid_hz_mean=float(spec["centroid_mean"]),
        spectral_flatness_mean=float(spec["flatness_mean"]),
        spectral_flux_mean=float(spec["flux_mean"]),
        transient_count=int(spec["transient_count"]),
        transient_count_ratio=trans_ratio,
        centroid_delta_pct=centroid_delta,
        flatness_delta_pct=flatness_delta,
        kpi_score=score,
    )


def _write_csv(path: Path, rows: list[MethodResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(r) for r in rows]
    fieldnames = list(records[0].keys()) if records else list(asdict(MethodResult("", "", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None)).keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _parse_methods(raw: str) -> list[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare time-stretch algorithms with numerical KPIs on real audio input."
    )
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--out-dir", default="out/kpi_compare", help="Output directory")
    parser.add_argument(
        "--methods",
        default="ffmpeg_atempo,scipy_resample,librosa_phase_vocoder",
        help="Comma-separated methods: ffmpeg_atempo,scipy_resample,librosa_phase_vocoder,pvx_external",
    )
    parser.add_argument("--factor", type=float, default=2.0, help="Stretch factor (>1 means longer/slower)")
    parser.add_argument("--sample-rate", type=int, default=None, help="Optional analysis/render sample rate")
    parser.add_argument(
        "--pvx-cmd",
        default=None,
        help="External pvx command template, e.g. 'pvx stretch --in {input} --out {output} --factor {factor}'",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if float(args.factor) <= 0.0:
        raise ValueError("--factor must be > 0")

    out_dir = Path(args.out_dir)
    renders_dir = out_dir / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    x, sr = _read_wav(input_path, target_sr=args.sample_rate)
    target_dur = float((x.shape[0] * float(args.factor)) / float(sr))
    input_ref = _spectral_summary(x, sr)

    methods = _parse_methods(args.methods)
    rows: list[MethodResult] = []

    for method in methods:
        out_wav = renders_dir / f"{method}.wav"
        t0 = time.perf_counter()
        try:
            if method == "ffmpeg_atempo":
                _render_ffmpeg_atempo(input_path=input_path, output_path=out_wav, stretch_factor=float(args.factor), sr=args.sample_rate)
            elif method == "scipy_resample":
                _render_scipy_resample(x=x, output_path=out_wav, sr=sr, stretch_factor=float(args.factor))
            elif method == "librosa_phase_vocoder":
                _render_librosa_phase_vocoder(x=x, output_path=out_wav, sr=sr, stretch_factor=float(args.factor))
            elif method == "pvx_external":
                _render_pvx_external(
                    input_path=input_path,
                    output_path=out_wav,
                    stretch_factor=float(args.factor),
                    template=str(args.pvx_cmd or ""),
                )
            else:
                raise RuntimeError(f"unknown method: {method}")
            runtime = time.perf_counter() - t0
            rows.append(
                _compute_result(
                    method=method,
                    output_wav=out_wav,
                    runtime_s=runtime,
                    target_duration_s=target_dur,
                    input_ref=input_ref,
                )
            )
        except Exception as exc:
            rows.append(
                MethodResult(
                    method=method,
                    status="error",
                    output_wav=str(out_wav) if out_wav.exists() else None,
                    runtime_s=None,
                    duration_s=None,
                    target_duration_s=target_dur,
                    duration_error_ms=None,
                    realtime_factor=None,
                    peak_abs=None,
                    clipping_ratio=None,
                    rms_linear=None,
                    crest_factor_db=None,
                    spectral_centroid_hz_mean=None,
                    spectral_flatness_mean=None,
                    spectral_flux_mean=None,
                    transient_count=None,
                    transient_count_ratio=None,
                    centroid_delta_pct=None,
                    flatness_delta_pct=None,
                    kpi_score=None,
                    error=str(exc),
                )
            )

    rows_sorted = sorted(
        rows,
        key=lambda r: (0 if r.status == "ok" else 1, -(r.kpi_score or -1.0)),
    )

    json_path = out_dir / "kpi_summary.json"
    csv_path = out_dir / "kpi_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_payload: dict[str, Any] = {
        "input": str(input_path),
        "sample_rate": sr,
        "stretch_factor": float(args.factor),
        "target_duration_s": target_dur,
        "methods": methods,
        "input_reference": input_ref,
        "results": [asdict(r) for r in rows_sorted],
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows_sorted)

    print(f"input: {input_path}")
    print(f"target_duration_s: {target_dur:.6f}")
    print(f"results_json: {json_path}")
    print(f"results_csv: {csv_path}")
    print("")
    print("method ranking (higher kpi_score is better):")
    for r in rows_sorted:
        if r.status != "ok":
            print(f"- {r.method:24s} ERROR: {r.error}")
            continue
        print(
            f"- {r.method:24s} "
            f"kpi_score={r.kpi_score:6.2f} "
            f"dur_err_ms={r.duration_error_ms:8.2f} "
            f"rtf={r.realtime_factor:6.3f} "
            f"clip={r.clipping_ratio:8.6f}"
        )

    return 0 if any(r.status == "ok" for r in rows_sorted) else 1


if __name__ == "__main__":
    raise SystemExit(main())

