"""Higher-level soundscape insight workflows.

These routines are intentionally transparent, deterministic baselines built on
the public esl feature and metric stack. They are not secret-sauce models.

References:
- Foote (2000), automatic audio segmentation with self-similarity/novelty.
- Sueur et al. (2008), acoustic indices for biodiversity assessment.
- Kasten et al. (2012), acoustic ecology sensor-network summaries.
- Slabbekoorn & den Boer-Visser (2006), urban noise and vocal adjustment.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, stft

from esl import __version__
from esl.core.audio import iter_supported_files, read_audio
from esl.core.similarity import (
    SUPPORTED_PATTERNS,
    SimilaritySearchConfig,
    _aggregate_feature_vector,
    run_similarity_search,
)
from esl.ml.device import device_resolution_dict, resolve_compute_device
from esl.schema import SCHEMA_VERSION
from esl.viz.feature_vectors import extract_feature_vectors

EPS = 1e-12


@dataclass(slots=True)
class InsightPaths:
    """Common return type for insight workflows."""

    primary: Path
    report: dict[str, Any]


def _base_report(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "esl_version": __version__,
        "insight_kind": kind,
        "config": config,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    out = (x - mean) / np.maximum(std, EPS)
    return np.where(np.isfinite(out), out, 0.0)


def _frame_starts(num_samples: int, frame_size: int, hop_size: int) -> list[int]:
    if num_samples <= 0:
        return []
    if num_samples < frame_size:
        return [0]
    return list(range(0, num_samples - frame_size + 1, hop_size))


def _metric_means_from_report(payload: dict[str, Any]) -> dict[str, float]:
    weighted = payload.get("weighted_metric_means")
    if isinstance(weighted, dict):
        return {str(k): float(v) for k, v in weighted.items() if isinstance(v, (int, float))}

    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        out: dict[str, float] = {}
        for name, body in metrics.items():
            if not isinstance(body, dict):
                continue
            summary = body.get("summary")
            if isinstance(summary, dict) and isinstance(summary.get("mean"), (int, float)):
                out[str(name)] = float(summary["mean"])
        return out
    return {}


def run_scene_changes(
    input_path: Path,
    output_dir: Path,
    *,
    frame_size: int = 2048,
    hop_size: int = 512,
    sample_rate: int | None = None,
    threshold_z: float = 1.5,
    max_changes: int | None = None,
    feature_set: str = "auto",
) -> InsightPaths:
    """Detect acoustic scene-change candidates from adjacent feature distances."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fv = extract_feature_vectors(
        input_path,
        feature_set=feature_set,
        frame_size=frame_size,
        hop_size=hop_size,
        sample_rate=sample_rate,
    )
    x = _zscore_columns(fv.matrix)
    if x.shape[0] <= 1:
        scores = np.zeros((x.shape[0],), dtype=np.float64)
    else:
        distances = np.linalg.norm(np.diff(x, axis=0), axis=1)
        scores = np.concatenate([[0.0], distances])
    score_mean = float(np.mean(scores)) if scores.size else 0.0
    score_std = float(np.std(scores)) if scores.size else 0.0
    cutoff = score_mean + float(threshold_z) * score_std
    peaks, _ = find_peaks(scores, height=cutoff if score_std > 0 else score_mean)
    order = sorted(peaks.tolist(), key=lambda i: float(scores[i]), reverse=True)
    if max_changes is not None:
        order = order[: max(0, int(max_changes))]
    order = sorted(order)

    rows = [
        {
            "rank": rank,
            "frame_index": int(i),
            "time_s": float(fv.times_s[i])
            if i < fv.times_s.size
            else float(i * hop_size / fv.sample_rate),
            "change_score": float(scores[i]),
            "threshold": float(cutoff),
        }
        for rank, i in enumerate(order, start=1)
    ]

    csv_path = output_dir / "scene_changes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "frame_index", "time_s", "change_score", "threshold"]
        )
        writer.writeheader()
        writer.writerows(rows)

    report = _base_report(
        "scene_changes",
        {
            "input_path": str(input_path),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "sample_rate": sample_rate,
            "threshold_z": float(threshold_z),
            "max_changes": max_changes,
            "feature_set": feature_set,
        },
    )
    report.update(
        {
            "method": "adjacent_zscored_feature_distance",
            "feature_backend": fv.backend,
            "num_frames": int(fv.matrix.shape[0]),
            "num_features": int(fv.matrix.shape[1]),
            "threshold": float(cutoff),
            "changes": rows,
            "csv_path": str(csv_path),
        }
    )
    return InsightPaths(_write_json(output_dir / "scene_changes.json", report), report)


def run_calmness(
    input_path: Path,
    output_path: Path,
    *,
    frame_size: int = 2048,
    hop_size: int = 512,
    sample_rate: int | None = None,
) -> InsightPaths:
    """Estimate calmness, chaos, and diversity from level and spectral dynamics."""
    fv = extract_feature_vectors(
        input_path,
        feature_set="core",
        frame_size=frame_size,
        hop_size=hop_size,
        sample_rate=sample_rate,
    )
    names = {name: i for i, name in enumerate(fv.feature_names)}
    rms = (
        fv.matrix[:, names.get("rms_linear", 0)]
        if fv.matrix.size
        else np.zeros((1,), dtype=np.float64)
    )
    rms_db = 20.0 * np.log10(np.maximum(rms, EPS))
    flux = (
        fv.matrix[:, names.get("spectral_flux", 0)]
        if "spectral_flux" in names
        else np.zeros_like(rms)
    )
    flux_norm = flux / max(float(np.nanpercentile(np.abs(flux), 95)), EPS)

    mel_cols = [i for i, name in enumerate(fv.feature_names) if name.startswith("log_mel_")]
    mel_energy = np.maximum(fv.matrix[:, mel_cols], 0.0) if mel_cols else np.maximum(fv.matrix, 0.0)
    band_profile = np.sum(mel_energy, axis=0)
    probs = band_profile / max(float(np.sum(band_profile)), EPS)
    diversity = float(
        -np.sum(probs * np.log2(np.maximum(probs, EPS))) / max(np.log2(max(probs.size, 2)), EPS)
    )
    level_variability = float(np.std(rms_db) / 60.0)
    flux_activity = float(np.mean(np.abs(flux_norm)))
    chaos = float(max(0.0, level_variability + flux_activity))
    calmness = float(1.0 / (1.0 + chaos))

    report = _base_report(
        "calmness_chaos_diversity",
        {
            "input_path": str(input_path),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "sample_rate": sample_rate,
        },
    )
    report.update(
        {
            "method": "rms_variability_plus_spectral_flux_entropy",
            "units": {
                "calmness_score": "unitless_0_to_1_heuristic",
                "chaos_score": "unitless_positive_heuristic",
                "diversity_score": "unitless_0_to_1_entropy",
            },
            "calmness_score": calmness,
            "chaos_score": chaos,
            "diversity_score": diversity,
            "rms_dbfs_mean": float(np.mean(rms_db)),
            "rms_dbfs_std": float(np.std(rms_db)),
            "spectral_flux_mean_normalized": flux_activity,
        }
    )
    return InsightPaths(_write_json(output_path, report), report)


def run_spatial_timeline(
    input_path: Path,
    output_dir: Path,
    *,
    frame_size: int = 2048,
    hop_size: int = 512,
    sample_rate: int | None = None,
) -> InsightPaths:
    """Write a lightweight frame-wise multichannel/spatial activity timeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    buf = read_audio(input_path, target_sr=sample_rate)
    samples = np.asarray(buf.samples, dtype=np.float64)
    starts = _frame_starts(samples.shape[0], frame_size, hop_size)
    rows: list[dict[str, Any]] = []
    maxlag = max(1, int(round(0.001 * buf.sample_rate)))
    for idx, start in enumerate(starts):
        frame = samples[start : start + frame_size]
        if frame.shape[0] < frame_size:
            pad = np.zeros((frame_size - frame.shape[0], frame.shape[1]), dtype=np.float64)
            frame = np.vstack([frame, pad])
        energy = np.mean(np.square(frame), axis=0)
        dominant = int(np.argmax(energy)) if energy.size else 0
        ild_db = None
        itd_s = None
        coherence = None
        azimuth_proxy_deg = None
        if frame.shape[1] >= 2:
            left = frame[:, 0] - np.mean(frame[:, 0])
            right = frame[:, 1] - np.mean(frame[:, 1])
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            coherence = float(np.dot(left, right) / denom) if denom > EPS else 0.0
            ild_db = float(10.0 * np.log10((energy[0] + EPS) / (energy[1] + EPS)))
            corr = np.correlate(left, right, mode="full")
            mid = corr.size // 2
            window = corr[mid - maxlag : mid + maxlag + 1]
            lag = int(np.argmax(window) - maxlag)
            itd_s = float(lag / buf.sample_rate)
            azimuth_proxy_deg = float(np.clip((lag / maxlag) * 90.0, -90.0, 90.0))
        rows.append(
            {
                "frame_index": idx,
                "time_s": float(start / buf.sample_rate),
                "dominant_channel": dominant,
                "total_energy": float(np.sum(energy)),
                "interchannel_coherence": coherence,
                "ild_db": ild_db,
                "itd_s": itd_s,
                "azimuth_proxy_deg": azimuth_proxy_deg,
            }
        )

    csv_path = output_dir / "spatial_timeline.csv"
    fields = [
        "frame_index",
        "time_s",
        "dominant_channel",
        "total_energy",
        "interchannel_coherence",
        "ild_db",
        "itd_s",
        "azimuth_proxy_deg",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = _base_report(
        "spatial_event_timeline",
        {
            "input_path": str(input_path),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "sample_rate": sample_rate,
        },
    )
    report.update(
        {
            "method": "frame_energy_interchannel_correlation_itd_proxy",
            "channels": int(buf.channels),
            "source_sample_rate": int(buf.sample_rate),
            "frames": len(rows),
            "csv_path": str(csv_path),
            "summary": {
                "mean_interchannel_coherence": float(
                    np.nanmean(
                        [
                            r["interchannel_coherence"]
                            for r in rows
                            if r["interchannel_coherence"] is not None
                        ]
                    )
                )
                if any(r["interchannel_coherence"] is not None for r in rows)
                else None,
            },
        }
    )
    return InsightPaths(_write_json(output_dir / "spatial_timeline.json", report), report)


def _parse_bands(raw: str) -> list[tuple[str, float, float]]:
    bands: list[tuple[str, float, float]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        name, _, rng = item.partition(":")
        lo, _, hi = rng.partition("-")
        bands.append((name.strip(), float(lo), float(hi)))
    if not bands:
        raise ValueError("At least one band is required, e.g. anthro:20-1000,bio:2000-8000")
    return bands


def run_bio_occupancy(
    input_path: Path,
    output_dir: Path,
    *,
    bands: str = "anthro:20-1000,bio:2000-8000",
    frame_size: int = 4096,
    hop_size: int = 2048,
    sample_rate: int | None = None,
    threshold_ratio: float = 0.2,
) -> InsightPaths:
    """Estimate per-band acoustic occupancy over time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed = _parse_bands(bands)
    buf = read_audio(input_path, target_sr=sample_rate)
    mono = _mono(buf.samples)
    freqs, times, z = stft(
        mono,
        fs=buf.sample_rate,
        nperseg=frame_size,
        noverlap=max(0, frame_size - hop_size),
        boundary=None,
    )
    power = np.square(np.abs(z))
    total = np.sum(power, axis=0) + EPS
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for name, lo, hi in parsed:
        mask = (freqs >= lo) & (freqs <= hi)
        band_power = np.sum(power[mask, :], axis=0) if np.any(mask) else np.zeros_like(total)
        ratio = band_power / total
        occupied = ratio >= float(threshold_ratio)
        summary[name] = {
            "low_hz": float(lo),
            "high_hz": float(hi),
            "occupancy_fraction": float(np.mean(occupied)) if occupied.size else 0.0,
            "mean_energy_ratio": float(np.mean(ratio)) if ratio.size else 0.0,
        }
        for i, t in enumerate(times):
            while len(rows) <= i:
                rows.append({"frame_index": i, "time_s": float(t)})
            rows[i][f"{name}_energy_ratio"] = float(ratio[i])
            rows[i][f"{name}_occupied"] = bool(occupied[i])

    csv_path = output_dir / "bio_occupancy.csv"
    fields = ["frame_index", "time_s"]
    for name, _, _ in parsed:
        fields.extend([f"{name}_energy_ratio", f"{name}_occupied"])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = _base_report(
        "bioacoustic_occupancy",
        {
            "input_path": str(input_path),
            "bands": bands,
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "sample_rate": sample_rate,
            "threshold_ratio": float(threshold_ratio),
        },
    )
    report.update(
        {"method": "stft_band_energy_ratio_threshold", "bands": summary, "csv_path": str(csv_path)}
    )
    return InsightPaths(_write_json(output_dir / "bio_occupancy.json", report), report)


def run_archive_drift(
    baseline_report: Path, candidate_report: Path, output_path: Path
) -> InsightPaths:
    """Compare metric means between two analysis or shard reports."""
    base = json.loads(baseline_report.read_text(encoding="utf-8"))
    cand = json.loads(candidate_report.read_text(encoding="utf-8"))
    b = _metric_means_from_report(base)
    c = _metric_means_from_report(cand)
    common = sorted(set(b) & set(c))
    rows = []
    normalized: list[float] = []
    for name in common:
        delta = float(c[name] - b[name])
        denom = max(abs(float(b[name])), abs(float(c[name])), 1.0)
        nd = float(delta / denom)
        normalized.append(abs(nd))
        rows.append(
            {
                "metric": name,
                "baseline": float(b[name]),
                "candidate": float(c[name]),
                "delta": delta,
                "normalized_delta": nd,
            }
        )
    report = _base_report(
        "archive_drift",
        {"baseline_report": str(baseline_report), "candidate_report": str(candidate_report)},
    )
    report.update(
        {
            "method": "common_metric_mean_normalized_delta",
            "common_metrics": common,
            "drift_score": float(np.mean(normalized)) if normalized else 0.0,
            "metric_deltas": rows,
        }
    )
    return InsightPaths(_write_json(output_path, report), report)


def run_event_retrieval(
    query_path: Path,
    corpus_dir: Path,
    output_dir: Path,
    *,
    top_k: int = 5,
    mode: str = "auto",
    metric: str = "novelty_curve",
    metrics: list[str] | None = None,
    distance: str = "cosine",
    feature_set: str = "auto",
    frame_size: int = 1024,
    hop_size: int = 256,
    sample_rate: int | None = None,
    max_files: int | None = None,
) -> InsightPaths:
    """Query-by-example retrieval wrapper around esl similarity search."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = run_similarity_search(
        SimilaritySearchConfig(
            input_path=query_path,
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            top_k=top_k,
            mode=mode,
            metric=metric,
            metrics=metrics,
            distance=distance,
            feature_set=feature_set,
            frame_size=frame_size,
            hop_size=hop_size,
            sample_rate=sample_rate,
            max_files=max_files,
        )
    )
    report = {**_base_report("query_by_example_retrieval", report.get("config", {})), **report}
    csv_path = output_dir / "event_retrieval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "path", "distance", "similarity"])
        writer.writeheader()
        for row in report.get("results", []):
            writer.writerow(
                {
                    "rank": row.get("rank"),
                    "path": row.get("path"),
                    "distance": row.get("distance"),
                    "similarity": row.get("similarity"),
                }
            )
    report["csv_path"] = str(csv_path)
    return InsightPaths(_write_json(output_dir / "event_retrieval.json", report), report)


def run_embeddings(
    input_dir: Path,
    output_dir: Path,
    *,
    feature_set: str = "auto",
    frame_size: int = 1024,
    hop_size: int = 256,
    sample_rate: int | None = None,
    max_files: int | None = None,
    device: str = "auto",
) -> InsightPaths:
    """Build deterministic clip-level feature embeddings for classical ML."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device_info = resolve_compute_device(device, strict=False)
    files = iter_supported_files(input_dir, patterns=SUPPORTED_PATTERNS, recursive=True)
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    vectors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    skipped: list[dict[str, str]] = []
    for p in files:
        try:
            fv = extract_feature_vectors(
                p,
                feature_set=feature_set,
                frame_size=frame_size,
                hop_size=hop_size,
                sample_rate=sample_rate,
            )
            vec = _aggregate_feature_vector(fv.matrix)
            if not feature_names:
                feature_names = [f"mean_{n}" for n in fv.feature_names] + [
                    f"std_{n}" for n in fv.feature_names
                ]
            vectors.append(vec)
            rows.append(
                {
                    "path": str(p),
                    "frames": int(fv.matrix.shape[0]),
                    "features": int(fv.matrix.shape[1]),
                    "backend": fv.backend,
                }
            )
        except Exception as exc:
            skipped.append({"path": str(p), "reason": str(exc)})
    matrix = (
        np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)
    )
    npz_path = output_dir / "embeddings.npz"
    np.savez_compressed(
        npz_path,
        embeddings=matrix,
        paths=np.array([r["path"] for r in rows], dtype=object),
        feature_names=np.array(feature_names, dtype=object),
    )
    csv_path = output_dir / "embeddings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", *feature_names])
        for row, vec in zip(rows, matrix, strict=True):
            writer.writerow([row["path"], *[float(v) for v in vec]])
    report = _base_report(
        "self_supervised_embedding_baseline",
        {
            "input_dir": str(input_dir),
            "feature_set": feature_set,
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "sample_rate": sample_rate,
            "max_files": max_files,
            "device": device,
        },
    )
    report.update(
        {
            "method": "clip_level_mean_std_feature_embedding",
            "device": device_resolution_dict(device_info),
            "num_files": len(rows),
            "num_features": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
            "npz_path": str(npz_path),
            "csv_path": str(csv_path),
            "items": rows,
            "skipped": skipped,
        }
    )
    return InsightPaths(_write_json(output_dir / "embeddings_manifest.json", report), report)


def run_soundscape_report(analysis_json: Path, output_dir: Path) -> InsightPaths:
    """Create a compact HTML soundscape report from an esl analysis JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(analysis_json.read_text(encoding="utf-8"))
    means = _metric_means_from_report(payload)
    selected = {k: means[k] for k in sorted(means)[:40]}
    report = _base_report("soundscape_report", {"analysis_json": str(analysis_json)})
    report.update({"metrics": selected})
    html_path = output_dir / "soundscape_report.html"
    rows = "\n".join(f"<tr><td>{k}</td><td>{v:.6g}</td></tr>" for k, v in selected.items())
    mermaid = """graph LR
  A["Audio file"] --> B["esl analyze"]
  B --> C["Metrics JSON"]
  C --> D["Soundscape report"]
  D --> E["Human decisions"]
"""
    script = (
        "<script type='module'>"
        "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
        "mermaid.initialize({startOnLoad:true});"
        "</script>"
    )
    style = (
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;line-height:1.5}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ddd;padding:.45rem}"
        "</style>"
    )
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>ecoSignalLab Soundscape Report</title>",
        script,
        style,
        "</head><body><h1>ecoSignalLab Soundscape Report</h1>",
        "<p>A compact, transparent summary of the analysis JSON. No wizard hat required.</p>",
        "<pre class='mermaid'>",
        mermaid,
        "</pre>",
        "<h2>Metric Means</h2><table><tr><th>Metric</th><th>Mean</th></tr>",
        rows,
        "</table></body></html>",
    ]
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    report["html_path"] = str(html_path)
    return InsightPaths(_write_json(output_dir / "soundscape_report.json", report), report)


def run_simulation_compare(
    simulated_json: Path, measured_json: Path, output_path: Path
) -> InsightPaths:
    """Compare simulated and measured analysis outputs by common metric means."""
    sim = json.loads(simulated_json.read_text(encoding="utf-8"))
    meas = json.loads(measured_json.read_text(encoding="utf-8"))
    s = _metric_means_from_report(sim)
    m = _metric_means_from_report(meas)
    preferred = ["rt60_s", "edt_s", "c50_db", "c80_db", "d50", "spl_a_db", "rms_dbfs", "snr_db"]
    common = [x for x in preferred if x in s and x in m] + sorted(
        (set(s) & set(m)) - set(preferred)
    )
    rows = [
        {
            "metric": name,
            "simulated": float(s[name]),
            "measured": float(m[name]),
            "measured_minus_simulated": float(m[name] - s[name]),
        }
        for name in common
    ]
    report = _base_report(
        "simulation_vs_field_comparison",
        {"simulated_json": str(simulated_json), "measured_json": str(measured_json)},
    )
    report.update({"method": "common_metric_mean_delta", "metric_deltas": rows})
    return InsightPaths(_write_json(output_path, report), report)


def run_storyboard(
    input_path: Path,
    output_dir: Path,
    *,
    clips: int = 12,
    window_s: float = 5.0,
    frame_size: int = 2048,
    hop_size: int = 512,
    sample_rate: int | None = None,
    feature_set: str = "auto",
    write_clips: bool = True,
) -> InsightPaths:
    """Create a timestamped acoustic storyboard from high-change moments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = run_scene_changes(
        input_path,
        output_dir / "_storyboard_scene",
        frame_size=frame_size,
        hop_size=hop_size,
        sample_rate=sample_rate,
        threshold_z=0.0,
        max_changes=max(1, int(clips) * 4),
        feature_set=feature_set,
    ).report
    candidates = sorted(
        scene.get("changes", []), key=lambda r: float(r.get("change_score", 0.0)), reverse=True
    )
    selected = sorted(candidates[: max(1, int(clips))], key=lambda r: float(r.get("time_s", 0.0)))
    buf = read_audio(input_path, target_sr=sample_rate) if write_clips else None
    clip_dir = output_dir / "clips"
    if write_clips:
        clip_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        center = float(row.get("time_s", 0.0))
        start_s = max(0.0, center - float(window_s) / 2.0)
        end_s = center + float(window_s) / 2.0
        clip_path = None
        if buf is not None:
            start = max(0, int(round(start_s * buf.sample_rate)))
            end = min(buf.samples.shape[0], int(round(end_s * buf.sample_rate)))
            if end > start:
                clip_path = clip_dir / f"story_{idx:03d}.wav"
                sf.write(clip_path, buf.samples[start:end], buf.sample_rate)
        rows.append(
            {
                "story_index": idx,
                "center_time_s": center,
                "start_s": start_s,
                "end_s": end_s,
                "change_score": float(row.get("change_score", 0.0)),
                "clip_path": str(clip_path) if clip_path else "",
            }
        )

    csv_path = output_dir / "storyboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "story_index",
                "center_time_s",
                "start_s",
                "end_s",
                "change_score",
                "clip_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    report = _base_report(
        "acoustic_storyboard",
        {
            "input_path": str(input_path),
            "clips": int(clips),
            "window_s": float(window_s),
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "sample_rate": sample_rate,
            "feature_set": feature_set,
            "write_clips": bool(write_clips),
        },
    )
    report.update(
        {
            "method": "top_scene_change_timestamp_storyboard",
            "csv_path": str(csv_path),
            "clips_dir": str(clip_dir),
            "items": rows,
        }
    )
    return InsightPaths(_write_json(output_dir / "storyboard.json", report), report)
