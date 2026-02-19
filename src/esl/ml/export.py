"""ML-oriented feature exports and anomaly helpers.

References:
- Isolation Forest:
  Liu, Ting, Zhou (2008), \"Isolation Forest\", ICDM.
- One-class representation and anomaly framing background:
  Pimentel et al. (2014), Signal Processing 99:215-249.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def frame_long_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build long-form frame table from all metric time series."""
    rows: list[dict[str, Any]] = []
    for metric, payload in result.get("metrics", {}).items():
        series = payload.get("series", [])
        times = payload.get("timestamps_s", [])
        units = payload.get("units", "")
        for t, v in zip(times, series):
            rows.append({"timestamp_s": float(t), "metric": metric, "value": float(v), "units": units})
    return rows


def frame_wide_table(result: dict[str, Any]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Build wide frame matrix (timestamps, metric names, matrix)."""
    rows = frame_long_table(result)
    if not rows:
        return np.array([], dtype=np.float64), [], np.empty((0, 0), dtype=np.float64)

    metrics = sorted({row["metric"] for row in rows})
    col_idx = {m: i for i, m in enumerate(metrics)}

    time_to_row: dict[float, np.ndarray] = {}
    for row in rows:
        t = float(row["timestamp_s"])
        if t not in time_to_row:
            time_to_row[t] = np.full((len(metrics),), np.nan, dtype=np.float64)
        time_to_row[t][col_idx[row["metric"]]] = float(row["value"])

    timestamps = np.array(sorted(time_to_row.keys()), dtype=np.float64)
    matrix = np.vstack([time_to_row[t] for t in timestamps]) if timestamps.size else np.empty((0, len(metrics)))
    return timestamps, metrics, matrix


def clip_feature_vector(result: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """Clip-level feature vector from per-metric summary means."""
    names = sorted(result.get("metrics", {}).keys())
    vals = []
    for name in names:
        val = result["metrics"][name].get("summary", {}).get("mean", np.nan)
        vals.append(np.nan if val is None else float(val))
    return names, np.array(vals, dtype=np.float64)


def run_isolation_forest(frame_matrix: np.ndarray, seed: int = 42) -> np.ndarray | None:
    """Optional anomaly score using IsolationForest."""
    if frame_matrix.ndim != 2 or frame_matrix.shape[0] < 8 or frame_matrix.shape[1] < 2:
        return None
    try:
        from sklearn.ensemble import IsolationForest
    except Exception:
        return None

    # Replace NaN columns with column means for sklearn compatibility.
    x = frame_matrix.copy()
    if np.isnan(x).any():
        col_means = np.nanmean(x, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(np.isnan(x))
        x[inds] = np.take(col_means, inds[1])

    # Isolation Forest baseline following Liu et al. (2008).
    model = IsolationForest(random_state=seed, n_estimators=200, contamination="auto")
    model.fit(x)
    score = -model.decision_function(x)
    return score.astype(np.float64)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_ml_features(
    result: dict[str, Any],
    output_dir: str | Path,
    prefix: str = "esl",
    seed: int = 42,
) -> dict[str, str]:
    """Export frame/clip features for NumPy, optional PyTorch, and HuggingFace datasets."""
    out = Path(output_dir)
    _ensure_dir(out)

    artifacts: dict[str, str] = {}

    names, clip_vec = clip_feature_vector(result)
    clip_npy = out / f"{prefix}_clip_features.npy"
    np.save(clip_npy, clip_vec)
    artifacts["clip_npy"] = str(clip_npy)

    long_rows = frame_long_table(result)
    frame_csv = out / f"{prefix}_frame_features.csv"
    _write_csv(frame_csv, ["timestamp_s", "metric", "value", "units"], long_rows)
    artifacts["frame_csv"] = str(frame_csv)

    timestamps, feature_cols, frame_matrix = frame_wide_table(result)
    frame_npy = out / f"{prefix}_frame_features.npy"
    np.save(frame_npy, frame_matrix)
    artifacts["frame_npy"] = str(frame_npy)

    meta = {
        "clip_feature_names": names,
        "frame_feature_columns": feature_cols,
        "frame_timestamps_s": timestamps.tolist(),
        "seed": seed,
        "source_config_hash": result.get("config_hash"),
        "esl_version": result.get("esl_version"),
    }
    meta_path = out / f"{prefix}_ml_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    artifacts["ml_metadata_json"] = str(meta_path)

    # Optional PyTorch tensors.
    try:
        import torch

        clip_pt = out / f"{prefix}_clip_features.pt"
        frame_pt = out / f"{prefix}_frame_features.pt"
        torch.save(torch.tensor(clip_vec, dtype=torch.float32), clip_pt)
        torch.save(torch.tensor(frame_matrix, dtype=torch.float32), frame_pt)
        artifacts["clip_pt"] = str(clip_pt)
        artifacts["frame_pt"] = str(frame_pt)
    except Exception:
        pass

    # Optional HuggingFace dataset export.
    try:
        from datasets import Dataset

        hf_dir = out / f"{prefix}_hf_dataset"
        hf_dir.mkdir(parents=True, exist_ok=True)
        ds = Dataset.from_dict(
            {
                "timestamp_s": timestamps.tolist(),
                "features": frame_matrix.tolist() if frame_matrix.size else [],
            }
        )
        ds.save_to_disk(str(hf_dir))
        artifacts["hf_dataset"] = str(hf_dir)
    except Exception:
        pass

    # Optional anomaly score using IsolationForest.
    scores = run_isolation_forest(frame_matrix, seed=seed)
    if scores is not None and timestamps.size:
        rows = [
            {"timestamp_s": float(ts), "anomaly_score": float(sc)}
            for ts, sc in zip(timestamps.tolist(), scores.tolist())
        ]
        score_path = out / f"{prefix}_anomaly_scores.csv"
        _write_csv(score_path, ["timestamp_s", "anomaly_score"], rows)
        artifacts["anomaly_scores_csv"] = str(score_path)

    return artifacts
