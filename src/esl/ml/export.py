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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FRAMETABLE_VERSION = "1.0.0"


@dataclass(slots=True)
class FrameTable:
    """Canonical frame-wise feature table used for tabular and tensor ML exports.

    Contract:
    - `timestamps_s`: shape `[frames]`
    - `feature_names`: shape `[features]`
    - `values`: shape `[frames, features]`
    - Tensor export layout: `[channels, frames, features]`
      where channel axis defaults to a single aggregate stream: `mix`.
    """

    timestamps_s: np.ndarray
    feature_names: list[str]
    values: np.ndarray
    channel_labels: list[str]
    metadata: dict[str, Any]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_frame_table(result: dict[str, Any]) -> FrameTable:
    """Build canonical FrameTable from metric series."""
    metrics = result.get("metrics", {})
    feature_names = sorted(
        [
            metric_name
            for metric_name, payload in metrics.items()
            if payload.get("series") and payload.get("timestamps_s")
        ]
    )

    if not feature_names:
        metadata = {
            "version": FRAMETABLE_VERSION,
            "frame_size": result.get("metadata", {}).get("frame_size"),
            "hop_size": result.get("metadata", {}).get("hop_size"),
            "source_channels": int(result.get("metadata", {}).get("channels", 1)),
            "channel_suffix_rule": "metric_id__chN for channel-specific columns; aggregate uses metric_id",
            "tensor_layout": "[channels, frames, features]",
            "channel_feature_mode": "aggregate_mixdown",
        }
        return FrameTable(
            timestamps_s=np.array([], dtype=np.float64),
            feature_names=[],
            values=np.empty((0, 0), dtype=np.float64),
            channel_labels=["mix"],
            metadata=metadata,
        )

    col_idx = {name: i for i, name in enumerate(feature_names)}
    time_to_row: dict[float, np.ndarray] = {}

    for metric_name in feature_names:
        payload = metrics.get(metric_name, {})
        ts = payload.get("timestamps_s", [])
        series = payload.get("series", [])
        for t, v in zip(ts, series):
            t_float = float(t)
            if t_float not in time_to_row:
                time_to_row[t_float] = np.full((len(feature_names),), np.nan, dtype=np.float64)
            time_to_row[t_float][col_idx[metric_name]] = float(v)

    timestamps = np.array(sorted(time_to_row.keys()), dtype=np.float64)
    values = (
        np.vstack([time_to_row[t] for t in timestamps])
        if timestamps.size
        else np.empty((0, len(feature_names)), dtype=np.float64)
    )

    metadata = {
        "version": FRAMETABLE_VERSION,
        "frame_size": result.get("metadata", {}).get("frame_size"),
        "hop_size": result.get("metadata", {}).get("hop_size"),
        "source_channels": int(result.get("metadata", {}).get("channels", 1)),
        "channel_suffix_rule": "metric_id__chN for channel-specific columns; aggregate uses metric_id",
        "tensor_layout": "[channels, frames, features]",
        "channel_feature_mode": "aggregate_mixdown",
        "source_config_hash": result.get("config_hash"),
        "esl_version": result.get("esl_version"),
    }
    return FrameTable(
        timestamps_s=timestamps,
        feature_names=feature_names,
        values=values,
        channel_labels=["mix"],
        metadata=metadata,
    )


def frame_table_tensor(
    frame_table: FrameTable,
    source_channels: int | None = None,
    replicate_aggregate: bool = False,
) -> tuple[np.ndarray, list[str], str]:
    """Export FrameTable as `[channels, frames, features]` tensor."""
    base = np.asarray(frame_table.values, dtype=np.float64)
    if base.ndim != 2:
        base = np.empty((0, 0), dtype=np.float64)

    ch = int(source_channels or frame_table.metadata.get("source_channels", 1))
    if replicate_aggregate and ch > 1:
        tensor = np.repeat(base[None, :, :], ch, axis=0)
        labels = [f"ch{i + 1}" for i in range(ch)]
        mode = "replicated_aggregate"
    else:
        tensor = base[None, :, :]
        labels = ["mix"]
        mode = "aggregate_mixdown"
    return tensor, labels, mode


def frame_table_rows(frame_table: FrameTable) -> list[dict[str, Any]]:
    """Build wide tabular rows from FrameTable."""
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(frame_table.timestamps_s.tolist()):
        row: dict[str, Any] = {"timestamp_s": float(ts)}
        for j, col in enumerate(frame_table.feature_names):
            val = float(frame_table.values[i, j]) if frame_table.values.size else float("nan")
            row[col] = None if np.isnan(val) else val
        rows.append(row)
    return rows


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
    frame_table = build_frame_table(result)
    return frame_table.timestamps_s, frame_table.feature_names, frame_table.values


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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False
    try:
        df = pd.DataFrame.from_records(rows)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


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

    frame_table = build_frame_table(result)
    timestamps = frame_table.timestamps_s
    feature_cols = frame_table.feature_names
    frame_matrix = frame_table.values

    frame_npy = out / f"{prefix}_frame_features.npy"
    np.save(frame_npy, frame_matrix)
    artifacts["frame_npy"] = str(frame_npy)

    frame_table_csv = out / f"{prefix}_frame_table.csv"
    wide_rows = frame_table_rows(frame_table)
    _write_csv(frame_table_csv, ["timestamp_s", *feature_cols], wide_rows)
    artifacts["frame_table_csv"] = str(frame_table_csv)

    frame_table_parquet = out / f"{prefix}_frame_table.parquet"
    if _write_parquet(frame_table_parquet, wide_rows):
        artifacts["frame_table_parquet"] = str(frame_table_parquet)

    frame_tensor, tensor_channel_labels, tensor_mode = frame_table_tensor(
        frame_table,
        source_channels=int(result.get("metadata", {}).get("channels", 1)),
        replicate_aggregate=False,
    )
    frame_tensor_npy = out / f"{prefix}_frame_tensor.npy"
    np.save(frame_tensor_npy, frame_tensor)
    artifacts["frame_tensor_npy"] = str(frame_tensor_npy)

    meta = {
        "clip_feature_names": names,
        "frame_feature_columns": feature_cols,
        "frame_timestamps_s": timestamps.tolist(),
        "frame_table_version": FRAMETABLE_VERSION,
        "frame_table": {
            "window_size": frame_table.metadata.get("frame_size"),
            "hop_size": frame_table.metadata.get("hop_size"),
            "channel_suffix_rule": frame_table.metadata.get("channel_suffix_rule"),
            "channel_feature_mode": frame_table.metadata.get("channel_feature_mode"),
            "tensor_layout": frame_table.metadata.get("tensor_layout"),
            "tensor_mode": tensor_mode,
            "tensor_channel_labels": tensor_channel_labels,
            "tensor_shape": list(frame_tensor.shape),
        },
        "seed": seed,
        "source_config_hash": result.get("config_hash"),
        "source_pipeline_hash": result.get("pipeline_hash"),
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
        frame_tensor_pt = out / f"{prefix}_frame_tensor.pt"
        torch.save(torch.tensor(clip_vec, dtype=torch.float32), clip_pt)
        torch.save(torch.tensor(frame_matrix, dtype=torch.float32), frame_pt)
        torch.save(torch.tensor(frame_tensor, dtype=torch.float32), frame_tensor_pt)
        artifacts["clip_pt"] = str(clip_pt)
        artifacts["frame_pt"] = str(frame_pt)
        artifacts["frame_tensor_pt"] = str(frame_tensor_pt)
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
