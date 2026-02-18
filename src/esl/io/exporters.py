"""Export analysis outputs to multiple interoperability formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in result["metrics"].items():
        units = payload.get("units")
        conf = payload.get("confidence")
        for stat, val in payload.get("summary", {}).items():
            rows.append(
                {
                    "metric": name,
                    "stat": stat,
                    "value": val,
                    "units": units,
                    "confidence": conf,
                }
            )
    return rows


def _series_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in result["metrics"].items():
        ts = payload.get("timestamps_s", [])
        vals = payload.get("series", [])
        for t, v in zip(ts, vals):
            rows.append({"metric": name, "timestamp_s": t, "value": v, "units": payload.get("units")})
    return rows


def save_json(result: dict[str, Any], path: str | Path) -> Path:
    """Write canonical JSON result document."""
    p = Path(path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return p


def _write_rows_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_csv(result: dict[str, Any], path: str | Path) -> Path:
    """Write summary CSV."""
    p = Path(path)
    _ensure_parent(p)
    rows = _summary_rows(result)
    _write_rows_csv(p, rows, ["metric", "stat", "value", "units", "confidence"])
    return p


def save_series_csv(result: dict[str, Any], path: str | Path) -> Path:
    """Write frame/series CSV."""
    p = Path(path)
    _ensure_parent(p)
    rows = _series_rows(result)
    _write_rows_csv(p, rows, ["metric", "timestamp_s", "value", "units"])
    return p


def save_parquet(result: dict[str, Any], path: str | Path) -> Path:
    """Write summary Parquet (requires pandas+pyarrow/fastparquet)."""
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("Parquet export requires pandas and pyarrow/fastparquet") from exc

    p = Path(path)
    _ensure_parent(p)
    df = pd.DataFrame.from_records(_summary_rows(result))
    df.to_parquet(p, index=False)
    return p


def save_hdf5(result: dict[str, Any], path: str | Path) -> Path:
    """Write HDF5 export with metadata and metric groups."""
    try:
        import h5py
    except Exception as exc:
        raise RuntimeError("HDF5 export requires h5py") from exc

    p = Path(path)
    _ensure_parent(p)
    with h5py.File(p, "w") as h5:
        meta = h5.create_group("metadata")
        for k, v in result.get("metadata", {}).items():
            if isinstance(v, (dict, list)):
                meta.attrs[k] = json.dumps(v)
            elif v is None:
                meta.attrs[k] = ""
            else:
                meta.attrs[k] = v

        metrics = h5.create_group("metrics")
        for name, payload in result["metrics"].items():
            g = metrics.create_group(name)
            g.attrs["units"] = payload.get("units", "")
            g.attrs["confidence"] = float(payload.get("confidence", 0.0))
            summary = payload.get("summary", {})
            for stat, val in summary.items():
                g.attrs[f"summary_{stat}"] = float(val)
            g.create_dataset("series", data=np.array(payload.get("series", []), dtype=np.float64))
            g.create_dataset("timestamps_s", data=np.array(payload.get("timestamps_s", []), dtype=np.float64))
    return p


def save_mat(result: dict[str, Any], path: str | Path) -> Path:
    """Write MATLAB MAT export."""
    try:
        from scipy.io import savemat
    except Exception as exc:
        raise RuntimeError("scipy is required for MATLAB export") from exc

    p = Path(path)
    _ensure_parent(p)
    payload: dict[str, Any] = {
        "esl_version": result.get("esl_version", ""),
        "config_hash": result.get("config_hash", ""),
    }
    for name, m in result["metrics"].items():
        key = name.replace("-", "_")
        payload[f"{key}_series"] = np.array(m.get("series", []), dtype=np.float64)
        payload[f"{key}_times"] = np.array(m.get("timestamps_s", []), dtype=np.float64)
        for stat, val in m.get("summary", {}).items():
            payload[f"{key}_{stat}"] = float(val)
    savemat(p, payload)
    return p


def _industrial_rows(result: dict[str, Any], system: str) -> list[dict[str, Any]]:
    rows = []
    for metric, payload in result["metrics"].items():
        summ = payload.get("summary", {})
        rows.append(
            {
                "system": system,
                "metric": metric,
                "value_mean": summ.get("mean"),
                "value_p95": summ.get("p95"),
                "units": payload.get("units"),
                "confidence": payload.get("confidence"),
                "weighting": "A/C/Z or unweighted per metric definition",
                "windowing": f"{result['metadata'].get('frame_size')}@hop{result['metadata'].get('hop_size')}",
            }
        )
    return rows


def save_head_csv(result: dict[str, Any], path: str | Path) -> Path:
    """Write HEAD Acoustics-oriented CSV mapping."""
    p = Path(path)
    _ensure_parent(p)
    _write_rows_csv(
        p,
        _industrial_rows(result, "HEAD Acoustics"),
        ["system", "metric", "value_mean", "value_p95", "units", "confidence", "weighting", "windowing"],
    )
    return p


def save_apx_csv(result: dict[str, Any], path: str | Path) -> Path:
    """Write Audio Precision APx-oriented CSV mapping."""
    p = Path(path)
    _ensure_parent(p)
    _write_rows_csv(
        p,
        _industrial_rows(result, "Audio Precision APx"),
        ["system", "metric", "value_mean", "value_p95", "units", "confidence", "weighting", "windowing"],
    )
    return p


def save_soundcheck_csv(result: dict[str, Any], path: str | Path) -> Path:
    """Write Listen Inc. SoundCheck-oriented CSV mapping."""
    p = Path(path)
    _ensure_parent(p)
    _write_rows_csv(
        p,
        _industrial_rows(result, "Listen SoundCheck"),
        ["system", "metric", "value_mean", "value_p95", "units", "confidence", "weighting", "windowing"],
    )
    return p
