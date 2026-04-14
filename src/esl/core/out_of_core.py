"""Out-of-core analysis helpers for long-duration audio processing.

References:
- Welford, B. P. (1962), "Note on a Method for Calculating Corrected Sums of
  Squares and Products", Technometrics 4(3):419-420.
- Vitter, J. S. (1985), "Random Sampling with a Reservoir", ACM TOMS 11(1):37-57.

These helpers intentionally avoid full-signal materialization so multi-hour,
multi-day, and multi-year recordings can be summarized in bounded memory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from esl.core.utils import canonicalize
from esl.metrics.base import MetricResult
from esl.ml.export import FRAMETABLE_VERSION


RUNNING_SUMMARY_METHOD = "welford_mean_std + reservoir_quantiles"
DEFAULT_QUANTILE_CAPACITY = 8192


def _nan_summary() -> dict[str, float]:
    return {
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "p50": float("nan"),
        "p95": float("nan"),
    }


@dataclass
class RunningStats:
    """Streaming summary with bounded-memory percentile approximation."""

    reservoir_capacity: int = DEFAULT_QUANTILE_CAPACITY
    rng_seed: int = 42
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float = float("inf")
    max_value: float = float("-inf")
    reservoir: list[float] = field(default_factory=list)
    rng_state: dict[str, Any] | None = None
    reservoir_seen: int = 0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)
        if self.rng_state is not None:
            self._rng.bit_generator.state = self.rng_state

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return

        batch_count = int(arr.size)
        batch_mean = float(np.mean(arr))
        batch_m2 = float(np.sum(np.square(arr - batch_mean)))
        batch_min = float(np.min(arr))
        batch_max = float(np.max(arr))

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min_value = batch_min
            self.max_value = batch_max
        else:
            total = self.count + batch_count
            delta = batch_mean - self.mean
            self.m2 = self.m2 + batch_m2 + ((delta * delta) * self.count * batch_count / total)
            self.mean = self.mean + delta * batch_count / total
            self.count = total
            self.min_value = min(self.min_value, batch_min)
            self.max_value = max(self.max_value, batch_max)

        self._update_reservoir(arr)

    def _update_reservoir(self, values: np.ndarray) -> None:
        for raw in values.tolist():
            value = float(raw)
            self.reservoir_seen += 1
            if len(self.reservoir) < self.reservoir_capacity:
                self.reservoir.append(value)
                continue
            j = int(self._rng.integers(0, max(self.reservoir_seen, 1)))
            if j < self.reservoir_capacity:
                self.reservoir[j] = value

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return _nan_summary()
        reservoir = np.asarray(self.reservoir, dtype=np.float64)
        if reservoir.size == 0:
            return _nan_summary()
        std = float(np.sqrt(max(self.m2 / max(self.count, 1), 0.0)))
        return {
            "mean": float(self.mean),
            "std": std,
            "min": float(self.min_value),
            "max": float(self.max_value),
            "p50": float(np.percentile(reservoir, 50.0)),
            "p95": float(np.percentile(reservoir, 95.0)),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservoir_capacity": int(self.reservoir_capacity),
            "rng_seed": int(self.rng_seed),
            "count": int(self.count),
            "mean": float(self.mean),
            "m2": float(self.m2),
            "min_value": float(self.min_value),
            "max_value": float(self.max_value),
            "reservoir": [float(v) for v in self.reservoir],
            "rng_state": canonicalize(self._rng.bit_generator.state),
            "reservoir_seen": int(self.reservoir_seen),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunningStats:
        return cls(
            reservoir_capacity=int(payload.get("reservoir_capacity", DEFAULT_QUANTILE_CAPACITY)),
            rng_seed=int(payload.get("rng_seed", 42)),
            count=int(payload.get("count", 0)),
            mean=float(payload.get("mean", 0.0)),
            m2=float(payload.get("m2", 0.0)),
            min_value=float(payload.get("min_value", float("inf"))),
            max_value=float(payload.get("max_value", float("-inf"))),
            reservoir=[float(v) for v in payload.get("reservoir", [])],
            rng_state=payload.get("rng_state"),
            reservoir_seen=int(payload.get("reservoir_seen", payload.get("count", 0))),
        )


@dataclass
class MetricAccumulator:
    """Accumulate metric summaries and optional series across chunks."""

    name: str
    store_series: bool = True
    max_series_points: int | None = None
    quantile_capacity: int = DEFAULT_QUANTILE_CAPACITY
    stats: RunningStats = field(default_factory=RunningStats)
    units: str = ""
    confidence_sum: float = 0.0
    confidence_count: int = 0
    series: list[float] = field(default_factory=list)
    timestamps_s: list[float] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    event_timestamps_s: list[float] = field(default_factory=list)
    series_truncated: bool = False

    def update(self, result: MetricResult, chunk_offset_s: float) -> None:
        values = np.asarray(result.series, dtype=np.float64)
        if values.size:
            self.stats.update(values)

        self.units = result.units or self.units
        self.confidence_sum += float(result.confidence)
        self.confidence_count += 1

        if result.extra:
            for key, value in result.extra.items():
                if key == "event_frame_indices" and result.timestamps_s:
                    continue
                self.extra.setdefault(key, value)

        if self.store_series and values.size:
            capacity = self.max_series_points
            remaining = None if capacity is None else max(capacity - len(self.series), 0)
            if remaining == 0:
                self.series_truncated = True
            else:
                count = values.size if remaining is None else min(int(values.size), int(remaining))
                if count < int(values.size):
                    self.series_truncated = True
                self.series.extend(float(v) for v in values[:count].tolist())
                abs_ts = [float(chunk_offset_s + float(t)) for t in result.timestamps_s[:count]]
                self.timestamps_s.extend(abs_ts)

        frame_events = result.extra.get("event_frame_indices") if isinstance(result.extra, dict) else None
        if isinstance(frame_events, list) and result.timestamps_s:
            for idx in frame_events:
                if isinstance(idx, int) and 0 <= idx < len(result.timestamps_s):
                    self.event_timestamps_s.append(float(chunk_offset_s + result.timestamps_s[idx]))

    def finalize(self) -> MetricResult:
        extra = dict(self.extra)
        extra["summary_method"] = RUNNING_SUMMARY_METHOD
        if self.series_truncated:
            extra["series_in_json"] = "truncated"
        elif not self.store_series:
            extra["series_in_json"] = "omitted"
        if self.event_timestamps_s:
            extra["event_timestamps_s"] = [float(v) for v in self.event_timestamps_s]
        return MetricResult(
            name=self.name,
            units=self.units,
            summary=self.stats.summary(),
            series=list(self.series) if self.store_series else [],
            timestamps_s=list(self.timestamps_s) if self.store_series else [],
            confidence=float(self.confidence_sum / self.confidence_count) if self.confidence_count else 0.0,
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "store_series": bool(self.store_series),
            "max_series_points": self.max_series_points,
            "stats": self.stats.to_dict(),
            "units": self.units,
            "confidence_sum": float(self.confidence_sum),
            "confidence_count": int(self.confidence_count),
            "series": [float(v) for v in self.series],
            "timestamps_s": [float(v) for v in self.timestamps_s],
            "extra": canonicalize(self.extra),
            "event_timestamps_s": [float(v) for v in self.event_timestamps_s],
            "series_truncated": bool(self.series_truncated),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricAccumulator:
        acc = cls(
            name=str(payload.get("name", "")),
            store_series=bool(payload.get("store_series", True)),
            max_series_points=payload.get("max_series_points"),
        )
        acc.stats = RunningStats.from_dict(payload.get("stats", {}))
        acc.units = str(payload.get("units", ""))
        acc.confidence_sum = float(payload.get("confidence_sum", 0.0))
        acc.confidence_count = int(payload.get("confidence_count", 0))
        acc.series = [float(v) for v in payload.get("series", [])]
        acc.timestamps_s = [float(v) for v in payload.get("timestamps_s", [])]
        acc.extra = dict(payload.get("extra", {}))
        acc.event_timestamps_s = [float(v) for v in payload.get("event_timestamps_s", [])]
        acc.series_truncated = bool(payload.get("series_truncated", False))
        return acc


@dataclass
class AudioAccumulator:
    """Aggregate multi-channel validity and level metadata without full loads."""

    sample_rate: int | None = None
    channels: int | None = None
    total_samples: int = 0
    sum_per_channel: np.ndarray | None = None
    sum_sq_per_channel: np.ndarray | None = None
    peak_per_channel: np.ndarray | None = None
    clip_count_per_channel: np.ndarray | None = None

    def update(self, samples: np.ndarray, sample_rate: int) -> None:
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim != 2 or arr.size == 0:
            return
        if self.channels is None:
            self.channels = int(arr.shape[1])
            self.sample_rate = int(sample_rate)
            self.sum_per_channel = np.zeros((self.channels,), dtype=np.float64)
            self.sum_sq_per_channel = np.zeros((self.channels,), dtype=np.float64)
            self.peak_per_channel = np.zeros((self.channels,), dtype=np.float64)
            self.clip_count_per_channel = np.zeros((self.channels,), dtype=np.int64)

        if self.channels != int(arr.shape[1]):
            raise RuntimeError("Channel count changed across audio chunks.")

        self.total_samples += int(arr.shape[0])
        assert self.sum_per_channel is not None
        assert self.sum_sq_per_channel is not None
        assert self.peak_per_channel is not None
        assert self.clip_count_per_channel is not None
        self.sum_per_channel += np.sum(arr, axis=0)
        self.sum_sq_per_channel += np.sum(np.square(arr), axis=0)
        self.peak_per_channel = np.maximum(self.peak_per_channel, np.max(np.abs(arr), axis=0))
        self.clip_count_per_channel += np.sum(np.abs(arr) >= 0.999, axis=0).astype(np.int64)

    def channel_summary(self) -> dict[str, Any]:
        if self.channels is None or self.total_samples <= 0:
            return {"channels": [], "aggregate": {}, "aggregation_rules": {}}
        assert self.sum_per_channel is not None
        assert self.sum_sq_per_channel is not None
        assert self.peak_per_channel is not None
        assert self.clip_count_per_channel is not None

        rms_ch = np.sqrt(self.sum_sq_per_channel / max(self.total_samples, 1))
        peak_ch = np.asarray(self.peak_per_channel, dtype=np.float64)
        dc_ch = self.sum_per_channel / max(self.total_samples, 1)
        clip_ch = self.clip_count_per_channel.astype(np.float64) / max(self.total_samples, 1)
        channels = []
        for i in range(int(self.channels)):
            channels.append(
                {
                    "id": f"ch{i + 1}",
                    "rms_dbfs": float(20.0 * np.log10(max(rms_ch[i], 1e-12))),
                    "peak_dbfs": float(20.0 * np.log10(max(peak_ch[i], 1e-12))),
                    "dc_offset": float(dc_ch[i]),
                    "clipping_ratio": float(clip_ch[i]),
                }
            )

        agg_rms = float(np.sqrt(np.mean(np.square(rms_ch))))
        aggregate = {
            "rms_dbfs": float(20.0 * np.log10(max(agg_rms, 1e-12))),
            "peak_dbfs": float(20.0 * np.log10(max(float(np.max(peak_ch)), 1e-12))),
            "dc_offset": float(np.mean(dc_ch)),
            "clipping_ratio": float(np.mean(clip_ch)),
        }
        return {
            "channels": channels,
            "aggregate": aggregate,
            "aggregation_rules": {
                "rms_dbfs": "20*log10(sqrt(mean(channel_rms_linear^2)))",
                "peak_dbfs": "max(channel_peak_dbfs)",
                "dc_offset": "mean(channel_dc_offset)",
                "clipping_ratio": "mean(channel_clipping_ratio)",
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "total_samples": int(self.total_samples),
            "sum_per_channel": canonicalize(self.sum_per_channel if self.sum_per_channel is not None else []),
            "sum_sq_per_channel": canonicalize(self.sum_sq_per_channel if self.sum_sq_per_channel is not None else []),
            "peak_per_channel": canonicalize(self.peak_per_channel if self.peak_per_channel is not None else []),
            "clip_count_per_channel": canonicalize(
                self.clip_count_per_channel if self.clip_count_per_channel is not None else []
            ),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AudioAccumulator:
        acc = cls(
            sample_rate=payload.get("sample_rate"),
            channels=payload.get("channels"),
            total_samples=int(payload.get("total_samples", 0)),
        )
        if payload.get("sum_per_channel") is not None:
            acc.sum_per_channel = np.asarray(payload.get("sum_per_channel", []), dtype=np.float64)
        if payload.get("sum_sq_per_channel") is not None:
            acc.sum_sq_per_channel = np.asarray(payload.get("sum_sq_per_channel", []), dtype=np.float64)
        if payload.get("peak_per_channel") is not None:
            acc.peak_per_channel = np.asarray(payload.get("peak_per_channel", []), dtype=np.float64)
        if payload.get("clip_count_per_channel") is not None:
            acc.clip_count_per_channel = np.asarray(payload.get("clip_count_per_channel", []), dtype=np.int64)
        return acc


class FrameTableCsvWriter:
    """Append canonical FrameTable rows to CSV without holding all frames."""

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._append = append
        self._file: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._fieldnames: list[str] | None = None

    def append_chunk(self, metric_results: dict[str, MetricResult], chunk_offset_s: float) -> None:
        rows = self._rows(metric_results=metric_results, chunk_offset_s=chunk_offset_s)
        if not rows:
            return
        fieldnames = ["timestamp_s", *sorted(name for name in metric_results.keys() if metric_results[name].series)]
        self._open(fieldnames)
        assert self._writer is not None
        for row in rows:
            self._writer.writerow(row)
        assert self._file is not None
        self._file.flush()

    def _rows(self, metric_results: dict[str, MetricResult], chunk_offset_s: float) -> list[dict[str, Any]]:
        feature_names = sorted(
            name for name, result in metric_results.items() if result.series and result.timestamps_s
        )
        if not feature_names:
            return []
        time_rows: dict[float, dict[str, Any]] = {}
        for name in feature_names:
            result = metric_results[name]
            for t, value in zip(result.timestamps_s, result.series):
                abs_t = float(chunk_offset_s + float(t))
                row = time_rows.setdefault(abs_t, {"timestamp_s": abs_t})
                row[name] = float(value)
        return [time_rows[t] for t in sorted(time_rows)]

    def _open(self, fieldnames: list[str]) -> None:
        if self._writer is not None:
            return
        self._fieldnames = fieldnames
        mode = "a" if self._append and self.path.exists() else "w"
        self._file = self.path.open(mode, encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        write_header = mode == "w" or self.path.stat().st_size == 0
        if write_header:
            self._writer.writeheader()

    def metadata_payload(
        self,
        *,
        frame_size: int,
        hop_size: int,
        source_channels: int,
        esl_version: str,
    ) -> dict[str, Any]:
        return {
            "frame_table_version": FRAMETABLE_VERSION,
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "source_channels": int(source_channels),
            "channel_suffix_rule": "metric_id__chN for channel-specific columns; aggregate uses metric_id",
            "tensor_layout": "[channels, frames, features]",
            "channel_feature_mode": "aggregate_mixdown",
            "esl_version": esl_version,
            "columns": self._fieldnames or ["timestamp_s"],
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None


class _FrameTableChunkRowsMixin:
    def _rows(self, metric_results: dict[str, MetricResult], chunk_offset_s: float) -> list[dict[str, Any]]:
        feature_names = sorted(
            name for name, result in metric_results.items() if result.series and result.timestamps_s
        )
        if not feature_names:
            return []
        time_rows: dict[float, dict[str, Any]] = {}
        for name in feature_names:
            result = metric_results[name]
            for t, value in zip(result.timestamps_s, result.series):
                abs_t = float(chunk_offset_s + float(t))
                row = time_rows.setdefault(abs_t, {"timestamp_s": abs_t})
                row[name] = float(value)
        return [time_rows[t] for t in sorted(time_rows)]


class FrameTableParquetDatasetWriter(_FrameTableChunkRowsMixin):
    """Append FrameTable chunks as a Parquet dataset directory.

    Each appended chunk is written as one `part-*.parquet` file. This keeps the
    write path resumable without rewriting a monolithic Parquet file.
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._append = append
        self._fieldnames: list[str] | None = None
        self._available = True
        self._error: str | None = None
        existing = sorted(self.path.glob("part-*.parquet")) if append else []
        self._part_index = len(existing)

    def append_chunk(self, metric_results: dict[str, MetricResult], chunk_offset_s: float) -> None:
        rows = self._rows(metric_results=metric_results, chunk_offset_s=chunk_offset_s)
        if not rows:
            return
        fieldnames = ["timestamp_s", *sorted(name for name in metric_results.keys() if metric_results[name].series)]
        self._fieldnames = fieldnames
        try:
            import pandas as pd
        except Exception as exc:
            self._available = False
            self._error = f"Parquet FrameTable export requires pandas and pyarrow/fastparquet: {exc}"
            return

        part_path = self.path / f"part-{self._part_index:08d}.parquet"
        self._part_index += 1
        df = pd.DataFrame.from_records(rows, columns=fieldnames)
        try:
            df.to_parquet(part_path, index=False)
        except Exception as exc:
            self._available = False
            self._error = str(exc)
            return

    def metadata_payload(
        self,
        *,
        frame_size: int,
        hop_size: int,
        source_channels: int,
        esl_version: str,
    ) -> dict[str, Any]:
        payload = {
            "frame_table_version": FRAMETABLE_VERSION,
            "storage_kind": "parquet_dataset_directory",
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "source_channels": int(source_channels),
            "channel_suffix_rule": "metric_id__chN for channel-specific columns; aggregate uses metric_id",
            "tensor_layout": "[channels, frames, features]",
            "channel_feature_mode": "aggregate_mixdown",
            "esl_version": esl_version,
            "columns": self._fieldnames or ["timestamp_s"],
            "available": bool(self._available),
            "error": self._error,
        }
        (self.path / "metadata.json").write_text(json.dumps(canonicalize(payload), indent=2), encoding="utf-8")
        return payload

    def close(self) -> None:
        return None


class FrameTableHdf5Writer(_FrameTableChunkRowsMixin):
    """Append FrameTable rows into a single resizable HDF5 file."""

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._append = append
        self._fieldnames: list[str] | None = None
        self._file: Any | None = None

    def _open(self) -> Any:
        if self._file is not None:
            return self._file
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError("HDF5 FrameTable export requires h5py") from exc
        mode = "a" if self._append and self.path.exists() else "w"
        self._file = h5py.File(self.path, mode)
        return self._file

    def append_chunk(self, metric_results: dict[str, MetricResult], chunk_offset_s: float) -> None:
        rows = self._rows(metric_results=metric_results, chunk_offset_s=chunk_offset_s)
        if not rows:
            return
        fieldnames = ["timestamp_s", *sorted(name for name in metric_results.keys() if metric_results[name].series)]
        h5 = self._open()
        data_rows = np.array(
            [[float(row.get(name, np.nan)) for name in fieldnames[1:]] for row in rows],
            dtype=np.float64,
        )
        timestamps = np.array([float(row["timestamp_s"]) for row in rows], dtype=np.float64)
        self._fieldnames = fieldnames

        cols_json = json.dumps(fieldnames[1:])
        if "feature_names_json" in h5.attrs:
            if str(h5.attrs["feature_names_json"]) != cols_json:
                raise RuntimeError("FrameTable HDF5 columns changed across appended chunks.")
        else:
            h5.attrs["feature_names_json"] = cols_json

        if "timestamps_s" not in h5:
            h5.create_dataset("timestamps_s", data=timestamps, maxshape=(None,), dtype="f8")
            h5.create_dataset(
                "values",
                data=data_rows,
                maxshape=(None, data_rows.shape[1]),
                dtype="f8",
            )
        else:
            ts_ds = h5["timestamps_s"]
            val_ds = h5["values"]
            old = int(ts_ds.shape[0])
            new = old + int(timestamps.shape[0])
            ts_ds.resize((new,))
            ts_ds[old:new] = timestamps
            val_ds.resize((new, data_rows.shape[1]))
            val_ds[old:new, :] = data_rows
        h5.flush()

    def metadata_payload(
        self,
        *,
        frame_size: int,
        hop_size: int,
        source_channels: int,
        esl_version: str,
    ) -> dict[str, Any]:
        payload = {
            "frame_table_version": FRAMETABLE_VERSION,
            "storage_kind": "hdf5_resizable_dataset",
            "frame_size": int(frame_size),
            "hop_size": int(hop_size),
            "source_channels": int(source_channels),
            "channel_suffix_rule": "metric_id__chN for channel-specific columns; aggregate uses metric_id",
            "tensor_layout": "[channels, frames, features]",
            "channel_feature_mode": "aggregate_mixdown",
            "esl_version": esl_version,
            "columns": self._fieldnames or ["timestamp_s"],
        }
        h5 = self._open()
        h5.attrs["frame_table_metadata_json"] = json.dumps(canonicalize(payload))
        h5.flush()
        return payload

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(canonicalize(payload), indent=2), encoding="utf-8")
    return p


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
