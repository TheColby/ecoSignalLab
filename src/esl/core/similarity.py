"""File-to-folder similarity search for acoustic workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esl.core.audio import iter_supported_files
from esl.core.config import AnalysisConfig, CalibrationProfile
from esl.core.utils import set_seed
from esl.metrics.registry import create_registry
from esl.viz.feature_vectors import extract_feature_vectors


SUPPORTED_PATTERNS = [
    "*.wav",
    "*.flac",
    "*.aiff",
    "*.aif",
    "*.rf64",
    "*.caf",
    "*.mp3",
    "*.aac",
    "*.ogg",
    "*.opus",
    "*.wma",
    "*.alac",
    "*.m4a",
    "*.sofa",
]

EPS = 1e-12


@dataclass(slots=True)
class SimilaritySearchConfig:
    """Configuration for query-to-corpus similarity ranking."""

    input_path: Path
    corpus_dir: Path
    output_dir: Path
    top_k: int = 5
    mode: str = "auto"  # auto|feature|metric|metrics
    metric: str = "novelty_curve"
    metrics: list[str] | None = None
    distance: str = "cosine"  # cosine|euclidean|manhattan
    feature_set: str = "auto"  # auto|core|librosa|all
    frame_size: int = 1024
    hop_size: int = 256
    sample_rate: int | None = None
    normalize: bool = True
    include_self: bool = False
    recursive: bool = True
    max_files: int | None = None
    calibration: CalibrationProfile | None = None
    seed: int = 42


def _distance(a: np.ndarray, b: np.ndarray, kind: str) -> tuple[float, float]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if kind == "cosine":
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom <= EPS:
            return 1.0, 0.0
        sim = float(np.dot(x, y) / denom)
        sim = float(np.clip(sim, -1.0, 1.0))
        return float(1.0 - sim), sim
    if kind == "euclidean":
        dist = float(np.linalg.norm(x - y))
        return dist, float(1.0 / (1.0 + dist))
    if kind == "manhattan":
        dist = float(np.sum(np.abs(x - y)))
        return dist, float(1.0 / (1.0 + dist))
    raise ValueError(f"Unsupported distance: {kind}")


def _aggregate_feature_vector(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.size == 0:
        return np.zeros((2,), dtype=np.float64)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    vec = np.concatenate([mean, std], axis=0)
    vec = np.where(np.isfinite(vec), vec, 0.0)
    return vec.astype(np.float64, copy=False)


def _collect_candidates(cfg: SimilaritySearchConfig) -> list[Path]:
    files = iter_supported_files(cfg.corpus_dir, patterns=SUPPORTED_PATTERNS, recursive=cfg.recursive)
    q = cfg.input_path.resolve()
    out: list[Path] = []
    for f in files:
        if not cfg.include_self and f.resolve() == q:
            continue
        out.append(f)
    if cfg.max_files is not None and cfg.max_files >= 0:
        out = out[: int(cfg.max_files)]
    return out


def _mode(cfg: SimilaritySearchConfig) -> str:
    raw = str(cfg.mode).strip().lower()
    if raw == "auto":
        return "feature"
    if raw not in {"feature", "metric", "metrics"}:
        raise ValueError("mode must be one of auto|feature|metric|metrics")
    return raw


def _feature_search(cfg: SimilaritySearchConfig, candidates: list[Path]) -> dict[str, Any]:
    query_fv = extract_feature_vectors(
        cfg.input_path,
        feature_set=cfg.feature_set,
        frame_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        sample_rate=cfg.sample_rate,
    )
    qvec = _aggregate_feature_vector(query_fv.matrix)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for p in candidates:
        try:
            fv = extract_feature_vectors(
                p,
                feature_set=cfg.feature_set,
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate,
            )
            cvec = _aggregate_feature_vector(fv.matrix)
            dist, sim = _distance(qvec, cvec, cfg.distance)
            rows.append(
                {
                    "path": str(p),
                    "distance": dist,
                    "similarity": sim,
                    "feature_backend": fv.backend,
                    "num_frames": int(fv.matrix.shape[0]),
                    "num_features": int(fv.matrix.shape[1]),
                }
            )
        except Exception as exc:
            skipped.append({"path": str(p), "reason": str(exc)})

    rows.sort(key=lambda r: (float(r["distance"]), str(r["path"])))
    rows = rows[: max(1, int(cfg.top_k))]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return {
        "mode": "feature",
        "feature_set": cfg.feature_set,
        "distance": cfg.distance,
        "query": {
            "path": str(cfg.input_path),
            "feature_backend": query_fv.backend,
            "num_frames": int(query_fv.matrix.shape[0]),
            "num_features": int(query_fv.matrix.shape[1]),
        },
        "results": rows,
        "skipped": skipped,
    }


def _metric_vector(result: dict[str, Any], metric_names: list[str]) -> np.ndarray:
    payload = result.get("metrics", {})
    if not isinstance(payload, dict):
        return np.full((len(metric_names),), np.nan, dtype=np.float64)
    out: list[float] = []
    for name in metric_names:
        v = np.nan
        metric_payload = payload.get(name, {})
        if isinstance(metric_payload, dict):
            summary = metric_payload.get("summary", {})
            if isinstance(summary, dict):
                mean_v = summary.get("mean")
                if isinstance(mean_v, (int, float)):
                    v = float(mean_v)
        out.append(v)
    return np.array(out, dtype=np.float64)


def _metric_search(cfg: SimilaritySearchConfig, candidates: list[Path], multi: bool) -> dict[str, Any]:
    from esl.core.analyzer import analyze

    metric_names = list(cfg.metrics or [])
    if not metric_names:
        metric_names = [cfg.metric]
    if not multi:
        metric_names = [metric_names[0]]

    registry = create_registry(with_external=True)
    qcfg = AnalysisConfig(
        input_path=cfg.input_path,
        output_dir=cfg.output_dir,
        metrics=list(metric_names),
        frame_size=cfg.frame_size,
        hop_size=cfg.hop_size,
        sample_rate=cfg.sample_rate,
        calibration=cfg.calibration,
        verbosity=0,
        debug=0,
        seed=cfg.seed,
    )
    qres = analyze(qcfg, registry=registry)
    qvec = _metric_vector(qres, metric_names)
    if not np.all(np.isfinite(qvec)):
        raise RuntimeError(f"Query file has non-finite metric means for: {metric_names}")

    cand_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    cand_vecs: list[np.ndarray] = []
    for p in candidates:
        try:
            ccfg = AnalysisConfig(
                input_path=p,
                output_dir=cfg.output_dir,
                metrics=list(metric_names),
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                sample_rate=cfg.sample_rate,
                calibration=cfg.calibration,
                verbosity=0,
                debug=0,
                seed=cfg.seed,
            )
            cres = analyze(ccfg, registry=registry)
            cvec = _metric_vector(cres, metric_names)
            if not np.all(np.isfinite(cvec)):
                skipped.append({"path": str(p), "reason": "non-finite metric mean(s)"})
                continue
            cand_vecs.append(cvec)
            cand_rows.append(
                {
                    "path": str(p),
                    "metric_means": {name: float(v) for name, v in zip(metric_names, cvec.tolist())},
                    "metadata": {
                        "duration_s": float(cres.get("metadata", {}).get("duration_s", 0.0)),
                        "channels": int(cres.get("metadata", {}).get("channels", 0)),
                        "sample_rate": int(cres.get("metadata", {}).get("sample_rate", 0)),
                    },
                }
            )
        except Exception as exc:
            skipped.append({"path": str(p), "reason": str(exc)})

    if not cand_rows:
        return {
            "mode": "metrics" if multi else "metric",
            "metrics": metric_names,
            "distance": cfg.distance,
            "normalize": bool(cfg.normalize),
            "query": {"path": str(cfg.input_path), "metric_means": {name: float(v) for name, v in zip(metric_names, qvec.tolist())}},
            "results": [],
            "skipped": skipped,
        }

    cand_mat = np.vstack(cand_vecs)
    q_work = qvec.copy()
    c_work = cand_mat.copy()
    if multi and cfg.normalize:
        all_mat = np.vstack([q_work[None, :], c_work])
        mu = np.nanmean(all_mat, axis=0)
        sigma = np.nanstd(all_mat, axis=0)
        sigma = np.where(sigma < EPS, 1.0, sigma)
        q_work = (q_work - mu) / sigma
        c_work = (c_work - mu[None, :]) / sigma[None, :]

    rows: list[dict[str, Any]] = []
    for row, cvec, cvec_work in zip(cand_rows, cand_mat, c_work):
        if multi:
            dist, sim = _distance(q_work, cvec_work, cfg.distance)
            dist_kind = cfg.distance
        else:
            dist = float(abs(cvec[0] - qvec[0]))
            sim = float(1.0 / (1.0 + dist))
            dist_kind = "abs_diff"
        rows.append(
            {
                "path": row["path"],
                "distance": float(dist),
                "similarity": float(sim),
                "distance_kind": dist_kind,
                "metric_means": row["metric_means"],
                **row["metadata"],
            }
        )

    rows.sort(key=lambda r: (float(r["distance"]), str(r["path"])))
    rows = rows[: max(1, int(cfg.top_k))]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    return {
        "mode": "metrics" if multi else "metric",
        "metrics": metric_names,
        "distance": cfg.distance,
        "normalize": bool(cfg.normalize),
        "query": {"path": str(cfg.input_path), "metric_means": {name: float(v) for name, v in zip(metric_names, qvec.tolist())}},
        "results": rows,
        "skipped": skipped,
    }


def run_similarity_search(cfg: SimilaritySearchConfig) -> dict[str, Any]:
    """Run similarity search from query file to corpus directory."""
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.input_path}")
    if not cfg.corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {cfg.corpus_dir}")
    if int(cfg.top_k) < 1:
        raise ValueError("top_k must be >= 1")

    set_seed(cfg.seed)
    candidates = _collect_candidates(cfg)
    selected_mode = _mode(cfg)

    if selected_mode == "feature":
        body = _feature_search(cfg, candidates)
    elif selected_mode == "metric":
        body = _metric_search(cfg, candidates, multi=False)
    else:
        body = _metric_search(cfg, candidates, multi=True)

    return {
        "query_path": str(cfg.input_path.resolve()),
        "corpus_dir": str(cfg.corpus_dir.resolve()),
        "top_k": int(cfg.top_k),
        "mode_requested": str(cfg.mode),
        "mode_used": body.get("mode"),
        "distance": cfg.distance,
        "candidates_scanned": len(candidates),
        "include_self": bool(cfg.include_self),
        "recursive": bool(cfg.recursive),
        "max_files": cfg.max_files,
        "config": {
            "feature_set": cfg.feature_set,
            "metric": cfg.metric,
            "metrics": cfg.metrics or [],
            "frame_size": int(cfg.frame_size),
            "hop_size": int(cfg.hop_size),
            "sample_rate": cfg.sample_rate,
            "normalize": bool(cfg.normalize),
            "seed": int(cfg.seed),
        },
        **body,
    }
