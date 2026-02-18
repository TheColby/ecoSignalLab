"""Online dataset ingestion backends (Freesound, HuggingFace, HTTP)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import soundfile as sf

from esl.core.config import IngestConfig


REQUEST_TIMEOUT_S = 30


def _ensure_dirs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    audio_dir = root / "audio"
    meta_dir = root / "metadata"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return audio_dir, meta_dir


def _download(url: str, out_path: Path, headers: dict[str, str] | None = None) -> None:
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_S, headers=headers) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def ingest_http(query: str, limit: int, output_dir: Path) -> list[dict[str, Any]]:
    """Ingest one or more direct HTTP URLs (newline/comma-separated)."""
    audio_dir, _ = _ensure_dirs(output_dir)
    urls = [u.strip() for u in query.replace("\n", ",").split(",") if u.strip()]
    urls = urls[: max(1, limit)]
    rows: list[dict[str, Any]] = []
    for i, url in enumerate(urls):
        suffix = Path(url.split("?")[0]).suffix or ".bin"
        out_path = audio_dir / f"http_{i:03d}{suffix}"
        _download(url, out_path)
        rows.append({"source": "http", "url": url, "local_path": str(out_path)})
    return rows


def ingest_freesound(query: str, limit: int, output_dir: Path, api_key: str) -> list[dict[str, Any]]:
    """Ingest Freesound previews using API token from env or parameter."""
    audio_dir, _ = _ensure_dirs(output_dir)
    headers = {"Authorization": f"Token {api_key}"}
    params = {
        "query": query,
        "page_size": min(max(limit, 1), 150),
        "fields": "id,name,username,license,previews,url,duration,tags",
    }
    r = requests.get(
        "https://freesound.org/apiv2/search/text/",
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT_S,
    )
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results", [])
    rows: list[dict[str, Any]] = []

    for i, item in enumerate(results[:limit]):
        previews = item.get("previews", {}) or {}
        preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
        if not preview_url:
            continue
        out_path = audio_dir / f"freesound_{item.get('id', i)}.mp3"
        _download(preview_url, out_path, headers=headers)
        rows.append(
            {
                "source": "freesound",
                "id": item.get("id"),
                "name": item.get("name"),
                "username": item.get("username"),
                "license": item.get("license"),
                "duration": item.get("duration"),
                "url": item.get("url"),
                "tags": item.get("tags", []),
                "local_path": str(out_path),
            }
        )
    return rows


def ingest_huggingface(dataset_name: str, limit: int, output_dir: Path) -> list[dict[str, Any]]:
    """Ingest audio samples from a HuggingFace dataset id."""
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError("datasets package is required for HuggingFace ingestion") from exc

    audio_dir, _ = _ensure_dirs(output_dir)
    ds = load_dataset(dataset_name, split="train")

    # Pick first likely audio column.
    audio_col = None
    for col in ds.column_names:
        if "audio" in col.lower() or col.lower() in {"wav", "waveform"}:
            audio_col = col
            break
    if audio_col is None:
        raise RuntimeError(f"No audio-like column found in dataset {dataset_name}")

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        audio_obj = item[audio_col]
        if not isinstance(audio_obj, dict) or "array" not in audio_obj or "sampling_rate" not in audio_obj:
            continue
        arr = audio_obj["array"]
        sr = int(audio_obj["sampling_rate"])
        out_path = audio_dir / f"hf_{i:04d}.wav"
        sf.write(out_path, arr, sr)
        rows.append(
            {
                "source": "huggingface",
                "dataset": dataset_name,
                "index": i,
                "sampling_rate": sr,
                "local_path": str(out_path),
            }
        )
    return rows


def ingest(config: IngestConfig) -> dict[str, Any]:
    """Run ingestion job and return manifest payload."""
    out_root = config.output_dir
    audio_dir, meta_dir = _ensure_dirs(out_root)
    _ = audio_dir  # explicit namespace anchor

    if config.source == "freesound":
        api_key = os.getenv("FREESOUND_API_KEY", "")
        if not api_key:
            raise RuntimeError("FREESOUND_API_KEY must be set for freesound ingestion.")
        items = ingest_freesound(config.query, config.limit, out_root, api_key)
    elif config.source == "huggingface":
        items = ingest_huggingface(config.query, config.limit, out_root)
    elif config.source == "http":
        items = ingest_http(config.query, config.limit, out_root)
    else:
        raise ValueError(f"Unsupported source: {config.source}")

    manifest = {
        "source": config.source,
        "query": config.query,
        "limit": config.limit,
        "output_dir": str(out_root.resolve()),
        "num_items": len(items),
        "items": items,
        "terms_notice": (
            "User is responsible for complying with source terms/licenses and attribution requirements."
        ),
    }
    manifest_path = meta_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
