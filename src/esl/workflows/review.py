"""Portable, bounded human annotation packs for native moments manifests."""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from esl.version import __version__

_MAX_BYTES = 16 * 1024 * 1024
_MAX_ROWS = 10_000
_FIELDS = ["clip_id", "start_s", "end_s", "rank_score", "status", "label", "notes"]
_STATUSES = ("pending", "accepted", "rejected", "uncertain")


def _read_text(path: Path) -> str:
    with path.open("rb") as stream:
        data = stream.read(_MAX_BYTES + 1)
    if len(data) > _MAX_BYTES:
        raise ValueError(f"Manifest exceeds {_MAX_BYTES} bytes: {path}")
    return data.decode("utf-8-sig")


def _read_csv(path: Path) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(_read_text(path)))
    rows = []
    for row in reader:
        if len(rows) >= _MAX_ROWS:
            raise ValueError(f"CSV exceeds {_MAX_ROWS} rows")
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row")
        rows.append(row)
    return rows


def _resolve(reference: str, base: Path) -> Path:
    path = Path(reference).expanduser()
    if path.is_absolute():
        return path
    # Native exporters can record paths relative to the invocation directory.
    local = base / path
    return local if local.exists() else path.resolve()


def _load_moments(manifest: Path) -> tuple[list[dict[str, str]], Path]:
    if manifest.suffix.lower() == ".json":
        report = json.loads(_read_text(manifest))
        if not isinstance(report, dict) or not isinstance(report.get("csv_path"), str):
            raise ValueError("Moments JSON must contain a csv_path string")
        manifest = _resolve(report["csv_path"], manifest.parent)
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError("Moments manifest has no rows")
    ids = [row.get("clip_id", "") for row in rows]
    if any(not value or value != value.strip() or value[0] in "=+-@" for value in ids):
        raise ValueError("clip_id must be nonempty, trimmed and spreadsheet-safe")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate clip_id in moments manifest")
    return rows, manifest.parent


def _score(row: dict[str, str]) -> float:
    try:
        score = float(row.get("rank_score", "0"))
    except ValueError as exc:
        raise ValueError("rank_score must be finite numeric data") from exc
    if not math.isfinite(score):
        raise ValueError("rank_score must be finite numeric data")
    return score


def _labels(path: Path | None, ids: set[str]) -> dict[str, dict[str, str]]:
    result = {}
    if path is None:
        return result
    for row in _read_csv(path):
        key = row.get("clip_id", "")
        if key not in ids or key in result:
            raise ValueError(f"Unknown or duplicate annotation clip_id: {key}")
        if row.get("status") not in _STATUSES:
            raise ValueError(f"Invalid annotation status for {key}")
        result[key] = row
    return result


def _preview(row: dict[str, str], base: Path, seconds: float) -> tuple[np.ndarray, int]:
    reference = row.get("wav_path", "")
    if not reference:
        raise ValueError(f"Missing wav_path for {row['clip_id']}; extract clips first")
    with sf.SoundFile(_resolve(reference, base)) as source:
        if source.channels > 32 or source.samplerate > 384_000:
            raise ValueError("Preview supports at most 32 channels and 384000 Hz")
        frames = min(source.frames, int(seconds * source.samplerate), 4_000_000)
        if frames < 2:
            raise ValueError(f"Clip is empty or too short: {reference}")
        # Read small blocks so channel count does not multiply peak preview memory.
        blocks = []
        remaining = frames
        while remaining:
            block = source.read(min(remaining, 65536), dtype="float32", always_2d=True)
            if not len(block):
                break
            blocks.append(block.mean(axis=1))
            remaining -= len(block)
        data = np.concatenate(blocks)
        if not np.isfinite(data).all():
            raise ValueError(f"Nonfinite audio samples: {reference}")
        return data, source.samplerate


def _plots(data: np.ndarray, rate: int, assets: Path, stem: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 2.3), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    # Min/max envelopes retain brief peaks without plotting millions of vertices.
    count = min(1200, len(data))
    edges = np.linspace(0, len(data), count + 1, dtype=int)
    lows = [float(data[a:b].min()) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    highs = [float(data[a:b].max()) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    axis.fill_between(edges[:-1] / rate, lows, highs, color="#21726a")
    axis.set(xlabel="Preview time (s)", ylabel="Amplitude (FS)")
    figure.savefig(assets / f"{stem}_waveform.png", dpi=100)
    figure.clear()
    axis = figure.subplots()
    window = min(1024, len(data))
    frequencies, times, power = spectrogram(
        data,
        fs=rate,
        window="hann",
        nperseg=window,
        noverlap=window // 2,
        detrend=False,
        scaling="density",
    )
    # Bound rendered columns while retaining the maximum energy in each group.
    width = max(1, math.ceil(power.shape[1] / 1200))
    columns = [power[:, i : i + width].max(axis=1) for i in range(0, power.shape[1], width)]
    image = axis.pcolormesh(
        times[::width],
        frequencies,
        10 * np.log10(np.maximum(np.stack(columns, axis=1), 1e-12)),
        shading="nearest",
        cmap="magma",
        vmin=-120,
    )
    axis.set(xlabel="Preview time (s)", ylabel="Frequency (Hz)")
    figure.colorbar(image, ax=axis, label="dB FS^2/Hz")
    figure.savefig(assets / f"{stem}_spectrogram.png", dpi=100)
    figure.clear()


def _safe_cell(value: str) -> str:
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def _write_html(items: list[dict[str, Any]], path: Path) -> None:
    cards = []
    for item in items:
        escaped = {key: html.escape(str(value), quote=True) for key, value in item.items()}
        options = "".join(
            f"<option{' selected' if status == item['status'] else ''}>{status}</option>"
            for status in _STATUSES
        )
        fields = "".join(f'<td class="meta">{escaped[key]}</td>' for key in _FIELDS[:4])
        cards.append(
            f"<section><h2>{escaped['clip_id']}</h2>"
            f"<p>Preview: {escaped['preview_seconds']} seconds; mono downmix.</p>"
            f'<audio controls preload="none" src="{escaped["audio"]}"></audio>'
            f'<img loading="lazy" src="{escaped["waveform"]}" alt="Waveform">'
            f'<img loading="lazy" src="{escaped["spectrogram"]}" alt="Spectrogram">'
            f"<table><thead><tr><th>ID</th><th>Start (s)</th><th>End (s)</th>"
            f"<th>Rank</th><th>Status</th><th>Label</th><th>Notes</th></tr></thead>"
            f'<tbody><tr>{fields}<td><select aria-label="Status">{options}</select></td>'
            f'<td><input aria-label="Label" value="{escaped["label"]}"></td>'
            f'<td><textarea aria-label="Notes">{escaped["notes"]}</textarea></td>'
            "</tr></tbody></table></section>"
        )
    header = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ecoSignalLab Review Queue</title><style>
body{font-family:Georgia,serif;background:#f5f1e8;color:#18352f;max-width:1000px;
margin:auto;padding:24px}section{border-top:2px solid #21726a;padding:20px 0}
img{display:block;max-width:100%;height:auto}table{display:block;overflow-x:auto}
th,td{text-align:left;padding:6px}input,textarea{max-width:180px}button{padding:12px}
</style><h1>Review Queue</h1><p>Edits are not saved automatically. Download labels before
closing this page. Original event times are retained; previews start at clip time zero.</p>
<button id="download" type="button">Download labels.csv</button>
"""
    script = r"""<script>
const fields=['clip_id','start_s','end_s','rank_score','status','label','notes'];
function cell(value){
  if (/^[\s]*[=+\-@]/.test(value)) value="'"+value;
  return '"'+value.replaceAll('"','""')+'"';
}
document.getElementById('download').addEventListener('click',()=>{
  const rows=[fields];
  document.querySelectorAll('tbody tr').forEach(tr=>{
    const meta=Array.from(tr.querySelectorAll('.meta'),td=>td.textContent);
    const values=Array.from(tr.querySelectorAll('select,input,textarea'),el=>el.value);
    rows.push([...meta,...values]);
  });
  const blob=new Blob([rows.map(r=>r.map(cell).join(',')).join('\r\n')+'\r\n'],
                      {type:'text/csv;charset=utf-8'});
  const url=URL.createObjectURL(blob), a=document.createElement('a');
  a.href=url;a.download='labels.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
</script></html>"""
    path.write_text(header + "\n".join(cards) + script, encoding="utf-8")


def build_review_pack(
    manifest: Path,
    output_dir: Path,
    *,
    max_items: int = 100,
    preview_seconds: float = 30.0,
    labels_path: Path | None = None,
) -> Path:
    """Build a new pack atomically; never overwrite an existing destination."""
    manifest, output_dir = Path(manifest).resolve(), Path(output_dir).absolute()
    if not 1 <= max_items <= 1000:
        raise ValueError("max_items must be between 1 and 1000")
    if not math.isfinite(preview_seconds) or not 0 < preview_seconds <= 120:
        raise ValueError("preview_seconds must be finite, greater than 0 and at most 120")
    if output_dir.exists():
        raise ValueError(f"Output already exists: {output_dir}")
    rows, base = _load_moments(manifest)
    annotations = _labels(labels_path, {row["clip_id"] for row in rows})
    rows.sort(key=_score, reverse=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".review-", dir=output_dir.parent))
    try:
        assets = temporary / "assets"
        assets.mkdir()
        items = []
        for index, row in enumerate(rows[:max_items], start=1):
            stem = f"item_{index:04d}"
            data, rate = _preview(row, base, preview_seconds)
            sf.write(assets / f"{stem}.wav", data, rate, subtype="PCM_16")
            _plots(data, rate, assets, stem)
            annotation = annotations.get(row["clip_id"], {})
            item = {key: row.get(key, "") for key in _FIELDS[:4]}
            item.update({key: annotation.get(key, "") for key in ("label", "notes")})
            item.update(
                status=annotation.get("status", "pending"),
                preview_seconds=round(len(data) / rate, 6),
                audio=f"assets/{stem}.wav",
                waveform=f"assets/{stem}_waveform.png",
                spectrogram=f"assets/{stem}_spectrogram.png",
            )
            items.append(item)
        with (temporary / "labels.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=_FIELDS)
            writer.writeheader()
            writer.writerows({key: _safe_cell(str(item[key])) for key in _FIELDS} for item in items)
        _write_html(items, temporary / "index.html")
        report = {
            "schema_version": "1.0.0",
            "artifact_kind": "review_pack",
            "esl_version": __version__,
            "provenance": {
                "moments_manifest": str(manifest),
                "labels_import": str(labels_path.resolve()) if labels_path else None,
            },
            "config": {
                "max_items": max_items,
                "preview_seconds": preview_seconds,
                "order": "descending_rank_score_stable",
                "downmix": "channel_mean",
                "max_frames_per_clip": 4_000_000,
            },
            "candidate_count": len(rows),
            "item_count": len(items),
            "items": items,
        }
        (temporary / "review_manifest.json").write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / "index.html"


def _run(args: argparse.Namespace) -> int:
    path = build_review_pack(
        Path(args.manifest),
        Path(args.out),
        max_items=args.max_items,
        preview_seconds=args.preview_seconds,
        labels_path=Path(args.labels) if args.labels else None,
    )
    print(path)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register ``esl review MANIFEST --out PACK``."""
    parser = subparsers.add_parser("review", help="Build an offline moments annotation pack")
    parser.add_argument("manifest", help="Native moments CSV or report JSON with csv_path")
    parser.add_argument("--out", required=True, help="New output directory (must not exist)")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--preview-seconds", type=float, default=30.0)
    parser.add_argument("--labels", help="Previously exported labels.csv to resume annotations")
    parser.set_defaults(func=_run)


def register_moments(subparsers: argparse._SubParsersAction) -> None:
    """Register the identical ``esl moments review`` alias."""
    register(subparsers)
