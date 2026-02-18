"""Visualization engine for static and interactive acoustic plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, stft

from esl.core.audio import read_audio


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_line(ts: list[float], values: list[float], title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, values, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    m_min, m_max = _hz_to_mel(np.array([fmin, fmax]))
    mel_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for j in range(left, center):
            if 0 <= j < fb.shape[1]:
                fb[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            if 0 <= j < fb.shape[1]:
                fb[i - 1, j] = (right - j) / max(right - center, 1)
    return fb


def _plot_spectral_suite(audio_path: str | Path, out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    buf = read_audio(audio_path)
    mono = np.mean(buf.samples, axis=1)

    n_fft = 2048
    hop = 512
    f, t, z = stft(mono, fs=buf.sample_rate, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    mag = np.abs(z)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    # Spectrogram
    p_spec = out_dir / "spectrogram.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=[t.min() if t.size else 0, t.max() if t.size else 0, f.min() if f.size else 0, f.max() if f.size else 0],
        cmap="magma",
    )
    ax.set_title("Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(p_spec, dpi=150)
    plt.close(fig)
    paths.append(p_spec)

    # Mel spectrogram
    fb = _mel_filterbank(buf.sample_rate, n_fft, n_mels=64)
    mel = np.dot(fb, np.square(mag))
    mel_db = 10.0 * np.log10(np.maximum(mel, 1e-12))
    p_mel = out_dir / "mel_spectrogram.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(mel_db, origin="lower", aspect="auto", extent=[t.min() if t.size else 0, t.max() if t.size else 0, 0, mel_db.shape[0]], cmap="viridis")
    ax.set_title("Mel Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel band")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(p_mel, dpi=150)
    plt.close(fig)
    paths.append(p_mel)

    # Log-frequency spectrogram
    p_log = out_dir / "log_frequency_spectrogram.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    f_safe = np.maximum(f, 1.0)
    mesh = ax.pcolormesh(t, f_safe, mag_db, shading="auto", cmap="plasma")
    ax.set_yscale("log")
    ax.set_ylim(max(20.0, float(np.min(f_safe))), float(np.max(f_safe)))
    ax.set_title("Log-Frequency Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(p_log, dpi=150)
    plt.close(fig)
    paths.append(p_log)

    # Waterfall
    p_water = out_dir / "waterfall.png"
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    if t.size and f.size:
        step_t = max(1, t.size // 30)
        step_f = max(1, f.size // 80)
        t_sub = t[::step_t]
        f_sub = f[::step_f]
        z_sub = mag_db[::step_f, ::step_t]
        tt, ff = np.meshgrid(t_sub, f_sub)
        ax.plot_surface(tt, ff, z_sub, cmap="cividis", linewidth=0, antialiased=False)
    ax.set_title("Waterfall Spectrum")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("dB")
    fig.tight_layout()
    fig.savefig(p_water, dpi=150)
    plt.close(fig)
    paths.append(p_water)

    # LTSA
    p_ltsa = out_dir / "ltsa.png"
    ltsa = np.mean(mag_db, axis=1) if mag_db.size else np.array([])
    fig, ax = plt.subplots(figsize=(10, 4))
    if ltsa.size:
        ax.plot(f, ltsa)
    ax.set_title("Long-Term Spectral Average")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean level (dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p_ltsa, dpi=150)
    plt.close(fig)
    paths.append(p_ltsa)

    return paths


def _compute_similarity_matrix(
    audio_path: str | Path,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute frame-level self-similarity matrix from log-mel features."""
    buf = read_audio(audio_path)
    mono = np.mean(buf.samples, axis=1)
    nperseg = min(n_fft, max(16, len(mono)))
    noverlap = min(nperseg - 1, max(0, nperseg - hop))
    _, t, z = stft(mono, fs=buf.sample_rate, nperseg=nperseg, noverlap=noverlap, boundary=None)
    if z.size == 0:
        return np.array([0.0]), np.zeros((1, 1), dtype=np.float64)

    mag = np.abs(z)
    fb = _mel_filterbank(buf.sample_rate, n_fft=nperseg, n_mels=n_mels)
    mel = np.dot(fb, np.square(mag))  # [mel, frames]
    feat = np.log1p(mel.T)  # [frames, mel]
    feat = feat - np.mean(feat, axis=1, keepdims=True)
    feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    ssm = np.dot(feat, feat.T)
    ssm = np.clip(ssm, 0.0, 1.0)
    return t, ssm


def _plot_similarity_matrix(audio_path: str | Path, out_dir: Path) -> Path:
    """Render self-similarity matrix plot."""
    t, ssm = _compute_similarity_matrix(audio_path)
    out_path = out_dir / "similarity_matrix.png"
    fig, ax = plt.subplots(figsize=(8, 7))
    if t.size > 1:
        extent = [float(t[0]), float(t[-1]), float(t[0]), float(t[-1])]
        im = ax.imshow(ssm, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Time (s)")
    else:
        im = ax.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Frame")
    ax.set_title("Self-Similarity Matrix")
    fig.colorbar(im, ax=ax, label="Similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _checkerboard_kernel(L: int, sigma_scale: float = 0.5) -> np.ndarray:
    """Build Foote-style Gaussian checkerboard kernel of size (2L+1)."""
    if L < 1:
        raise ValueError("L must be >= 1")
    axis = np.arange(-L, L + 1, dtype=np.float64)
    sigma = max(1e-6, L * float(sigma_scale))
    g = np.exp(-(axis**2) / (2.0 * sigma * sigma))
    gauss = np.outer(g, g)

    sign = np.zeros((2 * L + 1, 2 * L + 1), dtype=np.float64)
    sign[:L, :L] = 1.0
    sign[L + 1 :, L + 1 :] = 1.0
    sign[:L, L + 1 :] = -1.0
    sign[L + 1 :, :L] = -1.0
    # center row and center column remain zero by design.
    return gauss * sign


def compute_novelty_matrix(
    audio_path: str | Path,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    kernel_size: int = 32,
    kernel_sigma_scale: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Compute novelty-matrix components:
    - self-similarity matrix
    - checkerboard kernel
    - novelty curve
    - novelty peaks
    """
    t, ssm = _compute_similarity_matrix(audio_path, n_fft=n_fft, hop=hop, n_mels=n_mels)
    n_frames = int(ssm.shape[0])
    if n_frames <= 1:
        return {
            "times_s": np.array([0.0], dtype=np.float64),
            "ssm": np.array([[1.0]], dtype=np.float64),
            "kernel": np.array([[0.0]], dtype=np.float64),
            "novelty": np.array([0.0], dtype=np.float64),
            "peaks_idx": np.array([], dtype=np.int64),
            "peaks_s": np.array([], dtype=np.float64),
        }

    L_target = max(2, kernel_size // 2)
    L_max = max(1, (n_frames - 1) // 2)
    L = int(min(L_target, L_max))
    kernel = _checkerboard_kernel(L, sigma_scale=kernel_sigma_scale)

    novelty = np.zeros(n_frames, dtype=np.float64)
    for idx in range(L, n_frames - L):
        local = ssm[idx - L : idx + L + 1, idx - L : idx + L + 1]
        novelty[idx] = float(np.sum(local * kernel))

    novelty = np.maximum(novelty, 0.0)
    max_nov = float(np.max(novelty))
    if max_nov > 0:
        novelty = novelty / max_nov

    peak_distance = max(1, L // 2)
    prominence = 0.08 if max_nov > 0 else 0.0
    peaks_idx, _ = find_peaks(novelty, distance=peak_distance, prominence=prominence)
    peaks_s = t[peaks_idx] if t.size > 0 else np.array([], dtype=np.float64)

    return {
        "times_s": t.astype(np.float64),
        "ssm": ssm.astype(np.float64),
        "kernel": kernel.astype(np.float64),
        "novelty": novelty.astype(np.float64),
        "peaks_idx": peaks_idx.astype(np.int64),
        "peaks_s": peaks_s.astype(np.float64),
    }


def plot_novelty_matrix(
    audio_path: str | Path,
    output_dir: str | Path,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    kernel_size: int = 32,
    kernel_sigma_scale: float = 0.5,
) -> Path:
    """Render novelty matrix figure (SSM + kernel + novelty curve)."""
    out_dir = Path(output_dir)
    _ensure_dir(out_dir)
    data = compute_novelty_matrix(
        audio_path=audio_path,
        n_fft=n_fft,
        hop=hop,
        n_mels=n_mels,
        kernel_size=kernel_size,
        kernel_sigma_scale=kernel_sigma_scale,
    )
    t = data["times_s"]
    ssm = data["ssm"]
    kernel = data["kernel"]
    novelty = data["novelty"]
    peaks_idx = data["peaks_idx"]

    out_path = out_dir / "novelty_matrix.png"

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.32)

    # SSM
    ax0 = fig.add_subplot(gs[:, 0])
    if t.size > 1:
        extent = [float(t[0]), float(t[-1]), float(t[0]), float(t[-1])]
        im0 = ax0.imshow(ssm, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        for p in peaks_idx.tolist():
            if p < t.size:
                px = float(t[p])
                ax0.axvline(px, color="white", alpha=0.25, linewidth=0.7)
                ax0.axhline(px, color="white", alpha=0.25, linewidth=0.7)
        ax0.set_xlabel("Time (s)")
        ax0.set_ylabel("Time (s)")
    else:
        im0 = ax0.imshow(ssm, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.set_xlabel("Frame")
        ax0.set_ylabel("Frame")
    ax0.set_title("Self-Similarity Matrix")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="Similarity")

    # Checkerboard kernel
    ax1 = fig.add_subplot(gs[0, 1])
    vmax = float(np.max(np.abs(kernel))) if kernel.size else 1.0
    vmax = max(vmax, 1e-6)
    im1 = ax1.imshow(kernel, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_title("Checkerboard Kernel")
    ax1.set_xlabel("Kernel X")
    ax1.set_ylabel("Kernel Y")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Weight")

    # Novelty curve
    ax2 = fig.add_subplot(gs[1, 1])
    if t.size == novelty.size and t.size > 0:
        ax2.plot(t, novelty, linewidth=1.4, color="#0ea5e9")
        if peaks_idx.size > 0:
            ax2.scatter(t[peaks_idx], novelty[peaks_idx], color="#ef4444", s=18, zorder=5, label="Peaks")
        ax2.set_xlabel("Time (s)")
    else:
        x = np.arange(novelty.size)
        ax2.plot(x, novelty, linewidth=1.4, color="#0ea5e9")
        if peaks_idx.size > 0:
            ax2.scatter(peaks_idx, novelty[peaks_idx], color="#ef4444", s=18, zorder=5, label="Peaks")
        ax2.set_xlabel("Frame")
    ax2.set_ylabel("Novelty (norm.)")
    ax2.set_ylim(0.0, max(1.05, float(np.max(novelty) * 1.05 if novelty.size else 1.0)))
    ax2.set_title("Novelty Curve")
    ax2.grid(True, alpha=0.3)
    if peaks_idx.size > 0:
        ax2.legend(loc="upper right", frameon=False)

    fig.suptitle("Novelty Matrix (Foote-style)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_metric_lines(
    result: dict[str, Any],
    out_dir: Path,
    include_metrics: set[str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    to_plot = [
        ("spl_a_db", "SPL A over time", "dBA", "spl_a_over_time.png"),
        ("spl_c_db", "SPL C over time", "dBC", "spl_c_over_time.png"),
        ("spl_z_db", "SPL Z over time", "dB", "spl_z_over_time.png"),
        ("crest_factor_db", "Crest Factor over time", "dB", "crest_factor_over_time.png"),
        ("snr_db", "SNR over time", "dB", "snr_over_time.png"),
        ("novelty_curve", "Acoustic Novelty Curve", "a.u.", "novelty_curve.png"),
        ("bioacoustic_index", "Bioacoustic Index Trend", "a.u.", "bioacoustic_index_trend.png"),
        (
            "acoustic_complexity_index",
            "Acoustic Complexity Index Trend",
            "ratio",
            "acoustic_complexity_index_trend.png",
        ),
    ]

    metrics = result.get("metrics", {})
    for metric_name, title, ylabel, fname in to_plot:
        if include_metrics is not None and metric_name not in include_metrics:
            continue
        payload = metrics.get(metric_name)
        if not payload:
            continue
        ts = payload.get("timestamps_s", [])
        vals = payload.get("series", [])
        if not ts or not vals:
            continue
        p = out_dir / fname
        _save_line(ts, vals, title, ylabel, p)
        paths.append(p)

    rt = metrics.get("rt60_s")
    if include_metrics is not None and "rt60_s" not in include_metrics:
        rt = None
    if rt and rt.get("extra", {}).get("time") and rt.get("extra", {}).get("decay_db"):
        p_decay = out_dir / "rt60_decay_curve.png"
        _save_line(
            rt["extra"]["time"],
            rt["extra"]["decay_db"],
            "RT60 Decay Curve",
            "Decay (dB)",
            p_decay,
        )
        paths.append(p_decay)

    return paths


def _plot_interactive(
    result: dict[str, Any],
    out_dir: Path,
    include_metrics: set[str] | None = None,
) -> Path | None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return None

    fig = go.Figure()
    metrics = result.get("metrics", {})
    for name, payload in metrics.items():
        if include_metrics is not None and name not in include_metrics:
            continue
        ts = payload.get("timestamps_s", [])
        vals = payload.get("series", [])
        if ts and vals:
            fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=name))
    if not fig.data:
        return None

    fig.update_layout(
        title="esl interactive metric trends",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        template="plotly_white",
    )
    out_path = out_dir / "interactive_metrics.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def plot_analysis(
    result: dict[str, Any],
    output_dir: str | Path,
    audio_path: str | Path | None = None,
    interactive: bool = False,
    include_metrics: list[str] | None = None,
    include_spectral: bool = True,
    include_similarity_matrix: bool = False,
    include_novelty_matrix: bool = False,
) -> list[str]:
    """Create standard static and optional interactive plots."""
    out = Path(output_dir)
    _ensure_dir(out)
    include_set = set(include_metrics) if include_metrics else None

    artifacts: list[Path] = []
    artifacts.extend(_plot_metric_lines(result, out, include_metrics=include_set))

    source_audio = audio_path or result.get("metadata", {}).get("input_path")
    if source_audio and include_spectral:
        try:
            artifacts.extend(_plot_spectral_suite(source_audio, out))
        except Exception:
            pass
    if source_audio and include_similarity_matrix:
        try:
            artifacts.append(_plot_similarity_matrix(source_audio, out))
        except Exception:
            pass
    if source_audio and include_novelty_matrix:
        try:
            artifacts.append(plot_novelty_matrix(source_audio, out))
        except Exception:
            pass

    if interactive:
        html = _plot_interactive(result, out, include_metrics=include_set)
        if html is not None:
            artifacts.append(html)

    return [str(p) for p in artifacts]


def plot_from_json(
    json_path: str | Path,
    output_dir: str | Path,
    interactive: bool = False,
    audio_path: str | Path | None = None,
    include_metrics: list[str] | None = None,
    include_spectral: bool = True,
    include_similarity_matrix: bool = False,
    include_novelty_matrix: bool = False,
) -> list[str]:
    """Load analysis JSON and produce plots."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return plot_analysis(
        payload,
        output_dir=output_dir,
        audio_path=audio_path,
        interactive=interactive,
        include_metrics=include_metrics,
        include_spectral=include_spectral,
        include_similarity_matrix=include_similarity_matrix,
        include_novelty_matrix=include_novelty_matrix,
    )
