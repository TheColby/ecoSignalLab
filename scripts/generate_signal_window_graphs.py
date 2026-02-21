#!/usr/bin/env python3
"""Generate signal/window visualizations for docs.

The goal is to provide concrete DSP visuals for onboarding and reference docs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import chirp, get_window, stft

# Headless-safe backend for CI/sandbox environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _signal(sr: int, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(sr * duration_s))
    t = np.arange(n, dtype=np.float64) / float(sr)
    base = 0.25 * np.sin(2.0 * np.pi * 180.0 * t)
    sweep = 0.18 * chirp(t, f0=120.0, f1=2200.0, t1=duration_s, method="logarithmic")
    tone = 0.14 * np.sin(2.0 * np.pi * 510.0 * t)
    x = base + sweep + tone
    for pulse_t in (0.35, 0.95, 1.35, 1.75):
        idx = int(round(pulse_t * sr))
        span = int(0.012 * sr)
        lo = max(0, idx - span)
        hi = min(n, idx + span)
        win = np.hanning(max(8, hi - lo))
        x[lo:hi] += 0.45 * win
    x = np.clip(x, -0.95, 0.95)
    return t, x


def _plot_signal_waveform(t: np.ndarray, x: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, x, color="#0b4ea2", lw=1.0)
    ax.set_title("Synthetic Analysis Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "signal_waveform.png")


def _plot_frame_hop_overlay(t: np.ndarray, x: np.ndarray, sr: int, frame: int, hop: int, out_dir: Path) -> None:
    view_s = 0.28
    n = int(view_s * sr)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(t[:n], x[:n], color="#0f172a", lw=1.2)
    frame_starts = list(range(0, max(1, n - frame), hop))[:8]
    for idx, s in enumerate(frame_starts):
        a = s / sr
        b = min(n, s + frame) / sr
        ax.axvspan(a, b, color="#93c5fd", alpha=0.16 if idx % 2 == 0 else 0.08)
        ax.axvline(a, color="#1d4ed8", lw=0.8, alpha=0.75)
    ax.set_title(f"Frame/Hop Overlay (frame={frame} samples, hop={hop} samples)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.22)
    _save(fig, out_dir, "frame_hop_overlay.png")


def _plot_window_family(frame: int, out_dir: Path) -> None:
    n = np.arange(frame, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    for name, color in (
        ("boxcar", "#111827"),
        ("hann", "#1d4ed8"),
        ("hamming", "#0ea5e9"),
        ("blackman", "#7c3aed"),
    ):
        w = get_window(name, frame, fftbins=True)
        ax.plot(n, w, lw=1.3, label=name, color=color)
    ax.set_title("Window Function Family")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Window Weight")
    ax.grid(alpha=0.22)
    ax.legend(ncol=4, frameon=False)
    _save(fig, out_dir, "window_family.png")


def _plot_overlap_add(frame: int, out_dir: Path) -> None:
    hop = frame // 2
    w = get_window("hann", frame, fftbins=True)
    total = frame + 4 * hop
    acc = np.zeros(total, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 3.5))
    for k in range(4):
        s = k * hop
        acc[s : s + frame] += w
        xx = np.arange(s, s + frame, dtype=np.int64)
        ax.plot(xx, w, lw=0.9, alpha=0.55, color="#60a5fa")
    ax.plot(acc, color="#0b4ea2", lw=2.1, label="sum of shifted windows")
    ax.set_title("Hann Overlap-Add (50% overlap)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.24)
    ax.legend(frameon=False)
    _save(fig, out_dir, "overlap_add_hann_50pct.png")


def _plot_spectrogram_with_frames(x: np.ndarray, sr: int, frame: int, hop: int, out_dir: Path) -> None:
    f, t, z = stft(x, fs=sr, nperseg=frame, noverlap=frame - hop, boundary=None)
    mag = 20.0 * np.log10(np.abs(z) + 1e-8)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    im = ax.pcolormesh(t, f, mag, shading="auto", cmap="magma")
    for tc in t[:: max(1, len(t) // 18)]:
        ax.axvline(float(tc), color="white", alpha=0.16, lw=0.7)
    ax.set_title("Spectrogram with Frame Centers")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Magnitude (dB)")
    _save(fig, out_dir, "spectrogram_with_frames.png")


def _plot_spectral_flux_demo(x: np.ndarray, sr: int, frame: int, hop: int, out_dir: Path) -> None:
    f, _, z = stft(x, fs=sr, nperseg=frame, noverlap=frame - hop, boundary=None)
    mag = np.abs(z)
    k0 = max(2, mag.shape[1] // 3)
    a = mag[:, k0 - 1]
    b = mag[:, k0]
    pos_diff = np.maximum(b - a, 0.0)
    fig, ax = plt.subplots(figsize=(10.8, 3.8))
    ax.plot(f, a, lw=1.0, color="#111827", label="|X(f,k-1)|")
    ax.plot(f, b, lw=1.0, color="#1d4ed8", label="|X(f,k)|")
    ax.fill_between(f, 0.0, pos_diff, color="#93c5fd", alpha=0.6, label="positive diff")
    ax.set_xlim(0, min(sr / 2.0, 5000.0))
    ax.set_title("Spectral Flux Positive Difference")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _save(fig, out_dir, "spectral_flux_positive_diff.png")


def _plot_checkerboard_kernel(out_dir: Path, size: int = 33, sigma_scale: float = 0.5) -> None:
    if size % 2 == 0:
        size += 1
    half = size // 2
    u = np.arange(-half, half + 1)
    v = np.arange(-half, half + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    sigma = max(1.0, sigma_scale * half)
    gauss = np.exp(-(uu**2 + vv**2) / (2.0 * sigma**2))
    checker = np.sign(uu * vv)
    kernel = gauss * checker
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    im = ax.imshow(kernel, cmap="coolwarm", origin="lower")
    ax.set_title("Foote Checkerboard Kernel")
    ax.set_xlabel("v")
    ax.set_ylabel("u")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, "checkerboard_kernel.png")


def _plot_multichannel(sr: int, duration_s: float, out_dir: Path) -> None:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    ch1 = 0.4 * np.sin(2.0 * np.pi * 160.0 * t)
    ch2 = 0.33 * np.sin(2.0 * np.pi * 160.0 * (t - 0.0014))
    ch3 = 0.22 * np.sin(2.0 * np.pi * 350.0 * t)
    ch4 = 0.18 * np.sin(2.0 * np.pi * 600.0 * t) + 0.08 * np.sin(2.0 * np.pi * 90.0 * t)
    stack = np.stack([ch1, ch2, ch3, ch4], axis=1)
    fig, axs = plt.subplots(4, 1, figsize=(11, 6.2), sharex=True)
    labels = ("ch1", "ch2", "ch3", "ch4")
    for i, ax in enumerate(axs):
        ax.plot(t[: int(0.12 * sr)], stack[: int(0.12 * sr), i], lw=1.0, color="#0b4ea2")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.2)
    axs[0].set_title("Multichannel Waveforms (4 channels)")
    axs[-1].set_xlabel("Time (s)")
    _save(fig, out_dir, "multichannel_waveforms.png")


def _plot_foa(sr: int, duration_s: float, out_dir: Path) -> None:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    w = 0.35 * np.sin(2.0 * np.pi * 220.0 * t)
    x = 0.28 * np.sin(2.0 * np.pi * 220.0 * t + 0.5)
    y = 0.23 * np.sin(2.0 * np.pi * 220.0 * t - 0.7)
    z = 0.2 * np.sin(2.0 * np.pi * 110.0 * t)
    foa = np.stack([w, x, y, z], axis=1)
    fig, axs = plt.subplots(4, 1, figsize=(11, 6.2), sharex=True)
    labels = ("W", "X", "Y", "Z")
    for i, ax in enumerate(axs):
        ax.plot(t[: int(0.14 * sr)], foa[: int(0.14 * sr), i], lw=1.0, color="#312e81")
        ax.set_ylabel(labels[i])
        ax.grid(alpha=0.2)
    axs[0].set_title("FOA Channel Example (WXYZ)")
    axs[-1].set_xlabel("Time (s)")
    _save(fig, out_dir, "foa_wxyz_waveforms.png")


def _plot_chunk_vs_frame_timeline(duration_s: float, chunk_s: float, frame_s: float, hop_s: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 2.8))
    ax.set_title("Streaming Chunks vs Frame/Hop Timeline")
    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 2.2)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([0.75, 1.65])
    ax.set_yticklabels(["frames (hop)", "chunks"])
    ax.grid(axis="x", alpha=0.2)

    c = 0.0
    idx = 0
    while c < duration_s - 1e-9:
        end = min(duration_s, c + chunk_s)
        ax.add_patch(
            plt.Rectangle((c, 1.35), end - c, 0.55, color="#bfdbfe", alpha=0.65, ec="#1d4ed8", lw=0.6)
        )
        ax.text(c + 0.04, 1.62, f"chunk {idx}", fontsize=7.5, color="#1d4ed8")
        c += chunk_s
        idx += 1

    k = 0.0
    while k < duration_s - 1e-9:
        ax.add_patch(
            plt.Rectangle((k, 0.5), frame_s, 0.4, color="#93c5fd", alpha=0.35, ec="#1e3a8a", lw=0.35)
        )
        k += hop_s
    _save(fig, out_dir, "chunk_vs_frame_timeline.png")


def generate(out_dir: Path, sr: int = 16000, duration_s: float = 2.0, frame: int = 1024, hop: int = 256) -> None:
    _mkdir(out_dir)
    t, x = _signal(sr=sr, duration_s=duration_s)
    _plot_signal_waveform(t=t, x=x, out_dir=out_dir)
    _plot_frame_hop_overlay(t=t, x=x, sr=sr, frame=frame, hop=hop, out_dir=out_dir)
    _plot_window_family(frame=frame, out_dir=out_dir)
    _plot_overlap_add(frame=frame, out_dir=out_dir)
    _plot_spectrogram_with_frames(x=x, sr=sr, frame=frame, hop=hop, out_dir=out_dir)
    _plot_spectral_flux_demo(x=x, sr=sr, frame=frame, hop=hop, out_dir=out_dir)
    _plot_checkerboard_kernel(out_dir=out_dir)
    _plot_multichannel(sr=sr, duration_s=duration_s, out_dir=out_dir)
    _plot_foa(sr=sr, duration_s=duration_s, out_dir=out_dir)
    _plot_chunk_vs_frame_timeline(duration_s=12.0, chunk_s=3.0, frame_s=frame / sr, hop_s=hop / sr, out_dir=out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate signal/window visual assets for docs.")
    parser.add_argument("--out", default="docs/examples/signal_window_guide", help="Output directory")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Synthetic signal sample rate")
    parser.add_argument("--duration", type=float, default=2.0, help="Synthetic signal duration (seconds)")
    parser.add_argument("--frame-size", type=int, default=1024)
    parser.add_argument("--hop-size", type=int, default=256)
    args = parser.parse_args()

    generate(
        out_dir=Path(args.out),
        sr=int(args.sample_rate),
        duration_s=float(args.duration),
        frame=int(args.frame_size),
        hop=int(args.hop_size),
    )
    print(f"wrote assets to: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
