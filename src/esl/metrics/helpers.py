"""Shared helper functions for metric computations.

Algorithm references:
- STFT framing/spectral analysis:
  Allen & Rabiner (1977), \"A unified approach to short-time Fourier analysis and synthesis\"
  IEEE Proc. 65(11):1558-1564. doi:10.1109/PROC.1977.10770
- Spectral-flux novelty:
  Dixon (2006), \"Onset detection revisited\", DAFx.
- Schroeder energy decay integration:
  Schroeder (1965), JASA 37(3):409.
- Reverberation-time line fitting conventions:
  ISO 3382-1:2009; ISO 3382-2:2008.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

from esl.core.calibration import db, power_db


@dataclass(slots=True)
class FramedSignal:
    """Framed signal with frame start times."""

    frames: np.ndarray  # [num_frames, frame_size, channels]
    times_s: np.ndarray  # [num_frames]


def frame_signal(samples: np.ndarray, sample_rate: int, frame_size: int, hop_size: int) -> FramedSignal:
    """Frame [samples, channels] into overlapping windows with tail padding."""
    if samples.ndim != 2:
        raise ValueError("Expected [samples, channels] signal")
    n, ch = samples.shape
    if n == 0:
        pad = np.zeros((frame_size, ch), dtype=np.float32)
        return FramedSignal(frames=pad[None, :, :], times_s=np.array([0.0], dtype=np.float64))

    if n < frame_size:
        padded = np.zeros((frame_size, ch), dtype=np.float32)
        padded[:n] = samples
        return FramedSignal(frames=padded[None, :, :], times_s=np.array([0.0], dtype=np.float64))

    starts = list(range(0, n - frame_size + 1, hop_size))
    remainder = (n - frame_size) % hop_size
    if remainder != 0:
        starts.append(n - frame_size)

    out = np.zeros((len(starts), frame_size, ch), dtype=np.float32)
    for i, s in enumerate(starts):
        out[i] = samples[s : s + frame_size]
    times = np.array(starts, dtype=np.float64) / sample_rate
    return FramedSignal(frames=out, times_s=times)


def frame_rms(frames: np.ndarray) -> np.ndarray:
    """RMS per frame aggregated across channels."""
    return np.sqrt(np.mean(np.square(frames), axis=(1, 2)))


def frame_peak(frames: np.ndarray) -> np.ndarray:
    """Peak absolute amplitude per frame."""
    return np.max(np.abs(frames), axis=(1, 2))


def spectral_features(
    mono: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency bins, times, magnitude spectrum."""
    # Canonical STFT decomposition (Allen & Rabiner, 1977).
    nperseg = min(frame_size, max(8, len(mono)))
    noverlap = min(nperseg - 1, max(0, nperseg - hop_size))
    f, t, z = stft(
        mono,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    mag = np.abs(z) + 1e-12
    return f, t, mag


def summarize(series: np.ndarray) -> dict[str, float]:
    """Standard summary statistics."""
    return {
        "mean": float(np.mean(series)),
        "std": float(np.std(series)),
        "min": float(np.min(series)),
        "max": float(np.max(series)),
        "p50": float(np.percentile(series, 50.0)),
        "p95": float(np.percentile(series, 95.0)),
    }


def crest_factor_db(rms: np.ndarray, peak: np.ndarray) -> np.ndarray:
    """Crest factor in dB."""
    return db(np.maximum(peak, 1e-12) / np.maximum(rms, 1e-12))


def estimate_snr_db(series_db: np.ndarray) -> np.ndarray:
    """Estimate SNR by frame-level percentile separation."""
    noise_floor = np.percentile(series_db, 10.0)
    signal_level = np.percentile(series_db, 90.0)
    snr = signal_level - noise_floor
    return np.repeat(snr, len(series_db))


def novelty_from_spectrum(mag: np.ndarray) -> np.ndarray:
    """Novelty curve from positive spectral flux."""
    # Positive spectral flux as a lightweight onset/novelty function (Dixon, 2006).
    diff = np.diff(mag, axis=1, prepend=mag[:, :1])
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
    return flux


def schroeder_decay(ir: np.ndarray) -> np.ndarray:
    """Energy decay curve (dB) from impulse response."""
    # Reverse cumulative energy integration after Schroeder (1965).
    # Normalize and sanitize to avoid overflow on unstable decoded/filtered IR tails.
    x = np.nan_to_num(np.asarray(ir, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.0:
        x = x / peak
    e = np.square(x)
    rev_cumsum = np.cumsum(e[::-1])[::-1]
    rev_cumsum /= max(rev_cumsum[0], 1e-20)
    return power_db(rev_cumsum)


def fit_decay_time(decay_db: np.ndarray, t: np.ndarray, lo_db: float, hi_db: float) -> float:
    """Fit RT-style decay time using linear regression on decay segment."""
    # ISO 3382-family style RT extrapolation: RT60 = -60 / slope(decay segment).
    return float(fit_decay_diagnostics(decay_db, t, lo_db, hi_db)["rt60_s"])


def fit_decay_diagnostics(
    decay_db: np.ndarray,
    t: np.ndarray,
    lo_db: float,
    hi_db: float,
) -> dict[str, float | int | list[float]]:
    """Return linear-fit diagnostics used for RT-style extrapolation."""
    mask = (decay_db <= lo_db) & (decay_db >= hi_db)
    points = int(np.count_nonzero(mask))
    if points < 8:
        return {
            "rt60_s": float("nan"),
            "slope_db_per_s": float("nan"),
            "intercept_db": float("nan"),
            "r2": float("nan"),
            "num_points": points,
            "fit_range_db": [float(lo_db), float(hi_db)],
        }

    x = t[mask]
    y = decay_db[mask]
    a, b = np.polyfit(x, y, deg=1)
    y_hat = a * x + b

    ss_res = float(np.sum(np.square(y - y_hat)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-20)))
    rt60 = float(-60.0 / a) if a < 0 else float("nan")

    return {
        "rt60_s": rt60,
        "slope_db_per_s": float(a),
        "intercept_db": float(b),
        "r2": r2,
        "num_points": points,
        "fit_range_db": [float(lo_db), float(hi_db)],
    }
