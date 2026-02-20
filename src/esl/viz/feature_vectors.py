"""Frame-level feature vector extraction for similarity/novelty workflows.

References:
- Davis & Mermelstein (1980), MFCC/mel filterbank conventions.
- Foote (2000), similarity and novelty segmentation context.
- Librosa documentation for feature APIs (optional backend).
"""

from __future__ import annotations

import csv
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

from esl.core.audio import read_audio


EPS = 1e-12


@dataclass(slots=True)
class FeatureVectors:
    """Feature matrix and metadata for frame-level representations."""

    times_s: np.ndarray  # [frames]
    matrix: np.ndarray  # [frames, features]
    feature_names: list[str]
    backend: str
    sample_rate: int
    frame_size: int
    hop_size: int


def _has_librosa() -> bool:
    try:
        importlib.import_module("librosa")
        return True
    except Exception:
        return False


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 64, fmin: float = 20.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    m_min, m_max = _hz_to_mel(np.array([fmin, fmax], dtype=np.float64))
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


def _frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros((0, frame_size), dtype=np.float64)
    if x.size < frame_size:
        pad = np.zeros(frame_size - x.size, dtype=np.float64)
        x = np.concatenate([x, pad])
    frames: list[np.ndarray] = []
    for start in range(0, x.size - frame_size + 1, hop_size):
        frames.append(x[start : start + frame_size])
    if not frames:
        return np.zeros((0, frame_size), dtype=np.float64)
    return np.stack(frames, axis=0)


def _scipy_core_features(
    mono: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    n_mels: int,
) -> FeatureVectors:
    nperseg = min(frame_size, max(16, len(mono)))
    noverlap = min(nperseg - 1, max(0, nperseg - hop_size))
    freqs, times, z = stft(mono, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, boundary=None)
    if z.size == 0:
        return FeatureVectors(
            times_s=np.array([0.0], dtype=np.float64),
            matrix=np.zeros((1, 1), dtype=np.float64),
            feature_names=["constant_0"],
            backend="scipy-core",
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
        )
    mag = np.abs(z)
    power = np.square(mag)

    fb = _mel_filterbank(sample_rate, nperseg, n_mels=n_mels)
    mel = np.dot(fb, power)
    log_mel = np.log1p(np.maximum(mel, EPS))  # [mel, frames]

    den = np.sum(mag, axis=0) + EPS
    centroid = np.sum(freqs[:, None] * mag, axis=0) / den
    bandwidth = np.sqrt(np.sum(((freqs[:, None] - centroid[None, :]) ** 2) * mag, axis=0) / den)
    gm = np.exp(np.mean(np.log(np.maximum(mag, EPS)), axis=0))
    am = np.mean(mag, axis=0) + EPS
    flatness = gm / am
    csum = np.cumsum(power, axis=0)
    target = 0.85 * csum[-1]
    idx = np.argmax(csum >= target[None, :], axis=0)
    rolloff = freqs[idx]
    flux = np.zeros(mag.shape[1], dtype=np.float64)
    if mag.shape[1] > 1:
        flux[1:] = np.sqrt(np.sum(np.square(np.diff(mag, axis=1)), axis=0))

    frames = _frame_signal(mono, frame_size=frame_size, hop_size=hop_size)
    rms = np.sqrt(np.mean(np.square(frames), axis=1)) if frames.size else np.zeros((0,), dtype=np.float64)
    zcr = (
        np.mean(np.abs(np.diff(np.signbit(frames).astype(np.int8), axis=1)), axis=1) if frames.size else np.zeros((0,), dtype=np.float64)
    )

    frame_count = int(min(log_mel.shape[1], centroid.size, rms.size if rms.size else log_mel.shape[1]))
    if frame_count <= 0:
        frame_count = int(log_mel.shape[1])

    fixed = [
        ("spectral_centroid_hz", centroid[:frame_count]),
        ("spectral_bandwidth_hz", bandwidth[:frame_count]),
        ("spectral_flatness", flatness[:frame_count]),
        ("spectral_rolloff_hz", rolloff[:frame_count]),
        ("spectral_flux", flux[:frame_count]),
        ("rms_linear", rms[:frame_count] if rms.size else np.zeros(frame_count, dtype=np.float64)),
        ("zcr_ratio", zcr[:frame_count] if zcr.size else np.zeros(frame_count, dtype=np.float64)),
    ]
    mel_names = [f"log_mel_{i:02d}" for i in range(n_mels)]
    mel_data = log_mel[:n_mels, :frame_count]

    parts = [mel_data]
    names = list(mel_names)
    for nm, arr in fixed:
        parts.append(arr[None, :])
        names.append(nm)
    feat = np.vstack(parts).T  # [frames, features]
    return FeatureVectors(
        times_s=(times[:frame_count] if times.size >= frame_count else np.arange(frame_count) * (hop_size / sample_rate)),
        matrix=feat.astype(np.float64),
        feature_names=names,
        backend="scipy-core",
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )


def _librosa_rich_features(
    mono: np.ndarray,
    sample_rate: int,
    frame_size: int,
    hop_size: int,
    n_mels: int,
) -> FeatureVectors:
    # Optional dependency path: deliberately imported lazily.
    import librosa  # type: ignore

    y = mono.astype(np.float32)
    s = np.abs(
        librosa.stft(
            y=y,
            n_fft=frame_size,
            hop_length=hop_size,
            win_length=frame_size,
            center=False,
        )
    )
    p = np.square(s)

    mel = librosa.feature.melspectrogram(S=p, sr=sample_rate, n_mels=n_mels, fmin=20.0, fmax=sample_rate / 2.0)
    log_mel = np.log1p(np.maximum(mel, EPS))
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(np.maximum(mel, EPS), ref=np.max), sr=sample_rate, n_mfcc=20)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(S=p, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    contrast = librosa.feature.spectral_contrast(S=s, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
    rms = librosa.feature.rms(S=s)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_size, hop_length=hop_size, center=False)
    centroid = librosa.feature.spectral_centroid(S=s, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(S=s, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(S=s, sr=sample_rate, roll_percent=0.85)
    flatness = librosa.feature.spectral_flatness(S=p)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(np.maximum(p, EPS), ref=np.max), sr=sample_rate, hop_length=hop_size)
    onset = onset[None, :]

    blocks: list[tuple[str, np.ndarray]] = [
        ("log_mel", log_mel),
        ("mfcc", mfcc),
        ("mfcc_delta", d1),
        ("mfcc_delta2", d2),
        ("chroma", chroma),
        ("spectral_contrast", contrast),
        ("tonnetz", tonnetz),
        ("rms", rms),
        ("zcr", zcr),
        ("spectral_centroid", centroid),
        ("spectral_bandwidth", bandwidth),
        ("spectral_rolloff", rolloff),
        ("spectral_flatness", flatness),
        ("onset_strength", onset),
    ]
    min_frames = min(arr.shape[1] for _, arr in blocks if arr.ndim == 2 and arr.shape[1] > 0)
    if min_frames <= 0:
        return _scipy_core_features(mono, sample_rate, frame_size, hop_size, n_mels=n_mels)

    name_parts: list[str] = []
    feat_parts: list[np.ndarray] = []
    for prefix, arr in blocks:
        arr2 = arr[:, :min_frames]
        feat_parts.append(arr2)
        for i in range(arr2.shape[0]):
            name_parts.append(f"{prefix}_{i:02d}")
    feat = np.vstack(feat_parts).T
    times = np.arange(min_frames, dtype=np.float64) * (hop_size / sample_rate)
    return FeatureVectors(
        times_s=times,
        matrix=feat.astype(np.float64),
        feature_names=name_parts,
        backend="librosa-rich",
        sample_rate=sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
    )


def _combine_feature_sets(primary: FeatureVectors, secondary: FeatureVectors, backend_name: str) -> FeatureVectors:
    frame_count = int(min(primary.matrix.shape[0], secondary.matrix.shape[0], primary.times_s.size, secondary.times_s.size))
    if frame_count <= 0:
        return primary
    mat = np.concatenate([primary.matrix[:frame_count], secondary.matrix[:frame_count]], axis=1)
    names = list(primary.feature_names) + list(secondary.feature_names)
    return FeatureVectors(
        times_s=primary.times_s[:frame_count],
        matrix=mat,
        feature_names=names,
        backend=backend_name,
        sample_rate=primary.sample_rate,
        frame_size=primary.frame_size,
        hop_size=primary.hop_size,
    )


def extract_feature_vectors(
    audio_path: str | Path,
    *,
    feature_set: str = "auto",
    frame_size: int = 1024,
    hop_size: int = 256,
    n_mels: int = 64,
    sample_rate: int | None = None,
) -> FeatureVectors:
    """Extract frame-level features for similarity/anomaly workflows.

    Supported sets:
    - `auto`: prefers librosa-rich when available, otherwise scipy-core
    - `core`: scipy-core descriptors + log-mel bank
    - `librosa`: librosa-rich stack
    - `all`: librosa-rich + scipy-core concatenation
    """
    fset = str(feature_set).strip().lower()
    if fset not in {"auto", "core", "librosa", "all"}:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    buf = read_audio(audio_path, target_sr=sample_rate)
    mono = np.mean(buf.samples, axis=1).astype(np.float64)

    core = _scipy_core_features(
        mono=mono,
        sample_rate=buf.sample_rate,
        frame_size=frame_size,
        hop_size=hop_size,
        n_mels=n_mels,
    )

    if fset == "core":
        return core
    if fset in {"auto", "librosa", "all"} and _has_librosa():
        rich = _librosa_rich_features(
            mono=mono,
            sample_rate=buf.sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            n_mels=n_mels,
        )
        if fset == "librosa" or fset == "auto":
            return rich
        return _combine_feature_sets(rich, core, backend_name="librosa-rich+scipy-core")

    if fset == "librosa":
        raise RuntimeError("feature_set=librosa requested, but librosa is not installed.")
    return core


def similarity_matrix_from_features(matrix: np.ndarray) -> np.ndarray:
    """Compute cosine self-similarity matrix from [frames, features] vectors."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Feature matrix must be 2-D [frames, features].")
    if x.shape[0] == 0:
        return np.zeros((1, 1), dtype=np.float64)
    x = x - np.mean(x, axis=1, keepdims=True)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), EPS)
    ssm = np.dot(x, x.T)
    return np.clip(ssm, 0.0, 1.0)


def save_feature_vectors(features: FeatureVectors, out_path: str | Path) -> Path:
    """Save feature vectors to `.npz`, `.npy`, or `.csv`."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if ext == ".npz":
        np.savez_compressed(
            p,
            features=features.matrix.astype(np.float32),
            times_s=features.times_s.astype(np.float64),
            feature_names=np.array(features.feature_names, dtype=object),
            backend=np.array([features.backend], dtype=object),
            sample_rate=np.array([features.sample_rate], dtype=np.int64),
            frame_size=np.array([features.frame_size], dtype=np.int64),
            hop_size=np.array([features.hop_size], dtype=np.int64),
        )
        return p
    if ext == ".npy":
        np.save(p, features.matrix.astype(np.float32))
        return p
    if ext == ".csv":
        with p.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", *features.feature_names])
            for i in range(features.matrix.shape[0]):
                writer.writerow([float(features.times_s[i]), *[float(x) for x in features.matrix[i]]])
        return p
    raise ValueError(f"Unsupported feature output extension: {p.suffix}")


def load_feature_vectors(path: str | Path) -> FeatureVectors:
    """Load feature vectors from `.npz`, `.npy`, or `.csv`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature vectors file not found: {p}")
    ext = p.suffix.lower()
    if ext == ".npz":
        z = np.load(p, allow_pickle=True)
        matrix = np.asarray(z["features"], dtype=np.float64)
        times = np.asarray(z["times_s"], dtype=np.float64)
        names_raw = z["feature_names"]
        names = [str(x) for x in names_raw.tolist()]
        backend = str(z.get("backend", np.array(["unknown"], dtype=object))[0])
        sr_arr = z.get("sample_rate", np.array([0], dtype=np.int64))
        fs_arr = z.get("frame_size", np.array([0], dtype=np.int64))
        hs_arr = z.get("hop_size", np.array([0], dtype=np.int64))
        return FeatureVectors(
            times_s=times,
            matrix=matrix,
            feature_names=names,
            backend=backend,
            sample_rate=int(sr_arr[0]),
            frame_size=int(fs_arr[0]),
            hop_size=int(hs_arr[0]),
        )
    if ext == ".npy":
        matrix = np.asarray(np.load(p), dtype=np.float64)
        if matrix.ndim != 2:
            raise RuntimeError("Expected .npy feature vectors to be [frames, features].")
        times = np.arange(matrix.shape[0], dtype=np.float64)
        names = [f"f{i:03d}" for i in range(matrix.shape[1])]
        return FeatureVectors(
            times_s=times,
            matrix=matrix,
            feature_names=names,
            backend="npy",
            sample_rate=0,
            frame_size=0,
            hop_size=0,
        )
    if ext == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise RuntimeError("CSV feature file is empty.")
        header = rows[0]
        data_rows = rows[1:]
        if not data_rows:
            raise RuntimeError("CSV feature file has no data rows.")
        numeric = np.array([[float(x) for x in row] for row in data_rows], dtype=np.float64)
        if numeric.shape[1] < 2:
            raise RuntimeError("CSV feature file must contain time_s + at least one feature column.")
        return FeatureVectors(
            times_s=numeric[:, 0],
            matrix=numeric[:, 1:],
            feature_names=[str(x) for x in header[1:]],
            backend="csv",
            sample_rate=0,
            frame_size=0,
            hop_size=0,
        )
    raise ValueError(f"Unsupported feature vectors extension: {p.suffix}")
