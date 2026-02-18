"""Utility helpers for reproducibility and serialization."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Set deterministic seeds for supported RNG backends."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional; deterministic behavior still holds for numpy/random paths.
        pass


def canonicalize(value: Any) -> Any:
    """Convert nested values into JSON-serializable canonical structures."""
    if is_dataclass(value):
        return canonicalize(asdict(value))
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def config_hash(config: Any) -> str:
    """Create stable content hash for configuration provenance."""
    payload = json.dumps(canonicalize(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
