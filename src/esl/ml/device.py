"""Compute-device discovery and lightweight backend benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

import numpy as np


DeviceName = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(slots=True)
class DeviceResolution:
    """Resolved compute-device information."""

    requested: str
    resolved: str
    strict: bool
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    mps_available: bool
    device_name: str | None
    reason: str | None = None


def _torch_state() -> tuple[Any | None, bool, bool, str | None, str | None]:
    """Return torch module + capability flags without hard dependency."""
    try:
        import torch
    except Exception:
        return None, False, False, None, None

    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(
        hasattr(torch.backends, "mps")
        and getattr(torch.backends.mps, "is_available", lambda: False)()
    )
    dev_name: str | None = None
    if cuda_ok:
        try:
            dev_name = str(torch.cuda.get_device_name(0))
        except Exception:
            dev_name = "CUDA device"
    elif mps_ok:
        dev_name = "Apple Metal (MPS)"

    return torch, cuda_ok, mps_ok, str(getattr(torch, "__version__", None)), dev_name


def resolve_compute_device(requested: str = "auto", strict: bool = False) -> DeviceResolution:
    """Resolve compute device for optional tensor workflows.

    Policy:
    - `auto`: prefer `cuda`, then `mps`, else `cpu`.
    - explicit unavailable accelerators:
      - raise on `strict=True`
      - otherwise fall back to `cpu` with reason metadata.
    """
    req = str(requested or "auto").strip().lower()
    if req not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device '{requested}'. Use auto|cpu|cuda|mps.")

    torch_mod, cuda_ok, mps_ok, torch_ver, dev_name = _torch_state()
    torch_ok = torch_mod is not None

    def _fail_or_cpu(msg: str) -> DeviceResolution:
        if strict:
            raise RuntimeError(msg)
        return DeviceResolution(
            requested=req,
            resolved="cpu",
            strict=strict,
            torch_available=torch_ok,
            torch_version=torch_ver,
            cuda_available=cuda_ok,
            mps_available=mps_ok,
            device_name=None,
            reason=msg,
        )

    if req == "cpu":
        return DeviceResolution(
            requested=req,
            resolved="cpu",
            strict=strict,
            torch_available=torch_ok,
            torch_version=torch_ver,
            cuda_available=cuda_ok,
            mps_available=mps_ok,
            device_name=None,
            reason=None,
        )

    if not torch_ok:
        if req == "auto":
            return DeviceResolution(
                requested=req,
                resolved="cpu",
                strict=strict,
                torch_available=False,
                torch_version=None,
                cuda_available=False,
                mps_available=False,
                device_name=None,
                reason="PyTorch not installed; falling back to CPU.",
            )
        return _fail_or_cpu(f"Requested '{req}' but PyTorch is not installed.")

    if req == "auto":
        if cuda_ok:
            return DeviceResolution(
                requested=req,
                resolved="cuda",
                strict=strict,
                torch_available=True,
                torch_version=torch_ver,
                cuda_available=True,
                mps_available=mps_ok,
                device_name=dev_name,
                reason=None,
            )
        if mps_ok:
            return DeviceResolution(
                requested=req,
                resolved="mps",
                strict=strict,
                torch_available=True,
                torch_version=torch_ver,
                cuda_available=False,
                mps_available=True,
                device_name=dev_name,
                reason=None,
            )
        return DeviceResolution(
            requested=req,
            resolved="cpu",
            strict=strict,
            torch_available=True,
            torch_version=torch_ver,
            cuda_available=False,
            mps_available=False,
            device_name=None,
            reason="No CUDA/MPS accelerator available; using CPU.",
        )

    if req == "cuda":
        if cuda_ok:
            return DeviceResolution(
                requested=req,
                resolved="cuda",
                strict=strict,
                torch_available=True,
                torch_version=torch_ver,
                cuda_available=True,
                mps_available=mps_ok,
                device_name=dev_name,
                reason=None,
            )
        return _fail_or_cpu("Requested CUDA but no CUDA device is available.")

    # req == "mps"
    if mps_ok:
        return DeviceResolution(
            requested=req,
            resolved="mps",
            strict=strict,
            torch_available=True,
            torch_version=torch_ver,
            cuda_available=cuda_ok,
            mps_available=True,
            device_name=dev_name,
            reason=None,
        )
    return _fail_or_cpu("Requested MPS but MPS is not available.")


def device_resolution_dict(info: DeviceResolution) -> dict[str, Any]:
    """JSON-safe representation for metadata/provenance."""
    return {
        "requested": info.requested,
        "resolved": info.resolved,
        "strict": bool(info.strict),
        "torch_available": bool(info.torch_available),
        "torch_version": info.torch_version,
        "cuda_available": bool(info.cuda_available),
        "mps_available": bool(info.mps_available),
        "device_name": info.device_name,
        "reason": info.reason,
    }


def _sync_torch(torch_mod: Any, resolved: str) -> None:
    if resolved == "cuda":
        try:
            torch_mod.cuda.synchronize()
        except Exception:
            pass
    elif resolved == "mps" and hasattr(torch_mod, "mps"):
        try:
            torch_mod.mps.synchronize()
        except Exception:
            pass


def benchmark_tensor_backend(
    device: DeviceName = "auto",
    channels: int = 1,
    frames: int = 16_384,
    features: int = 256,
    iters: int = 20,
    seed: int = 42,
    strict: bool = False,
) -> dict[str, Any]:
    """Benchmark a simple tensor workload for device sanity/perf checks."""
    info = resolve_compute_device(requested=device, strict=strict)
    channels_i = max(1, int(channels))
    frames_i = max(8, int(frames))
    features_i = max(8, int(features))
    iters_i = max(1, int(iters))
    rows = channels_i * frames_i

    torch_mod, *_ = _torch_state()
    if torch_mod is not None and info.resolved in {"cpu", "cuda", "mps"}:
        torch_mod.manual_seed(int(seed))
        x = torch_mod.rand((rows, features_i), dtype=torch_mod.float32, device=info.resolved)
        w = torch_mod.rand((features_i, features_i), dtype=torch_mod.float32, device=info.resolved)
        # Warmup
        _ = torch_mod.relu(x @ w)
        _sync_torch(torch_mod, info.resolved)
        t0 = time.perf_counter()
        for _ in range(iters_i):
            y = torch_mod.relu(x @ w)
        _sync_torch(torch_mod, info.resolved)
        dt = time.perf_counter() - t0
        elems = float(rows * features_i * iters_i)
        return {
            "backend": "torch",
            "device": device_resolution_dict(info),
            "shape": {"channels": channels_i, "frames": frames_i, "features": features_i},
            "iters": iters_i,
            "seconds_total": float(dt),
            "seconds_per_iter": float(dt / iters_i),
            "throughput_mel_per_s": float(elems / max(dt, 1e-12) / 1_000_000.0),
            "result_abs_mean": float(torch_mod.mean(torch_mod.abs(y)).item()),
        }

    # Guaranteed fallback benchmark so command works without torch.
    rng = np.random.default_rng(int(seed))
    x_np = rng.random((rows, features_i), dtype=np.float32)
    w_np = rng.random((features_i, features_i), dtype=np.float32)
    _ = np.maximum(x_np @ w_np, 0.0)
    t0 = time.perf_counter()
    for _ in range(iters_i):
        y_np = np.maximum(x_np @ w_np, 0.0)
    dt = time.perf_counter() - t0
    elems = float(rows * features_i * iters_i)
    return {
        "backend": "numpy",
        "device": device_resolution_dict(info),
        "shape": {"channels": channels_i, "frames": frames_i, "features": features_i},
        "iters": iters_i,
        "seconds_total": float(dt),
        "seconds_per_iter": float(dt / iters_i),
        "throughput_mel_per_s": float(elems / max(dt, 1e-12) / 1_000_000.0),
        "result_abs_mean": float(np.mean(np.abs(y_np))),
    }
