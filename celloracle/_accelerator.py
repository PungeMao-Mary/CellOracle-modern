# -*- coding: utf-8 -*-
"""
CellOracle accelerator backend
==============================

This module provides a small, dependency-light abstraction over the dense
linear-algebra primitives that dominate the runtime of CellOracle.

Two acceleration paths are supported on macOS:

``"cpu"``
    NumPy / SciPy.  On macOS the official NumPy wheels link BLAS and LAPACK
    against Apple's **Accelerate** framework (``vecLib``), so dense ``matmul``
    already runs on Apple's matrix accelerator.  Use
    :func:`blas_info` to confirm what NumPy is actually linked against.

``"gpu"``
    Apple **Metal** GPU through `MLX <https://github.com/ml-explore/mlx>`_
    (Apple silicon only).  MLX is an optional dependency: when it is not
    installed CellOracle silently falls back to the CPU path.

Backend selection
-----------------
The backend can be selected with, in order of precedence:

1. :func:`set_backend` (programmatic).
2. The ``CELLORACLE_ACCELERATOR`` environment variable, one of
   ``"auto"``, ``"cpu"`` or ``"gpu"``.
3. The default, ``"auto"``: use the GPU when MLX is importable *and* a Metal
   device is available, otherwise the CPU.

``"auto"`` never uses the GPU for tiny problems: every operation has a
threshold below which the host CPU is faster because of the memory-transfer
overhead.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "backend",
    "set_backend",
    "get_backend",
    "gpu_available",
    "blas_info",
    "accelerator_info",
    "to_gpu",
    "to_numpy",
    "mx_matmul",
    "using_gpu",
]


ENV_VAR = "CELLORACLE_ACCELERATOR"
_VALID_BACKENDS = ("auto", "cpu", "gpu")

# Operations smaller than these never go to the GPU: the round-trip through
# unified memory would cost more than the computation itself.
MATMUL_GPU_MIN_FLOPS = 2_000_000
CORRELATION_GPU_MIN_WORK = 2_000_000

_lock = threading.RLock()
_mlx: Optional[Any] = None
_mlx_checked = False
_mlx_error: Optional[str] = None
_backend: Optional[str] = None
_requested: Optional[str] = None


# ---------------------------------------------------------------------------
# MLX discovery
# ---------------------------------------------------------------------------
def _import_mlx() -> Optional[Any]:
    """Import :mod:`mlx.core` once and cache the result (or the failure)."""
    global _mlx, _mlx_checked, _mlx_error
    if _mlx_checked:
        return _mlx
    with _lock:
        if _mlx_checked:
            return _mlx
        _mlx_checked = True
        if os.environ.get("CELLORACLE_DISABLE_MLX", "").lower() in ("1", "true", "yes"):
            _mlx_error = "disabled by CELLORACLE_DISABLE_MLX"
            _mlx = None
            return None
        try:
            import mlx.core as mx  # type: ignore

            # Force device initialisation so that failures surface here.
            mx.eval(mx.array([0.0]))
            _mlx = mx
        except Exception as exc:  # pragma: no cover - depends on host
            _mlx = None
            _mlx_error = f"{type(exc).__name__}: {exc}"
        return _mlx


def _metal_available() -> bool:
    mx = _import_mlx()
    if mx is None:
        return False
    try:
        metal = getattr(mx, "metal", None)
        if metal is not None and hasattr(metal, "is_available"):
            return bool(metal.is_available())
        # mlx >= 0.22: device reported by mx.default_device()
        return "gpu" in str(mx.default_device()).lower()
    except Exception:
        return False


def gpu_available() -> bool:
    """Return ``True`` when an Apple Metal GPU usable by MLX is present."""
    return _metal_available()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
def _env_backend() -> str:
    value = os.environ.get(ENV_VAR, "auto").strip().lower()
    return value if value in _VALID_BACKENDS else "auto"


def _resolve(name: str) -> str:
    if name == "cpu":
        return "cpu"
    if name == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "GPU backend requested but no Metal GPU / MLX runtime is available. "
                "Install MLX (`pip install mlx`) on an Apple-silicon Mac, or use backend='cpu'."
            )
        return "gpu"
    # auto
    return "gpu" if gpu_available() else "cpu"


def _requested_backend() -> str:
    """The backend explicitly requested (set_backend) or from the environment."""
    if _requested is not None:
        return _requested
    return _env_backend()


def set_backend(name: str = "auto") -> str:
    """Select the accelerator backend.

    Args:
        name: ``"auto"``, ``"cpu"`` or ``"gpu"``.

    Returns:
        The resolved backend (``"cpu"`` or ``"gpu"``).
    """
    global _backend, _requested
    name = (name or "auto").strip().lower()
    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; expected one of {_VALID_BACKENDS}.")
    with _lock:
        _requested = name
        _backend = _resolve(name)
        return _backend


def get_backend() -> str:
    """Return the currently resolved backend (``"cpu"`` or ``"gpu"``)."""
    global _backend
    with _lock:
        if _backend is None:
            _backend = _resolve(_env_backend())
        return _backend


def backend() -> str:
    """Alias of :func:`get_backend`."""
    return get_backend()


def using_gpu() -> bool:
    return get_backend() == "gpu"


def _mx() -> Optional[Any]:
    return _import_mlx() if using_gpu() else None


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------
def blas_info() -> Dict[str, Any]:
    """Report the BLAS/LAPACK implementation NumPy is linked against.

    On macOS this is normally ``"accelerate"``, i.e. Apple's matrix accelerator.
    """
    info: Dict[str, Any] = {"blas": None, "lapack": None, "openblas": None}
    try:
        import numpy.__config__ as cfg  # type: ignore

        d = cfg.show(mode="dicts").get("Build Dependencies", {})
        info["blas"] = (d.get("blas") or {}).get("name")
        info["lapack"] = (d.get("lapack") or {}).get("name")
        info["openblas"] = (d.get("blas") or {}).get("openblas configuration")
    except Exception:
        pass
    return info


def accelerator_info() -> Dict[str, Any]:
    """Return a dictionary summarising the available acceleration hardware."""
    mx = _import_mlx()
    out: Dict[str, Any] = {
        "backend": get_backend(),
        "requested": _requested_backend(),
        "gpu_available": gpu_available(),
        "mlx_installed": mx is not None,
        "mlx_version": getattr(mx, "__version__", None) if mx is not None else None,
        "device": str(mx.default_device()) if mx is not None else None,
        "mlx_error": _mlx_error,
        "blas": blas_info(),
        "platform": None,
    }
    try:
        import platform

        out["platform"] = {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Array transfer helpers
# ---------------------------------------------------------------------------
def to_gpu(array: np.ndarray, dtype: Any = None):
    """Move a NumPy array to the MLX device used by CellOracle.

    MLX/Metal does not support ``float64``; arrays are cast to ``float32``
    unless an explicit ``dtype`` is given.
    """
    mx = _mx()
    if mx is None:
        return array
    arr = np.ascontiguousarray(array)
    if dtype is None:
        dtype = mx.float32 if arr.dtype == np.float64 else None
    return mx.array(arr, dtype=dtype) if dtype is not None else mx.array(arr)


def to_numpy(array) -> np.ndarray:
    """Bring an array (NumPy or MLX) back to a NumPy array."""
    mx = _import_mlx()
    if mx is not None and isinstance(array, mx.array):
        mx.eval(array)
        return np.array(array)
    return np.asarray(array)


def _should_use_gpu(flops: float, threshold: int) -> bool:
    return using_gpu() and flops >= threshold


# ---------------------------------------------------------------------------
# Elementary operations
# ---------------------------------------------------------------------------
def mx_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense matrix multiplication, on the GPU when it pays off.

    Falls back to :func:`numpy.matmul` (Accelerate BLAS on macOS) for small
    problems, for integer/bool inputs, or when the GPU backend is disabled.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.dtype.kind not in "fc" or b.dtype.kind not in "fc":
        return np.matmul(a, b)

    flops = float(a.shape[-1]) * float(a.shape[-2] if a.ndim > 1 else 1) * float(
        b.shape[-1] if b.ndim > 1 else 1
    )
    if not _should_use_gpu(flops, MATMUL_GPU_MIN_FLOPS):
        return np.matmul(a, b)

    mx = _mx()
    if mx is None:
        return np.matmul(a, b)
    try:
        out = mx.matmul(to_gpu(a), to_gpu(b))
        return to_numpy(out)
    except Exception:
        return np.matmul(a, b)
