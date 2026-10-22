# -*- coding: utf-8 -*-
"""
Built-in, dependency-free replacements for the small set of ``velocyto``
routines that CellOracle relies on for velocity / transition-probability
calculation.

Historically CellOracle imported these from the third-party ``velocyto``
package.  ``velocyto`` is effectively unmaintained and does not build against
modern NumPy, so its Cython/OpenMP kernels are re-implemented here with the
exact same mathematics, using vectorised NumPy (Accelerate BLAS on macOS) and,
when available, the Apple GPU through MLX.

The functions mirror the public velocyto API:

* :func:`colDeltaCor`, :func:`colDeltaCorpartial`
* :func:`colDeltaCorSqrt`, :func:`colDeltaCorSqrtpartial`
* :func:`colDeltaCorLog10`, :func:`colDeltaCorLog10partial`
* :class:`Diffusion`

Reference implementation: ``velocyto.py`` (BSD 2-Clause,
Copyright (c) 2017, Peter Kharchenko, Sten Linnarsson and Gioele La Manno),
https://github.com/velocyto-team/velocyto.py
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .. import _accelerator as _acc

__all__ = [
    "colDeltaCor",
    "colDeltaCorpartial",
    "colDeltaCorSqrt",
    "colDeltaCorSqrtpartial",
    "colDeltaCorLog10",
    "colDeltaCorLog10partial",
    "Diffusion",
]

# Maximum number of elements kept alive in the largest temporary array.
_MAX_BLOCK_ELEMENTS = 2 ** 25


# ---------------------------------------------------------------------------
# transforms applied to the expression difference e[:, i] - e[:, c]
# ---------------------------------------------------------------------------
def _transform_none(x: np.ndarray, psc: float = 0.0) -> np.ndarray:
    return x


def _transform_sqrt(x: np.ndarray, psc: float = 0.0) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x) + psc)


def _transform_log10(x: np.ndarray, psc: float = 0.0) -> np.ndarray:
    return np.sign(x) * np.log10(np.abs(x) + psc)


def _safe_corr(numerator, ss_a, ss_b):
    """corr = numerator / (sqrt(ss_a) * sqrt(ss_b)), NaN-safe like velocyto."""
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt(ss_a) * np.sqrt(ss_b)
        out = numerator / denom
    return out


# ---------------------------------------------------------------------------
# GPU (MLX) kernels
# ---------------------------------------------------------------------------
def _mx_apply_transform(mx, x, psc, kind="none"):
    if kind == "none":
        return x
    if kind == "sqrt":
        return mx.sign(x) * mx.sqrt(mx.abs(x) + psc)
    if kind == "log10":
        return mx.sign(x) * mx.log10(mx.abs(x) + psc)
    raise ValueError(kind)


def _mlx_partial_chunk(mx, e_gpu, d_gpu, ix_chunk, block, psc, kind="none"):
    """Partial-correlation block for ``block`` cells, with GPU operands.

    ``e_gpu``/``d_gpu`` are the *full* ``(genes, ncells)`` device arrays;
    ``ix_chunk`` are the absolute neighbour indices of the cells in ``block``.
    """
    ix_gpu = mx.array(np.ascontiguousarray(ix_chunk).astype(np.int32))
    en = mx.take(e_gpu, ix_gpu, axis=1)              # (genes, k, N)
    a = en - e_gpu[:, block, None]                   # (genes, k, N)
    a = _mx_apply_transform(mx, a, psc, kind)
    a = a - mx.mean(a, axis=0, keepdims=True)
    b = d_gpu[:, block]                              # (genes, k)
    b = b - mx.mean(b, axis=0, keepdims=True)
    num = mx.einsum("gcn,gc->cn", a, b)
    ss_a = mx.sum(a * a, axis=0)
    ss_b = mx.sum(b * b, axis=0)
    return _acc.to_numpy(num / (mx.sqrt(ss_a) * mx.sqrt(ss_b)[:, None]))


def _mlx_full_chunk(mx, e_gpu, d_gpu, block, psc, kind="none"):
    """Full-correlation block for ``block`` cells, with GPU operands."""
    a = e_gpu[:, None, :] - e_gpu[:, block, None]        # (genes, k, ncells)
    a = _mx_apply_transform(mx, a, psc, kind)
    a = a - mx.mean(a, axis=0, keepdims=True)
    b = d_gpu[:, block]
    b = b - mx.mean(b, axis=0, keepdims=True)
    num = mx.einsum("gci,gc->ci", a, b)
    ss_a = mx.sum(a * a, axis=0)
    ss_b = mx.sum(b * b, axis=0)
    return _acc.to_numpy(num / (mx.sqrt(ss_a) * mx.sqrt(ss_b)[:, None]))


# ---------------------------------------------------------------------------
# CPU kernels
# ---------------------------------------------------------------------------
def _cpu_partial_chunk(Ec, Dc, ix_chunk, block, transform, psc):
    """Correlation block for the cells selected by ``block``."""
    en = Ec[:, ix_chunk]                                 # (genes, k, N)
    a = en - Ec[:, block][:, :, None]                    # (genes, k, N)
    a = transform(a, psc)
    a = a - a.mean(axis=0, keepdims=True)
    b = Dc[:, block]                                     # (genes, k)
    b = b - b.mean(axis=0, keepdims=True)
    num = np.einsum("gcn,gc->cn", a, b)
    ss_a = np.einsum("gcn,gcn->cn", a, a)
    ss_b = np.einsum("gc,gc->c", b, b)
    return _safe_corr(num, ss_a, ss_b[:, None])


def _cpu_full_chunk(Ec, Dc, block, transform, psc):
    """Full correlation block ``out[c, i] = corr(f(e[:,i]-e[:,c]), d[:,c])``.

    Args:
        Ec: full mean-centred expression matrix ``(genes, ncells)``.
        Dc: full displacement matrix ``(genes, ncells)``.
        block: slice object selecting the rows (cells) ``c`` of this block.
    """
    e_c = Ec[:, block]                                   # (genes, k)
    a = Ec[:, None, :] - e_c[:, :, None]                 # (genes, k, ncells)
    a = transform(a, psc)
    a = a - a.mean(axis=0, keepdims=True)
    b = Dc[:, block]                                     # (genes, k)
    b = b - b.mean(axis=0, keepdims=True)
    num = np.einsum("gci,gc->ci", a, b)
    ss_a = np.einsum("gci,gci->ci", a, a)
    ss_b = np.einsum("gc,gc->c", b, b)
    return _safe_corr(num, ss_a, ss_b[:, None])


# ---------------------------------------------------------------------------
# public correlators
# ---------------------------------------------------------------------------
def _split(cells: int, work_per_cell: int) -> int:
    return max(1, int(_MAX_BLOCK_ELEMENTS // max(1, work_per_cell)))


def _transform_kind(transform) -> str:
    return {_transform_none: "none", _transform_sqrt: "sqrt", _transform_log10: "log10"}[transform]


def _correlate(emat, dmat, ixs=None, transform=_transform_none, psc=0.0):
    emat = np.asarray(emat, dtype=np.float64)
    dmat = np.asarray(dmat, dtype=np.float64)
    n_genes, n_cells = emat.shape
    out = np.zeros((n_cells, n_cells), dtype=np.float64)

    gpu = _acc.get_backend() == "gpu" and _acc._import_mlx() is not None
    kind = _transform_kind(transform)

    if ixs is None:
        chunk = _split(n_cells, n_genes * n_cells)
        if gpu and n_genes * n_cells * n_cells >= _acc.CORRELATION_GPU_MIN_WORK:
            mx = _acc._import_mlx()
            e_gpu, d_gpu = _acc.to_gpu(emat), _acc.to_gpu(dmat)
            for start in range(0, n_cells, chunk):
                stop = min(start + chunk, n_cells)
                out[start:stop, :] = _mlx_full_chunk(mx, e_gpu, d_gpu, slice(start, stop), psc, kind)
            return out
        for start in range(0, n_cells, chunk):
            stop = min(start + chunk, n_cells)
            out[start:stop, :] = _cpu_full_chunk(emat, dmat, slice(start, stop), transform, psc)
        return out

    ixs = np.ascontiguousarray(ixs).astype(np.intp)
    n_neighbors = ixs.shape[1]
    chunk = _split(n_cells, n_genes * n_neighbors)

    if gpu and n_genes * n_cells * n_neighbors >= _acc.CORRELATION_GPU_MIN_WORK:
        mx = _acc._import_mlx()
        e_gpu, d_gpu = _acc.to_gpu(emat), _acc.to_gpu(dmat)
        for start in range(0, n_cells, chunk):
            stop = min(start + chunk, n_cells)
            values = _mlx_partial_chunk(mx, e_gpu, d_gpu, ixs[start:stop], slice(start, stop), psc, kind)
            rows = np.arange(start, stop)[:, None]
            np.add.at(out, (rows, ixs[start:stop]), values)
        return out

    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        values = _cpu_partial_chunk(emat, dmat, ixs[start:stop], slice(start, stop), transform, psc)
        rows = np.arange(start, stop)[:, None]
        np.add.at(out, (rows, ixs[start:stop]), values)
    return out


def colDeltaCor(emat: np.ndarray, dmat: np.ndarray, threads: int = None) -> np.ndarray:
    """Correlation between displacement ``d[:, i]`` and ``e[:, j] - e[:, i]``.

    Args:
        emat: gene expression matrix, shape ``(ngenes, ncells)``.
        dmat: displacement (velocity) matrix, shape ``(ngenes, ncells)``.
        threads: kept for API compatibility; computation is vectorised.

    Returns:
        ``(ncells, ncells)`` array.
    """
    return _correlate(emat, dmat, ixs=None)


def colDeltaCorpartial(emat: np.ndarray, dmat: np.ndarray, ixs: np.ndarray, threads: int = None) -> np.ndarray:
    """Partial correlation using only the neighbours listed in ``ixs``.

    ``ixs[i, k]`` is the k-th neighbour of cell ``i``.

    Returns:
        ``(ncells, ncells)`` array with non-zeros at the neighbour positions.
    """
    return _correlate(emat, dmat, ixs=ixs)


def colDeltaCorSqrt(emat: np.ndarray, dmat: np.ndarray, threads: int = None, psc: float = 0.0) -> np.ndarray:
    return _correlate(emat, dmat, ixs=None, transform=_transform_sqrt, psc=psc)


def colDeltaCorSqrtpartial(emat: np.ndarray, dmat: np.ndarray, ixs: np.ndarray, threads: int = None, psc: float = 0.0) -> np.ndarray:
    return _correlate(emat, dmat, ixs=ixs, transform=_transform_sqrt, psc=psc)


def colDeltaCorLog10(emat: np.ndarray, dmat: np.ndarray, threads: int = None, psc: float = 1.0) -> np.ndarray:
    return _correlate(emat, dmat, ixs=None, transform=_transform_log10, psc=psc)


def colDeltaCorLog10partial(emat: np.ndarray, dmat: np.ndarray, ixs: np.ndarray, threads: int = None, psc: float = 1.0) -> np.ndarray:
    return _correlate(emat, dmat, ixs=ixs, transform=_transform_log10, psc=psc)


# ---------------------------------------------------------------------------
# Diffusion (Markov process helper) -- reduced velocyto-compatible version
# ---------------------------------------------------------------------------
class Diffusion:
    """Minimal re-implementation of ``velocyto.diffusion.Diffusion``.

    Only the methods used by CellOracle are provided.
    """

    def __init__(self) -> None:
        pass

    def compute_transition_matrix(self, knn, x, v, epsilon: float = 0.0, reverse: bool = False):
        knn = knn.tocoo()
        v0, v1 = knn.row, knn.col
        uv = x[v1] - x[v0]
        norms = np.linalg.norm(uv, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = uv / norms[:, None]
        scalar_projection = np.einsum("ij,ij->i", v[v0], uv)
        if reverse:
            scalar_projection = -scalar_projection
        scalar_projection = scalar_projection + epsilon
        np.clip(scalar_projection, a_min=0, a_max=None, out=scalar_projection)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = scalar_projection * (1 / norms)
        from sklearn.preprocessing import normalize

        return normalize(sparse.coo_matrix((p, (v0, v1))).tocsr(), axis=1, norm="l1")

    def diffuse(self, x: np.ndarray, tr, n_steps: int = 10, mode: str = "path_integral"):
        x = np.asarray(x, dtype=np.float64)
        if x.sum() == 0:
            raise ValueError("The starting distribution sums to zero.")
        state = sparse.csr_matrix(x / x.sum())
        tr = sparse.csr_matrix(tr)

        if mode == "path_integral":
            result = np.zeros(x.shape)
            for _ in range(n_steps):
                state = state.dot(tr)
                result = result + state
            return result
        if mode == "time_evolution":
            for _ in range(n_steps):
                state = state.dot(tr)
            return state.toarray()
        if mode == "map_trajectory":
            result = [np.argmax(state.toarray())]
            for _ in range(n_steps):
                state = state.dot(tr)
                result.append(int(np.argmax(state.toarray())))
            return result
        if mode == "frontier":
            result = [np.argmax(state.toarray())]
            for _ in range(n_steps):
                nxt = state.dot(tr)
                result.append(int(np.argmax((nxt.toarray() + 1) / (state.toarray() + 1))))
                state = nxt
            return result
        if mode == "trajectory":
            from sklearn.preprocessing import normalize

            node = int(np.random.choice(np.arange(x.shape[0]), p=x / x.sum()))
            trajectories = [node]
            for _ in range(n_steps):
                onehot = np.zeros(tr.shape[0], dtype=np.float64)
                onehot[node] = 1
                nxt = sparse.csr_matrix(onehot).dot(tr).toarray()
                nxt = normalize(nxt, norm="l1")[0]
                if nxt.sum() == 0:
                    nxt = onehot
                node = int(np.random.choice(np.arange(nxt.shape[0]), p=nxt / nxt.sum()))
                trajectories.append(node)
            return trajectories
        raise ValueError(f"Unknown diffusion mode {mode!r}")
