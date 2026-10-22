# -*- coding: utf-8 -*-
"""Regression tests for the modern NumPy / pandas / SciPy stack.

The tests in this module are deliberately dependency-light: they build small
synthetic matrices and check that the bundled velocity kernels and the GRN
simulation loop behave like their mathematical definitions, and that the CPU
and Apple-GPU implementations agree.
"""

import numpy as np
import pandas as pd
import pytest

import celloracle as co
from celloracle import _accelerator as acc
from celloracle.trajectory._velocity_compat import (
    colDeltaCor,
    colDeltaCorpartial,
)
from celloracle.trajectory.oracle_GRN import _do_simulation


def _reference_partial(emat, dmat, ixs):
    """Naive, loop-based reference for colDeltaCorpartial."""
    emat = np.asarray(emat, dtype=np.float64)
    dmat = np.asarray(dmat, dtype=np.float64)
    n_cells = emat.shape[1]
    out = np.zeros((n_cells, n_cells))
    for c in range(n_cells):
        for n in range(ixs.shape[1]):
            i = ixs[c, n]
            a = emat[:, i] - emat[:, c]
            a = a - a.mean()
            b = dmat[:, c]
            b = b - b.mean()
            num = float(np.dot(a, b))
            den = np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))
            with np.errstate(divide="ignore", invalid="ignore"):
                out[c, i] = num / den
    return out


def _reference_full(emat, dmat):
    emat = np.asarray(emat, dtype=np.float64)
    dmat = np.asarray(dmat, dtype=np.float64)
    n_cells = emat.shape[1]
    out = np.zeros((n_cells, n_cells))
    for c in range(n_cells):
        for i in range(n_cells):
            a = emat[:, i] - emat[:, c]
            a = a - a.mean()
            b = dmat[:, c]
            b = b - b.mean()
            den = np.sqrt(np.dot(a, a)) * np.sqrt(np.dot(b, b))
            with np.errstate(divide="ignore", invalid="ignore"):
                out[c, i] = float(np.dot(a, b)) / den
    return out


@pytest.fixture(scope="module")
def small_problem():
    rng = np.random.default_rng(42)
    n_genes, n_cells, n_nb = 40, 60, 8
    emat = rng.random((n_genes, n_cells))
    dmat = rng.normal(size=(n_genes, n_cells)) * 0.1
    ixs = np.stack([rng.choice(n_cells, n_nb, replace=False) for _ in range(n_cells)])
    return emat, dmat, ixs


def test_coldeltacorpartial_matches_reference(small_problem):
    emat, dmat, ixs = small_problem
    got = colDeltaCorpartial(emat, dmat, ixs)
    ref = _reference_partial(emat, dmat, ixs)
    m = np.isfinite(ref)
    assert np.allclose(got[m], ref[m], rtol=1e-9, atol=1e-9)


def test_coldeltacor_matches_reference(small_problem):
    emat, dmat, _ = small_problem
    got = colDeltaCor(emat, dmat)
    ref = _reference_full(emat, dmat)
    m = np.isfinite(ref)
    assert np.allclose(got[m], ref[m], rtol=1e-9, atol=1e-9)


def test_simulation_matches_manual_loop():
    rng = np.random.default_rng(7)
    n_cells, n_genes = 200, 30
    coef = pd.DataFrame(rng.normal(size=(n_genes, n_genes)) * 0.05)
    gem = pd.DataFrame(np.abs(rng.random((n_cells, n_genes))))
    sim_in = gem.copy()
    sim_in.iloc[:, 0] = 0.0

    prev = acc.get_backend()
    try:
        co.set_accelerator("cpu")
        got = _do_simulation(coef, sim_in, gem, 3)
    finally:
        co.set_accelerator(prev)

    delta_in = sim_in - gem
    delta = delta_in.copy()
    for _ in range(3):
        delta = delta.dot(coef)
        delta[delta_in != 0] = delta_in
        tmp = gem + delta
        tmp[tmp < 0] = 0
        delta = tmp - gem
    expected = gem + delta
    assert np.allclose(got.values, expected.values)


@pytest.mark.skipif(not acc.gpu_available(), reason="No Apple Metal GPU / MLX available")
def test_simulation_gpu_matches_cpu():
    rng = np.random.default_rng(11)
    n_cells, n_genes = 1500, 400
    coef = pd.DataFrame(rng.normal(size=(n_genes, n_genes)) * 0.01)
    gem = pd.DataFrame(np.abs(rng.random((n_cells, n_genes))))
    sim_in = gem.copy()
    sim_in.iloc[:, :3] = 0.0

    prev = acc.get_backend()
    try:
        co.set_accelerator("cpu")
        cpu = _do_simulation(coef, sim_in, gem, 3)
        co.set_accelerator("gpu")
        gpu = _do_simulation(coef, sim_in, gem, 3)
    finally:
        co.set_accelerator(prev)

    scale = np.abs(cpu.values).max()
    assert np.abs(cpu.values - gpu.values).max() / scale < 1e-3


def test_pandas_concat_and_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    path = tmp_path / "df.parquet"
    df.to_parquet(path)
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, back)
