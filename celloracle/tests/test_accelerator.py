# -*- coding: utf-8 -*-
"""Tests for the optional hardware acceleration layer.

These tests are hardware-aware: they always test the CPU path and only exercise
the GPU path when an Apple Metal device + MLX are available.
"""

import numpy as np
import pytest

import celloracle as co
from celloracle import _accelerator as acc


def test_backend_resolution():
    backend = co.get_accelerator()
    assert backend in ("cpu", "gpu")
    assert acc.get_backend() in ("cpu", "gpu")


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        co.set_accelerator("quantum")


def test_accelerator_info_has_blas():
    info = co.accelerator_info()
    assert "blas" in info
    assert info["backend"] in ("cpu", "gpu")


def test_cpu_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 300))
    b = rng.normal(size=(300, 40))
    prev = acc.get_backend()
    try:
        co.set_accelerator("cpu")
        out = acc.mx_matmul(a, b)
    finally:
        co.set_accelerator(prev)
    assert np.allclose(out, a @ b)


@pytest.mark.skipif(not acc.gpu_available(), reason="No Apple Metal GPU / MLX available")
def test_gpu_matmul_matches_cpu():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(800, 900))
    b = rng.normal(size=(900, 900))
    prev = acc.get_backend()
    try:
        co.set_accelerator("cpu")
        cpu = acc.mx_matmul(a, b)
        co.set_accelerator("gpu")
        gpu = acc.mx_matmul(a, b)
    finally:
        co.set_accelerator(prev)
    # float32 GPU accumulation: compare Frobenius norms (a single entry that
    # nearly cancels can have a large relative error).
    rel = np.linalg.norm(cpu - gpu) / np.linalg.norm(cpu)
    assert rel < 5e-3, f"GPU/CPU matmul relative error {rel:.2e} too large"
