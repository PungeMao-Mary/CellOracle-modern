# -*- coding: utf-8 -*-
"""Regression tests for the vector-field plotting wrapper.

These cover the upstream ``UnboundLocalError`` in
``_plot_simulation_flow_on_grid``: the grid coordinates were looked up with the
wrong attribute (``mass_filter`` instead of ``flow_grid``) and the mass-filter
local was left unbound when ``calculate_mass_filter()`` had not been called.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from celloracle.visualizations.development_module_visualization import (
    _plot_simulation_flow_on_grid,
)


class _OracleLike:
    """Mimics a plain ``Oracle`` after ``calculate_p_mass()``.

    Has ``flow`` / ``flow_grid`` / ``flow_rndm``; ``mass_filter`` and
    ``gridpoints_coordinates`` are optional, exactly like the real class.
    """

    def __init__(self, n_grid=10, mass_filter=None):
        self.embedding = np.random.RandomState(0).rand(50, 2)
        meshes = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 1, n_grid))
        self.flow_grid = np.vstack([m.flat for m in meshes]).T
        self.flow = np.random.RandomState(1).rand(n_grid * n_grid, 2)
        self.flow_rndm = np.random.RandomState(2).rand(n_grid * n_grid, 2)
        if mass_filter is not None:
            self.mass_filter = mass_filter


def _fresh_ax():
    fig, ax = plt.subplots()
    return fig, ax


def test_flow_plot_without_mass_filter():
    """The reported bug: Oracle has flow_grid but no mass_filter."""
    obj = _OracleLike()
    assert not hasattr(obj, "mass_filter")
    assert not hasattr(obj, "gridpoints_coordinates")
    fig, ax = _fresh_ax()
    # Must not raise UnboundLocalError.
    _plot_simulation_flow_on_grid(self=obj, ax=ax, show_background=False)
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_flow_plot_with_mass_filter():
    """Tutorial path: calculate_mass_filter() was called."""
    obj = _OracleLike(mass_filter=np.zeros(100, dtype=bool))
    fig, ax = _fresh_ax()
    _plot_simulation_flow_on_grid(self=obj, ax=ax, show_background=False)
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_flow_plot_with_dev_module_gridpoints():
    """Development-module path: gridpoints_coordinates takes precedence."""
    obj = _OracleLike(mass_filter=np.zeros(100, dtype=bool))
    obj.gridpoints_coordinates = obj.flow_grid.copy()
    obj.mass_filter_simulation = np.zeros(100, dtype=bool)
    fig, ax = _fresh_ax()
    _plot_simulation_flow_on_grid(self=obj, ax=ax, show_background=False)
    assert len(ax.collections) >= 1
    plt.close(fig)


def test_flow_plot_random_requires_flow_rndm():
    obj = _OracleLike()
    del obj.flow_rndm
    fig, ax = _fresh_ax()
    with pytest.raises(AttributeError, match="flow_rndm"):
        _plot_simulation_flow_on_grid(self=obj, ax=ax, show_background=False, data_random=True)
    plt.close(fig)


def test_flow_plot_missing_coordinates_raises_clear_error():
    class _NoGrid:
        embedding = np.random.RandomState(0).rand(10, 2)
        flow = np.random.RandomState(1).rand(100, 2)

    fig, ax = _fresh_ax()
    with pytest.raises(AttributeError, match="gridpoints_coordinates.*flow_grid"):
        _plot_simulation_flow_on_grid(self=_NoGrid(), ax=ax, show_background=False)
    plt.close(fig)


def test_flow_plot_missing_flow_raises_clear_error():
    class _NoFlow:
        embedding = np.random.RandomState(0).rand(10, 2)
        flow_grid = np.random.RandomState(1).rand(100, 2)

    fig, ax = _fresh_ax()
    with pytest.raises(AttributeError, match="`flow` is unavailable"):
        _plot_simulation_flow_on_grid(self=_NoFlow(), ax=ax, show_background=False)
    plt.close(fig)
