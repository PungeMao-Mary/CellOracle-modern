# -*- coding: utf-8 -*-
"""
The :mod:`.trajectory` module implements CellOracle's GRN simulation,
in-silico perturbation and trajectory analysis.

The two public objects, :class:`~celloracle.trajectory.oracle_core.Oracle` and
:func:`~celloracle.trajectory.oracle_core.load_oracle`, are resolved lazily to
avoid a circular import during :mod:`celloracle` package initialisation.
"""

__all__ = ["Oracle", "load_oracle"]


def __getattr__(name):
    if name in __all__:
        from . import oracle_core

        return getattr(oracle_core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
