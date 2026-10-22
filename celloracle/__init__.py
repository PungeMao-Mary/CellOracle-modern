# -*- coding: utf-8 -*-

import sys
import re
import warnings
import logging

from . import utility, network, network_analysis, go_analysis, data, data_conversion, oracle_utility
from .trajectory.oracle_core import Oracle
from .network import Net
from .network_analysis import Links
from .utility.load_hdf5 import load_hdf5
from .utility import check_python_requirements
from .network_analysis import test_R_libraries_installation
from . import _accelerator as accelerator
from ._accelerator import (set_backend as set_accelerator,
                           get_backend as get_accelerator,
                           accelerator_info)
from .version import __version__
#from . import motif_analysis

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}\.'.format(re.escape(__name__)))

__copyright__    = 'Copyright (C) 2020 Kenji Kamimoto'
# Modified for CellOracle-modern: the upstream non-commercial academic-use
# restriction remains part of the license terms; this is not standard Apache-2.0.
__license__      = 'Modified Apache License 2.0 (non-commercial academic use only)'
__author__       = 'Kenji Kamimoto'
__author_email__ = 'kamimoto@wustl.edu'
__url__          = 'https://github.com/morris-lab/CellOracle'
#__version__ = "0.10.2"
__all__ = ["utility", "motif_analysis", "network", "network_analysis",
           "go_analysis", "data", "data_conversion",
           "Oracle",
           "Links",
           "Net",
           "load_hdf5",
           "test_R_libraries_installation",
           "check_python_requirements",
           "accelerator",
           "set_accelerator",
           "get_accelerator",
           "accelerator_info",
           ]

# Resolve the accelerator backend once at import time so that the choice is
# logged early.  This never raises: an unavailable GPU silently falls back.
try:
    accelerator.get_backend()
except Exception:  # pragma: no cover - defensive
    pass

check_python_requirements(return_detail=False, print_warning=True)
