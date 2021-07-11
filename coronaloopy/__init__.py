# Licensed under a 3-clause BSD style license - see LICENSE.rst
from .loop import *

try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"
