"""
DismalPy
========
"""
from numpy.testing import Tester
test = Tester().test

try:
    from .version import version as __version__
except ImportError:
    __version__ = 'not-yet-built'

from . import ssm