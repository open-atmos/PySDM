"""
.. include:: ../docs/pysdm_examples_landing.md
"""

from importlib.metadata import PackageNotFoundError, version
import PySDM

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

# assert PySDM.__version__ == __version__
