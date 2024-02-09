"""
PySDM_examples package includes common Python modules used in PySDM smoke tests
and in example notebooks (but the package wheels do not include the notebooks)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
