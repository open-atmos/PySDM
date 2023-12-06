"""
PySDM_examples package includes common Python modules used in PySDM smoke tests
and in example notebooks (but the package wheels do not include the notebooks)
"""
from pkg_resources import DistributionNotFound, VersionConflict, get_distribution

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
