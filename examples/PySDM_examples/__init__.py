# pylint disable=fixme
from pkg_resources import DistributionNotFound, VersionConflict, get_distribution

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass
