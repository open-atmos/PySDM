"""
Created at 30.04.2020

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .builder import Builder

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
