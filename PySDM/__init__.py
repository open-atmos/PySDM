# pylint:disable=invalid-name
"""
.. include:: ../docs/markdown/pysdm_landing.md
"""

from importlib.metadata import PackageNotFoundError, version

from PySDM.attributes.impl.attribute_registry import register_attribute

from . import attributes
from . import environments, exporters, products
from .builder import Builder
from .formulae import Formulae
from .particulator import Particulator

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
