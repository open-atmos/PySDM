"""
A context manager (for use with the `with` statement)
for use in unit tests which disables Numba and enables Pint
"""
from importlib import reload
from PySDM import formulae
from . import constants
from .impl import flag


class DimensionalAnalysis:

    def __enter__(*_):  # pylint: disable=no-method-argument
        flag.DIMENSIONAL_ANALYSIS = True
        reload(constants)
        reload(formulae)

    def __exit__(*_):  # pylint: disable=no-method-argument
        flag.DIMENSIONAL_ANALYSIS = False
        reload(constants)
        reload(formulae)
