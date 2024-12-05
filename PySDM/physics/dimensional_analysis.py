"""
A context manager (for use with the `with` statement) which
enables Pint physical-units checks and disables Numba in `PySDM.formulae.Formulae`
"""

from importlib import reload

from PySDM import formulae
from PySDM import physics
from . import constants, constants_defaults
from .impl import flag


class DimensionalAnalysis:
    def __enter__(*_):  # pylint: disable=no-method-argument,no-self-argument
        flag.DIMENSIONAL_ANALYSIS = True
        reload(constants)
        reload(constants_defaults)
        reload(formulae)
        reload(physics)

    def __exit__(*_):  # pylint: disable=no-method-argument,no-self-argument
        flag.DIMENSIONAL_ANALYSIS = False
        reload(constants)
        reload(constants_defaults)
        reload(formulae)
        reload(physics)
