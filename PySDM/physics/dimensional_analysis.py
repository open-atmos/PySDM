"""
A context manager (for use with the `with` statement)
for use in unit tests which disables Numba and enables Pint
"""

from importlib import reload

class DimensionalAnalysis:
    def __enter__(*_):  # pylint: disable=no-method-argument,no-self-argument,import-outside-toplevel
        from PySDM import formulae
        from PySDM import physics
        from . import constants, constants_defaults
        from .impl import flag
        
        flag.DIMENSIONAL_ANALYSIS = True
        reload(constants)
        reload(constants_defaults)
        reload(formulae)
        reload(physics)

    def __exit__(*_):  # pylint: disable=no-method-argument,no-self-argument,import-outside-toplevel
        from PySDM import formulae
        from PySDM import physics
        from . import constants, constants_defaults
        from .impl import flag

        flag.DIMENSIONAL_ANALYSIS = False
        reload(constants)
        reload(constants_defaults)
        reload(formulae)
        reload(physics)
