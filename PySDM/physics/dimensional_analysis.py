"""
Crated at 2019
"""

from importlib import reload
from PySDM.physics import _flag
from PySDM.physics import constants
from PySDM.physics import formulae
from PySDM.backends import formulae as backend_formulae

class DimensionalAnalysis:

    def __enter__(*_):
        _flag.DIMENSIONAL_ANALYSIS = True
        reload(constants)
        reload(backend_formulae)
        reload(formulae)

    def __exit__(*_):
        _flag.DIMENSIONAL_ANALYSIS = False
        reload(constants)
        reload(backend_formulae)
        reload(formulae)
