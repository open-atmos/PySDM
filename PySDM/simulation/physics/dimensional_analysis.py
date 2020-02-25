from importlib import reload
from PySDM.simulation.physics import _flag
from PySDM.simulation.physics import constants
from PySDM.simulation.physics import formulae
from PySDM.backends.numba import numba_helpers


class DimensionalAnalysis:
    def __enter__(*_):
        _flag.DIMENSIONAL_ANALYSIS = True
        reload(constants)
        reload(numba_helpers)
        reload(formulae)

    def __exit__(*_):
        _flag.DIMENSIONAL_ANALYSIS = False
        reload(constants)
        reload(numba_helpers)
        reload(formulae)
