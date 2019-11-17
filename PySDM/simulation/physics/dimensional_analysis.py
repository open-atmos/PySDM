from importlib import reload
from PySDM.simulation.physics import _flag
from PySDM.simulation.physics import constants
from PySDM.simulation.physics import formulae
from PySDM.backends.numba import _physics_methods


class DimensionalAnalysis:
    def __enter__(*_):
        _flag.DIMENSIONAL_ANALYSIS = True
        reload(constants)
        reload(_physics_methods)
        reload(formulae)

    def __exit__(*_):
        _flag.DIMENSIONAL_ANALYSIS = False
        reload(constants)
        reload(_physics_methods)
        reload(formulae)
