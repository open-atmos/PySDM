"""
Created at 24.07.2019
"""

from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba.impl._pair_methods import PairMethods
from PySDM.backends.numba.impl._physics_methods import PhysicsMethods
from PySDM.backends.numba.impl._index_methods import IndexMethods
from PySDM.backends.numba.impl.condensation_methods import CondensationMethods
from PySDM.backends.numba.random import Random as ImportedRandom
from PySDM.backends.numba.storage import Storage as ImportedStorage


class Numba(
    AlgorithmicMethods,
    PairMethods,
    IndexMethods,
    PhysicsMethods,
    CondensationMethods
):
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = 'local'

    def __init__(self):
        raise Exception("Backend is stateless.")

    @staticmethod
    def sanity_check():
        pass
