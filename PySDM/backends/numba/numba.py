"""
Created at 24.07.2019
"""

from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.numba.impl._maths_methods import MathsMethods
from PySDM.backends.numba.impl._physics_methods import PhysicsMethods
from PySDM.backends.numba.impl._storage_methods import StorageMethods
from PySDM.backends.numba.impl.condensation_methods import CondensationMethods
from PySDM.backends.numba.random import Random as ImportedRandom
from PySDM.backends.numba.storage.index import Index as ImportedIndex
from PySDM.backends.numba.storage.indexed_storage import IndexedStorage as ImportedIndexedStorage
from PySDM.backends.numba.storage.pair_indicator import PairIndicator as ImportedPairIndicator
from PySDM.backends.numba.storage.pairwise_storage import PairwiseStorage as ImportedPairwiseStorage
from PySDM.backends.numba.storage.storage import Storage as ImportedStorage


class Numba(
    AlgorithmicMethods,
    AlgorithmicStepMethods,
    StorageMethods,
    MathsMethods,
    PhysicsMethods,
    CondensationMethods
):
    Storage = ImportedStorage
    Index = ImportedIndex
    IndexedStorage = ImportedIndexedStorage
    PairIndicator = ImportedPairIndicator
    PairwiseStorage = ImportedPairwiseStorage
    Random = ImportedRandom

    def __init__(self):
        raise Exception("Backend is stateless.")
