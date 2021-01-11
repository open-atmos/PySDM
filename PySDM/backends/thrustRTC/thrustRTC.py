"""
Created at 01.08.2019
"""

from PySDM.backends.thrustRTC.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.thrustRTC.impl._algorithmic_step_methods import AlgorithmicStepMethods
from PySDM.backends.thrustRTC.impl._storage_methods import StorageMethods
from PySDM.backends.thrustRTC.impl._maths_methods import MathsMethods
from PySDM.backends.thrustRTC.impl._physics_methods import PhysicsMethods
from .storage.storage import Storage as ImportedStorage
from PySDM.backends.thrustRTC.storage.indexed_storage import IndexedStorage as ImportedIndexedStorage
from PySDM.backends.thrustRTC.random import Random as ImportedRandom
from .storage.pairwise_storage import PairwiseStorage as ImportedPairwiseStorage
from .storage.pair_indicator import PairIndicator as ImportedPairIndicator
from .storage.index import Index as ImportedIndex


class ThrustRTC(
    AlgorithmicMethods,
    AlgorithmicStepMethods,
    StorageMethods,
    MathsMethods,
    PhysicsMethods,
):
    ENABLE = True
    Storage = ImportedStorage
    IndexedStorage = ImportedIndexedStorage
    Random = ImportedRandom
    PairIndicator = ImportedPairIndicator
    PairwiseStorage = ImportedPairwiseStorage
    Index = ImportedIndex

    default_croupier = 'global'

    def __init__(self):
        raise Exception("Backend is stateless.")
