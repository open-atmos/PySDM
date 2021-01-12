"""
Created at 01.08.2019
"""

import os
import sys
import warnings
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

    @staticmethod
    def sanity_check():
        if not ThrustRTC.ENABLE \
           and 'CI' not in os.environ:
            if 'google.colab' in sys.modules:
                raise RuntimeError("to use GPU on Colab set hardware accelerator to 'GPU' before session start"
                                   'in the "Runtime :: Change runtime type :: Hardware accelerator" menu')
            else:
                warnings.warn('CUDA is not available, using FakeThrustRTC!')
