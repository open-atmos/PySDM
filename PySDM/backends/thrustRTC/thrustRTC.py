"""
GPU-resident backend using NVRTC runtime compilation library for CUDA
"""

import os
import warnings
from PySDM.backends.thrustRTC.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.thrustRTC.impl._pair_methods import PairMethods
from PySDM.backends.thrustRTC.impl._index_methods import IndexMethods
from PySDM.backends.thrustRTC.impl._physics_methods import PhysicsMethods
from PySDM.backends.thrustRTC.impl.moments_methods import MomentsMethods
from PySDM.backends.thrustRTC.impl.condensation_methods import CondensationMethods
from PySDM.backends.thrustRTC.storage import Storage as ImportedStorage
from PySDM.backends.thrustRTC.random import Random as ImportedRandom


class ThrustRTC(
    AlgorithmicMethods,
    PairMethods,
    IndexMethods,
    PhysicsMethods,
    CondensationMethods,
    MomentsMethods
):
    ENABLE = True
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = 'global'

    def __init__(self, formulae):
        self.formulae = formulae
        PhysicsMethods.__init__(self)
        CondensationMethods.__init__(self)
        AlgorithmicMethods.__init__(self)
        MomentsMethods.__init__(self)

    @staticmethod
    def sanity_check():
        if not ThrustRTC.ENABLE \
           and 'CI' not in os.environ:
            warnings.warn('CUDA is not available, using FakeThrustRTC!')
