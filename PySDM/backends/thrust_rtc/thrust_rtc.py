"""
GPU-resident backend using NVRTC runtime compilation library for CUDA
"""

import os
import warnings
from PySDM.backends.thrust_rtc.impl.algorithmic_methods import AlgorithmicMethods
from PySDM.backends.thrust_rtc.impl.pair_methods import PairMethods
from PySDM.backends.thrust_rtc.impl.index_methods import IndexMethods
from PySDM.backends.thrust_rtc.impl.physics_methods import PhysicsMethods
from PySDM.backends.thrust_rtc.impl.moments_methods import MomentsMethods
from PySDM.backends.thrust_rtc.impl.condensation_methods import CondensationMethods
from PySDM.backends.thrust_rtc.storage import Storage as ImportedStorage
from PySDM.backends.thrust_rtc.random import Random as ImportedRandom
from PySDM.physics import Formulae


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

    def __init__(self, formulae=None):
        self.formulae = formulae or Formulae()
        AlgorithmicMethods.__init__(self)
        PairMethods.__init__(self)
        IndexMethods.__init__(self)
        PhysicsMethods.__init__(self)
        CondensationMethods.__init__(self)
        MomentsMethods.__init__(self)

        if not ThrustRTC.ENABLE \
           and 'CI' not in os.environ:
            warnings.warn('CUDA is not available, using FakeThrustRTC!')
