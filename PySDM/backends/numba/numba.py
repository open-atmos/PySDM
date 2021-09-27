"""
Multi-threaded CPU backend using LLVM-powered just-in-time compilation
"""

from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba.impl._pair_methods import PairMethods
from PySDM.backends.numba.impl._physics_methods import PhysicsMethods
from PySDM.backends.numba.impl._index_methods import IndexMethods
from PySDM.backends.numba.impl.condensation_methods import CondensationMethods
from PySDM.backends.numba.impl.moments_methods import MomentsMethods
from PySDM.backends.numba.impl.freezing_methods import FreezingMethods
from PySDM.backends.numba.impl._chemistry_methods import ChemistryMethods
from PySDM.backends.numba.random import Random as ImportedRandom
from PySDM.backends.numba.storage import Storage as ImportedStorage
from PySDM.physics import Formulae


class Numba(
    AlgorithmicMethods,
    PairMethods,
    IndexMethods,
    PhysicsMethods,
    CondensationMethods,
    ChemistryMethods,
    MomentsMethods,
    FreezingMethods
):
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = 'local'

    def __init__(self, formulae=None):
        self.formulae = formulae or Formulae()
        PhysicsMethods.__init__(self)
        ChemistryMethods.__init__(self)
        FreezingMethods.__init__(self)
