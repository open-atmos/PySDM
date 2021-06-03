"""
Multi-threaded CPU backend using LLVM-powered just-in-time compilation
"""

from PySDM.backends.numba.impl._algorithmic_methods import AlgorithmicMethods
from PySDM.backends.numba.impl._pair_methods import PairMethods
from PySDM.backends.numba.impl._physics_methods import PhysicsMethods
from PySDM.backends.numba.impl._index_methods import IndexMethods
from PySDM.backends.numba.impl.condensation_methods import CondensationMethods
from .impl.moments_methods import MomentsMethods
from PySDM.backends.numba.impl._chemistry_methods import ChemistryMethods
from PySDM.backends.numba.random import Random as ImportedRandom
from PySDM.backends.numba.storage import Storage as ImportedStorage


class Numba(
    AlgorithmicMethods,
    PairMethods,
    IndexMethods,
    PhysicsMethods,
    CondensationMethods,
    ChemistryMethods,
    MomentsMethods
):
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = 'local'

    def __init__(self, formulae):
        self.formulae = formulae
        PhysicsMethods.__init__(self)
        ChemistryMethods.__init__(self)

    @staticmethod
    def sanity_check():
        pass
