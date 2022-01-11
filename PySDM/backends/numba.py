"""
Multi-threaded CPU backend using LLVM-powered just-in-time compilation
"""

from PySDM.backends.impl_numba.methods.collisions_methods import CollisionsMethods
from PySDM.backends.impl_numba.methods.pair_methods import PairMethods
from PySDM.backends.impl_numba.methods.physics_methods import PhysicsMethods
from PySDM.backends.impl_numba.methods.index_methods import IndexMethods
from PySDM.backends.impl_numba.methods.condensation_methods import CondensationMethods
from PySDM.backends.impl_numba.methods.moments_methods import MomentsMethods
from PySDM.backends.impl_numba.methods.freezing_methods import FreezingMethods
from PySDM.backends.impl_numba.methods.chemistry_methods import ChemistryMethods
from PySDM.backends.impl_numba.methods.displacement_methods import DisplacementMethods
from PySDM.backends.impl_numba.methods.terminal_velocity_methods import TerminalVelocityMethods
from PySDM.backends.impl_numba.random import Random as ImportedRandom
from PySDM.backends.impl_numba.storage import Storage as ImportedStorage
from PySDM.formulae import Formulae


class Numba(  # pylint: disable=too-many-ancestors,duplicate-code
    CollisionsMethods,
    PairMethods,
    IndexMethods,
    PhysicsMethods,
    CondensationMethods,
    ChemistryMethods,
    MomentsMethods,
    FreezingMethods,
    DisplacementMethods,
    TerminalVelocityMethods
):
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = 'local'

    def __init__(self, formulae=None):
        self.formulae = formulae or Formulae()
        CollisionsMethods.__init__(self)
        PairMethods.__init__(self)
        IndexMethods.__init__(self)
        PhysicsMethods.__init__(self)
        CondensationMethods.__init__(self)
        ChemistryMethods.__init__(self)
        MomentsMethods.__init__(self)
        FreezingMethods.__init__(self)
        DisplacementMethods.__init__(self)
        TerminalVelocityMethods.__init__(self)
