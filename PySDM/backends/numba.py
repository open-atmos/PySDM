"""
Multi-threaded CPU backend using LLVM-powered just-in-time compilation
"""

from PySDM.backends.impl_numba import methods
from PySDM.backends.impl_numba.random import Random as ImportedRandom
from PySDM.backends.impl_numba.storage import Storage as ImportedStorage
from PySDM.formulae import Formulae
from PySDM.backends.impl_numba.conf import JIT_FLAGS


class Numba(  # pylint: disable=too-many-ancestors,duplicate-code
    methods.CollisionsMethods,
    methods.FragmentationMethods,
    methods.PairMethods,
    methods.IndexMethods,
    methods.PhysicsMethods,
    methods.CondensationMethods,
    methods.ChemistryMethods,
    methods.MomentsMethods,
    methods.FreezingMethods,
    methods.DisplacementMethods,
    methods.TerminalVelocityMethods,
    methods.IsotopeMethods,
):
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = "local"

    def __init__(self, formulae=None, double_precision=True, override_jit_flags=None):
        if not double_precision:
            raise NotImplementedError()
        self.formulae = formulae or Formulae()
        self.formulae_flattened = self.formulae.flatten

        assert "fastmath" not in (override_jit_flags or {})
        self.default_jit_flags = {
            **JIT_FLAGS,
            **{"fastmath": self.formulae.fastmath},
            **(override_jit_flags or {}),
        }

        methods.CollisionsMethods.__init__(self)
        methods.FragmentationMethods.__init__(self)
        methods.PairMethods.__init__(self)
        methods.IndexMethods.__init__(self)
        methods.PhysicsMethods.__init__(self)
        methods.CondensationMethods.__init__(self)
        methods.ChemistryMethods.__init__(self)
        methods.MomentsMethods.__init__(self)
        methods.FreezingMethods.__init__(self)
        methods.DisplacementMethods.__init__(self)
        methods.TerminalVelocityMethods.__init__(self)
        methods.IsotopeMethods.__init__(self)
