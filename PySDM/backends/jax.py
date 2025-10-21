"""
Multi-threaded CPU backend using LLVM-powered just-in-time compilation
"""

from PySDM.backends.impl_jax import methods
from PySDM.backends.impl_numba.random import Random as ImportedRandom
from PySDM.backends.impl_jax.storage import Storage as ImportedStorage
from PySDM.formulae import Formulae


class Jax(
    methods.CollisionsMethods,
    methods.PairMethods,
    methods.IndexMethods,
    methods.PhysicsMethods,
    methods.MomentsMethods,
):
    Storage = ImportedStorage
    Random = ImportedRandom

    default_croupier = "local"

    def __init__(
        self, formulae=None, *, double_precision=True, override_jit_flags=None
    ):
        if not double_precision:
            raise NotImplementedError()
        
        self.formulae = formulae or Formulae()
        self.formulae_flattened = self.formulae.flatten

        # assert "fastmath" not in (override_jit_flags or {})
        # self.default_jit_flags = {
        #     **JIT_FLAGS,  # here parallel=False (for out-of-backend code)
        #     **{"fastmath": self.formulae.fastmath, "parallel": parallel_default},
        #     **(override_jit_flags or {}),
        # }
        self.default_jit_flags = {
            "parallel": False
        }

        methods.CollisionsMethods.__init__(self)
        methods.PairMethods.__init__(self)
        methods.IndexMethods.__init__(self)
        methods.PhysicsMethods.__init__(self)
        methods.MomentsMethods.__init__(self)
