"""
Multi-threaded CPU backend using LLVM-powered just-in-time compilation
"""

import jax

from PySDM.backends.impl_jax import methods
from PySDM.backends.impl_jax.random import Random as ImportedRandom
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
        self,
        formulae=None,
        *,
        double_precision=True,
        override_jit_flags=None,  # pylint: disable=unused-argument
        # TODO #1913: investigate if there are any jit/jax flags we can add configuration for
        block_until_ready=False,
    ):
        jax.config.update("jax_enable_x64", True)
        if not double_precision:
            raise NotImplementedError()

        self.block_until_ready = block_until_ready  # TODO #1913: implement switch in jit code
        self.formulae = formulae or Formulae()
        self.formulae_flattened = self.formulae.flatten

        self.default_jit_flags = {"parallel": False}

        methods.CollisionsMethods.__init__(self)
        methods.PairMethods.__init__(self)
        methods.IndexMethods.__init__(self)
        methods.PhysicsMethods.__init__(self)
        methods.MomentsMethods.__init__(self)
