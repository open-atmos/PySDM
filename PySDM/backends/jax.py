"""
CPU/GPU Backend using the JAX library
"""

from functools import wraps

import jax

from PySDM.backends.impl_jax import methods
from PySDM.backends.impl_jax.random import Random as ImportedRandom
from PySDM.backends.impl_jax.storage import Storage as ImportedStorage
from PySDM.formulae import Formulae


def with_default_device(cls):
    for name, method in cls.__dict__.items():
        if name.startswith("__") or not callable(method):
            continue

        @wraps(method)
        def wrapper(self, *args, __method=method, **kwargs):
            with jax.default_device(self.default_device):
                return __method(self, *args, **kwargs)

        setattr(cls, name, wrapper)

    return cls


@with_default_device
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
        jax_backend=None,
        *,
        double_precision=True,
        override_jit_flags=None,  # pylint: disable=unused-argument
        # TODO #1913: investigate if there are any jit/jax flags we can add configuration for
        block_until_ready=False,
    ):
        jax.config.update("jax_enable_x64", True)
        if not double_precision:
            raise NotImplementedError()

        self.default_device = jax.devices(backend=jax_backend)[0]

        self.block_until_ready = (
            block_until_ready  # TODO #1913: implement switch in jit code
        )
        self.formulae = formulae or Formulae()
        self.formulae_flattened = self.formulae.flatten

        self.default_jit_flags = {"parallel": False}

        methods.CollisionsMethods.__init__(self)
        methods.PairMethods.__init__(self)
        methods.IndexMethods.__init__(self)
        methods.PhysicsMethods.__init__(self)
        methods.MomentsMethods.__init__(self)
