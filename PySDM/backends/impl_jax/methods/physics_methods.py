"""
JAX implementation of backend methods wrapping basic physics formulae
"""

from functools import cached_property

import jax


from PySDM.backends.impl_common.backend_methods import BackendMethods


class PhysicsMethods(BackendMethods):  # pylint: disable=too-few-public-methods
    # TODO #1913: implement more physics methods to alleviate this pylint error
    def __init__(self):
        BackendMethods.__init__(self)

    @cached_property
    def _volume_of_mass_body(self):
        ff = self.formulae

        # @jax.jit
        # TODO #1913: fix test fails for this method
        def body(mass):
            return ff.particle_shape_and_density.mass_to_volume.jax(mass)

        return body

    def volume_of_water_mass(self, volume, mass):
        mapped_func = jax.vmap(self._volume_of_mass_body, (0))
        volume.data = mapped_func(mass.data).block_until_ready()
