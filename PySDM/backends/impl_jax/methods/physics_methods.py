"""
CPU implementation of backend methods wrapping basic physics formulae
"""

from functools import cached_property

import jax


from PySDM.backends.impl_common.backend_methods import BackendMethods


class PhysicsMethods(BackendMethods):
    def __init__(self):
        BackendMethods.__init__(self)

    @cached_property
    def _volume_of_mass_body(self):
        # particle_shape_and_density__mass_to_volume = jax.jit(
        #     self.formulae_flattened.particle_shape_and_density__mass_to_volume.py_func)
        ff = self.formulae_flattened

        @jax.jit
        def body(mass):
            # volume = particle_shape_and_density__mass_to_volume(mass)
            return ff.particle_shape_and_density__mass_to_volume.py_func(mass)

            # return volume

        return body

    def volume_of_water_mass(self, volume, mass):
        mapped_func = jax.vmap(self._volume_of_mass_body, (0))
        volume.data = mapped_func(mass.data)
