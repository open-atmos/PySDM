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
        ff = self.formulae_flattened

        @jax.jit
        def body(volume, mass):
            volume = ff.particle_shape_and_density__mass_to_volume.py_func(mass)

            return volume

        return body

    def volume_of_water_mass(self, volume, mass):
        volume.data = self._volume_of_mass_body(volume.data, mass.data)
