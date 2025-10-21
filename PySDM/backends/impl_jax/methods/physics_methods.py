"""
CPU implementation of backend methods wrapping basic physics formulae
"""

from functools import cached_property

import numba
from numba import prange

from PySDM.backends.impl_common.backend_methods import BackendMethods


class PhysicsMethods(BackendMethods):
    def __init__(self):
        BackendMethods.__init__(self)

    @cached_property
    def _volume_of_mass_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(volume, mass):
            for i in prange(volume.shape[0]):  # pylint: disable=not-an-iterable
                volume[i] = ff.particle_shape_and_density__mass_to_volume(mass[i])

        return body

    def volume_of_water_mass(self, volume, mass):
        self._volume_of_mass_body(volume.data, mass.data)
