from functools import cached_property
from PySDM.backends.impl_common.backend_methods import BackendMethods
import numba


class DepositionMethods(BackendMethods):
    @cached_property
    def _deposition(self):
        assert self.formulae.particle_shape_and_density.supports_mixed_phase()

        mass_to_volume = self.formulae.particle_shape_and_density.mass_to_volume

        @numba.jit(**self.default_jit_flags)
        def body(water_mass):
            volume = mass_to_volume(water_mass[0])
            print(volume)

            if water_mass[0] > 0:
                water_mass[0] *= 1.1

        return body

    def deposition(self, water_mass):
        self._deposition(
            water_mass=water_mass.data
        )
