from functools import cached_property
from PySDM.backends.impl_common.backend_methods import BackendMethods
import numba


class DepositionMethods(BackendMethods):
    @cached_property
    def _deposition(self):
        assert self.formulae.particle_shape_and_density.supports_mixed_phase()

        mass_to_volume = self.formulae.particle_shape_and_density.mass_to_volume
        diffusion_coefficient_function = self.formulae.diffusion_thermics.D

        @numba.jit(**self.default_jit_flags)
        def body(
            water_mass, ambient_temperature, ambient_total_pressure, time_step, cell_id
        ):

            n_sd = len(water_mass)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable

                if water_mass[i] < 0:

                    cid = cell_id[i]

                    volume = mass_to_volume(water_mass[i])

                    temperature = ambient_temperature[cid]
                    pressure = ambient_total_pressure[cid]

                    diffusion_coefficient = diffusion_coefficient_function(
                        temperature, pressure
                    )

                    print(volume, temperature, pressure, diffusion_coefficient)

                    water_mass[i] *= 1.1

        return body

    def deposition(
        self,
        *,
        water_mass,
        ambient_temperature,
        ambient_total_pressure,
        time_step,
        cell_id
    ):
        self._deposition(
            water_mass=water_mass.data,
            ambient_temperature=ambient_temperature.data,
            ambient_total_pressure=ambient_total_pressure.data,
            time_step=time_step,
            cell_id=cell_id.data,
        )
