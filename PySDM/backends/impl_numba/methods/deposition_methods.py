""" basic water vapor deposition on ice for cpu backend """

from functools import cached_property
import numba
import numpy as np
from PySDM.backends.impl_common.backend_methods import BackendMethods


class DepositionMethods(BackendMethods):  # pylint:disable=too-few-public-methods
    @cached_property
    def _deposition(self):
        assert self.formulae.particle_shape_and_density.supports_mixed_phase()

        formulae = self.formulae_flattened
        liquid = formulae.trivia__unfrozen

        diffusion_coefficient_function = self.formulae.diffusion_thermics.D

        @numba.jit(**self.default_jit_flags)
        def body(
            water_mass,
            ambient_temperature,
            ambient_total_pressure,
            ambient_humidity,
            ambient_water_activity,
            ambient_vapour_mixing_ratio,
            ambient_dry_air_density,
            cell_volume,
            time_step,
            cell_id,
            reynolds_number,
            schmidt_number,
        ):
            n_sd = len(water_mass)
            # for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
            for i in range(n_sd):
                if not liquid(water_mass[i]):
                    ice_mass = -water_mass[i]
                    cid = cell_id[i]

                    temperature = ambient_temperature[cid]
                    pressure = ambient_total_pressure[cid]
                    capacity = formulae.particle_shape_and_density__ice_mass_to_radius(
                        water_mass[i]
                    )

                    ventilation_factor = formulae.ventilation__ventilation_coefficient(
                        sqrt_re_times_cbrt_sc=formulae.trivia__sqrt_re_times_cbrt_sc(
                            Re=reynolds_number[i],
                            Sc=schmidt_number[cid],
                        )
                    )

                    diffusion_coefficient = diffusion_coefficient_function(
                        temperature, pressure
                    )

                    saturation_ratio_ice = (
                        ambient_humidity[cid] / ambient_water_activity[cid]
                    )

                    dm_dt = (
                        4
                        * np.pi
                        * capacity
                        * diffusion_coefficient
                        * (saturation_ratio_ice - 1)
                    )
                    # print("dm_dt", dm_dt)
                    if dm_dt == 0:
                        continue
                    delta_rv_i = (
                        -dm_dt
                        * time_step
                        / (cell_volume * ambient_dry_air_density[cid])
                    )
                    if -delta_rv_i > ambient_vapour_mixing_ratio[cid]:
                        assert False
                    # print('delta_rv_i', delta_rv_i)
                    # print('A', ambient_vapour_mixing_ratio[cid])
                    ambient_vapour_mixing_ratio[cid] += delta_rv_i
                    # print('B', ambient_vapour_mixing_ratio[cid])
                    # print(
                    #     f" {volume=}, {temperature=}, {pressure=}, {diffusion_coefficient=},"
                    # )
                    # print(f"  {time_step=},  {saturation_ratio_ice=}, {dm_dt=}")

                    x_old = formulae.diffusion_coordinate__x(ice_mass)
                    dx_dt_old = formulae.diffusion_coordinate__dx_dt(x_old, dm_dt)
                    x_new = formulae.trivia__explicit_euler(x_old, time_step, dx_dt_old)

                    # print(x_old, x_new, dm_dt)

                    water_mass[i] = -formulae.diffusion_coordinate__mass(x_new)

        return body

    def deposition(
        self,
        *,
        water_mass,
        ambient_temperature,
        ambient_total_pressure,
        ambient_humidity,
        ambient_water_activity,
        ambient_vapour_mixing_ratio,
        ambient_dry_air_density,
        cell_volume,
        time_step,
        cell_id,
        reynolds_number,
        schmidt_number,
    ):
        self._deposition(
            water_mass=water_mass.data,
            ambient_temperature=ambient_temperature.data,
            ambient_total_pressure=ambient_total_pressure.data,
            ambient_humidity=ambient_humidity.data,
            ambient_water_activity=ambient_water_activity.data,
            ambient_vapour_mixing_ratio=ambient_vapour_mixing_ratio.data,
            ambient_dry_air_density=ambient_dry_air_density.data,
            cell_volume=cell_volume,
            time_step=time_step,
            cell_id=cell_id.data,
            reynolds_number=reynolds_number.data,
            schmidt_number=schmidt_number.data,
        )
