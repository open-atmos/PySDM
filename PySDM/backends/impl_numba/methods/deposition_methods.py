"""basic water vapor deposition on ice for cpu backend"""

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

        @numba.jit(**self.default_jit_flags)
        def body(  # pylint: disable=too-many-arguments
            multiplicity,
            signed_water_mass,
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
            # pylint: disable=too-many-locals
            n_sd = len(signed_water_mass)
            for i in range(n_sd):
                if not liquid(signed_water_mass[i]):
                    ice_mass = -signed_water_mass[i]
                    cid = cell_id[i]

                    radius = formulae.particle_shape_and_density__mass_to_radius(
                        signed_water_mass[i]
                    )

                    diameter = radius * 2.0

                    temperature = ambient_temperature[cid]
                    pressure = ambient_total_pressure[cid]
                    rho = ambient_dry_air_density[cid]
                    Rv = formulae.constants.Rv
                    pvs_ice = formulae.saturation_vapour_pressure__pvs_ice(temperature)
                    latent_heat_sub = formulae.latent_heat_sublimation__ls(temperature)

                    capacity = formulae.diffusion_ice_capacity__capacity(diameter)

                    # TODO #1389
                    # pylint: disable=unused-variable
                    ventilation_factor = formulae.ventilation__ventilation_coefficient(
                        sqrt_re_times_cbrt_sc=formulae.trivia__sqrt_re_times_cbrt_sc(
                            Re=reynolds_number[i],
                            Sc=schmidt_number[cid],
                        )
                    )

                    Dv_const = formulae.diffusion_thermics__D(temperature, pressure)
                    lambdaD = formulae.diffusion_ice_kinetics__lambdaD(
                        temperature, pressure
                    )
                    diffusion_coefficient = formulae.diffusion_ice_kinetics__D(
                        Dv_const, radius, lambdaD, temperature
                    )

                    Ka_const = formulae.diffusion_thermics__K(temperature, pressure)
                    lambdaK = formulae.diffusion_ice_kinetics__lambdaK(
                        temperature, pressure
                    )
                    thermal_conductivity = formulae.diffusion_ice_kinetics__K(
                        Ka_const, radius, lambdaK, temperature, rho
                    )

                    howell_factor = 1.0 / (
                        (latent_heat_sub / Rv / temperature - 1.0)
                        * latent_heat_sub
                        * diffusion_coefficient
                        / temperature
                        / thermal_conductivity
                        + Rv * temperature / pvs_ice
                    )

                    saturation_ratio_ice = (
                        ambient_humidity[cid] / ambient_water_activity[cid]
                    )

                    rho_vs_ice = pvs_ice / Rv / temperature

                    dm_dt = (
                        4
                        * np.pi
                        * capacity
                        * diffusion_coefficient
                        * howell_factor
                        * (saturation_ratio_ice - 1)
                    ) * rho_vs_ice

                    if dm_dt == 0:
                        continue

                    delta_rv_i = (
                        -dm_dt * multiplicity[i] * time_step / (cell_volume * rho)
                    )
                    if -delta_rv_i > ambient_vapour_mixing_ratio[cid]:
                        assert False
                    ambient_vapour_mixing_ratio[cid] += delta_rv_i

                    delta_T = -delta_rv_i * latent_heat_sub / formulae.constants.c_pd
                    ambient_temperature[cid] += delta_T

                    x_old = formulae.diffusion_coordinate__x(ice_mass)
                    dx_dt_old = formulae.diffusion_coordinate__dx_dt(x_old, dm_dt)
                    x_new = formulae.trivia__explicit_euler(x_old, time_step, dx_dt_old)
                    signed_water_mass[i] = -formulae.diffusion_coordinate__mass(x_new)

        return body

    def deposition(
        self,
        *,
        multiplicity,
        signed_water_mass,
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
            multiplicity=multiplicity.data,
            signed_water_mass=signed_water_mass.data,
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
