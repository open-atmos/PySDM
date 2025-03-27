"""basic water vapor deposition on ice for CPU backend, for Howell factor see:
[Howell 1949](https://doi.org/10.1175/1520-0469(1949)006%3C0134:TGOCDI%3E2.0.CO;2)
"""

from functools import cached_property
import numba
import numpy as np
from PySDM.backends.impl_common.backend_methods import BackendMethods


class DepositionMethods(BackendMethods):  # pylint:disable=too-few-public-methods
    @cached_property
    def _deposition(self):
        assert self.formulae.particle_shape_and_density.supports_mixed_phase()

        formulae = self.formulae_flattened

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(  # pylint: disable=too-many-arguments
            *,
            multiplicity,
            signed_water_mass,
            current_temperature,
            current_total_pressure,
            current_relative_humidity,
            current_water_activity,
            current_vapour_mixing_ratio,
            current_dry_air_density,
            current_dry_potential_temperature,
            cell_volume,
            time_step,
            cell_id,
            reynolds_number,
            schmidt_number,
            # to be modified
            predicted_vapour_mixing_ratio,
            predicted_dry_potential_temperature,
        ):
            # pylint: disable=too-many-locals
            n_sd = len(signed_water_mass)
            for i in range(n_sd):
                if not formulae.trivia__unfrozen(signed_water_mass[i]):
                    ice_mass = -signed_water_mass[i]
                    cid = cell_id[i]

                    radius = formulae.particle_shape_and_density__mass_to_radius(
                        signed_water_mass[i]
                    )

                    diameter = radius * 2.0

                    temperature = current_temperature[cid]
                    pressure = current_total_pressure[cid]
                    rho = current_dry_air_density[cid]
                    pvs_ice = formulae.saturation_vapour_pressure__pvs_ice(temperature)
                    latent_heat_sub = formulae.latent_heat_sublimation__ls(temperature)

                    capacity = formulae.diffusion_ice_capacity__capacity(diameter)

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
                    saturation_ratio_ice = (
                        current_relative_humidity[cid] / current_water_activity[cid]
                    )

                    if saturation_ratio_ice == 1:
                        continue

                    howell_factor_x_diffcoef_x_rhovsice_x_icess = (
                        formulae.drop_growth__r_dr_dt(
                            RH_eq=1,
                            T=temperature,
                            RH=saturation_ratio_ice,
                            lv=latent_heat_sub,
                            pvs=pvs_ice,
                            D=diffusion_coefficient,
                            K=thermal_conductivity,
                            ventilation_factor=ventilation_factor,
                        )
                        * formulae.constants.rho_w
                    )

                    dm_dt = (
                        4
                        * np.pi
                        * capacity
                        * howell_factor_x_diffcoef_x_rhovsice_x_icess
                    )

                    delta_rv_i = (
                        -dm_dt * multiplicity[i] * time_step / (cell_volume * rho)
                    )
                    if -delta_rv_i > current_vapour_mixing_ratio[cid]:
                        assert False
                    predicted_vapour_mixing_ratio[cid] += delta_rv_i

                    predicted_dry_potential_temperature[cid] += (
                        formulae.state_variable_triplet__dthd_dt(
                            rhod=current_dry_air_density[cid],
                            thd=current_dry_potential_temperature[cid],
                            T=temperature,
                            d_water_vapour_mixing_ratio__dt=delta_rv_i / time_step,
                            lv=latent_heat_sub,
                        )
                        * time_step
                    )

                    x_old = formulae.diffusion_coordinate__x(ice_mass)
                    dx_dt_old = formulae.diffusion_coordinate__dx_dt(ice_mass, dm_dt)
                    x_new = formulae.trivia__explicit_euler(x_old, time_step, dx_dt_old)
                    signed_water_mass[i] = -formulae.diffusion_coordinate__mass(x_new)

        return body

    def deposition(  # pylint: disable=too-many-locals
        self,
        *,
        multiplicity,
        signed_water_mass,
        current_temperature,
        current_total_pressure,
        current_relative_humidity,
        current_water_activity,
        current_vapour_mixing_ratio,
        current_dry_air_density,
        current_dry_potential_temperature,
        cell_volume,
        time_step,
        cell_id,
        reynolds_number,
        schmidt_number,
        predicted_vapour_mixing_ratio,
        predicted_dry_potential_temperature,
    ):
        self._deposition(
            multiplicity=multiplicity.data,
            signed_water_mass=signed_water_mass.data,
            current_temperature=current_temperature.data,
            current_total_pressure=current_total_pressure.data,
            current_relative_humidity=current_relative_humidity.data,
            current_water_activity=current_water_activity.data,
            current_vapour_mixing_ratio=current_vapour_mixing_ratio.data,
            current_dry_air_density=current_dry_air_density.data,
            current_dry_potential_temperature=current_dry_potential_temperature.data,
            cell_volume=cell_volume,
            time_step=time_step,
            cell_id=cell_id.data,
            reynolds_number=reynolds_number.data,
            schmidt_number=schmidt_number.data,
            predicted_vapour_mixing_ratio=predicted_vapour_mixing_ratio.data,
            predicted_dry_potential_temperature=predicted_dry_potential_temperature.data,
        )
