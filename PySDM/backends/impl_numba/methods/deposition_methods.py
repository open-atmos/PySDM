"""basic water vapor deposition on ice for CPU backend, for Howell factor see:
[Howell 1949](https://doi.org/10.1175/1520-0469(1949)006%3C0134:TGOCDI%3E2.0.CO;2)
"""

from functools import cached_property
import numba
import numpy as np
from PyMPDATA_examples.Shipway_and_Hill_2012.formulae import rho_d

from PySDM.backends.impl_common.backend_methods import BackendMethods


class DepositionMethods(BackendMethods):  # pylint:disable=too-few-public-methods
    @cached_property
    def _deposition(self):
        assert self.formulae.particle_shape_and_density.supports_mixed_phase()

        formulae = self.formulae_flattened

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def calc_saturation_ratio_ice_and_temperature(
            vapour_mixing_ratio, dry_air_potential_temperature, dry_air_density
        ):
            temperature = formulae.state_variable_triplet__T(
                rhod=dry_air_density,
                thd=dry_air_potential_temperature,
            )
            total_pressure = formulae.state_variable_triplet__p(
                rhod=dry_air_density,
                T=temperature,
                water_vapour_mixing_ratio=vapour_mixing_ratio,
            )
            vapour_partial_pressure = formulae.state_variable_triplet__pv(
                p=total_pressure,
                water_vapour_mixing_ratio=vapour_mixing_ratio,
            )
            saturation_ratio_ice = (
                vapour_partial_pressure
                / formulae.saturation_vapour_pressure__pvs_ice(temperature)
            )
            return saturation_ratio_ice, temperature

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def mass_deposition_rate_per_droplet(
            temperature: float,
            rho_d: float,
            signed_mass_old: float,
            latent_heat_sub: float,
            saturation_ratio_ice: float,
            pressure: float,
            reynolds_number: float,
            schmidt_number: float,
        ):
            radius = formulae.particle_shape_and_density__mass_to_radius(
                signed_mass_old
            )

            diameter = radius * 2.0

            pvs_ice = formulae.saturation_vapour_pressure__pvs_ice(temperature)

            capacity = formulae.diffusion_ice_capacity__capacity(diameter)

            mass_ventilation_factor = formulae.ventilation__ventilation_coefficient(
                sqrt_re_times_cbrt_sc=formulae.trivia__sqrt_re_times_cbrt_sc(
                    Re=reynolds_number,
                    Sc=schmidt_number,
                )
            )
            heat_ventilation_factor = mass_ventilation_factor  # TODO #1588

            Dv_const = formulae.diffusion_thermics__D(temperature, pressure)
            lambdaD = formulae.diffusion_ice_kinetics__lambdaD(temperature, pressure)
            diffusion_coefficient = formulae.diffusion_ice_kinetics__D(
                Dv_const, radius, lambdaD, temperature
            )

            Ka_const = formulae.diffusion_thermics__K(temperature, pressure)
            lambdaK = formulae.diffusion_ice_kinetics__lambdaK(temperature, pressure)
            thermal_conductivity = formulae.diffusion_ice_kinetics__K(
                Ka_const, radius, lambdaK, temperature, rho_d
            )

            Fk = formulae.drop_growth__Fk(
                T=temperature,
                K=thermal_conductivity * heat_ventilation_factor,
                lv=latent_heat_sub,
            )
            Fd = formulae.drop_growth__Fd(
                T=temperature,
                D=diffusion_coefficient * mass_ventilation_factor,
                pvs=pvs_ice,
            )
            howell_factor_x_diffcoef_x_rhovsice_x_icess = (
                formulae.drop_growth__r_dr_dt(
                    RH_eq=1,
                    RH=saturation_ratio_ice,
                    Fk=Fk,
                    Fd=Fd,
                )
                * formulae.constants.rho_w
            )

            mass_deposition_rate = (
                4 * np.pi * capacity * howell_factor_x_diffcoef_x_rhovsice_x_icess
            )
            return mass_deposition_rate

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(  # pylint: disable=too-many-arguments
            *,
            adaptive,
            multiplicity,
            signed_water_mass,
            current_total_pressure,
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
            """simplest adaptivity:
            - no physical tolerance - just checking if ambient vapour is positive
            - global dt - no cell-wise logic (we don't have any test/example for it!)
            - no mechanism to retain shorter dt over timesteps (and hence cannot make it adapt towards longer)
            - explicit Euler mass integration (vs. implicit in condensation)
            - no safeguards for infinite loop in substep number search
            note: condensation uses theta for tolerance, we could use RH here (and later also in cond)
            """
            # pylint: disable=too-many-locals
            multiplier = 10
            midpoint = True
            rel_tol_rh = 1e-6

            cid = 0  # TODO

            n_sd = len(signed_water_mass)
            n_substeps = 1

            rv_tendency = (
                predicted_vapour_mixing_ratio[cid] - current_vapour_mixing_ratio[cid]
            ) / time_step
            thd_tendency = (
                predicted_dry_potential_temperature[cid]
                - current_dry_potential_temperature[cid]
            ) / time_step
            rho_d = current_dry_air_density[cid]

            if adaptive:
                n_substeps = 1 / multiplier
                delta_rh_long = np.nan
                while True:
                    sub_time_step = time_step / n_substeps
                    saturation_ratio_ice, temperature = (
                        calc_saturation_ratio_ice_and_temperature(
                            vapour_mixing_ratio=current_vapour_mixing_ratio[cid]
                            + rv_tendency * sub_time_step,
                            dry_air_potential_temperature=current_dry_potential_temperature[
                                cid
                            ]
                            + thd_tendency * sub_time_step,
                            dry_air_density=rho_d,
                        )
                    )
                    rv = current_vapour_mixing_ratio[cid]
                    for i in range(n_sd):
                        if not formulae.trivia__unfrozen(signed_water_mass[i]):

                            latent_heat_sub = formulae.latent_heat_sublimation__ls(
                                temperature
                            )

                            mass_deposition_rate = mass_deposition_rate_per_droplet(
                                temperature=temperature,
                                rho_d=rho_d,
                                signed_mass_old=signed_water_mass[i],
                                latent_heat_sub=latent_heat_sub,
                                saturation_ratio_ice=saturation_ratio_ice,
                                pressure=current_total_pressure[cid],
                                reynolds_number=reynolds_number[i],
                                schmidt_number=schmidt_number[cid],
                            )

                            rv += (
                                -mass_deposition_rate
                                * multiplicity[i]
                                * sub_time_step
                                / (cell_volume * rho_d)
                            )
                    delta_thd = formulae.state_variable_triplet__dthd_dt(
                        rhod=rho_d,
                        thd=current_dry_potential_temperature[cid],
                        T=temperature,
                        d_water_vapour_mixing_ratio__dt=(
                            rv - current_vapour_mixing_ratio[cid]
                        )
                        / sub_time_step,
                        lv=latent_heat_sub,
                    )
                    delta_rh_short = (
                        calc_saturation_ratio_ice_and_temperature(
                            vapour_mixing_ratio=rv,
                            dry_air_potential_temperature=current_dry_potential_temperature[
                                cid
                            ]
                            + thd_tendency * sub_time_step
                            + delta_thd,
                            dry_air_density=rho_d,
                        )[0]
                        - saturation_ratio_ice
                    )
                    if (
                        n_substeps < 1
                        or rv < 0
                        # or abs(delta_rh_long - multiplier * delta_rh_short)
                        # / saturation_ratio_ice
                        # > rel_tol_rh
                    ):
                        delta_rh_long = delta_rh_short
                        n_substeps *= multiplier
                        rv_tendency /= multiplier
                        thd_tendency /= multiplier
                    else:
                        break
            sub_time_step = time_step / n_substeps
            rv = current_vapour_mixing_ratio[cid]
            thd = current_dry_potential_temperature[cid]
            print("n", n_substeps)
            for _ in range(int(n_substeps)):
                # midpoint -> computer the sink with midpoint source
                rv += sub_time_step * rv_tendency * (0.5 if midpoint else 1)
                thd += sub_time_step * thd_tendency * (0.5 if midpoint else 1)
                saturation_ratio_ice, temperature = (
                    calc_saturation_ratio_ice_and_temperature(
                        vapour_mixing_ratio=rv,
                        dry_air_potential_temperature=thd,
                        dry_air_density=rho_d,
                    )
                )
                for i in range(n_sd):
                    if not formulae.trivia__unfrozen(signed_water_mass[i]):
                        cid = cell_id[i]

                        latent_heat_sub = formulae.latent_heat_sublimation__ls(
                            temperature
                        )
                        rho_d = current_dry_air_density[cid]

                        mass_deposition_rate = mass_deposition_rate_per_droplet(
                            temperature=temperature,
                            rho_d=rho_d,
                            signed_mass_old=signed_water_mass[i],
                            latent_heat_sub=latent_heat_sub,
                            saturation_ratio_ice=saturation_ratio_ice,
                            pressure=current_total_pressure[cid],
                            reynolds_number=reynolds_number[i],
                            schmidt_number=schmidt_number[cid],
                        )

                        delta_rv_i = (
                            -mass_deposition_rate
                            * multiplicity[i]
                            * sub_time_step
                            / (cell_volume * rho_d)
                        )
                        rv += delta_rv_i
                        assert rv >= 0

                        thd += (
                            formulae.state_variable_triplet__dthd_dt(
                                rhod=current_dry_air_density[cid],
                                thd=current_dry_potential_temperature[cid],
                                T=temperature,
                                d_water_vapour_mixing_ratio__dt=delta_rv_i
                                / sub_time_step,
                                lv=latent_heat_sub,
                            )
                            * sub_time_step
                        )

                        x_old = formulae.diffusion_coordinate__x(-signed_water_mass[i])
                        dx_dt_old = formulae.diffusion_coordinate__dx_dt(
                            -signed_water_mass[i], mass_deposition_rate
                        )
                        x_new = formulae.trivia__explicit_euler(
                            x_old, sub_time_step, dx_dt_old
                        )
                        signed_water_mass[i] = -formulae.diffusion_coordinate__mass(
                            x_new
                        )
                if midpoint:
                    thd += sub_time_step * thd_tendency / 2
                    rv += sub_time_step * rv_tendency / 2
            predicted_dry_potential_temperature[cid] = thd
            predicted_vapour_mixing_ratio[cid] = rv

        return body

    def deposition(  # pylint: disable=too-many-locals
        self,
        *,
        adaptive,
        multiplicity,
        signed_water_mass,
        current_total_pressure,  # TODO: do we need it ?
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
            adaptive=adaptive,
            multiplicity=multiplicity.data,
            signed_water_mass=signed_water_mass.data,
            current_total_pressure=current_total_pressure.data,
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
