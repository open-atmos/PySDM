"""basic water vapor deposition on ice for CPU backend, for Howell factor see:
[Howell 1949](https://doi.org/10.1175/1520-0469(1949)006%3C0134:TGOCDI%3E2.0.CO;2)
"""

from functools import cached_property
import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods


# TODO #1524
# pylint: disable=too-many-arguments,too-many-locals,too-many-statements


class DepositionMethods(BackendMethods):  # pylint:disable=too-few-public-methods
    @cached_property
    def _deposition(self):
        assert self.formulae.particle_shape_and_density.supports_mixed_phase()

        formulae = self.formulae_flattened
        multiplier = 2
        midpoint = True
        rel_tol_rh = 1e-2
        fuse = 16

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def calc_saturation_ratio_ice_temperature_and_pressure(
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
            return saturation_ratio_ice, temperature, total_pressure

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def mass_deposition_rate_per_droplet(
            temperature: float,
            rho_d: float,
            signed_mass_old: float,
            latent_heat_sub: float,
            saturation_ratio_ice: float,
            pressure: float,
        ):
            radius = formulae.particle_shape_and_density__mass_to_radius(
                signed_mass_old
            )
            pvs_ice = formulae.saturation_vapour_pressure__pvs_ice(temperature)

            capacity = formulae.diffusion_ice_capacity__capacity(abs(signed_mass_old))

            mass_ventilation_factor = 1  # TODO #1655
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
            if mass_deposition_rate > 1:
                print(
                    radius,
                    Fk,
                    Fd,
                    mass_deposition_rate,
                    howell_factor_x_diffcoef_x_rhovsice_x_icess,
                    pvs_ice,
                )
                assert False
            return mass_deposition_rate

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def _loop(
            fake,
            temperature,
            rhod,
            thd,
            signed_water_mass,
            saturation_ratio_ice,
            total_pressure,
            multiplicity,
            sub_time_step,
            mass_of_dry_air,
        ):
            latent_heat_sub = formulae.latent_heat_sublimation__ls(temperature)
            delta_rv = 0
            for i, ksi in enumerate(multiplicity):
                if not formulae.trivia__unfrozen(signed_water_mass[i]):
                    mass_deposition_rate = mass_deposition_rate_per_droplet(
                        temperature=temperature,
                        rho_d=rhod,
                        signed_mass_old=signed_water_mass[i],
                        latent_heat_sub=latent_heat_sub,
                        saturation_ratio_ice=saturation_ratio_ice,
                        pressure=total_pressure,
                    )
                    delta_rv += (
                        -mass_deposition_rate * ksi * sub_time_step / mass_of_dry_air
                    )
                    if not fake:
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
                        if x_new > 1:
                            print(x_old, dx_dt_old, x_new, signed_water_mass[i])
                            assert False
            delta_thd = (
                formulae.state_variable_triplet__dthd_dt(
                    rhod=rhod,
                    thd=thd,
                    T=temperature,
                    d_water_vapour_mixing_ratio__dt=delta_rv / sub_time_step,
                    lv=latent_heat_sub,
                )
                * sub_time_step
            )
            if delta_rv == 0:
                assert delta_thd == 0
            else:
                assert (delta_rv < 0 < delta_thd) or (delta_rv > 0 > delta_thd)
            return delta_rv, delta_thd

        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(  # pylint: disable=too-many-arguments
            *,
            adaptive,
            multiplicity,
            signed_water_mass,
            current_vapour_mixing_ratio,
            current_dry_air_density,
            current_dry_potential_temperature,
            cell_volume,
            time_step,
            cell_id,
            # to be modified
            predicted_vapour_mixing_ratio,
            predicted_dry_potential_temperature,
            predicted_dry_air_density,
        ):
            """simplest adaptivity:
            - global dt - no cell-wise logic (we don't have any test/example for it!)
            - no mechanism to retain dt value over timesteps
            - explicit Euler mass integration (vs. implicit in condensation)
            """
            # pylint: disable=too-many-locals
            n_substeps = 1
            cid = cell_id[0]  # TODO #1524: add support for multi-cell environments

            rv_tendency = (
                predicted_vapour_mixing_ratio[cid] - current_vapour_mixing_ratio[cid]
            ) / time_step
            thd_tendency = (
                predicted_dry_potential_temperature[cid]
                - current_dry_potential_temperature[cid]
            ) / time_step
            rhod_tendency = (
                predicted_dry_air_density[cid] - current_dry_air_density[cid]
            ) / time_step
            dry_air_mass_mean = (
                cell_volume
                * (predicted_dry_air_density[cid] + current_dry_air_density[cid])
                / 2
            )

            if adaptive:
                n_substeps = 1 / multiplier
                delta_rh_long = np.nan
                for burnout in range(fuse + 1):
                    if burnout == fuse:
                        assert False
                    sub_time_step = time_step / n_substeps
                    rhod = (
                        current_dry_air_density[cid]
                        + rhod_tendency * (0.5 if midpoint else 1) * sub_time_step
                    )
                    rv = (
                        current_vapour_mixing_ratio[cid]
                        + rv_tendency * (0.5 if midpoint else 1) * sub_time_step
                    )
                    thd = (
                        current_dry_potential_temperature[cid]
                        + thd_tendency * (0.5 if midpoint else 1) * sub_time_step
                    )

                    saturation_ratio_ice, temperature, total_pressure = (
                        calc_saturation_ratio_ice_temperature_and_pressure(
                            vapour_mixing_ratio=rv,
                            dry_air_potential_temperature=thd,
                            dry_air_density=rhod,
                        )
                    )
                    delta_rv, delta_thd = _loop(
                        fake=True,
                        temperature=temperature,
                        rhod=rhod,
                        thd=thd,
                        signed_water_mass=signed_water_mass,
                        saturation_ratio_ice=saturation_ratio_ice,
                        total_pressure=total_pressure,
                        multiplicity=multiplicity,
                        sub_time_step=sub_time_step,
                        mass_of_dry_air=dry_air_mass_mean,
                    )
                    delta_rh_short = (
                        calc_saturation_ratio_ice_temperature_and_pressure(
                            vapour_mixing_ratio=rv + delta_rv,
                            dry_air_potential_temperature=thd + delta_thd,
                            dry_air_density=rhod,
                        )[0]
                        - saturation_ratio_ice
                    )
                    if (
                        n_substeps < 1
                        or rv < -delta_rv
                        or not formulae.trivia__within_tolerance(
                            abs(delta_rh_long - multiplier * delta_rh_short),
                            saturation_ratio_ice,
                            rel_tol_rh,
                        )
                    ):
                        delta_rh_long = delta_rh_short
                        n_substeps *= multiplier
                    else:
                        break
            sub_time_step = time_step / n_substeps

            rv = current_vapour_mixing_ratio[cid]
            thd = current_dry_potential_temperature[cid]
            rhod = current_dry_air_density[cid]

            assert n_substeps == int(n_substeps)
            for _ in range(int(n_substeps)):
                rv += sub_time_step * rv_tendency * (0.5 if midpoint else 1)
                thd += sub_time_step * thd_tendency * (0.5 if midpoint else 1)
                rhod += sub_time_step * rhod_tendency * (0.5 if midpoint else 1)

                saturation_ratio_ice, temperature, total_pressure = (
                    calc_saturation_ratio_ice_temperature_and_pressure(
                        vapour_mixing_ratio=rv,
                        dry_air_potential_temperature=thd,
                        dry_air_density=rhod,
                    )
                )
                delta_rv, delta_thd = _loop(
                    fake=False,
                    temperature=temperature,
                    rhod=rhod,
                    thd=thd,
                    signed_water_mass=signed_water_mass,
                    saturation_ratio_ice=saturation_ratio_ice,
                    total_pressure=total_pressure,
                    multiplicity=multiplicity,
                    sub_time_step=sub_time_step,
                    mass_of_dry_air=dry_air_mass_mean,
                )
                thd += delta_thd
                rv += delta_rv
                assert rv >= 0

                if midpoint:
                    thd += sub_time_step * thd_tendency / 2
                    rv += sub_time_step * rv_tendency / 2
                    rhod += sub_time_step * rhod_tendency / 2

            predicted_dry_potential_temperature[cid] = thd
            predicted_vapour_mixing_ratio[cid] = rv

        return body

    def deposition(  # pylint: disable=too-many-locals
        self,
        *,
        adaptive,
        multiplicity,
        signed_water_mass,
        current_vapour_mixing_ratio,
        current_dry_air_density,
        current_dry_potential_temperature,
        cell_volume,
        time_step,
        cell_id,
        predicted_vapour_mixing_ratio,
        predicted_dry_potential_temperature,
        predicted_dry_air_density,
    ):
        self._deposition(
            adaptive=adaptive,
            multiplicity=multiplicity.data,
            signed_water_mass=signed_water_mass.data,
            current_vapour_mixing_ratio=current_vapour_mixing_ratio.data,
            current_dry_air_density=current_dry_air_density.data,
            current_dry_potential_temperature=current_dry_potential_temperature.data,
            cell_volume=cell_volume,
            time_step=time_step,
            cell_id=cell_id.data,
            predicted_vapour_mixing_ratio=predicted_vapour_mixing_ratio.data,
            predicted_dry_potential_temperature=predicted_dry_potential_temperature.data,
            predicted_dry_air_density=predicted_dry_air_density.data,
        )
