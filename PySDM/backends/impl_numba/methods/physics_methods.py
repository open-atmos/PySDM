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
    def _critical_volume_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(*, v_cr, kappa, f_org, v_dry, v_wet, T, cell):
            for i in prange(len(v_cr)):  # pylint: disable=not-an-iterable
                sigma = ff.surface_tension__sigma(
                    T[cell[i]], v_wet[i], v_dry[i], f_org[i]
                )
                v_cr[i] = ff.trivia__volume(
                    ff.hygroscopicity__r_cr(
                        kp=kappa[i],
                        rd3=v_dry[i] / ff.constants.PI_4_3,
                        T=T[cell[i]],
                        sgm=sigma,
                    )
                )

        return body

    def critical_volume(self, *, v_cr, kappa, f_org, v_dry, v_wet, T, cell):
        self._critical_volume_body(
            v_cr=v_cr.data,
            kappa=kappa.data,
            f_org=f_org.data,
            v_dry=v_dry.data,
            v_wet=v_wet.data,
            T=T.data,
            cell=cell.data,
        )

    @cached_property
    def _temperature_pressure_rh_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(*, rhod, thd, water_vapour_mixing_ratio, T, p, RH):
            for i in prange(T.shape[0]):  # pylint: disable=not-an-iterable
                T[i] = ff.state_variable_triplet__T(rhod[i], thd[i])
                p[i] = ff.state_variable_triplet__p(
                    rhod[i], T[i], water_vapour_mixing_ratio[i]
                )
                RH[i] = ff.state_variable_triplet__pv(
                    p[i], water_vapour_mixing_ratio[i]
                ) / ff.saturation_vapour_pressure__pvs_Celsius(T[i] - ff.constants.T0)

        return body

    def temperature_pressure_rh(
        self, *, rhod, thd, water_vapour_mixing_ratio, T, p, RH
    ):
        self._temperature_pressure_rh_body(
            rhod=rhod.data,
            thd=thd.data,
            water_vapour_mixing_ratio=water_vapour_mixing_ratio.data,
            T=T.data,
            p=p.data,
            RH=RH.data,
        )

    @cached_property
    def _a_w_ice_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(*, T_in, p_in, RH_in, water_vapour_mixing_ratio_in, a_w_ice_out):
            for i in prange(T_in.shape[0]):  # pylint: disable=not-an-iterable
                pvi = ff.saturation_vapour_pressure__ice_Celsius(
                    T_in[i] - ff.constants.T0
                )
                pv = ff.state_variable_triplet__pv(
                    p_in[i], water_vapour_mixing_ratio_in[i]
                )
                pvs = pv / RH_in[i]
                a_w_ice_out[i] = pvi / pvs

        return body

    def a_w_ice(self, *, T, p, RH, water_vapour_mixing_ratio, a_w_ice):
        self._a_w_ice_body(
            T_in=T.data,
            p_in=p.data,
            RH_in=RH.data,
            water_vapour_mixing_ratio_in=water_vapour_mixing_ratio.data,
            a_w_ice_out=a_w_ice.data,
        )

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

    @cached_property
    def _mass_of_volume_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(mass, volume):
            for i in prange(volume.shape[0]):  # pylint: disable=not-an-iterable
                mass[i] = ff.particle_shape_and_density__volume_to_mass(volume[i])

        return body

    def mass_of_water_volume(self, mass, volume):
        self._mass_of_volume_body(mass.data, volume.data)

    @cached_property
    def __air_density_body(self):
        formulae = self.formulae.flatten

        @numba.njit(**self.default_jit_flags)
        def body(output, rhod, water_vapour_mixing_ratio):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = (
                    formulae.state_variable_triplet__rho_of_rhod_and_water_vapour_mixing_ratio(
                        rhod[i], water_vapour_mixing_ratio[i]
                    )
                )

        return body

    def air_density(self, *, output, rhod, water_vapour_mixing_ratio):
        self.__air_density_body(output.data, rhod.data, water_vapour_mixing_ratio.data)

    @cached_property
    def __air_dynamic_viscosity_body(self):
        formulae = self.formulae.flatten

        @numba.njit(**self.default_jit_flags)
        def body(output, temperature):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = formulae.air_dynamic_viscosity__eta_air(temperature[i])

        return body

    def air_dynamic_viscosity(self, *, output, temperature):
        self.__air_dynamic_viscosity_body(output.data, temperature.data)

    @cached_property
    def __reynolds_number_body(self):
        formulae = self.formulae.flatten

        @numba.njit(**self.default_jit_flags)
        def body(  # pylint: disable=too-many-arguments
            output,
            cell_id,
            air_dynamic_viscosity,
            air_density,
            radius,
            velocity_wrt_air,
        ):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = formulae.particle_shape_and_density__reynolds_number(
                    radius=radius[i],
                    velocity_wrt_air=velocity_wrt_air[i],
                    dynamic_viscosity=air_dynamic_viscosity[cell_id[i]],
                    density=air_density[cell_id[i]],
                )

        return body

    def reynolds_number(
        self, *, output, cell_id, dynamic_viscosity, density, radius, velocity_wrt_air
    ):
        self.__reynolds_number_body(
            output.data,
            cell_id.data,
            dynamic_viscosity.data,
            density.data,
            radius.data,
            velocity_wrt_air.data,
        )

    @cached_property
    def _explicit_euler_body(self):
        ff = self.formulae_flattened

        @numba.njit(**self.default_jit_flags)
        def body(y, dt, dy_dt):
            y[:] = ff.trivia__explicit_euler(y, dt, dy_dt)

        return body

    def explicit_euler(self, y, dt, dy_dt):
        self._explicit_euler_body(y.data, dt, dy_dt)
