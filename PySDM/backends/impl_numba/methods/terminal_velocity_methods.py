"""
CPU implementation of backend methods for terminal velocities
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods


class TerminalVelocityMethods(BackendMethods):

    @cached_property
    def _gunn_and_kinzer_interpolation_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(output, radius, factor, b, c):
            for i in numba.prange(len(radius)):  # pylint: disable=not-an-iterable
                if radius[i] > 0:
                    r_id = int(factor * radius[i])
                    r_rest = ((factor * radius[i]) % 1) / factor
                    output[i] = b[r_id] + r_rest * c[r_id]
                elif radius[i] == 0:
                    output[i] = 0

        return body

    def gunn_and_kinzer_interpolation(self, *, output, radius, factor, b, c):
        return self._gunn_and_kinzer_interpolation_body(
            output.data, radius.data, factor, b.data, c.data
        )

    @cached_property
    def _rogers_and_yau_terminal_velocity_body(self):
        v_term = self.formulae.terminal_velocity.v_term

        @numba.njit(**self.default_jit_flags)
        def body(*, values, radius):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                if radius[i] >= 0.0:
                    values[i] = v_term(radius[i])

        return body

    def rogers_and_yau_terminal_velocity(self, *, values, radius):
        self._rogers_and_yau_terminal_velocity_body(values=values, radius=radius)

    @cached_property
    def _power_series_body(self):
        @numba.njit(**self.default_jit_flags)
        def body(*, values, radius, num_terms, prefactors, powers):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                values[i] = 0.0
                for j in range(num_terms):
                    values[i] = values[i] + prefactors[j] * radius[i] ** (powers[j] * 3)

        return body

    def power_series(self, *, values, radius, num_terms, prefactors, powers):
        self._power_series_body(
            values=values,
            radius=radius,
            num_terms=num_terms,
            prefactors=prefactors,
            powers=powers,
        )

    def terminal_velocity_columnar_ice_crystals(
        self, *, values, signed_water_mass, cell_id, temperature, pressure
    ):
        self._terminal_velocity_columnar_ice_crystals_body(
            values=values,
            signed_water_mass=signed_water_mass,
            cell_id=cell_id,
            temperature=temperature,
            pressure=pressure,
        )

    @cached_property
    def _terminal_velocity_columnar_ice_crystals_body(self):
        v_base_term = self.formulae.terminal_velocity_ice.v_base_term
        atmospheric_correction_factor = (
            self.formulae.terminal_velocity_ice.atmospheric_correction_factor
        )

        @numba.njit(**self.default_jit_flags)
        def body(*, values, signed_water_mass, cell_id, temperature, pressure):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                if signed_water_mass[i] < 0:
                    cid = cell_id[i]
                    correction = atmospheric_correction_factor(
                        temperature[cid], pressure[cid]
                    )
                    values[i] = v_base_term(-signed_water_mass[i]) * correction

        return body

    def terminal_velocity_ice_spheres(
        self, *, values, signed_water_mass, cell_id, temperature, pressure
    ):  # pylint: disable=unused-argument
        self._terminal_velocity_ice_spheres_body(
            values=values,
            signed_water_mass=signed_water_mass,
            cell_id=cell_id,
            temperature=temperature,
        )

    @cached_property
    def _terminal_velocity_ice_spheres_body(self):
        v_base_term = self.formulae.terminal_velocity_ice.v_base_term
        stokes_prefactor = self.formulae.terminal_velocity_ice.stokes_regime
        formulae = self.formulae_flattened

        def body(*, values, signed_water_mass, cell_id, temperature):
            for i in numba.prange(len(values)):  # pylint: disable=not-an-iterable
                if signed_water_mass[i] < 0:
                    cid = cell_id[i]
                    radius = formulae.particle_shape_and_density__mass_to_radius(
                        signed_water_mass[i]
                    )
                    dynamic_viscosity = formulae.air_dynamic_viscosity__eta_air(
                        temperature[cid]
                    )
                    prefactor = stokes_prefactor(radius, dynamic_viscosity)
                    values[i] = v_base_term(prefactor, radius)

        return body
