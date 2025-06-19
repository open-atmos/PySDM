"""
CPU implementation of backend methods for homogeneous freezing and
heterogeneous freezing (singular and time-dependent immersion freezing)
"""

from functools import cached_property

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods

from ...impl_common.freezing_attributes import (
    SingularAttributes,
    TimeDependentAttributes,
    TimeDependentHomogeneousAttributes,
)


class FreezingMethods(BackendMethods):
    @cached_property
    def _freeze(self):
        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(signed_water_mass, i):
            signed_water_mass[i] = -1 * signed_water_mass[i]
            # TODO #599: change thd (latent heat)!

        return body

    @cached_property
    def _thaw(self):
        @numba.njit(**{**self.default_jit_flags, **{"parallel": False}})
        def body(signed_water_mass, i):
            signed_water_mass[i] = -1 * signed_water_mass[i]
            # TODO #599: change thd (latent heat)!

        return body

    @cached_property
    def _freeze_singular_body(self):
        _thaw = self._thaw
        _freeze = self._freeze
        frozen_and_above_freezing_point = (
            self.formulae.trivia.frozen_and_above_freezing_point
        )
        unfrozen_and_saturated = self.formulae.trivia.unfrozen_and_saturated

        @numba.njit(**self.default_jit_flags)
        def body(attributes, temperature, relative_humidity, cell, thaw):
            n_sd = len(attributes.freezing_temperature)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                if attributes.freezing_temperature[i] == 0:
                    continue
                if thaw and frozen_and_above_freezing_point(
                    attributes.signed_water_mass[i], temperature[cell[i]]
                ):
                    _thaw(attributes.signed_water_mass, i)
                elif (
                    unfrozen_and_saturated(
                        attributes.signed_water_mass[i], relative_humidity[cell[i]]
                    )
                    and temperature[cell[i]] <= attributes.freezing_temperature[i]
                ):
                    _freeze(attributes.signed_water_mass, i)

        return body

    @cached_property
    def _freeze_time_dependent_body(self):
        _thaw = self._thaw
        _freeze = self._freeze
        frozen_and_above_freezing_point = (
            self.formulae.trivia.frozen_and_above_freezing_point
        )
        unfrozen_and_saturated = self.formulae.trivia.unfrozen_and_saturated
        j_het = self.formulae.heterogeneous_ice_nucleation_rate.j_het
        prob_zero_events = self.formulae.trivia.poissonian_avoidance_function

        @numba.njit(**self.default_jit_flags)
        def body(  # pylint: disable=too-many-arguments
            rand,
            attributes,
            timestep,
            cell,
            a_w_ice,
            temperature,
            relative_humidity,
            thaw,
        ):
            n_sd = len(attributes.signed_water_mass)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                if attributes.immersed_surface_area[i] == 0:
                    continue
                cell_id = cell[i]
                if thaw and frozen_and_above_freezing_point(
                    attributes.signed_water_mass[i], temperature[cell_id]
                ):
                    _thaw(attributes.signed_water_mass, i)
                elif unfrozen_and_saturated(
                    attributes.signed_water_mass[i], relative_humidity[cell_id]
                ):
                    rate_assuming_constant_temperature_within_dt = (
                        j_het(a_w_ice[cell_id]) * attributes.immersed_surface_area[i]
                    )
                    prob = 1 - prob_zero_events(
                        r=rate_assuming_constant_temperature_within_dt, dt=timestep
                    )
                    if rand[i] < prob:
                        _freeze(attributes.signed_water_mass, i)

        return body

    @cached_property
    def _freeze_time_dependent_homogeneous_body(self):
        _thaw = self._thaw
        _freeze = self._freeze
        frozen_and_above_freezing_point = (
            self.formulae.trivia.frozen_and_above_freezing_point
        )
        unfrozen_and_ice_saturated = self.formulae.trivia.unfrozen_and_ice_saturated
        j_hom = self.formulae.homogeneous_ice_nucleation_rate.j_hom
        prob_zero_events = self.formulae.trivia.poissonian_avoidance_function
        d_a_w_ice_within_range = (
            self.formulae.homogeneous_ice_nucleation_rate.d_a_w_ice_within_range
        )
        d_a_w_ice_maximum = (
            self.formulae.homogeneous_ice_nucleation_rate.d_a_w_ice_maximum
        )

        @numba.njit(**self.default_jit_flags)
        def body(  # pylint: disable=unused-argument,too-many-arguments
            rand,
            attributes,
            timestep,
            cell,
            a_w_ice,
            temperature,
            relative_humidity_ice,
            thaw,
        ):

            n_sd = len(attributes.signed_water_mass)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                cell_id = cell[i]
                if thaw and frozen_and_above_freezing_point(
                    attributes.signed_water_mass[i], temperature[cell_id]
                ):
                    _thaw(attributes.signed_water_mass, i)
                elif unfrozen_and_ice_saturated(
                    attributes.signed_water_mass[i], relative_humidity_ice[cell_id]
                ):
                    d_a_w_ice = (relative_humidity_ice[cell_id] - 1.0) * a_w_ice[
                        cell_id
                    ]

                    if d_a_w_ice_within_range(d_a_w_ice):
                        d_a_w_ice = d_a_w_ice_maximum(d_a_w_ice)
                        rate_assuming_constant_temperature_within_dt = (
                            j_hom(temperature[cell_id], d_a_w_ice)
                            * attributes.volume[i]
                        )
                        prob = 1 - prob_zero_events(
                            r=rate_assuming_constant_temperature_within_dt, dt=timestep
                        )
                        if rand[i] < prob:
                            _freeze(attributes.signed_water_mass, i)

        return body

    def freeze_singular(
        self, *, attributes, temperature, relative_humidity, cell, thaw: bool
    ):
        self._freeze_singular_body(
            SingularAttributes(
                freezing_temperature=attributes.freezing_temperature.data,
                signed_water_mass=attributes.signed_water_mass.data,
            ),
            temperature.data,
            relative_humidity.data,
            cell.data,
            thaw=thaw,
        )

    def freeze_time_dependent(
        self,
        *,
        rand,
        attributes,
        timestep,
        cell,
        a_w_ice,
        temperature,
        relative_humidity,
        thaw: bool,
    ):
        self._freeze_time_dependent_body(
            rand.data,
            TimeDependentAttributes(
                immersed_surface_area=attributes.immersed_surface_area.data,
                signed_water_mass=attributes.signed_water_mass.data,
            ),
            timestep,
            cell.data,
            a_w_ice.data,
            temperature.data,
            relative_humidity.data,
            thaw=thaw,
        )

    def freeze_time_dependent_homogeneous(
        self,
        *,
        rand,
        attributes,
        timestep,
        cell,
        a_w_ice,
        temperature,
        relative_humidity_ice,
        thaw: bool,
    ):
        self._freeze_time_dependent_homogeneous_body(
            rand.data,
            TimeDependentHomogeneousAttributes(
                volume=attributes.volume.data,
                signed_water_mass=attributes.signed_water_mass.data,
            ),
            timestep,
            cell.data,
            a_w_ice.data,
            temperature.data,
            relative_humidity_ice.data,
            thaw=thaw,
        )

    @cached_property
    def _record_freezing_temperatures_body(self):
        ff = self.formulae_flattened

        @numba.njit(**{**self.default_jit_flags, "fastmath": False})
        def body(data, cell_id, temperature, signed_water_mass):
            for drop_id in numba.prange(len(data)):  # pylint: disable=not-an-iterable
                if ff.trivia__unfrozen(signed_water_mass[drop_id]):
                    if data[drop_id] > 0:
                        data[drop_id] = np.nan
                else:
                    if np.isnan(data[drop_id]):
                        data[drop_id] = temperature[cell_id[drop_id]]

        return body

    def record_freezing_temperatures(
        self, *, data, cell_id, temperature, signed_water_mass
    ):
        self._record_freezing_temperatures_body(
            data=data.data,
            cell_id=cell_id.data,
            temperature=temperature.data,
            signed_water_mass=signed_water_mass.data,
        )
