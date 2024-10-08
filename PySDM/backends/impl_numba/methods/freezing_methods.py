"""
CPU implementation of backend methods for freezing (singular and time-dependent immersion freezing)
"""

import numba
import numpy as np

from PySDM.backends.impl_common.backend_methods import BackendMethods

from ...impl_common.freezing_attributes import (
    SingularAttributes,
    TimeDependentAttributes,
)


class FreezingMethods(BackendMethods):
    def __init__(self):
        BackendMethods.__init__(self)
        unfrozen_and_saturated = self.formulae.trivia.unfrozen_and_saturated
        frozen_and_above_freezing_point = (
            self.formulae.trivia.frozen_and_above_freezing_point
        )

        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def _freeze(water_mass, i):
            water_mass[i] = -1 * water_mass[i]
            # TODO #599: change thd (latent heat)!

        @numba.njit(**{**self.default_jit_flags, "parallel": False})
        def _thaw(water_mass, i):
            water_mass[i] = -1 * water_mass[i]
            # TODO #599: change thd (latent heat)!

        @numba.njit(**self.default_jit_flags)
        def freeze_singular_body(
            attributes, temperature, relative_humidity, cell, thaw
        ):
            n_sd = len(attributes.freezing_temperature)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                if attributes.freezing_temperature[i] == 0:
                    continue
                if thaw and frozen_and_above_freezing_point(
                    attributes.water_mass[i], temperature[cell[i]]
                ):
                    _thaw(attributes.water_mass, i)
                elif (
                    unfrozen_and_saturated(
                        attributes.water_mass[i], relative_humidity[cell[i]]
                    )
                    and temperature[cell[i]] <= attributes.freezing_temperature[i]
                ):
                    _freeze(attributes.water_mass, i)

        self.freeze_singular_body = freeze_singular_body

        j_het = self.formulae.heterogeneous_ice_nucleation_rate.j_het

        @numba.njit(**self.default_jit_flags)
        def freeze_time_dependent_body(  # pylint: disable=unused-argument,too-many-arguments
            rand,
            attributes,
            timestep,
            cell,
            a_w_ice,
            temperature,
            relative_humidity,
            record_freezing_temperature,
            freezing_temperature,
            thaw,
        ):
            n_sd = len(attributes.water_mass)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                if attributes.immersed_surface_area[i] == 0:
                    continue
                cell_id = cell[i]
                if thaw and frozen_and_above_freezing_point(
                    attributes.water_mass[i], temperature[cell_id]
                ):
                    _thaw(attributes.water_mass, i)
                elif unfrozen_and_saturated(
                    attributes.water_mass[i], relative_humidity[cell_id]
                ):
                    rate = j_het(a_w_ice[cell_id])
                    # TODO #594: this assumes constant T throughout timestep, can we do better?
                    prob = 1 - np.exp(  # TODO #599: common code for Poissonian prob
                        -rate * attributes.immersed_surface_area[i] * timestep
                    )
                    if rand[i] < prob:
                        _freeze(attributes.water_mass, i)
                        # if record_freezing_temperature:
                        #     freezing_temperature[i] = temperature[cell_id]

        self.freeze_time_dependent_body = freeze_time_dependent_body

    def freeze_singular(
        self, *, attributes, temperature, relative_humidity, cell, thaw: bool
    ):
        self.freeze_singular_body(
            SingularAttributes(
                freezing_temperature=attributes.freezing_temperature.data,
                water_mass=attributes.water_mass.data,
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
        record_freezing_temperature,
        freezing_temperature,
        thaw: bool
    ):
        self.freeze_time_dependent_body(
            rand.data,
            TimeDependentAttributes(
                immersed_surface_area=attributes.immersed_surface_area.data,
                water_mass=attributes.water_mass.data,
            ),
            timestep,
            cell.data,
            a_w_ice.data,
            temperature.data,
            relative_humidity.data,
            record_freezing_temperature=record_freezing_temperature,
            freezing_temperature=(
                freezing_temperature.data if record_freezing_temperature else None
            ),
            thaw=thaw,
        )
