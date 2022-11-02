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
from ...impl_numba import conf


class FreezingMethods(BackendMethods):
    def __init__(self):
        BackendMethods.__init__(self)
        const = self.formulae.constants
        unfrozen_and_saturated = self.formulae.trivia.unfrozen_and_saturated

        @numba.njit(
            **{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath, "parallel": False}
        )
        def _freeze(volume, i):
            volume[i] = -1 * volume[i] * const.rho_w / const.rho_i
            # TODO #599: change thd (latent heat)!
            # TODO #599: handle the negative volume in tests, attributes, products, dynamics, ...

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def freeze_singular_body(attributes, temperature, relative_humidity, cell):
            n_sd = len(attributes.freezing_temperature)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                if attributes.freezing_temperature[i] == 0:
                    continue
                if (
                    unfrozen_and_saturated(
                        attributes.wet_volume[i], relative_humidity[cell[i]]
                    )
                    and temperature[cell[i]] <= attributes.freezing_temperature[i]
                ):
                    _freeze(attributes.wet_volume, i)

        self.freeze_singular_body = freeze_singular_body

        j_het = self.formulae.heterogeneous_ice_nucleation_rate.j_het

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
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
        ):
            n_sd = len(attributes.wet_volume)
            for i in numba.prange(n_sd):  # pylint: disable=not-an-iterable
                if attributes.immersed_surface_area[i] == 0:
                    continue
                cell_id = cell[i]
                if unfrozen_and_saturated(
                    attributes.wet_volume[i], relative_humidity[cell_id]
                ):
                    rate = j_het(a_w_ice[cell_id])
                    # TODO #594: this assumes constant T throughout timestep, can we do better?
                    prob = 1 - np.exp(  # TODO #599: common code for Poissonian prob
                        -rate * attributes.immersed_surface_area[i] * timestep
                    )
                    if rand[i] < prob:
                        _freeze(attributes.wet_volume, i)
                        # if record_freezing_temperature:
                        #     freezing_temperature[i] = temperature[cell_id]

        self.freeze_time_dependent_body = freeze_time_dependent_body

    def freeze_singular(self, *, attributes, temperature, relative_humidity, cell):
        self.freeze_singular_body(
            SingularAttributes(
                freezing_temperature=attributes.freezing_temperature.data,
                wet_volume=attributes.wet_volume.data,
            ),
            temperature.data,
            relative_humidity.data,
            cell.data,
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
        freezing_temperature
    ):
        self.freeze_time_dependent_body(
            rand.data,
            TimeDependentAttributes(
                immersed_surface_area=attributes.immersed_surface_area.data,
                wet_volume=attributes.wet_volume.data,
            ),
            timestep,
            cell.data,
            a_w_ice.data,
            temperature.data if record_freezing_temperature else None,
            relative_humidity.data,
            record_freezing_temperature=record_freezing_temperature,
            freezing_temperature=freezing_temperature.data
            if record_freezing_temperature
            else None,
        )
