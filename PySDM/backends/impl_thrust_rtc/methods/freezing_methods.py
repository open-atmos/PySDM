"""
GPU implementation of backend methods for freezing (singular and time-dependent immersion freezing)
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class FreezingMethods(ThrustRTCBackendMethods):
    @cached_property
    def freeze_time_dependent_body(self):
        return trtc.For(
            param_names=(
                "rand",
                "immersed_surface_area",
                "water_mass",
                "timestep",
                "cell",
                "a_w_ice",
                "relative_humidity",
                "thaw",
                "temperature",
            ),
            name_iter="i",
            body=f"""
                if (immersed_surface_area[i] == 0) {{
                    return;
                }}
                if (thaw && {self.formulae.trivia.frozen_and_above_freezing_point.c_inline(
                    water_mass="water_mass[i]",
                    temperature="temperature[cell[i]]"
                )}) {{
                    water_mass[i] = -1 * water_mass[i];
                }} else if ({self.formulae.trivia.unfrozen_and_saturated.c_inline(
                        water_mass="water_mass[i]",
                        relative_humidity="relative_humidity[cell[i]]"
                    )}) {{
                    auto rate = {self.formulae.heterogeneous_ice_nucleation_rate.j_het.c_inline(
                        a_w_ice="a_w_ice[cell[i]]"
                    )};
                    auto prob = 1 - exp(
                        -rate * immersed_surface_area[i] * timestep
                    );
                    if (rand[i] < prob) {{
                        water_mass[i] = -1 * water_mass[i];
                    }}
                }}
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @cached_property
    def freeze_singular_body(self):
        return trtc.For(
            param_names=(
                "freezing_temperature",
                "water_mass",
                "temperature",
                "relative_humidity",
                "cell",
                "thaw",
            ),
            name_iter="i",
            body=f"""
                if (freezing_temperature[i] == 0) {{
                    return;
                }}
                if (thaw && {self.formulae.trivia.frozen_and_above_freezing_point.c_inline(
                    water_mass="water_mass[i]",
                    temperature="temperature[cell[i]]"
                )}) {{
                    water_mass[i] = -1 * water_mass[i];
                }} else if (
                    {self.formulae.trivia.unfrozen_and_saturated.c_inline(
                        water_mass="water_mass[i]",
                        relative_humidity="relative_humidity[cell[i]]"
                    )} && temperature[cell[i]] <= freezing_temperature[i]
                ) {{
                    water_mass[i] = -1 * water_mass[i];
                }}
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def freeze_singular(
        self, *, attributes, temperature, relative_humidity, cell, thaw
    ):
        n_sd = len(attributes.freezing_temperature)
        self.freeze_singular_body.launch_n(
            n=n_sd,
            args=(
                attributes.freezing_temperature.data,
                attributes.water_mass.data,
                temperature.data,
                relative_humidity.data,
                cell.data,
                trtc.DVBool(thaw),
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def freeze_time_dependent(  # pylint: disable=unused-argument
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
        thaw,
    ):
        n_sd = len(attributes.immersed_surface_area)
        self.freeze_time_dependent_body.launch_n(
            n=n_sd,
            args=(
                rand.data,
                attributes.immersed_surface_area.data,
                attributes.water_mass.data,
                self._get_floating_point(timestep),
                cell.data,
                a_w_ice.data,
                relative_humidity.data,
                trtc.DVBool(thaw),
                temperature.data,
            ),
        )
