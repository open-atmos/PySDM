"""
GPU implementation of backend methods for terminal velocities
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class TerminalVelocityMethods(ThrustRTCBackendMethods):
    @cached_property
    def __gunn_and_kinzer_interpolation_body(self):
        # TODO #599 r<0
        return trtc.For(
            ("output", "radius", "factor", "a", "b"),
            "i",
            """
            auto r_id = (int64_t)(factor * radius[i]);
            auto r_rest = (factor * radius[i] - r_id) / factor;
            output[i] = a[r_id] + r_rest * b[r_id];
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def gunn_and_kinzer_interpolation(self, *, output, radius, factor, b, c):
        factor_device = trtc.DVInt64(factor)
        self.__gunn_and_kinzer_interpolation_body.launch_n(
            len(radius), (output.data, radius.data, factor_device, b.data, c.data)
        )

    @cached_property
    def __power_series_body(self):
        # TODO #599 r<0
        return trtc.For(
            ("output", "radius", "num_terms", "prefactors", "powers"),
            "i",
            """
            output[i] = 0.0
            int kmax = num_terms
            for (auto k = 0; k < kmax; k+=1) {
                auto sumterm = prefactors[k] * pow(radius[i], 3*powers[k])
                output[i] = output[i] + sumterm
            """,
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def power_series(self, *, values, radius, num_terms, prefactors, powers):
        prefactors = self._get_floating_point(prefactors)
        powers = self._get_floating_point(powers)
        num_terms = self._get_floating_point(num_terms)
        self.__power_series_body.launch_n(
            values.size(), (values, radius, num_terms, prefactors, powers)
        )

    @cached_property
    def __rogers_and_yau_terminal_velocity_body(self):
        return trtc.For(
            param_names=("values", "radius"),
            name_iter="i",
            body=f"""
            values[i] = {self.formulae.terminal_velocity.v_term.c_inline(
                radius="radius[i]"
            )};
            """.replace("real_type", self._get_c_type()),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def rogers_and_yau_terminal_velocity(self, *, values, radius):
        self.__rogers_and_yau_terminal_velocity_body.launch_n(
            n=values.size(), args=[values, radius]
        )
