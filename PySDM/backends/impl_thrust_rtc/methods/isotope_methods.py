"""
GPU implementation of isotope-relates backend methods
"""

from functools import cached_property

from PySDM.backends.impl_thrust_rtc.conf import NICE_THRUST_FLAGS
from PySDM.backends.impl_thrust_rtc.nice_thrust import nice_thrust

from ..conf import trtc
from ..methods.thrust_rtc_backend_methods import ThrustRTCBackendMethods


class IsotopeMethods(ThrustRTCBackendMethods):
    @cached_property
    def __isotopic_delta(self):
        return trtc.For(
            param_names=("output", "ratio", "reference_ratio"),
            name_iter="i",
            body=f"""
            output[i] = {self.formulae.trivia.isotopic_ratio_2_delta.c_inline(
                ratio="ratio[i]",
                reference_ratio="reference_ratio"
            )};
            """.replace(
                "real_type", self._get_c_type()
            ),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def isotopic_delta(self, output, ratio, reference_ratio):
        self.__isotopic_delta.launch_n(
            n=output.shape[0],
            args=(output.data, ratio.data, self._get_floating_point(reference_ratio)),
        )

    @nice_thrust(**NICE_THRUST_FLAGS)
    def isotopic_fractionation(self):
        pass
