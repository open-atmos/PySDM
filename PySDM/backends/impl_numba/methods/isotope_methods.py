"""
CPU implementation of isotope-relates backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


class IsotopeMethods(BackendMethods):
    @cached_property
    def _isotopic_delta_body(self):
        ff = self.formulae_flattened

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def body(output, ratio, reference_ratio):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = ff.trivia__isotopic_ratio_2_delta(ratio[i], reference_ratio)

        return body

    def isotopic_delta(self, output, ratio, reference_ratio):
        self._isotopic_delta_body(output.data, ratio.data, reference_ratio)

    def isotopic_fractionation(self):
        pass
