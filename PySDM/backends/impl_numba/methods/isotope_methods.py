"""
CPU implementation of isotope-relates backend methods
"""

from functools import cached_property

import numba

from PySDM.backends.impl_common.backend_methods import BackendMethods
from PySDM.backends.impl_numba import conf


class IsotopeMethods(BackendMethods):
    @cached_property
    def __isotopic_delta_body(self):
        phys_isotopic_delta = self.formulae.trivia.isotopic_ratio_2_delta

        @numba.njit(**{**conf.JIT_FLAGS, "fastmath": self.formulae.fastmath})
        def isotopic_delta(output, ratio, reference_ratio):
            for i in numba.prange(output.shape[0]):  # pylint: disable=not-an-iterable
                output[i] = phys_isotopic_delta(ratio[i], reference_ratio)

        return isotopic_delta

    def isotopic_delta(self, output, ratio, reference_ratio):
        self.__isotopic_delta_body(output.data, ratio.data, reference_ratio)

    def isotopic_fractionation(self):
        pass
