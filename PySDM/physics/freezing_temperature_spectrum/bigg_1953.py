"""
freezing temperature spectrum based on [Bigg 1953](https://doi.org/10.1088/0370-1301/66/8/309)
 formulae (i.e. immersed surface independent)
"""

import numpy as np


class Bigg_1953:
    def __init__(self, const):
        assert np.isfinite(const.BIGG_DT_MEDIAN)

    @staticmethod
    def pdf(const, T, A_insol):  # pylint: disable=unused-argument
        A = np.log(1 - 0.5)
        B = const.BIGG_DT_MEDIAN - const.T0
        return -A * np.exp(A * np.exp(B + T) + B + T)

    @staticmethod
    def cdf(const, T, A_insol):  # pylint: disable=unused-argument
        return np.exp(np.log(1 - 0.5) * np.exp(const.BIGG_DT_MEDIAN - (const.T0 - T)))

    @staticmethod
    def median(const):
        return const.T0 - const.BIGG_DT_median
