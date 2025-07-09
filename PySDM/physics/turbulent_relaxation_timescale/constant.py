"""
constant timescale formulation (for tests)
"""

import numpy as np


class Constant:
    def __init__(self, const):
        assert np.isfinite(const.TURBULENT_RELAXATION_TIMESCALE_FOR_TESTS)

    @staticmethod
    def tau(
        const, linear_eddy_length_scale, tke_dissipation_rate
    ):  # pylint: disable=unused-argument
        return const.TURBULENT_RELAXATION_TIMESCALE_FOR_TESTS
