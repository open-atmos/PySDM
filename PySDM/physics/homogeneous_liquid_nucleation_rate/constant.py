"""
constant rate formulation (for tests)
"""

import numpy as np


class Constant:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.J_LIQ_HOMO)

    @staticmethod
    def j_liq_homo(const, T, S):  # pylint: disable=unused-argument
        return const.J_LIQ_HOMO
