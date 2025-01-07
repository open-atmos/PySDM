"""
constant rate formulation (for tests)
"""

import numpy as np


class Constant:
    def __init__(self, const):
        assert np.isfinite(const.J_LIQ_HOMO)
        assert np.isfinite(const.R_LIQ_HOMO)

    @staticmethod
    def j_liq_homo(const, T, S, e_s):  # pylint: disable=unused-argument
        return const.J_LIQ_HOMO

    @staticmethod
    def r_liq_homo(const, T, S):  # pylint: disable=unused-argument
        return const.R_LIQ_HOMO
