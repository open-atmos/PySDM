"""
constant rate formulation (for tests)
"""

import numpy as np


class Constant:
    def __init__(self, const):
        assert np.isfinite(const.J_HOM)

    @staticmethod
    def d_a_w_ice_within_range(const, da_w_ice):  # pylint: disable=unused-argument
        return True

    @staticmethod
    def d_a_w_ice_maximum(const, da_w_ice):  # pylint: disable=unused-argument
        return da_w_ice

    @staticmethod
    def j_hom(const, T, a_w_ice):  # pylint: disable=unused-argument
        return const.J_HOM
