"""
constant rate formulation (for tests)
"""

import numpy as np


class Constant:  # pylint: disable=too-few-public-methods
    def __init__(self, const):
        assert np.isfinite(const.J_HOM)

    @staticmethod
    def d_a_w_ice_within_range(const, da_w_ice):
        return da_w_ice >= const.KOOP_MIN_DA_W_ICE

    @staticmethod
    def d_a_w_ice_maximum(const, da_w_ice):
        return np.where(
            da_w_ice > const.KOOP_MAX_DA_W_ICE, const.KOOP_MAX_DA_W_ICE, da_w_ice
        )

    @staticmethod
    def j_hom(const, T, a_w_ice):  # pylint: disable=unused-argument
        return const.J_HOM
