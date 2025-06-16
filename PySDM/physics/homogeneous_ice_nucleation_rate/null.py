"""
do-nothing null formulation (needed as other formulations require parameters
 to be set before instantiation of Formulae)
"""

import numpy as np


class Null:  # pylint: disable=too-few-public-methods,unused-argument
    def __init__(self, _):
        pass

    @staticmethod
    def d_a_w_ice_within_range(const, da_w_ice):
        return da_w_ice >= const.KOOP_MIN_DA_W_ICE

    @staticmethod
    def d_a_w_ice_maximum(const, da_w_ice):
        return np.where(
            da_w_ice > const.KOOP_MAX_DA_W_ICE, const.KOOP_MAX_DA_W_ICE, da_w_ice
        )

    @staticmethod
    def j_hom(const, T, d_a_w_ice):  # pylint: disable=unused-argument
        return np.nan
