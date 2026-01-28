"""
Koop and Murray homogeneous nucleation rate parameterization for pure water droplets
adapted for water activity formulation
 ([Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023))
"""

import numpy as np


class KoopMurray2016_DAW:
    def __init__(self, const):
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
    def j_hom(const, T, da_w_ice):  # pylint: disable=unused-argument
        return (
            10
            ** (
                const.KOOP_MURRAY_DAW_C0
                + const.KOOP_MURRAY_DAW_C1 * da_w_ice
                + const.KOOP_MURRAY_DAW_C2 * da_w_ice**2.0
            )
            * const.KOOP_MURRAY_DAW_UNIT
        )
