"""
Koop homogeneous nucleation rate parameterization for solution droplets [Koop et al. 2000] corrected
such that it coincides with homogeneous nucleation rate parameterization for pure water droplets
[Koop and Murray 2016] at water saturation between 235K < T < 240K
 ([Spichtinger et al. 2023](https://doi.org/10.5194/acp-23-2035-2023))
"""

import numpy as np


class Koop_Correction:
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
                const.KOOP_2000_C1
                + const.KOOP_2000_C2 * da_w_ice
                + const.KOOP_2000_C3 * da_w_ice**2.0
                + const.KOOP_2000_C4 * da_w_ice**3.0
                + const.KOOP_CORR
            )
            * const.KOOP_UNIT
        )
