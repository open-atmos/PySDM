"""
Koop and Murray homogeneous nucleation rate parameterization for pure water droplets
at water saturation
([eq. A9, Tab VII in Koop and Murray 2016](https://doi.org/10.1063/1.4962355))
"""

import numpy as np


class KoopMurray2016:
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
                const.KOOP_MURRAY_C0
                + const.KOOP_MURRAY_C1 * (T - const.T0)
                + const.KOOP_MURRAY_C2 * (T - const.T0) ** 2.0
                + const.KOOP_MURRAY_C3 * (T - const.T0) ** 3.0
                + const.KOOP_MURRAY_C4 * (T - const.T0) ** 4.0
                + const.KOOP_MURRAY_C5 * (T - const.T0) ** 5.0
                + const.KOOP_MURRAY_C6 * (T - const.T0) ** 6.0
            )
            * const.KOOP_UNIT
        )
