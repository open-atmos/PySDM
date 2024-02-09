"""
Equilibrium fractionation factors from
[Horita and Wesolowski 1994](https://doi.org/10.1016/0016-7037(94)90096-5)
"""

import numpy as np


class HoritaAndWesolowski1994:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_18O(const, T):
        return np.exp(
            const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T3 / T**3
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T2 / T**2
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T1 / T
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_18O_T0
        )

    @staticmethod
    def alpha_l_2H(const, T):
        return np.exp(
            const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T3 / T**3
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_0
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_1 * T
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_2 * T**2
            + const.HORITA_AND_WESOLOWSKI_1994_ALPHA_L_2H_T_3 * T**3
        )
