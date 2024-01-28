"""
Equilibrium fractionation factors from [Majoube 1971](https://doi.org/10.1051/jcp/1971681423)
"""

import numpy as np


class Majoube1971:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_18O(const, T):
        return np.exp(
            const.MAJOUBE_1971_ALPHA_L_18O_T2 / T**2
            + const.MAJOUBE_1971_ALPHA_L_18O_T1 / T
            + const.MAJOUBE_1971_ALPHA_L_18O_T0
        )

    @staticmethod
    def alpha_l_2H(const, T):
        return np.exp(
            const.MAJOUBE_1971_ALPHA_L_2H_T2 / T**2
            + const.MAJOUBE_1971_ALPHA_L_2H_T1 / T
            + const.MAJOUBE_1971_ALPHA_L_2H_T0
        )
