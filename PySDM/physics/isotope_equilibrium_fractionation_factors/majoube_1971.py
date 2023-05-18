"""
Equilibrium fractionation factors from [Majoube 1971](https://doi.org/10.1051/jcp/1971681423)
"""
import numpy as np


class Majoube1971:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_H2_18O(const, T):
        return np.exp(
            const.MAJOUBE_1971_ALPHA_L_H218O_T2 / T**2
            + const.MAJOUBE_1971_ALPHA_L_H218O_T1 / T
            + const.MAJOUBE_1971_ALPHA_L_H218O_T0
        )
