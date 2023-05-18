"""
Equilibrium fractionation factors from [Majoube 1970](https://doi.org/10.1038/2261242a0)
"""
import numpy as np


class Majoube1970:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_i_H2_18O(const, T):
        return np.exp(
            const.MAJOUBE_1970_ALPHA_I_H218O_T2 / T**2
            + const.MAJOUBE_1970_ALPHA_I_H218O_T1 / T
            + const.MAJOUBE_1970_ALPHA_I_H218O_T0
        )
