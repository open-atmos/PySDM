"""
Equilibrium fractionation factors used in [Bolot et al. 2013](https://10.5194/acp-13-7903-2013)
"""
import numpy as np


class BolotEtAl2013:
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_l_HDO(const, T):
        return np.exp(
            const.MERLIVAT_NIEF_1967_ALPHA_L_HDO_T2 / T**2
            + const.MERLIVAT_NIEF_1967_ALPHA_L_HDO_T1 / T
            + const.MERLIVAT_NIEF_1967_ALPHA_L_HDO_T0
        )

    @staticmethod
    def alpha_i_HDO(const, T):
        return np.exp(
            const.MERLIVAT_NIEF_1967_ALPHA_I_HDO_T2 / T**2
            + const.MERLIVAT_NIEF_1967_ALPHA_I_HDO_T1 / T
            + const.MERLIVAT_NIEF_1967_ALPHA_I_HDO_T0
        )

    @staticmethod
    def alpha_l_H2_18O(const, T):
        return np.exp(
            const.MAJOUBE_1971_ALPHA_L_H218O_T2 / T**2
            + const.MAJOUBE_1971_ALPHA_L_H218O_T1 / T
            + const.MAJOUBE_1971_ALPHA_L_H218O_T0
        )

    @staticmethod
    def alpha_i_H2_18O(const, T):
        return np.exp(
            const.MAJOUBE_1970_ALPHA_I_H218O_T2 / T**2
            + const.MAJOUBE_1970_ALPHA_I_H218O_T1 / T
            + const.MAJOUBE_1970_ALPHA_I_H218O_T0
        )
