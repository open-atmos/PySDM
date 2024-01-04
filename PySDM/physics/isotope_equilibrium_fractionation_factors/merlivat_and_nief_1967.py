"""
Equilibrium fractionation factors from
[Merlivat and Nief 1967](https://doi.org/10.3402/tellusa.v19i1.9756)
"""
import numpy as np


class MerlivatAndNief1967:
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
