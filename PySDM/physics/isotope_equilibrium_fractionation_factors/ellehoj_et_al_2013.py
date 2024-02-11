"""
Equilibrium fractionation factors from
[Ellehoj et al. 2013](https://doi.org/10.1002/rcm.6668)
"""

import numpy as np


class EllehojEtAl2013:  # pylint: disable=too-few-public-methods
    def __init__(self, _):
        pass

    @staticmethod
    def alpha_i_2H(const, T):
        return np.exp(
            const.ELLEHOJ_ET_AL_2013_ALPHA_I_2H_T2 / T**2
            + const.ELLEHOJ_ET_AL_2013_ALPHA_I_2H_T1 / T
            + const.ELLEHOJ_ET_AL_2013_ALPHA_I_2H_T0
        )
